"""Generate and save sample training/test data for the text-to-Cypher pipeline.

Usage:
    uv run python generate_samples.py
    uv run python generate_samples.py --max-samples 50 --output-dir data/samples
"""

import argparse
import json
from pathlib import Path

from graph_rag.data_preparation import (
    convert_bc5cdr_to_training_format,
    load_bc5cdr_dataset,
)
from graph_rag.graph_store import parse_cypher_to_triplets


def generate_samples(
    max_samples: int | None = None,
    output_dir: str = "data/samples",
    include_negative_samples: bool = True,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": "train",
        "validation": "validation",
        "test": "test",
    }

    for split_name, split_key in splits.items():
        print(f"\n{'=' * 60}")
        print(f"Loading {split_name} split...")
        print(f"{'=' * 60}")

        raw_samples = load_bc5cdr_dataset(split_key)
        print(f"  Raw documents: {len(raw_samples)}")

        data = convert_bc5cdr_to_training_format(
            raw_samples,
            include_negative_samples=include_negative_samples,
        )
        samples = data["samples"]

        if max_samples:
            samples = samples[:max_samples]

        # Count positives vs negatives
        positives = [s for s in samples if s["target"] != "// No relations"]
        negatives = [s for s in samples if s["target"] == "// No relations"]
        print(f"  Training pairs: {len(samples)} ({len(positives)} positive, {len(negatives)} negative)")

        # Save full dataset
        out_file = output_path / f"{split_name}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        print(f"  Saved to: {out_file}")

        # Save a human-readable preview
        preview_file = output_path / f"{split_name}_preview.txt"
        with open(preview_file, "w", encoding="utf-8") as f:
            f.write(f"{'=' * 80}\n")
            f.write(f"{split_name.upper()} SPLIT — {len(samples)} samples ")
            f.write(f"({len(positives)} positive, {len(negatives)} negative)\n")
            f.write(f"{'=' * 80}\n\n")

            for i, sample in enumerate(samples[:10]):  # preview first 10
                f.write(f"--- Sample {i + 1} ---\n\n")
                f.write(f"PROMPT:\n{sample['prompt']}\n\n")
                f.write(f"TARGET:\n{sample['target']}\n\n")

                if sample["target"] != "// No relations":
                    triplets = parse_cypher_to_triplets(sample["target"])
                    f.write("GRAPH EDGES:\n")
                    for t in triplets:
                        f.write(
                            f"  (:{t['head_label']}) {t['head']}  "
                            f"--[:{t['type'].upper().replace(' ', '_')}]-->  "
                            f"(:{t['tail_label']}) {t['tail']}\n"
                        )
                f.write("\n")

        print(f"  Preview:  {preview_file}")

    # Print a quick stat summary
    print(f"\n{'=' * 60}")
    print("Done. Files written to:", output_path)
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text-to-Cypher training samples")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap samples per split")
    parser.add_argument("--output-dir", type=str, default="data/samples", help="Output directory")
    parser.add_argument("--no-negative-samples", action="store_true", help="Exclude negative samples")
    args = parser.parse_args()

    generate_samples(
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        include_negative_samples=not args.no_negative_samples,
    )
