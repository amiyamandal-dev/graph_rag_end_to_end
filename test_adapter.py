"""Test a fine-tuned LoRA adapter on sample medical texts.

Usage:
    # Test on built-in examples
    uv run python test_adapter.py

    # Test on validation split from BC5CDR
    uv run python test_adapter.py --eval-split validation --max-samples 20

    # Custom adapter path
    uv run python test_adapter.py --model-path models/biogpt_bc5cdr

    # Test on your own text
    uv run python test_adapter.py --text "Aspirin causes Reye syndrome in children."
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import BioGptTokenizer

from graph_rag.graph_store import parse_cypher_to_triplets
from graph_rag.medical_re_model import BioGPTREPipeline, get_device

SAMPLE_TEXTS = [
    (
        "Cisplatin is widely used in cancer chemotherapy but frequently causes "
        "nephrotoxicity and ototoxicity. Adequate hydration may reduce the risk "
        "of renal damage."
    ),
    (
        "A 45-year-old patient developed severe hepatotoxicity after receiving "
        "high-dose methotrexate for rheumatoid arthritis. Liver enzymes were "
        "elevated within two weeks of treatment initiation."
    ),
    (
        "Doxorubicin-induced cardiotoxicity remains a major concern in breast "
        "cancer treatment. Cumulative doses exceeding 550 mg/m2 significantly "
        "increase the risk of congestive heart failure."
    ),
    (
        "Administration of vancomycin was associated with red man syndrome "
        "characterized by flushing, erythema, and pruritus. Slowing the "
        "infusion rate typically resolves symptoms."
    ),
    (
        "The pharmacokinetics of amoxicillin were studied in 30 healthy "
        "volunteers. No significant adverse effects were observed during "
        "the 14-day trial period."
    ),
]


def load_pipeline(model_path: str, device: torch.device) -> BioGPTREPipeline:
    """Load the fine-tuned pipeline from a checkpoint directory."""
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No config.json found at {model_path}. "
            "Train the model first with: uv run python -m graph_rag.train"
        )

    print(f"Loading adapter from: {model_path}")
    print(f"Device: {device}")
    start = time.time()
    pipeline = BioGPTREPipeline.from_pretrained(model_path, device=device)
    print(f"Loaded in {time.time() - start:.1f}s\n")
    return pipeline


def test_on_texts(
    pipeline: BioGPTREPipeline,
    texts: list[str],
    labels: list[str] | None = None,
) -> list[dict]:
    """Run inference on a list of texts and print results."""
    results = []

    for i, text in enumerate(texts):
        label = labels[i] if labels else f"Sample {i + 1}"
        print(f"{'=' * 80}")
        print(f"[{label}]")
        print(f"{'=' * 80}")
        print(f"\nINPUT TEXT:\n  {text[:200]}{'...' if len(text) > 200 else ''}\n")

        start = time.time()
        cypher = pipeline.predict_cypher(text)
        elapsed = time.time() - start

        print(f"GENERATED CYPHER ({elapsed:.2f}s):")
        if cypher.strip():
            for line in cypher.strip().split("\n"):
                print(f"  {line}")
        else:
            print("  (empty)")

        triplets = parse_cypher_to_triplets(cypher)
        print(f"\nPARSED TRIPLETS ({len(triplets)}):")
        if triplets:
            for t in triplets:
                print(
                    f"  (:{t['head_label']}) {t['head']}  "
                    f"--[:{t['type'].upper().replace(' ', '_')}]-->  "
                    f"(:{t['tail_label']}) {t['tail']}"
                )
        else:
            print("  No relations extracted")

        print()

        results.append({
            "text": text,
            "cypher": cypher,
            "triplets": triplets,
            "time": elapsed,
        })

    return results


def test_on_bc5cdr(
    pipeline: BioGPTREPipeline,
    split: str = "test",
    max_samples: int = 10,
) -> dict:
    """Evaluate on BC5CDR split and report accuracy metrics."""
    from graph_rag.data_preparation import (
        convert_bc5cdr_to_training_format,
        load_bc5cdr_dataset,
    )

    print(f"\n{'=' * 80}")
    print(f"EVALUATION ON BC5CDR {split.upper()} SPLIT")
    print(f"{'=' * 80}\n")

    raw_samples = load_bc5cdr_dataset(split)
    data = convert_bc5cdr_to_training_format(
        raw_samples, include_negative_samples=True
    )
    samples = data["samples"][:max_samples]
    print(f"Evaluating on {len(samples)} samples...\n")

    total = 0
    correct_has_relations = 0  # predicted relations when expected
    correct_no_relations = 0  # predicted no relations when expected
    total_expected_triplets = 0
    total_predicted_triplets = 0
    total_matched_triplets = 0
    total_time = 0.0

    for i, sample in enumerate(samples):
        prompt_text = sample["prompt"]
        expected_target = sample["target"]

        # Extract the raw text from prompt (strip prefix/suffix)
        raw_text = prompt_text
        if "from text:" in raw_text:
            raw_text = raw_text.split("from text:", 1)[1]
        if "\nCypher:" in raw_text:
            raw_text = raw_text.rsplit("\nCypher:", 1)[0]
        raw_text = raw_text.strip()

        start = time.time()
        predicted_cypher = pipeline.predict_cypher(raw_text)
        elapsed = time.time() - start
        total_time += elapsed

        expected_triplets = parse_cypher_to_triplets(expected_target)
        predicted_triplets = parse_cypher_to_triplets(predicted_cypher)

        has_expected = len(expected_triplets) > 0
        has_predicted = len(predicted_triplets) > 0

        # Direction check: did the model correctly predict presence/absence?
        if has_expected and has_predicted:
            correct_has_relations += 1
        elif not has_expected and not has_predicted:
            correct_no_relations += 1

        # Triplet-level matching (head, type, tail) — case-insensitive
        expected_keys = {
            (t["head"].lower(), t["type"].lower(), t["tail"].lower())
            for t in expected_triplets
        }
        predicted_keys = {
            (t["head"].lower(), t["type"].lower(), t["tail"].lower())
            for t in predicted_triplets
        }
        matched = expected_keys & predicted_keys

        total_expected_triplets += len(expected_keys)
        total_predicted_triplets += len(predicted_keys)
        total_matched_triplets += len(matched)
        total += 1

        status = "HIT" if matched else ("OK" if not has_expected and not has_predicted else "MISS")
        print(
            f"  [{i + 1:3d}/{len(samples)}] {status:4s}  "
            f"expected={len(expected_keys):2d}  predicted={len(predicted_keys):2d}  "
            f"matched={len(matched):2d}  ({elapsed:.2f}s)"
        )

    # Summary
    print(f"\n{'─' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'─' * 60}")
    print(f"  Samples evaluated:     {total}")
    print(f"  Avg inference time:    {total_time / max(total, 1):.2f}s")

    direction_acc = (correct_has_relations + correct_no_relations) / max(total, 1)
    print(f"  Direction accuracy:    {direction_acc:.1%} (relation present/absent)")

    precision = total_matched_triplets / max(total_predicted_triplets, 1)
    recall = total_matched_triplets / max(total_expected_triplets, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    print(f"  Triplet precision:     {precision:.1%} ({total_matched_triplets}/{total_predicted_triplets})")
    print(f"  Triplet recall:        {recall:.1%} ({total_matched_triplets}/{total_expected_triplets})")
    print(f"  Triplet F1:            {f1:.1%}")
    print()

    return {
        "total": total,
        "direction_accuracy": direction_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_time": total_time / max(total, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned BioGPT adapter")
    parser.add_argument(
        "--model-path", type=str, default="models/biogpt_bc5cdr",
        help="Path to adapter checkpoint directory",
    )
    parser.add_argument(
        "--text", type=str, default=None,
        help="Custom text to test (overrides built-in samples)",
    )
    parser.add_argument(
        "--eval-split", type=str, default=None, choices=["train", "validation", "test"],
        help="Evaluate on a BC5CDR split instead of sample texts",
    )
    parser.add_argument(
        "--max-samples", type=int, default=20,
        help="Max samples for BC5CDR evaluation",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: auto, cpu, mps, cuda",
    )
    parser.add_argument(
        "--save-results", type=str, default=None,
        help="Save results to JSON file",
    )
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)

    # Load
    pipeline = load_pipeline(args.model_path, device)

    # Run
    if args.text:
        results = test_on_texts(pipeline, [args.text], labels=["Custom input"])
    elif args.eval_split:
        metrics = test_on_bc5cdr(pipeline, args.eval_split, args.max_samples)
        results = [{"metrics": metrics}]
    else:
        results = test_on_texts(pipeline, SAMPLE_TEXTS)

    # Optionally save
    if args.save_results:
        out_path = Path(args.save_results)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
