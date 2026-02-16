"""Merge a LoRA adapter into the base model to produce a standalone model.

After merging, the model can be loaded without the PEFT library and runs
at full speed (no adapter overhead).

Usage:
    # Merge and save to default output path
    uv run python merge_adapter.py

    # Custom paths
    uv run python merge_adapter.py --adapter-path models/biogpt_bc5cdr --output-path models/biogpt_merged

    # Merge and run a quick sanity test
    uv run python merge_adapter.py --test

    # Also export to float16 for smaller disk footprint
    uv run python merge_adapter.py --half
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from transformers import BioGptForCausalLM, BioGptTokenizer


def merge_adapter(
    adapter_path: str = "models/biogpt_bc5cdr",
    output_path: str | None = None,
    half: bool = False,
) -> Path:
    """Merge LoRA adapter weights into the base model.

    Args:
        adapter_path: Path to the fine-tuned adapter checkpoint.
        output_path: Where to save the merged model.
                     Defaults to {adapter_path}_merged.
        half: Cast merged weights to float16 before saving.

    Returns:
        Path to the saved merged model directory.
    """
    adapter_dir = Path(adapter_path)
    if output_path is None:
        output_dir = adapter_dir.parent / f"{adapter_dir.name}_merged"
    else:
        output_dir = Path(output_path)

    # Load adapter config
    config_path = adapter_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json at {adapter_dir}")

    with open(config_path) as f:
        config = json.load(f)

    base_model_name = config.get("base_model", "microsoft/BioGPT-Large")
    use_peft = config.get("use_peft", False)
    has_adapter = (adapter_dir / "adapter_config.json").exists()

    if not use_peft and not has_adapter:
        print("Model is already a full (non-PEFT) checkpoint. Copying as-is.")
        if output_dir != adapter_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            for f in adapter_dir.iterdir():
                if f.is_file():
                    shutil.copy2(f, output_dir / f.name)
        print(f"Saved to: {output_dir}")
        return output_dir

    # ── Step 1: Load base model ──────────────────────────────────────────
    print(f"Loading base model: {base_model_name}")
    start = time.time()
    base_model = BioGptForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    print(f"  Base model loaded ({time.time() - start:.1f}s)")

    # ── Step 2: Load LoRA adapter on top ─────────────────────────────────
    print(f"Loading LoRA adapter from: {adapter_dir}")
    try:
        from peft import PeftModel
    except ImportError:
        raise ImportError(
            "PEFT library is required for merging. Install with: uv add peft"
        )

    start = time.time()
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    print(f"  Adapter loaded ({time.time() - start:.1f}s)")

    # Print adapter info
    peft_model.print_trainable_parameters()

    # ── Step 3: Merge adapter into base weights ──────────────────────────
    print("Merging adapter into base model...")
    start = time.time()
    merged_model = peft_model.merge_and_unload()
    print(f"  Merged ({time.time() - start:.1f}s)")

    # ── Step 4: Optionally cast to half precision ────────────────────────
    if half:
        print("Casting to float16...")
        merged_model = merged_model.half()

    # ── Step 5: Save merged model ────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to: {output_dir}")

    start = time.time()
    merged_model.save_pretrained(output_dir)

    # Save tokenizer
    tokenizer = BioGptTokenizer.from_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(output_dir)

    # Save updated config (mark as non-PEFT so it loads without peft library)
    merged_config = {
        "base_model": base_model_name,
        "use_peft": False,
        "merged_from": str(adapter_dir),
        "entity_types": config.get("entity_types", ["Chemical", "Disease"]),
        "relation_types": config.get("relation_types", ["induces", "no_relation"]),
        "dtype": "float16" if half else "float32",
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(merged_config, f, indent=2)

    elapsed = time.time() - start
    model_size = sum(f.stat().st_size for f in output_dir.iterdir() if f.is_file())
    print(f"  Saved ({elapsed:.1f}s, {model_size / 1e9:.2f} GB)")

    return output_dir


def sanity_test(model_path: Path) -> None:
    """Run a quick inference test on the merged model."""
    from graph_rag.graph_store import parse_cypher_to_triplets
    from graph_rag.medical_re_model import BioGPTREPipeline, get_device

    device = get_device()
    print(f"\nRunning sanity test on merged model (device={device})...")

    pipeline = BioGPTREPipeline.from_pretrained(str(model_path), device=device)

    test_texts = [
        "Cisplatin causes nephrotoxicity and ototoxicity in cancer patients.",
        "Ibuprofen pharmacokinetics were studied in healthy volunteers.",
    ]

    for text in test_texts:
        print(f"\n  Input:  {text}")
        cypher = pipeline.predict_cypher(text)
        triplets = parse_cypher_to_triplets(cypher)

        if triplets:
            for t in triplets:
                print(
                    f"  Output: (:{t['head_label']}) {t['head']} "
                    f"--[:{t['type']}]--> "
                    f"(:{t['tail_label']}) {t['tail']}"
                )
        else:
            cypher_preview = cypher[:120] if cypher else "(empty)"
            print(f"  Output: {cypher_preview}")

    print("\nSanity test complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter into base model"
    )
    parser.add_argument(
        "--adapter-path", type=str, default="models/biogpt_bc5cdr",
        help="Path to the LoRA adapter checkpoint",
    )
    parser.add_argument(
        "--output-path", type=str, default=None,
        help="Output path for merged model (default: {adapter_path}_merged)",
    )
    parser.add_argument(
        "--half", action="store_true",
        help="Save merged model in float16 (halves disk size)",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run sanity test after merging",
    )
    args = parser.parse_args()

    output_dir = merge_adapter(
        adapter_path=args.adapter_path,
        output_path=args.output_path,
        half=args.half,
    )

    print(f"\nMerged model saved to: {output_dir}")
    print("Load it without PEFT:")
    print(f'  pipeline = BioGPTREPipeline.from_pretrained("{output_dir}")')

    if args.test:
        sanity_test(output_dir)


if __name__ == "__main__":
    main()
