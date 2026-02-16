"""Training pipeline for BioGPT Text-to-Cypher with PEFT and optimizations.

Fine-tunes BioGPT-Large on BC5CDR to generate Cypher MERGE statements
from medical/pharmaceutical text.
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    BioGptForCausalLM,
    BioGptTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from graph_rag.data_preparation import (
    CYPHER_PROMPT_PREFIX,
    CYPHER_PROMPT_SUFFIX,
    convert_bc5cdr_to_training_format,
    load_bc5cdr_dataset,
)
from graph_rag.graph_store import parse_cypher_to_triplets

DEFAULT_MODEL = "microsoft/BioGPT"


@dataclass
class TrainingConfig:
    """Configuration for BioGPT text-to-Cypher training.

    Optimised for the smaller BioGPT (390M, hidden=1024, max_pos=1024).
    Key differences vs a Large config:
      - LoRA rank 32 → more adapter capacity to compensate for smaller base
      - All four attention projections targeted (q/k/v/out)
      - Higher dropout (0.1) to prevent overfitting
      - More epochs (6) with patient early stopping
      - Larger effective batch (8×2=16) — fits easily in memory
      - Sequence budget split 448+192=640 (within 1024 position limit)
    """

    # Model
    base_model: str = DEFAULT_MODEL

    # PEFT — higher rank + more modules for smaller base model
    use_peft: bool = True
    lora_r: int = 32
    lora_alpha: int = 64  # 2× rank — standard scaling
    lora_dropout: float = 0.1  # stronger regularisation for small model
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "out_proj"]
    )

    # Training — larger effective batch, more epochs
    batch_size: int = 8
    num_epochs: int = 6
    learning_rate: float = 3e-4  # slightly higher for LoRA + small model
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    warmup_steps: int = 0  # overrides warmup_ratio if > 0
    max_grad_norm: float = 1.0

    # Gradient optimisation
    gradient_accumulation_steps: int = 2  # effective batch = 8×2 = 16
    gradient_checkpointing: bool = True

    # Precision
    fp16: bool = False
    bf16: bool = False  # Apple Silicon: set True for speed

    # Early stopping — more patience for longer training
    early_stopping: bool = True
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.005

    # Data — fits within BioGPT's 1024 position limit (448 + 192 = 640)
    max_input_length: int = 448
    max_target_length: int = 192
    include_negative_samples: bool = True
    max_samples: int | None = None

    # Output
    output_dir: str = "models/biogpt_bc5cdr"
    save_best_only: bool = True
    eval_steps: int = 100
    log_steps: int = 10

    # Scheduler
    scheduler_type: str = "cosine"

    # Device
    device: str = "auto"

    # Misc
    seed: int = 42
    dataloader_num_workers: int = 0


class EarlyStoppingCallback:
    """Early stopping callback to prevent overfitting."""

    def __init__(
        self,
        patience: int = 3,
        threshold: float = 0.01,
        mode: str = "min",
    ):
        self.patience = patience
        self.threshold = threshold
        self.mode = mode
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.best_step = 0

    def __call__(self, value: float, step: int) -> bool:
        """Check if training should stop.

        Returns:
            True if training should stop
        """
        if self.mode == "min":
            improved = value < self.best_value - self.threshold
        else:
            improved = value > self.best_value + self.threshold

        if improved:
            self.best_value = value
            self.counter = 0
            self.best_step = step
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


def setup_peft(model: Any, config: TrainingConfig) -> Any:
    """Setup PEFT (LoRA) for efficient fine-tuning.

    PEFT avoids catastrophic forgetting by only training a small subset of parameters.
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        raise ImportError(
            "PEFT library required for LoRA fine-tuning. "
            "Install with: uv add peft"
        )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model


def setup_gradient_checkpointing(model: Any, enabled: bool = True) -> None:
    """Enable gradient checkpointing to reduce memory usage.

    Trades compute for memory by recomputing activations during backward pass.
    """
    if enabled and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")


def get_optimizer_grouped_parameters(model: Any, weight_decay: float) -> list[dict]:
    """Get parameter groups with different weight decay for biases and LayerNorm."""
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


class CypherDataset(Dataset):
    """Dataset for BioGPT text-to-Cypher training.

    Each sample has a 'prompt' (medical text) and 'target' (Cypher statements).
    Prompt and target are tokenized separately and concatenated so the label
    mask boundary is exact (no BPE boundary ambiguity).
    """

    def __init__(
        self,
        samples: list[dict],
        tokenizer: BioGptTokenizer,
        max_input_length: int = 512,
        max_target_length: int = 128,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.max_total_length = max_input_length + max_target_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        prompt = sample["prompt"]
        target = " " + sample["target"]  # leading space as separator

        # Tokenize prompt and target separately to get exact boundary
        prompt_encoding = self.tokenizer(
            prompt,
            max_length=self.max_input_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        target_encoding = self.tokenizer(
            target,
            max_length=self.max_target_length,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt",
        )

        prompt_ids = prompt_encoding["input_ids"].squeeze(0)
        target_ids = target_encoding["input_ids"].squeeze(0)
        prompt_len = prompt_ids.shape[0]

        # Concatenate
        input_ids = torch.cat([prompt_ids, target_ids])

        # Truncate to max total length
        if input_ids.shape[0] > self.max_total_length:
            input_ids = input_ids[: self.max_total_length]

        # Build labels: -100 for prompt tokens and padding
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        # Pad to max total length
        current_len = input_ids.shape[0]
        pad_len = self.max_total_length - current_len
        pad_id = self.tokenizer.pad_token_id

        if pad_len > 0:
            input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_id, dtype=input_ids.dtype)])
            labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=labels.dtype)])

        attention_mask = (input_ids != pad_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate batch of samples."""
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }


def get_device(config: TrainingConfig) -> torch.device:
    """Get the target device."""
    if config.device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(config.device)


def get_autocast_context(device: torch.device, config: TrainingConfig):
    """Get autocast context for mixed precision training."""
    if not (config.bf16 or config.fp16):
        return torch.enable_grad()

    if config.bf16:
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    elif device.type == "mps":
        # MPS supports float16 autocast; bf16 falls back to fp16
        return torch.autocast(device_type="mps", dtype=torch.float16)
    else:
        return torch.enable_grad()


def preview_test_samples(
    model: Any,
    tokenizer: BioGptTokenizer,
    test_samples: list[dict],
    device: torch.device,
    epoch: int,
    config: TrainingConfig,
) -> None:
    """Run inference on a few test samples and print generated vs expected Cypher.

    Called after each epoch's evaluation to give a qualitative view of model
    progress during training.
    """
    model.eval()
    max_gen_length = config.max_input_length + config.max_target_length

    print(f"\n{'─' * 70}")
    print(f"  TEST PREVIEW — Epoch {epoch} ({len(test_samples)} samples)")
    print(f"{'─' * 70}")

    total_expected = 0
    total_matched = 0

    for i, sample in enumerate(test_samples):
        prompt = sample["prompt"]
        expected_target = sample["target"]

        # Tokenize the prompt
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=config.max_input_length,
            truncation=True,
            padding=True,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_gen_length,
                num_beams=3,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Strip prompt from generated text
        if CYPHER_PROMPT_PREFIX in generated:
            cypher_part = generated.split("Cypher:")[-1].strip()
        else:
            cypher_part = generated.strip()

        # Parse triplets for comparison
        expected_triplets = parse_cypher_to_triplets(expected_target)
        predicted_triplets = parse_cypher_to_triplets(cypher_part)

        expected_keys = {
            (t["head"].lower(), t["type"].lower(), t["tail"].lower())
            for t in expected_triplets
        }
        predicted_keys = {
            (t["head"].lower(), t["type"].lower(), t["tail"].lower())
            for t in predicted_triplets
        }
        matched = expected_keys & predicted_keys

        total_expected += len(expected_keys)
        total_matched += len(matched)

        # Extract short text snippet from prompt
        raw_text = prompt
        if "from text:" in raw_text:
            raw_text = raw_text.split("from text:", 1)[1]
        if "\nCypher:" in raw_text:
            raw_text = raw_text.rsplit("\nCypher:", 1)[0]
        raw_text = raw_text.strip()[:100]

        status = "MATCH" if matched else ("OK" if not expected_keys and not predicted_keys else "MISS")

        print(f"\n  [{i + 1}] {status}  text: {raw_text}...")
        print(f"      Expected ({len(expected_keys)} triplets): {expected_target[:120]}")
        print(f"      Generated({len(predicted_keys)} triplets): {cypher_part[:120]}")
        if matched:
            print(f"      Matched: {len(matched)}/{len(expected_keys)}")

    # Summary
    recall = total_matched / max(total_expected, 1)
    print(f"\n  Summary: {total_matched}/{total_expected} triplets matched (recall: {recall:.1%})")
    print(f"{'─' * 70}\n")

    model.train()


def train(config: TrainingConfig) -> None:
    """Train BioGPT text-to-Cypher with PEFT and optimizations."""
    # Set seed for reproducibility
    torch.manual_seed(config.seed)

    # Device setup
    device = get_device(config)
    print(f"Training on device: {device}")

    # Precision settings — GradScaler only works on CUDA
    use_amp = config.fp16 or config.bf16
    use_cuda_scaler = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_scaler)
    if use_amp and device.type != "cuda":
        print(f"Note: AMP enabled via autocast on {device.type}; GradScaler disabled (CUDA only)")

    # Load tokenizer
    tokenizer = BioGptTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading base model: {config.base_model}")
    model = BioGptForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        low_cpu_mem_usage=True,
    )

    # Enable gradient checkpointing before PEFT
    setup_gradient_checkpointing(model, config.gradient_checkpointing)

    # Setup PEFT (LoRA) to avoid catastrophic forgetting
    if config.use_peft:
        print("Setting up PEFT with LoRA...")
        model = setup_peft(model, config)

    model = model.to(device)

    # Load dataset
    print("Loading BC5CDR dataset...")
    train_samples = load_bc5cdr_dataset("train")
    eval_samples = load_bc5cdr_dataset("validation")

    train_data = convert_bc5cdr_to_training_format(
        train_samples, include_negative_samples=config.include_negative_samples
    )
    eval_data = convert_bc5cdr_to_training_format(
        eval_samples, include_negative_samples=config.include_negative_samples
    )

    train_samples = train_data["samples"]
    eval_samples = eval_data["samples"]

    if config.max_samples:
        train_samples = train_samples[: config.max_samples]
        eval_samples = eval_samples[: min(config.max_samples, len(eval_samples))]

    # Load test samples for periodic inference preview (max 5)
    test_raw = load_bc5cdr_dataset("test")
    test_data = convert_bc5cdr_to_training_format(
        test_raw, include_negative_samples=False
    )
    test_preview_samples = test_data["samples"][:5]

    print(f"Training samples: {len(train_samples)}")
    print(f"Evaluation samples: {len(eval_samples)}")
    print(f"Test preview samples: {len(test_preview_samples)}")

    # Create datasets
    train_dataset = CypherDataset(
        samples=train_samples,
        tokenizer=tokenizer,
        max_input_length=config.max_input_length,
        max_target_length=config.max_target_length,
    )

    eval_dataset = CypherDataset(
        samples=eval_samples,
        tokenizer=tokenizer,
        max_input_length=config.max_input_length,
        max_target_length=config.max_target_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.dataloader_num_workers,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        num_workers=config.dataloader_num_workers,
    )

    # Optimizer with parameter groups
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, config.weight_decay
    )
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # Scheduler
    num_update_steps_per_epoch = max(len(train_loader) // config.gradient_accumulation_steps, 1)
    total_steps = num_update_steps_per_epoch * config.num_epochs

    if config.warmup_steps > 0:
        warmup_steps = config.warmup_steps
    else:
        warmup_steps = int(total_steps * config.warmup_ratio)

    if config.scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
    elif config.scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
    else:
        scheduler = None

    print(f"Total training steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")

    # Output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_to_save = {
        "base_model": config.base_model,
        "use_peft": config.use_peft,
        "lora_r": config.lora_r if config.use_peft else None,
        "lora_alpha": config.lora_alpha if config.use_peft else None,
        "entity_types": ["Chemical", "Disease"],
        "relation_types": ["induces", "no_relation"],
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_to_save, f, indent=2)

    # Early stopping
    early_stopper = EarlyStoppingCallback(
        patience=config.early_stopping_patience,
        threshold=config.early_stopping_threshold,
    ) if config.early_stopping else None

    # Training state
    global_step = 0
    best_loss = float("inf")

    # Build autocast context once (Bug fix: was recreated every step)
    autocast_context = get_autocast_context(device, config)

    # Training loop
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{config.num_epochs}",
        )

        num_steps = len(train_loader)

        for step, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward with autocast
            with autocast_context:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / config.gradient_accumulation_steps

            # Backward with gradient scaling
            scaler.scale(loss).backward()

            is_accumulation_boundary = (step + 1) % config.gradient_accumulation_steps == 0
            is_last_step = (step + 1) == num_steps

            # Flush gradients at accumulation boundary OR at end of epoch
            if is_accumulation_boundary or is_last_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if scheduler is not None:
                    scheduler.step()

                global_step += 1

            epoch_loss += loss.item() * config.gradient_accumulation_steps

            current_lr = optimizer.param_groups[0]["lr"]
            progress_bar.set_postfix({
                "loss": f"{loss.item() * config.gradient_accumulation_steps:.4f}",
                "lr": f"{current_lr:.2e}",
            })

        # End of epoch
        avg_train_loss = epoch_loss / num_steps
        print(f"\nEpoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}")

        # Evaluation
        eval_loss = evaluate(model, eval_loader, device, config)
        print(f"Evaluation Loss: {eval_loss:.4f}")

        # Save best model only
        if eval_loss < best_loss:
            improvement = best_loss - eval_loss
            best_loss = eval_loss

            # Save model
            save_model(model, tokenizer, output_dir, config.use_peft)
            print(f"New best model saved (loss: {best_loss:.4f}, improvement: {improvement:.4f})")

            # Save training state
            training_state = {
                "global_step": global_step,
                "best_loss": best_loss,
                "epoch": epoch + 1,
            }
            with open(output_dir / "training_state.json", "w") as f:
                json.dump(training_state, f, indent=2)
        else:
            print(f"No improvement (best: {best_loss:.4f})")

        # Test sample preview — qualitative check on generation quality
        if test_preview_samples:
            preview_test_samples(
                model, tokenizer, test_preview_samples, device, epoch + 1, config
            )

        # Early stopping check
        if early_stopper:
            should_stop = early_stopper(eval_loss, global_step)
            remaining = config.early_stopping_patience - early_stopper.counter
            if early_stopper.counter > 0:
                print(
                    f"Early stopping: no improvement for {early_stopper.counter}/{config.early_stopping_patience} "
                    f"epochs (best: {early_stopper.best_value:.4f})"
                )
            else:
                print(f"Early stopping: patience reset (best: {early_stopper.best_value:.4f})")
            if should_stop:
                print(f"\n*** Early stopping triggered at epoch {epoch + 1} ***")
                print(f"Best loss: {early_stopper.best_value:.4f} at step {early_stopper.best_step}")
                break

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Model saved to: {output_dir}")


def evaluate(
    model: Any,
    eval_loader: DataLoader,
    device: torch.device,
    config: TrainingConfig,
) -> float:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    autocast_context = get_autocast_context(device, config)

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast_context:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

            total_loss += outputs.loss.item()
            num_batches += 1

    model.train()
    return total_loss / max(num_batches, 1)


def save_model(model: Any, tokenizer: BioGptTokenizer, output_dir: Path, use_peft: bool) -> None:
    """Save model and tokenizer."""
    if use_peft:
        # PEFT model has its own save method
        model.save_pretrained(output_dir)
    else:
        # Full model save
        torch.save(model.state_dict(), output_dir / "pytorch_model.bin")

    tokenizer.save_pretrained(output_dir)


def main():
    parser = argparse.ArgumentParser(description="Train BioGPT text-to-Cypher with LoRA")

    # Model
    parser.add_argument("--base-model", type=str, default=DEFAULT_MODEL)

    # PEFT
    parser.add_argument("--use-peft", action="store_true", default=True)
    parser.add_argument("--no-peft", action="store_false", dest="use_peft")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--output-dir", type=str, default="models/biogpt_bc5cdr")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # Precision
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")

    # Gradient checkpointing
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", action="store_false", dest="gradient_checkpointing")

    # Early stopping
    parser.add_argument("--early-stopping", action="store_true", default=True)
    parser.add_argument("--no-early-stopping", action="store_false", dest="early_stopping")
    parser.add_argument("--early-stopping-patience", type=int, default=5)

    # Scheduler
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine", "constant"])

    # Data
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-negative-samples", action="store_true")

    # Device
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    config = TrainingConfig(
        base_model=args.base_model,
        use_peft=args.use_peft,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        early_stopping=args.early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        scheduler_type=args.scheduler,
        max_samples=args.max_samples,
        include_negative_samples=not args.no_negative_samples,
        device=args.device,
        seed=args.seed,
    )

    train(config)


if __name__ == "__main__":
    main()
