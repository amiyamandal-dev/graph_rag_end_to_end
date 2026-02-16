import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from typing import Literal

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DEFAULT_MODEL = "Babelscape/rebel-large"
DEFAULT_BIOGPT_MODEL = "microsoft/BioGPT-Large"
DEFAULT_MEDICAL_MODEL = "models/biogpt_bc5cdr"

ModelType = Literal["rebel", "medical", "biogpt"]


def get_device() -> torch.device:
    """Return the best available device (MPS on Apple Silicon, else CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(
    model_name: str = DEFAULT_MODEL, device: torch.device | None = None
) -> tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    """Load the REBEL model and tokenizer, moving the model to the given device."""
    if device is None:
        device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def extract_triplets_from_text(text: str) -> list[dict]:
    """Parse REBEL's generated text into structured triplets.

    REBEL outputs text in the format:
        <triplet> subject <subj> relation <obj> object <triplet> ...

    Returns list of dicts with keys: head, type, tail.
    """
    triplets = []
    # Clean up special tokens
    text = text.replace("<s>", "").replace("</s>", "").replace("<pad>", "").strip()

    for triplet_str in text.split("<triplet>"):
        triplet_str = triplet_str.strip()
        if not triplet_str:
            continue

        # Split on <subj> to get head and the rest
        parts = triplet_str.split("<subj>")
        if len(parts) != 2:
            continue

        head = parts[0].strip()
        rest = parts[1]

        # Split on <obj> to get relation and tail
        obj_parts = rest.split("<obj>")
        if len(obj_parts) != 2:
            continue

        relation = obj_parts[0].strip()
        tail = obj_parts[1].strip()

        if head and relation and tail:
            triplets.append({"head": head, "type": relation, "tail": tail})

    return triplets


def extract_relations(
    chunks: list[str],
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
) -> list[dict]:
    """Run REBEL inference on text chunks and return deduplicated triplets.

    Each chunk is tokenized and passed through the model with beam search.
    Results are deduplicated by (head, type, tail) tuple.
    """
    device = next(model.parameters()).device
    seen = set()
    triplets = []

    for chunk in chunks:
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                num_beams=3,
                num_return_sequences=3,
                max_length=256,
            )

        for seq in outputs:
            decoded = tokenizer.decode(seq, skip_special_tokens=False)
            for triplet in extract_triplets_from_text(decoded):
                key = (triplet["head"], triplet["type"], triplet["tail"])
                if key not in seen:
                    seen.add(key)
                    triplets.append(triplet)

    return triplets


def load_medical_model(
    model_path: str = DEFAULT_MEDICAL_MODEL,
    device: torch.device | None = None,
):
    """Load the fine-tuned BioGPT medical relation extraction model.

    Args:
        model_path: Path to the fine-tuned model checkpoint
        device: Target device (auto-detected if None)

    Returns:
        BioGPTREPipeline ready for inference
    """
    from graph_rag.medical_re_model import BioGPTREPipeline, get_device as _get_device

    if device is None:
        device = _get_device()

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Medical model not found at {model_path}. "
            "Train the model first using: python -m graph_rag.train"
        )

    return BioGPTREPipeline.from_pretrained(model_path, device=device)


def load_biogpt_model(
    model_path: str | None = None,
    device: torch.device | None = None,
):
    """Load BioGPT model (fine-tuned or base).

    Args:
        model_path: Path to fine-tuned model (uses base model if None)
        device: Target device

    Returns:
        BioGPTREPipeline instance
    """
    from graph_rag.medical_re_model import load_model

    if device is None:
        device = get_device()

    return load_model(model_path, device)


def extract_relations_medical(
    chunks: list[str],
    pipeline,  # BioGPTREPipeline
) -> list[dict]:
    """Extract relations using the fine-tuned BioGPT medical model.

    Args:
        chunks: List of text chunks to process
        pipeline: BioGPTREPipeline instance

    Returns:
        List of unique triplets: {"head": str, "type": str, "tail": str}
    """
    seen = set()
    triplets = []

    for chunk in chunks:
        chunk_triplets = pipeline.predict(chunk)
        for triplet in chunk_triplets:
            key = (triplet["head"], triplet["type"], triplet["tail"])
            if key not in seen:
                seen.add(key)
                triplets.append(triplet)

    return triplets


def extract_relations_biogpt(
    chunks: list[str],
    pipeline,  # BioGPTREPipeline
) -> list[dict]:
    """Extract relations using BioGPT model.

    Args:
        chunks: List of text chunks to process
        pipeline: BioGPTREPipeline instance

    Returns:
        List of unique triplets: {"head": str, "type": str, "tail": str}
    """
    from graph_rag.medical_re_model import extract_relations_biogpt as _extract

    return _extract(chunks, pipeline)


def load_model_by_type(
    model_type: ModelType = "rebel",
    model_path: str | None = None,
    device: torch.device | None = None,
):
    """Load the appropriate model based on type.

    Args:
        model_type: "rebel" for REBEL, "medical" for fine-tuned BioGPT, "biogpt" for base BioGPT
        model_path: Path to model (uses defaults if None)
        device: Target device

    Returns:
        Tuple of (model, tokenizer) for REBEL, or BioGPTREPipeline for medical/biogpt
    """
    if device is None:
        device = get_device()

    if model_type == "medical":
        path = model_path or DEFAULT_MEDICAL_MODEL
        return load_medical_model(path, device)
    elif model_type == "biogpt":
        return load_biogpt_model(model_path, device)
    else:
        path = model_path or DEFAULT_MODEL
        return load_model(model_name=path, device=device)
