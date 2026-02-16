"""BioGPT-based Medical Text-to-Cypher Model.

Fine-tuned to convert medical/pharmaceutical text into Cypher MERGE statements
for direct ingestion into a graph database (Memgraph / Neo4j).
"""

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from typing import Any

import torch
import torch.nn as nn
from transformers import BioGptForCausalLM, BioGptTokenizer

from graph_rag.data_preparation import CYPHER_PROMPT_PREFIX, CYPHER_PROMPT_SUFFIX
from graph_rag.graph_store import parse_cypher_to_triplets

DEFAULT_MODEL = "microsoft/BioGPT"

# BC5CDR entity types
DEFAULT_ENTITY_TYPES = [
    "Chemical",
    "Disease",
]

# BC5CDR relation types (Chemical-Induced Disease)
DEFAULT_RELATION_TYPES = [
    "CID",  # Chemical Induces Disease
    "no_relation",
]


class BioGPTREModel(nn.Module):
    """BioGPT-based Text-to-Cypher Model.

    Uses BioGPT-Large for generative Cypher production from medical text.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        dropout: float = 0.1,
        model: Any = None,
    ):
        super().__init__()
        if model is not None:
            self.biogpt = model
        else:
            self.biogpt = BioGptForCausalLM.from_pretrained(model_name)
        self.hidden_size = self.biogpt.config.hidden_size
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for causal language modeling."""
        outputs = self.biogpt(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        return {
            "logits": outputs.logits,
            "loss": outputs.loss if labels is not None else None,
        }

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 256,
        num_beams: int = 3,
        num_return_sequences: int = 1,
    ) -> torch.Tensor:
        """Generate output sequences."""
        return self.biogpt.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.biogpt.config.eos_token_id,
        )


class BioGPTREPipeline:
    """Inference pipeline for BioGPT Text-to-Cypher generation.

    Prompt format (must match training):
        "Generate Cypher to store medical relations from text: {text}\\nCypher:"

    Output: one or more MERGE statements, or "// No relations".
    """

    def __init__(
        self,
        model: BioGPTREModel,
        tokenizer: BioGptTokenizer,
        entity_types: list[str] | None = None,
        relation_types: list[str] | None = None,
        device: torch.device | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.entity_types = entity_types or DEFAULT_ENTITY_TYPES
        self.relation_types = relation_types or DEFAULT_RELATION_TYPES

        if device is None:
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        self.model = self.model.to(device)
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: torch.device | None = None,
    ) -> "BioGPTREPipeline":
        """Load a fine-tuned model from a checkpoint directory.

        Supports both full model and PEFT (LoRA) checkpoints.
        """
        import json

        if device is None:
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # Load config
        config_path = f"{model_path}/config.json"
        with open(config_path) as f:
            config = json.load(f)

        # Load tokenizer
        tokenizer = BioGptTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Check if this is a PEFT model
        use_peft = config.get("use_peft", False)
        peft_config_path = os.path.join(model_path, "adapter_config.json")

        if use_peft or os.path.exists(peft_config_path):
            try:
                from peft import PeftModel

                base_model = BioGptForCausalLM.from_pretrained(
                    config.get("base_model", DEFAULT_MODEL),
                    torch_dtype=torch.float32,
                )

                peft_model = PeftModel.from_pretrained(base_model, model_path)
                peft_model = peft_model.to(device)

                wrapped_model = BioGPTREModel(model=peft_model)

            except ImportError:
                raise ImportError(
                    "PEFT library required to load LoRA model. "
                    "Install with: uv add peft"
                )
        else:
            # Load full model
            wrapped_model = BioGPTREModel(
                model_name=config.get("base_model", DEFAULT_MODEL),
            )

            weights_path = f"{model_path}/pytorch_model.bin"
            if os.path.exists(weights_path):
                state_dict = torch.load(
                    weights_path, map_location=device, weights_only=True
                )
                wrapped_model.load_state_dict(state_dict)

            wrapped_model = wrapped_model.to(device)

        return cls(
            model=wrapped_model,
            tokenizer=tokenizer,
            entity_types=config.get("entity_types", DEFAULT_ENTITY_TYPES),
            relation_types=config.get("relation_types", DEFAULT_RELATION_TYPES),
            device=device,
        )

    def save(self, model_path: str) -> None:
        """Save model and configuration."""
        import json

        os.makedirs(model_path, exist_ok=True)

        config = {
            "base_model": DEFAULT_MODEL,
            "entity_types": self.entity_types,
            "relation_types": self.relation_types,
        }
        with open(f"{model_path}/config.json", "w") as f:
            json.dump(config, f, indent=2)

        torch.save(self.model.state_dict(), f"{model_path}/pytorch_model.bin")
        self.tokenizer.save_pretrained(model_path)

    def _build_prompt(self, text: str, max_length: int = 512) -> str:
        """Build inference prompt matching the training format."""
        if len(text) > max_length:
            text = text[:max_length]
        return f"{CYPHER_PROMPT_PREFIX} {text}{CYPHER_PROMPT_SUFFIX}"

    def predict_cypher(self, text: str) -> str:
        """Generate raw Cypher from medical text.

        Args:
            text: Medical text to process.

        Returns:
            Generated Cypher string (may contain MERGE statements or "// No relations").
        """
        prompt = self._build_prompt(text)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=512,
                num_beams=3,
                num_return_sequences=1,
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Strip the prompt portion from the generated output
        if CYPHER_PROMPT_PREFIX in generated:
            cypher_part = generated.split("Cypher:")[-1].strip()
        else:
            cypher_part = generated.strip()

        return cypher_part

    def predict(self, text: str) -> list[dict[str, str]]:
        """Extract relations from medical text as triplet dicts.

        Generates Cypher internally and parses it back to triplets for
        compatibility with the rest of the pipeline.

        Args:
            text: Medical text to process.

        Returns:
            List of triplets: {"head": str, "type": str, "tail": str, ...}
        """
        cypher = self.predict_cypher(text)
        return parse_cypher_to_triplets(cypher)


def get_device() -> torch.device:
    """Return the best available device (MPS on Apple Silicon, else CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(
    model_path: str | None = None,
    device: torch.device | None = None,
) -> BioGPTREPipeline:
    """Load the BioGPT RE model for inference.

    Args:
        model_path: Path to fine-tuned model checkpoint (uses base model if None)
        device: Target device

    Returns:
        BioGPTREPipeline ready for inference
    """
    if device is None:
        device = get_device()

    if model_path and os.path.exists(model_path):
        return BioGPTREPipeline.from_pretrained(model_path, device=device)

    # Load base model without fine-tuning
    tokenizer = BioGptTokenizer.from_pretrained(DEFAULT_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = BioGPTREModel()
    return BioGPTREPipeline(model, tokenizer, device=device)


def extract_relations_biogpt(
    chunks: list[str],
    pipeline: BioGPTREPipeline,
) -> list[dict]:
    """Run BioGPT RE inference on text chunks and return deduplicated triplets.

    Args:
        chunks: List of text chunks to process
        pipeline: BioGPTREPipeline for inference

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
