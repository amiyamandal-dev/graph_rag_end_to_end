# Graph RAG Medical Relation Extraction

A pipeline for extracting medical knowledge triplets from PDF documents and storing them in a graph database.

## Architecture

```
PDF Document
    |
    v
[PDF Ingestion] ---> Text Chunks
    |
    v
[Relation Extraction] ---> Triplets (head, type, tail)
    |
    v
[Graph Storage] ---> Memgraph/Neo4j
```

## Model Options

| Model | Type | Description |
|-------|------|-------------|
| REBEL | Seq2Seq | General relation extraction (Babelscape/rebel-large) |
| BioGPT | Causal LM | Biomedical generative model (microsoft/BioGPT-Large) |
| Fine-tuned BioGPT | Causal LM | BioGPT fine-tuned on BC5CDR dataset |

## Modules

### 1. PDF Ingestion (`graph_rag/pdf_ingestion.py`)

Extracts and chunks text from PDF documents.

**Functions:**

| Function | Description |
|----------|-------------|
| `extract_text_by_page(pdf_path)` | Extract text from each page, returns `list[dict]` with `page` and `text` keys |
| `chunk_text(text, max_tokens=128, overlap_tokens=16)` | Split text into overlapping word-based chunks |

**Usage:**
```python
from graph_rag.pdf_ingestion import extract_text_by_page, chunk_text

pages = extract_text_by_page("medical_document.pdf")
chunks = chunk_text(pages[0]["text"], max_tokens=128)
```

---

### 2. Relation Extraction (`graph_rag/relation_extraction.py`)

Extracts entity-relation triplets using REBEL or BioGPT.

**Model Types:**
- `rebel`: General-purpose relation extraction
- `biogpt`: Base BioGPT-Large for biomedical text
- `medical`: Fine-tuned BioGPT on BC5CDR

**Functions:**

| Function | Description |
|----------|-------------|
| `get_device()` | Returns best available device (MPS/CPU) |
| `load_model(model_name, device)` | Load REBEL Seq2Seq model |
| `load_biogpt_model(model_path, device)` | Load BioGPT model |
| `load_medical_model(model_path, device)` | Load fine-tuned BioGPT |
| `extract_relations(chunks, model, tokenizer)` | Extract triplets using REBEL |
| `extract_relations_biogpt(chunks, pipeline)` | Extract triplets using BioGPT |

**Usage:**
```python
# Using REBEL
from graph_rag.relation_extraction import load_model, extract_relations

model, tokenizer = load_model()
triplets = extract_relations(chunks, model, tokenizer)

# Using BioGPT
from graph_rag.relation_extraction import load_biogpt_model, extract_relations_biogpt

pipeline = load_biogpt_model()
triplets = extract_relations_biogpt(chunks, pipeline)

# Using fine-tuned BioGPT
from graph_rag.relation_extraction import load_medical_model, extract_relations_medical

pipeline = load_medical_model("models/biogpt_bc5cdr")
triplets = extract_relations_medical(chunks, pipeline)
```

---

### 3. Graph Storage (`graph_rag/graph_store.py`)

Stores triplets in Memgraph/Neo4j graph database.

**Functions:**

| Function | Description |
|----------|-------------|
| `store_triplets(triplets, uri, auth, database)` | Store triplets using MERGE (idempotent) |
| `query_all_relations(uri, auth, database)` | Retrieve all entity relations |
| `clear_graph(uri, auth, database)` | Delete all nodes and relationships |
| `build_merge_query(triplet)` | Build parameterized Cypher query |

**Configuration:**
```python
DEFAULT_URI = "bolt://192.168.0.222:7687"
DEFAULT_AUTH = ("", "")
DEFAULT_DB = "memgraph"
```

---

### 4. Data Preparation (`graph_rag/data_preparation.py`)

Utilities for preparing training data, including BC5CDR dataset loading.

**BC5CDR Functions:**

| Function | Description |
|----------|-------------|
| `load_bc5cdr_dataset(split)` | Load BC5CDR from HuggingFace (train/validation/test) |
| `load_bc5cdr_all_splits()` | Load all splits as dict |
| `convert_bc5cdr_to_training_format(samples)` | Convert to BioGPT training format |
| `format_biogpt_prompt(chemical, disease, context)` | Format input prompt for BioGPT |

**Usage:**
```python
from graph_rag.data_preparation import load_bc5cdr_dataset, convert_bc5cdr_to_training_format

# Load BC5CDR dataset
train_data = load_bc5cdr_dataset("train")
# Returns list of samples with entities (Chemical, Disease) and relations (CID)

# Convert to training format
training_samples = convert_bc5cdr_to_training_format(train_data)
# Creates prompts: "Chemical: {chem}, Disease: {disease}, Context: {text}, Relation:"
# Targets: "CID" or "no_relation"
```

---

### 5. BioGPT Model (`graph_rag/medical_re_model.py`)

BioGPT-Large based relation extraction model.

**Classes:**

| Class | Description |
|-------|-------------|
| `BioGPTREModel` | PyTorch model wrapping BioGPT for causal LM |
| `BioGPTREPipeline` | Inference pipeline with prompt-based extraction |
| `Entity` | Dataclass for detected entities |
| `Relation` | Dataclass for relations between entities |

**Entity Types (BC5CDR):**
- Chemical, Disease

**Relation Types:**
- CID (Chemical Induces Disease)
- no_relation

**Usage:**
```python
from graph_rag.medical_re_model import BioGPTREPipeline, load_model

# Load fine-tuned model
pipeline = load_model("models/biogpt_bc5cdr")

# Extract relations
triplets = pipeline.predict("Aspirin induces asthma in some patients.")
# Returns: [{"head": "Aspirin", "type": "induces", "tail": "asthma"}]
```

---

### 6. Training Pipeline (`graph_rag/train.py`)

Training script for fine-tuning BioGPT on BC5CDR with PEFT (LoRA) and optimizations.

**Key Features:**
- **PEFT/LoRA**: Parameter-efficient fine-tuning to avoid catastrophic forgetting
- **Early Stopping**: Stops training when validation loss stops improving
- **Gradient Checkpointing**: Reduces memory usage by recomputing activations
- **Mixed Precision**: FP16/BF16 training for faster inference
- **Gradient Accumulation**: Effective batch size multiplication
- **Cosine Scheduler**: Learning rate warmup and decay
- **Save Best Only**: Only saves the best model checkpoint

**Usage:**
```bash
# Train with defaults (LoRA, early stopping, gradient checkpointing)
uv run python -m graph_rag.train

# Custom configuration
uv run python -m graph_rag.train \
    --epochs 5 \
    --batch-size 4 \
    --lr 2e-4 \
    --lora-r 16 \
    --early-stopping-patience 3 \
    --bf16

# Quick test with limited samples
uv run python -m graph_rag.train --max-samples 100 --epochs 1
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--base-model` | microsoft/BioGPT-Large | Base model name |
| `--output-dir` | models/biogpt_bc5cdr | Output directory |
| `--batch-size` | 4 | Batch size |
| `--epochs` | 3 | Number of epochs |
| `--lr` | 2e-4 | Learning rate |
| `--use-peft` | True | Use LoRA for efficient fine-tuning |
| `--no-peft` | - | Disable PEFT (full fine-tuning) |
| `--lora-r` | 16 | LoRA rank |
| `--lora-alpha` | 32 | LoRA alpha |
| `--lora-dropout` | 0.05 | LoRA dropout |
| `--gradient-accumulation-steps` | 4 | Gradient accumulation |
| `--gradient-checkpointing` | True | Enable gradient checkpointing |
| `--fp16` | False | Use FP16 precision |
| `--bf16` | False | Use BF16 precision |
| `--early-stopping` | True | Enable early stopping |
| `--early-stopping-patience` | 3 | Epochs without improvement |
| `--scheduler` | cosine | LR scheduler (linear/cosine) |
| `--max-samples` | None | Limit samples for testing |

**Why PEFT/LoRA?**
- Trains only ~1% of parameters (adapter weights)
- Preserves pre-trained knowledge (no catastrophic forgetting)
- Much smaller checkpoint files (~50MB vs ~3GB)
- Faster training with less memory

---

## Main Pipeline (`main.py`)

**Configuration:**
```python
PDF_PATH = "medical_diagnosis_manual.pdf"
MAX_PAGES = None
MODEL_TYPE = "rebel"  # "rebel", "biogpt", or "medical"
MEDICAL_MODEL_PATH = "models/biogpt_bc5cdr"
```

**Run:**
```bash
uv run python main.py
```

---

## Output Format

All models output triplets in the same format for graph storage:

```python
{
    "head": "Entity name",
    "type": "relation_type",
    "tail": "Entity name"
}
```

---

## BC5CDR Dataset

The BC5CDR corpus contains:
- 1,500 PubMed articles
- 15,935 chemical mentions
- 12,852 disease mentions
- Chemical-Induced Disease (CID) relations

**Loading:**
```python
from graph_rag.data_preparation import load_bc5cdr_dataset

# Downloads from HuggingFace automatically
train = load_bc5cdr_dataset("train")      # 500 articles
val = load_bc5cdr_dataset("validation")   # 500 articles
test = load_bc5cdr_dataset("test")        # 500 articles
```

---

## Dependencies

```toml
[project.dependencies]
neo4j = ">=6.1.0"
pymupdf = ">=1.27.1"
torch = ">=2.10.0"
transformers = ">=5.1.0"
datasets = ">=3.0.0"
accelerate = ">=1.0.0"
seqeval = ">=1.2.2"
evaluate = ">=0.4.0"
tqdm = ">=4.66.0"
```

---

## Workflow

### Using REBEL (Default)

```bash
# Set MODEL_TYPE = "rebel" in main.py
uv run python main.py
```

### Using BioGPT

```bash
# Set MODEL_TYPE = "biogpt" in main.py
uv run python main.py
```

### Training Custom BioGPT

```bash
# 1. Train on BC5CDR
uv run python -m graph_rag.train --epochs 3

# 2. Set MODEL_TYPE = "medical" in main.py

# 3. Run pipeline
uv run python main.py
```
