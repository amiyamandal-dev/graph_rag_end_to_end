"""Data preparation utilities for text-to-Cypher medical relation extraction training.

Converts BC5CDR (BioCreative V Chemical-Disease Relation) annotations into
(medical_text, cypher_statements) training pairs for fine-tuning BioGPT to
generate Cypher directly from medical/pharmaceutical documents.
"""

import json
import os
import tempfile
import zipfile
from pathlib import Path

from graph_rag.graph_store import _sanitize_rel_type, escape_cypher_string, _sanitize_label

# BC5CDR entity and relation types
BC5CDR_ENTITY_TYPES = ["Chemical", "Disease"]
BC5CDR_RELATION_TYPES = ["CID", "no_relation"]  # CID = Chemical Induces Disease

CYPHER_PROMPT_PREFIX = "Generate Cypher to store medical relations from text:"
CYPHER_PROMPT_SUFFIX = "\nCypher:"


# ---------------------------------------------------------------------------
# Cypher generation helpers
# ---------------------------------------------------------------------------


def triplet_to_cypher(
    head: str,
    head_label: str,
    tail: str,
    tail_label: str,
    rel_type: str,
) -> str:
    """Convert a single (head, relation, tail) to a Cypher MERGE statement.

    Example output:
        MERGE (h:Chemical {name: 'Aspirin'}) MERGE (t:Disease {name: 'Reye syndrome'}) MERGE (h)-[:INDUCES]->(t)
    """
    sanitized_rel = _sanitize_rel_type(rel_type)
    h_label = _sanitize_label(head_label)
    t_label = _sanitize_label(tail_label)
    h_name = escape_cypher_string(head)
    t_name = escape_cypher_string(tail)

    return (
        f"MERGE (h:{h_label} {{name: '{h_name}'}}) "
        f"MERGE (t:{t_label} {{name: '{t_name}'}}) "
        f"MERGE (h)-[:{sanitized_rel}]->(t)"
    )


def sample_to_cypher(entities: list[dict], relations: list[dict]) -> str:
    """Convert a BC5CDR sample's annotated relations to Cypher statements.

    Each relation becomes a self-contained MERGE statement.
    Multiple statements are separated by newlines.

    Returns empty string if there are no relations.
    """
    if not relations:
        return ""

    statements: list[str] = []
    seen: set[tuple[str, str, str]] = set()

    for rel in relations:
        head_idx = rel["head_idx"]
        tail_idx = rel["tail_idx"]
        if head_idx >= len(entities) or tail_idx >= len(entities):
            continue

        head = entities[head_idx]
        tail = entities[tail_idx]
        rel_type = rel["type"]

        key = (head["text"], rel_type, tail["text"])
        if key in seen:
            continue
        seen.add(key)

        stmt = triplet_to_cypher(
            head=head["text"],
            head_label=head.get("label", "Entity"),
            tail=tail["text"],
            tail_label=tail.get("label", "Entity"),
            rel_type=rel_type,
        )
        statements.append(stmt)

    return "\n".join(statements)


# ---------------------------------------------------------------------------
# Prompt / target formatting for BioGPT fine-tuning
# ---------------------------------------------------------------------------


def format_biogpt_prompt(text: str, max_context_length: int = 512) -> str:
    """Format input prompt for Cypher generation.

    Args:
        text: Medical/pharma source text.
        max_context_length: Truncate text to this many characters.

    Returns:
        Prompt string ready for tokenization.
    """
    if len(text) > max_context_length:
        text = text[:max_context_length]
    return f"{CYPHER_PROMPT_PREFIX} {text}{CYPHER_PROMPT_SUFFIX}"


def format_biogpt_target(cypher: str) -> str:
    """Format the target (Cypher statements) for training.

    Args:
        cypher: One or more newline-separated MERGE statements,
                or empty string for negative samples.
    """
    return cypher if cypher else "// No relations"


# ---------------------------------------------------------------------------
# BC5CDR dataset loading
# ---------------------------------------------------------------------------


def load_bc5cdr_dataset(
    split: str = "train",
    cache_dir: str | None = None,
) -> list[dict]:
    """Load BC5CDR dataset from HuggingFace Hub.

    Args:
        split: Dataset split ("train", "validation", "test")
        cache_dir: Optional cache directory for datasets

    Returns:
        List of samples with entities and relations
    """
    from huggingface_hub import hf_hub_download

    if cache_dir is None:
        cache_dir = os.path.join(tempfile.gettempdir(), "bc5cdr_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Download CDR corpus zip from bigbio/bc5cdr
    try:
        zip_path = hf_hub_download(
            repo_id="bigbio/bc5cdr",
            filename="CDR_Data.zip",
            cache_dir=cache_dir,
            repo_type="dataset",
        )
        return _parse_cdr_zip(zip_path, split)
    except Exception as e:
        print(f"Failed to download from bigbio/bc5cdr: {e}")

    # Fallback: try alternative dataset without relations
    try:
        return _load_bc5cdr_from_ghadeermobasher(split, cache_dir)
    except Exception as e:
        print(f"Failed to load from ghadeermobasher: {e}")

    raise RuntimeError(
        "Could not load BC5CDR dataset. Please ensure you have internet access."
    )


def _parse_cdr_zip(zip_path: str, split: str) -> list[dict]:
    """Parse CDR corpus from downloaded zip file."""
    cache_dir = os.path.dirname(zip_path)
    extract_dir = os.path.join(cache_dir, "CDR_Data_extracted")

    # Extract if not exists
    if not os.path.exists(extract_dir):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    # Map splits to files (nested structure: CDR_Data/CDR.Corpus.v010516/)
    split_files = {
        "train": os.path.join(extract_dir, "CDR_Data", "CDR.Corpus.v010516", "CDR_TrainingSet.BioC.xml"),
        "validation": os.path.join(extract_dir, "CDR_Data", "CDR.Corpus.v010516", "CDR_DevelopmentSet.BioC.xml"),
        "test": os.path.join(extract_dir, "CDR_Data", "CDR.Corpus.v010516", "CDR_TestSet.BioC.xml"),
    }

    file_path = split_files.get(split)
    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(f"CDR data file not found for split: {split}. Path: {file_path}")

    return _parse_bioc_xml(file_path)


def _load_bc5cdr_from_ghadeermobasher(split: str, cache_dir: str | None = None) -> list[dict]:
    """Load BC5CDR from ghadeermobasher/BC5CDR-Chemical-Disease (standard format)."""
    from datasets import load_dataset

    split_map = {
        "train": "train",
        "validation": "validation",
        "test": "test",
    }

    hf_split = split_map.get(split, split)

    # Try loading via datasets library first
    try:
        dataset = load_dataset(
            "ghadeermobasher/BC5CDR-Chemical-Disease",
            split=hf_split,
            cache_dir=cache_dir,
            trust_remote_code=False,
        )
        return _convert_ghadeermobasher_format(dataset)
    except Exception:
        pass

    # Fallback: try loading from JSON directly
    import urllib.request
    import json as json_module

    base_url = "https://huggingface.co/datasets/ghadeermobasher/BC5CDR-Chemical-Disease/resolve/main"
    url = f"{base_url}/{hf_split}.json"
    if cache_dir:
        local_path = os.path.join(cache_dir, f"{hf_split}.json")
        os.makedirs(cache_dir, exist_ok=True)
        urllib.request.urlretrieve(url, local_path)

        with open(local_path) as f:
            data = json_module.load(f)
    else:
        with urllib.request.urlopen(url) as response:
            data = json_module.loads(response.read().decode())

    return _convert_ghadeermobasher_format(data if isinstance(data, list) else data.get("data", []))


def _convert_ghadeermobasher_format(data: list) -> list[dict]:
    """Convert ghadeermobasher format to our standard format."""
    samples = []

    for item in data:
        tokens = item.get("tokens", [])
        ner_tags = item.get("ner_tags", [])

        # Decode BIO tags
        entities = _decode_bio_tags(tokens, ner_tags)
        text = " ".join(tokens)

        samples.append({
            "id": item.get("id", str(len(samples))),
            "text": text,
            "entities": entities,
            "relations": [],  # This dataset doesn't have relation annotations
        })

    return samples


# ---------------------------------------------------------------------------
# BioC XML parser
# ---------------------------------------------------------------------------


def _parse_bioc_xml(file_path: str) -> list[dict]:
    """Parse BioC XML format CDR corpus."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(file_path)
    root = tree.getroot()

    samples = []

    for document in root.findall(".//document"):
        doc_id = document.find("id").text if document.find("id") is not None else ""

        # Extract passages (title + abstract)
        passages = []
        for passage in document.findall("passage"):
            text_elem = passage.find("text")
            offset_elem = passage.find("offset")

            if text_elem is not None and offset_elem is not None:
                passages.append({
                    "text": text_elem.text or "",
                    "offset": int(offset_elem.text or 0),
                })

        # Combine passage texts
        full_text = " ".join(p["text"] for p in passages)

        # Extract annotations (entities)
        entities = []
        mesh_to_indices: dict[str, list[int]] = {}

        for i, annotation in enumerate(document.findall(".//annotation")):
            annot_id = annotation.get("id", str(i))

            # Get entity text
            text_elem = annotation.find("text")
            entity_text = text_elem.text if text_elem is not None else ""

            # Get infons
            infons = {}
            for infon in annotation.findall("infon"):
                key = infon.get("key")
                value = infon.text
                infons[key] = value

            entity_type = infons.get("type", "Unknown")
            mesh_id = infons.get("MESH", None)

            # Get location/offset
            location = annotation.find("location")
            if location is not None:
                start = int(location.get("offset", 0))
                length = int(location.get("length", 0))
                end = start + length
            else:
                start, end = 0, len(entity_text)

            # Normalize entity type
            if entity_type.lower() == "chemical":
                entity_type = "Chemical"
            elif entity_type.lower() == "disease":
                entity_type = "Disease"

            idx = len(entities)

            # Build MeSH ID mapping for relations
            if mesh_id and mesh_id != "-1":
                if mesh_id not in mesh_to_indices:
                    mesh_to_indices[mesh_id] = []
                mesh_to_indices[mesh_id].append(idx)

            entities.append({
                "id": annot_id,
                "text": entity_text,
                "start": start,
                "end": end,
                "label": entity_type,
                "mesh_id": mesh_id,
            })

        # Extract relations (CID: Chemical Induces Disease)
        relations = []
        for relation in document.findall(".//relation"):
            # Get relation type
            rel_type_elem = relation.find("infon[@key='relation']")
            rel_type = rel_type_elem.text if rel_type_elem is not None else ""

            # Get Chemical and Disease MeSH IDs from infons
            chem_mesh_elem = relation.find("infon[@key='Chemical']")
            disease_mesh_elem = relation.find("infon[@key='Disease']")

            chem_mesh = chem_mesh_elem.text if chem_mesh_elem is not None else None
            disease_mesh = disease_mesh_elem.text if disease_mesh_elem is not None else None

            if chem_mesh and disease_mesh:
                # Find all entities with these MeSH IDs
                chem_indices = mesh_to_indices.get(chem_mesh, [])
                disease_indices = mesh_to_indices.get(disease_mesh, [])

                # Create relations for all combinations
                for chem_idx in chem_indices:
                    for disease_idx in disease_indices:
                        # Normalize relation type
                        if rel_type.lower() == "cid":
                            rel_type = "induces"

                        relations.append({
                            "head_idx": chem_idx,
                            "tail_idx": disease_idx,
                            "type": rel_type,
                        })

        samples.append({
            "id": doc_id,
            "text": full_text,
            "entities": entities,
            "relations": relations,
        })

    return samples


def _decode_bio_tags(tokens: list[str], tags: list[int]) -> list[dict]:
    """Decode BIO tagged tokens to entity spans.

    Tag format (BC5CDR):
    - 0: O (outside)
    - 1: B-Chemical
    - 2: I-Chemical
    - 3: B-Disease
    - 4: I-Disease
    """
    entities = []
    current_entity = None
    char_offset = 0

    for i, (token, tag) in enumerate(zip(tokens, tags)):
        token_start = char_offset
        token_end = char_offset + len(token)

        if tag == 0:  # O
            if current_entity:
                entities.append(current_entity)
                current_entity = None
        elif tag == 1:  # B-Chemical
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                "text": token,
                "start": token_start,
                "end": token_end,
                "label": "Chemical",
            }
        elif tag == 2:  # I-Chemical
            if current_entity and current_entity["label"] == "Chemical":
                current_entity["text"] += " " + token
                current_entity["end"] = token_end
            else:
                current_entity = {
                    "text": token,
                    "start": token_start,
                    "end": token_end,
                    "label": "Chemical",
                }
        elif tag == 3:  # B-Disease
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                "text": token,
                "start": token_start,
                "end": token_end,
                "label": "Disease",
            }
        elif tag == 4:  # I-Disease
            if current_entity and current_entity["label"] == "Disease":
                current_entity["text"] += " " + token
                current_entity["end"] = token_end
            else:
                current_entity = {
                    "text": token,
                    "start": token_start,
                    "end": token_end,
                    "label": "Disease",
                }

        # Update char offset (add 1 for space)
        char_offset = token_end + 1

    # Don't forget last entity
    if current_entity:
        entities.append(current_entity)

    return entities


# ---------------------------------------------------------------------------
# Training format conversion
# ---------------------------------------------------------------------------


def load_bc5cdr_all_splits(cache_dir: str | None = None) -> dict[str, list[dict]]:
    """Load all BC5CDR splits.

    Args:
        cache_dir: Optional cache directory

    Returns:
        Dict with 'train', 'validation', 'test' keys
    """
    return {
        "train": load_bc5cdr_dataset("train", cache_dir),
        "validation": load_bc5cdr_dataset("validation", cache_dir),
        "test": load_bc5cdr_dataset("test", cache_dir),
    }


def convert_bc5cdr_to_training_format(
    samples: list[dict],
    include_negative_samples: bool = True,
    max_context_length: int = 512,
) -> dict:
    """Convert BC5CDR samples to text-to-Cypher training pairs.

    For each document:
      - Input prompt : "Generate Cypher to store medical relations from text: {doc_text}\\nCypher:"
      - Target output: newline-separated MERGE statements for every annotated relation

    Negative samples (documents with entities but no CID relations) are
    included with target "// No relations" when *include_negative_samples* is True.

    Args:
        samples: BC5CDR samples from load_bc5cdr_dataset
        include_negative_samples: Include documents with no CID relations
        max_context_length: Truncate source text to this many characters

    Returns:
        Dict with 'samples' key containing list of {prompt, target} dicts.
    """
    training_samples: list[dict] = []

    for sample in samples:
        text = sample["text"]
        entities = sample["entities"]
        relations = sample["relations"]

        cypher = sample_to_cypher(entities, relations)

        if cypher:
            training_samples.append({
                "prompt": format_biogpt_prompt(text, max_context_length),
                "target": format_biogpt_target(cypher),
            })
        elif include_negative_samples and entities:
            # Document has entities but no CID relation — useful negative
            training_samples.append({
                "prompt": format_biogpt_prompt(text, max_context_length),
                "target": format_biogpt_target(""),
            })

    return {"samples": training_samples}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def entities_to_triplets(sample: dict) -> list[dict]:
    """Convert annotated sample to triplet format for graph storage.

    Args:
        sample: Annotated sample with 'text', 'entities', 'relations'

    Returns:
        List of triplets: {"head": str, "type": str, "tail": str}
    """
    triplets = []
    entities = sample.get("entities", [])
    relations = sample.get("relations", [])

    for rel in relations:
        head_idx = rel["head_idx"]
        tail_idx = rel["tail_idx"]
        if 0 <= head_idx < len(entities) and 0 <= tail_idx < len(entities):
            triplets.append({
                "head": entities[head_idx]["text"],
                "type": rel["type"],
                "tail": entities[tail_idx]["text"],
            })

    return triplets


def load_medical_docs(path: str | Path) -> list[str]:
    """Load raw medical texts from a directory or file.

    Args:
        path: Path to a directory containing .txt files or a single .txt file

    Returns:
        List of text strings, one per document
    """
    path = Path(path)
    texts = []

    if path.is_file():
        if path.suffix == ".txt":
            texts.append(path.read_text(encoding="utf-8"))
        elif path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                texts.extend(item.get("text", str(item)) for item in data)
            else:
                texts.append(data.get("text", str(data)))
    elif path.is_dir():
        for file_path in sorted(path.glob("*.txt")):
            texts.append(file_path.read_text(encoding="utf-8"))
        for file_path in sorted(path.glob("*.json")):
            data = json.loads(file_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                texts.extend(item.get("text", str(item)) for item in data)
            else:
                texts.append(data.get("text", str(data)))

    return texts


def save_training_data(data: dict, path: str | Path) -> None:
    """Save training data to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_training_data(path: str | Path) -> dict:
    """Load training data from JSON file."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)
