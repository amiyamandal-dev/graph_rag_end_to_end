from graph_rag.pdf_ingestion import chunk_text, extract_text_by_page
from graph_rag.relation_extraction import (
    extract_relations,
    extract_relations_biogpt,
    extract_relations_medical,
    get_device,
    load_biogpt_model,
    load_medical_model,
    load_model,
)
from graph_rag.graph_store import clear_graph, query_all_relations, store_triplets

PDF_PATH = "medical_diagnosis_manual.pdf"
MAX_PAGES = None  # Set to an integer (e.g. 3) to limit pages, None for full PDF
MODEL_TYPE = "rebel"  # "rebel", "medical", or "biogpt"
MEDICAL_MODEL_PATH = "models/biogpt_bc5cdr"  # Path to fine-tuned BioGPT model


def main():
    # --- Step 1: Extract text from PDF ---
    pages = extract_text_by_page(PDF_PATH)
    if MAX_PAGES is not None:
        pages = pages[:MAX_PAGES]
    print(f"Processing {len(pages)} pages from {PDF_PATH}...")
    print(f"Got {len(pages)} pages with text")

    all_chunks = []
    for page in pages:
        chunks = chunk_text(page["text"])
        all_chunks.extend(chunks)
    print(f"Split into {len(all_chunks)} chunks")

    # --- Step 2: Extract relations ---
    device = get_device()

    if MODEL_TYPE == "medical":
        print(f"Loading fine-tuned BioGPT model from {MEDICAL_MODEL_PATH} on {device}...")
        pipeline = load_medical_model(MEDICAL_MODEL_PATH, device=device)
        print("Model loaded")

        print("Extracting relations with BioGPT...")
        triplets = extract_relations_medical(all_chunks, pipeline)
    elif MODEL_TYPE == "biogpt":
        print(f"Loading base BioGPT model on {device}...")
        pipeline = load_biogpt_model(device=device)
        print("Model loaded")

        print("Extracting relations with BioGPT...")
        triplets = extract_relations_biogpt(all_chunks, pipeline)
    else:
        print(f"Loading REBEL model on {device}...")
        model, tokenizer = load_model(device=device)
        print("Model loaded")

        print("Extracting relations...")
        triplets = extract_relations(all_chunks, model, tokenizer)
    print(f"\nFound {len(triplets)} unique relations:\n")

    for t in triplets:
        print(f"  {t['head']} --[{t['type']}]--> {t['tail']}")

    # --- Step 3: Store in Memgraph ---
    print("\n--- Storing triplets in Memgraph ---")
    print("Clearing existing graph...")
    clear_graph()

    print(f"Storing {len(triplets)} triplets...")
    stored = store_triplets(triplets)
    print(f"Successfully stored {stored}/{len(triplets)} triplets")

    # --- Step 4: Verify by querying back ---
    print("\n--- Verifying: querying all relations from Memgraph ---")
    results = query_all_relations()
    print(f"Retrieved {len(results)} relations from graph:\n")

    for r in results:
        label = r.get("relation_label", r["relation"])
        print(f"  {r['head']} --[{label}]--> {r['tail']}")


if __name__ == "__main__":
    main()
