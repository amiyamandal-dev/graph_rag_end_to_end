import re
from pathlib import Path

import pymupdf

_WATERMARK_PATTERNS = [
    # Full watermark block (email + code + disclaimer)
    r"suchit\.sharma@telomereconsulting\.io\s*\n*\s*ARMPF2GQD8\s*\n*\s*This file is meant for personal use by suchit\.sharma@telomereconsulting\.io only\.\s*\n*\s*Sharing or publishing the contents in part or full is liable for legal action\.",
    # Disclaimer-only block
    r"This file is meant for personal use by suchit\.sharma@telomereconsulting\.io only\.\s*\n*\s*Sharing or publishing the contents in part or full is liable for legal action\.",
    # Fragmented watermark (PyMuPDF sometimes splits text across lines)
    r"personal use by suchit\.sharma@telome\n?shing the contents in part or full is liable\s*",
    # Email + code on their own
    r"suchit\.sharma@telomereconsulting\.io\s*\n*\s*ARMPF2GQD8",
    r"suchit\.sharma@telomereconsulting\.io",
    r"ARMPF2GQD8",
]


def _strip_watermarks(text: str) -> str:
    """Remove known watermark/DRM text from extracted PDF content."""
    for pattern in _WATERMARK_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    # Collapse excessive blank lines left behind
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_by_page(pdf_path: str | Path) -> list[dict]:
    """Extract text from each page of a PDF.

    Returns a list of dicts with 'page' (1-indexed) and 'text' keys.
    Skips pages with no meaningful text content.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = []
    with pymupdf.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            text = _strip_watermarks(page.get_text())
            if text:
                pages.append({"page": i + 1, "text": text})
    return pages


def chunk_text(text: str, max_tokens: int = 128, overlap_tokens: int = 16) -> list[str]:
    """Split text into overlapping word-based chunks.

    Uses whitespace tokenization as an approximation of subword token count.
    Each chunk has at most max_tokens words, with overlap_tokens words
    carried over from the previous chunk for context continuity.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + max_tokens
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap_tokens
    return chunks
