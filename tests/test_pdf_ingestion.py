from pathlib import Path

import pytest

from graph_rag.pdf_ingestion import chunk_text, extract_text_by_page

PDF_PATH = Path(__file__).resolve().parent.parent / "medical_diagnosis_manual.pdf"


class TestExtractTextByPage:
    def test_returns_list_of_dicts(self):
        pages = extract_text_by_page(PDF_PATH)
        assert isinstance(pages, list)
        assert len(pages) > 0
        assert all(isinstance(p, dict) for p in pages)

    def test_dict_has_correct_keys(self):
        pages = extract_text_by_page(PDF_PATH)
        for p in pages:
            assert "page" in p
            assert "text" in p

    def test_page_numbers_start_at_one(self):
        pages = extract_text_by_page(PDF_PATH)
        assert pages[0]["page"] >= 1

    def test_no_empty_text_pages(self):
        pages = extract_text_by_page(PDF_PATH)
        for p in pages:
            assert p["text"].strip() != ""

    def test_invalid_path_raises(self):
        with pytest.raises(FileNotFoundError):
            extract_text_by_page("/nonexistent/file.pdf")


class TestChunkText:
    def test_basic_chunking(self):
        text = " ".join(f"word{i}" for i in range(300))
        chunks = chunk_text(text, max_tokens=128, overlap_tokens=16)
        assert len(chunks) > 1

    def test_each_chunk_within_limit(self):
        text = " ".join(f"word{i}" for i in range(300))
        chunks = chunk_text(text, max_tokens=128, overlap_tokens=16)
        for chunk in chunks:
            assert len(chunk.split()) <= 128

    def test_overlap_present(self):
        text = " ".join(f"word{i}" for i in range(300))
        chunks = chunk_text(text, max_tokens=128, overlap_tokens=16)
        # Last 16 words of chunk 0 should appear as first 16 words of chunk 1
        words_0 = chunks[0].split()
        words_1 = chunks[1].split()
        assert words_0[-16:] == words_1[:16]

    def test_empty_text_returns_empty(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_short_text_single_chunk(self):
        chunks = chunk_text("hello world", max_tokens=128)
        assert len(chunks) == 1
        assert chunks[0] == "hello world"
