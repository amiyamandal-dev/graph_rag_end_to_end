import pytest

from graph_rag.relation_extraction import extract_triplets_from_text


class TestExtractTripletsFromText:
    def test_single_triplet(self):
        text = "<triplet> Paris <subj> capital of <obj> France"
        result = extract_triplets_from_text(text)
        assert len(result) == 1
        assert result[0] == {"head": "Paris", "type": "capital of", "tail": "France"}

    def test_multiple_triplets(self):
        text = (
            "<triplet> Paris <subj> capital of <obj> France "
            "<triplet> Berlin <subj> capital of <obj> Germany"
        )
        result = extract_triplets_from_text(text)
        assert len(result) == 2
        assert result[0]["head"] == "Paris"
        assert result[1]["head"] == "Berlin"

    def test_empty_input(self):
        assert extract_triplets_from_text("") == []

    def test_pad_token_stripping(self):
        text = "<s><pad><triplet> Paris <subj> capital of <obj> France</s>"
        result = extract_triplets_from_text(text)
        assert len(result) == 1
        assert result[0]["head"] == "Paris"

    def test_correct_dict_keys(self):
        text = "<triplet> Aspirin <subj> instance of <obj> drug"
        result = extract_triplets_from_text(text)
        assert len(result) == 1
        assert set(result[0].keys()) == {"head", "type", "tail"}

    def test_malformed_input_skipped(self):
        text = "<triplet> no subj marker here"
        result = extract_triplets_from_text(text)
        assert result == []

    def test_whitespace_handling(self):
        text = "<triplet>  Paris  <subj>  capital of  <obj>  France  "
        result = extract_triplets_from_text(text)
        assert len(result) == 1
        assert result[0]["head"] == "Paris"
        assert result[0]["tail"] == "France"


@pytest.mark.slow
class TestExtractRelationsIntegration:
    def test_end_to_end(self):
        from graph_rag.relation_extraction import (
            extract_relations,
            get_device,
            load_model,
        )

        device = get_device()
        model, tokenizer = load_model(device=device)

        chunks = [
            "Barack Obama was born in Honolulu, Hawaii. He served as the 44th president of the United States."
        ]
        triplets = extract_relations(chunks, model, tokenizer)

        assert len(triplets) > 0
        assert all(isinstance(t, dict) for t in triplets)
        assert all("head" in t and "type" in t and "tail" in t for t in triplets)
