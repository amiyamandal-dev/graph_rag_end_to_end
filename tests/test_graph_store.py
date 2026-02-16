import pytest

from graph_rag.graph_store import (
    _sanitize_label,
    _sanitize_rel_type,
    build_merge_query,
)


class TestSanitizeLabel:
    def test_simple_label(self):
        assert _sanitize_label("Person") == "Person"

    def test_spaces_replaced(self):
        assert _sanitize_label("blood type") == "blood_type"

    def test_special_chars_replaced(self):
        assert _sanitize_label("has-symptom(s)") == "has_symptom_s"

    def test_leading_number_gets_prefix(self):
        assert _sanitize_label("123abc").startswith("Entity_")

    def test_empty_string_gets_prefix(self):
        result = _sanitize_label("")
        assert result.startswith("Entity_")


class TestSanitizeRelType:
    def test_simple_relation(self):
        assert _sanitize_rel_type("capital of") == "CAPITAL_OF"

    def test_hyphens_replaced(self):
        assert _sanitize_rel_type("instance-of") == "INSTANCE_OF"

    def test_multiple_spaces(self):
        result = _sanitize_rel_type("born  in")
        assert result == "BORN_IN"

    def test_special_chars(self):
        result = _sanitize_rel_type("has/type")
        assert result == "HAS_TYPE"

    def test_empty_string(self):
        result = _sanitize_rel_type("")
        assert result == "RELATED_TO"


class TestBuildMergeQuery:
    def test_returns_query_and_params(self):
        triplet = {"head": "Paris", "type": "capital of", "tail": "France"}
        query, params = build_merge_query(triplet)
        assert isinstance(query, str)
        assert isinstance(params, dict)

    def test_query_contains_merge(self):
        triplet = {"head": "Paris", "type": "capital of", "tail": "France"}
        query, _ = build_merge_query(triplet)
        assert "MERGE" in query

    def test_params_have_correct_values(self):
        triplet = {"head": "Paris", "type": "capital of", "tail": "France"}
        _, params = build_merge_query(triplet)
        assert params["head"] == "Paris"
        assert params["tail"] == "France"
        assert params["rel_label"] == "capital of"

    def test_rel_type_sanitized_in_query(self):
        triplet = {"head": "X", "type": "instance of", "tail": "Y"}
        query, _ = build_merge_query(triplet)
        assert "INSTANCE_OF" in query

    def test_preserves_original_label_in_params(self):
        triplet = {"head": "X", "type": "instance of", "tail": "Y"}
        _, params = build_merge_query(triplet)
        assert params["rel_label"] == "instance of"


@pytest.mark.memgraph
class TestMemgraphIntegration:
    """Integration tests that require a running Memgraph instance."""

    def test_store_and_query(self):
        from graph_rag.graph_store import (
            clear_graph,
            query_all_relations,
            store_triplets,
        )

        clear_graph()

        triplets = [
            {"head": "Paris", "type": "capital of", "tail": "France"},
            {"head": "Berlin", "type": "capital of", "tail": "Germany"},
            {"head": "France", "type": "instance of", "tail": "country"},
        ]
        stored = store_triplets(triplets)
        assert stored == 3

        results = query_all_relations()
        assert len(results) == 3
        heads = {r["head"] for r in results}
        assert "Paris" in heads
        assert "Berlin" in heads

        clear_graph()

    def test_merge_deduplicates(self):
        from graph_rag.graph_store import (
            clear_graph,
            query_all_relations,
            store_triplets,
        )

        clear_graph()

        triplet = [{"head": "Paris", "type": "capital of", "tail": "France"}]
        store_triplets(triplet)
        store_triplets(triplet)  # store same triplet again

        results = query_all_relations()
        assert len(results) == 1

        clear_graph()
