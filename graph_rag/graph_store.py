import os
import re

from neo4j import GraphDatabase

DEFAULT_URI = os.environ.get("MEMGRAPH_URI", "bolt://192.168.0.222:7687")
DEFAULT_AUTH = (
    os.environ.get("MEMGRAPH_USER", ""),
    os.environ.get("MEMGRAPH_PASSWORD", ""),
)
DEFAULT_DB = os.environ.get("MEMGRAPH_DB", "memgraph")


def _sanitize_label(label: str) -> str:
    """Sanitize a string for use as a Cypher node label.

    Replaces non-alphanumeric/underscore chars with underscores,
    strips leading/trailing underscores, and ensures it starts with a letter.
    """
    label = re.sub(r"[^a-zA-Z0-9_]", "_", label)
    label = label.strip("_")
    if not label or not label[0].isalpha():
        label = "Entity_" + label
    return label


def _sanitize_rel_type(rel_type: str) -> str:
    """Sanitize a string for use as a Cypher relationship type.

    Converts to UPPER_SNAKE_CASE.
    """
    rel = re.sub(r"[^a-zA-Z0-9_]", "_", rel_type)
    rel = re.sub(r"_+", "_", rel)
    rel = rel.strip("_").upper()
    if not rel:
        return "RELATED_TO"
    if not rel[0].isalpha():
        rel = "RELATED_TO_" + rel
    return rel


def escape_cypher_string(s: str) -> str:
    """Escape a string for use inside a Cypher single-quoted literal."""
    return s.replace("\\", "\\\\").replace("'", "\\'")


def build_merge_query(triplet: dict) -> tuple[str, dict]:
    """Build a parameterized MERGE Cypher query for a single triplet.

    Creates/matches nodes by name and creates the relationship between them.
    Supports optional 'head_label' and 'tail_label' keys for typed nodes
    (e.g. Chemical, Disease); defaults to 'Entity' when absent.

    Returns (query_string, params_dict).
    """
    rel_type = _sanitize_rel_type(triplet["type"])
    head_label = _sanitize_label(triplet.get("head_label", "Entity"))
    tail_label = _sanitize_label(triplet.get("tail_label", "Entity"))

    query = (
        f"MERGE (h:{head_label} {{name: $head}}) "
        f"MERGE (t:{tail_label} {{name: $tail}}) "
        f"MERGE (h)-[r:{rel_type}]->(t) "
        "SET r.label = $rel_label "
        "RETURN h.name AS head, type(r) AS relation, t.name AS tail"
    )
    params = {
        "head": triplet["head"],
        "tail": triplet["tail"],
        "rel_label": triplet["type"],
    }
    return query, params


def triplet_to_cypher(
    head: str,
    head_label: str,
    tail: str,
    tail_label: str,
    rel_type: str,
) -> str:
    """Convert a single triplet to a Cypher MERGE statement.

    Returns a self-contained MERGE statement that can be executed directly
    or used as a training target for text-to-Cypher models.
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


_CYPHER_MERGE_PATTERN = re.compile(
    r"MERGE\s+\(\w+:(\w+)\s+\{name:\s*'([^']*(?:\\'[^']*)*)'\}\)\s+"
    r"MERGE\s+\(\w+:(\w+)\s+\{name:\s*'([^']*(?:\\'[^']*)*)'\}\)\s+"
    r"MERGE\s+\(\w+\)-\[:(\w+)\]->\(\w+\)",
    re.IGNORECASE,
)


def parse_cypher_to_triplets(cypher_text: str) -> list[dict]:
    """Parse model-generated Cypher MERGE statements back to triplet dicts.

    Expected format per statement:
        MERGE (h:Label {name: 'X'}) MERGE (t:Label {name: 'Y'}) MERGE (h)-[:REL]->(t)

    Returns list of dicts with keys: head, head_label, tail, tail_label, type.
    """
    triplets = []
    seen = set()

    for match in _CYPHER_MERGE_PATTERN.finditer(cypher_text):
        head_label, head_name, tail_label, tail_name, rel_type = match.groups()
        # Unescape
        head_name = head_name.replace("\\'", "'")
        tail_name = tail_name.replace("\\'", "'")

        key = (head_name, rel_type, tail_name)
        if key not in seen:
            seen.add(key)
            triplets.append({
                "head": head_name,
                "head_label": head_label,
                "tail": tail_name,
                "tail_label": tail_label,
                "type": rel_type.lower().replace("_", " "),
            })

    return triplets


def store_triplets(
    triplets: list[dict],
    uri: str = DEFAULT_URI,
    auth: tuple[str, str] = DEFAULT_AUTH,
    database: str = DEFAULT_DB,
) -> int:
    """Store a list of triplets into Memgraph.

    Each triplet is expected to have keys: head, type, tail.
    Optionally head_label and tail_label for typed nodes (default: Entity).
    Uses MERGE to avoid duplicates.

    Returns the number of triplets successfully stored.
    """
    stored = 0
    with GraphDatabase.driver(uri, auth=auth) as driver:
        driver.verify_connectivity()
        for triplet in triplets:
            query, params = build_merge_query(triplet)
            records, summary, _ = driver.execute_query(
                query, **params, database_=database
            )
            if records:
                stored += 1
    return stored


def store_from_cypher(
    cypher_text: str,
    uri: str = DEFAULT_URI,
    auth: tuple[str, str] = DEFAULT_AUTH,
    database: str = DEFAULT_DB,
) -> int:
    """Parse model-generated Cypher and store the extracted triplets safely.

    Instead of executing raw generated Cypher (injection risk), this parses
    the Cypher back to triplets and uses the parameterized store_triplets path.

    Returns number of triplets stored.
    """
    triplets = parse_cypher_to_triplets(cypher_text)
    if not triplets:
        return 0
    return store_triplets(triplets, uri, auth, database)


def query_all_relations(
    uri: str = DEFAULT_URI,
    auth: tuple[str, str] = DEFAULT_AUTH,
    database: str = DEFAULT_DB,
) -> list[dict]:
    """Retrieve all node-relation-node triplets from Memgraph.

    Returns list of dicts with keys: head, relation, tail, relation_label.
    """
    query = (
        "MATCH (h)-[r]->(t) "
        "RETURN h.name AS head, type(r) AS relation, "
        "r.label AS relation_label, t.name AS tail, "
        "labels(h)[0] AS head_label, labels(t)[0] AS tail_label"
    )
    with GraphDatabase.driver(uri, auth=auth) as driver:
        records, _, _ = driver.execute_query(query, database_=database)
    return [dict(r) for r in records]


def clear_graph(
    uri: str = DEFAULT_URI,
    auth: tuple[str, str] = DEFAULT_AUTH,
    database: str = DEFAULT_DB,
) -> None:
    """Delete all nodes and relationships from the graph. Use with caution."""
    with GraphDatabase.driver(uri, auth=auth) as driver:
        driver.execute_query("MATCH (n) DETACH DELETE n", database_=database)
