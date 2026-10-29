"""SPARQL generation for a configurable event-centric RDF schema."""
DEFAULT_SCHEMA = {
    "actor_predicate": "http://example.org/kg/actor",
    "relation_predicate": "http://example.org/kg/relation",
    "target_predicate": "http://example.org/kg/target",
    "date_predicate": "http://example.org/kg/date",
}


def escape_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")


def iri(value: str) -> str:
    if not value.startswith(("http://", "https://", "urn:")):
        raise ValueError(f"Predicate must be an absolute IRI: {value}")
    return f"<{value}>"


def _label_filter(variable: str, value: str) -> str:
    safe = escape_literal(value.casefold())
    return f'FILTER(LCASE(STR({variable})) = "{safe}")'


def build_event_query(
    qtype: str,
    slots: dict,
    answer_role: str = "actor",
    schema: dict | None = None,
    limit: int = 100,
) -> str:
    """Build a query from known slots; user text is always encoded as literals."""
    cfg = {**DEFAULT_SCHEMA, **(schema or {})}
    if answer_role not in {"actor", "target", "relation", "date"}:
        raise ValueError("answer_role must be actor, target, relation, or date")
    if qtype not in {"equal", "first_last", "before_after", "before_last", "after_first", "equal_multi"}:
        raise ValueError(f"No SPARQL generator for qtype {qtype!r}")
    if not 1 <= limit <= 10000:
        raise ValueError("limit must be between 1 and 10000")

    actor_p = iri(cfg["actor_predicate"])
    relation_p = iri(cfg["relation_predicate"])
    target_p = iri(cfg["target_predicate"])
    date_p = iri(cfg["date_predicate"])
    answer_var = {"actor": "?actor", "target": "?target", "relation": "?relation", "date": "?date"}[answer_role]
    select_vars = answer_var if answer_var == "?date" else answer_var + " ?date"
    lines = [
        "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>",
        "SELECT DISTINCT " + select_vars + " WHERE {",
        f"  ?event {actor_p} ?actor ;",
        f"          {relation_p} ?relation ;",
        f"          {target_p} ?target ;",
        f"          {date_p} ?date .",
        "  OPTIONAL { ?actor rdfs:label ?actorLabel . }",
        "  OPTIONAL { ?relation rdfs:label ?relationLabel . }",
        "  OPTIONAL { ?target rdfs:label ?targetLabel . }",
    ]
    if slots.get("actor"):
        lines.append("  " + _label_filter("?actorLabel", slots["actor"]))
    if slots.get("relation"):
        lines.append("  " + _label_filter("?relationLabel", slots["relation"]))
    if slots.get("target"):
        lines.append("  " + _label_filter("?targetLabel", slots["target"]))

    anchor = slots.get("anchor")
    if qtype in {"before_last", "after_first"} and anchor:
        anchor_literal = escape_literal(anchor.casefold())
        lines.extend([
            f"  ?anchorEvent {actor_p} ?anchorActor ; {target_p} ?anchorTarget ; {date_p} ?anchorDate .",
            "  OPTIONAL { ?anchorActor rdfs:label ?anchorActorLabel . }",
            "  OPTIONAL { ?anchorTarget rdfs:label ?anchorTargetLabel . }",
            f'  FILTER(LCASE(STR(?anchorActorLabel)) = "{anchor_literal}" || LCASE(STR(?anchorTargetLabel)) = "{anchor_literal}")',
            "  FILTER(?date " + ("<" if qtype == "before_last" else ">") + " ?anchorDate)",
        ])
    elif slots.get("date"):
        date_value = escape_literal(slots["date"])
        operator = slots.get("temporal_operator")
        if qtype == "before_last":
            operator = "<"
        elif qtype == "after_first":
            operator = ">"
        if operator in {"before", "<"}:
            lines.append(f'  FILTER(STR(?date) < "{date_value}")')
        elif operator in {"after", ">"}:
            lines.append(f'  FILTER(STR(?date) > "{date_value}")')
        else:
            lines.append(f'  FILTER(STRSTARTS(STR(?date), "{date_value}"))')

    lines.append("}")
    if qtype == "first_last":
        order = "DESC" if slots.get("extreme", "last") == "last" else "ASC"
        lines.append(f"ORDER BY {order}(?date)")
        lines.append("LIMIT 1")
    elif qtype == "before_last":
        lines.append("ORDER BY DESC(?date)")
        lines.append("LIMIT 1")
    elif qtype == "after_first":
        lines.append("ORDER BY ASC(?date)")
        lines.append("LIMIT 1")
    else:
        lines.append("ORDER BY ?date")
        lines.append(f"LIMIT {limit}")
    return "\n".join(lines)


def execute_local_rdf(path: str, query: str) -> list[dict]:
    """Execute generated SPARQL against a local RDF file using RDFLib."""
    from rdflib import Graph

    graph = Graph()
    graph.parse(path)
    results = graph.query(query)
    return [{str(key): str(value) for key, value in row.asdict().items()} for row in results]
