"""Question template inventory and lightweight template induction."""
from dataclasses import asdict, dataclass
import re


@dataclass(frozen=True)
class TemplateSpec:
    qtype: str
    name: str
    intent: str
    answer_slot: str
    ordering: str = "none"
    dataset_supported: bool = True


TEMPLATE_SPECS = {
    "equal": TemplateSpec("equal", "fact_lookup", "Find an event matching the stated relation and optional time.", "answer", "none"),
    "first_last": TemplateSpec("first_last", "temporal_extreme", "Return the earliest or latest matching event.", "answer", "first_or_last"),
    "before_after": TemplateSpec("before_after", "temporal_filter", "Find matching events before or after a temporal constraint.", "answer", "date_filter"),
    "before_last": TemplateSpec("before_last", "last_before_anchor", "Return the latest matching event before an anchor event.", "answer", "descending"),
    "after_first": TemplateSpec("after_first", "first_after_anchor", "Return the earliest matching event after an anchor event.", "answer", "ascending"),
    "equal_multi": TemplateSpec("equal_multi", "same_period_multi_answer", "Return all matching answers for a shared temporal context.", "answers", "none"),
}

EXTENSION_TEMPLATES = {
    "comparison": TemplateSpec("comparison", "compare_entities", "Compare two entities on a named property.", "comparison", "none", False),
    "causal_inference": TemplateSpec("causal_inference", "causal_relation", "Retrieve an explicitly represented cause-effect relation.", "effect", "none", False),
}


def infer_qtype(question: str) -> str:
    """Map surface wording to an existing temporal qtype; default to equal."""
    q = question.casefold()
    before = bool(re.search(r"\b(before|prior to)\b", q))
    after = bool(re.search(r"\b(after|following)\b", q))
    first = bool(re.search(r"\b(first|earliest)\b", q))
    last = bool(re.search(r"\b(last|latest|most recent)\b", q))

    if before and last:
        return "before_last"
    if after and first:
        return "after_first"
    if first or last:
        return "first_last"
    if before or after:
        return "before_after"
    if re.search(r"\bsame (day|month|year)\b", q):
        return "equal_multi"
    return "equal"


def template_for(qtype: str) -> dict:
    spec = TEMPLATE_SPECS.get(qtype) or EXTENSION_TEMPLATES.get(qtype)
    if spec is None:
        raise ValueError(f"Unsupported question type: {qtype}")
    return asdict(spec)


def induce_template_catalog(rows: list[dict]) -> dict:
    counts = {}
    examples = {}
    for row in rows:
        qtype = row.get("qtype") or infer_qtype(row.get("question", ""))
        counts[qtype] = counts.get(qtype, 0) + 1
        examples.setdefault(qtype, row.get("question", ""))
    observed = {}
    for qtype, count in sorted(counts.items()):
        spec = template_for(qtype)
        observed[qtype] = {**spec, "count": count, "example_question": examples[qtype]}
    return {
        "dataset_question_count": len(rows),
        "observed_templates": observed,
        "extension_templates": {k: asdict(v) for k, v in EXTENSION_TEMPLATES.items()},
        "note": "Extension templates are not supervised by the supplied qtype labels.",
    }
