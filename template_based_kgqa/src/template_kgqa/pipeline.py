"""Question-to-template orchestration and transparent slot extraction."""
import re
from .templates import infer_qtype, template_for
from .sparql import build_event_query


DATE_RE = re.compile(r"\b(19|20)\d{2}(?:-\d{1,2}(?:-\d{1,2})?)?\b")
RELATION_WORDS = (
    "praised", "praise", "criticised", "criticized", "condemned", "accused",
    "threatened", "appealed", "appeal", "requested", "request", "negotiated",
    "negotiate", "visited", "visit", "cooperate", "cooperated", "offered",
    "rejected", "investigated", "expressed", "announced", "telephoned",
)


def extract_temporal_constraint(question: str) -> tuple[str | None, str | None]:
    match = DATE_RE.search(question)
    if not match:
        return None, None
    date = match.group(0)
    lower = question.casefold()
    operator = "before" if re.search(r"\b(before|prior to)\b", lower) else (
        "after" if re.search(r"\b(after|following)\b", lower) else None
    )
    return date, operator


def extract_relation_candidate(question: str) -> str | None:
    lower = question.casefold()
    for phrase in sorted(RELATION_WORDS, key=len, reverse=True):
        if re.search(r"\b" + re.escape(phrase) + r"\b", lower):
            return phrase
    return None


def prepare_question(question: str, supplied_slots: dict | None = None) -> dict:
    slots = dict(supplied_slots or {})
    qtype = slots.pop("qtype", None) or infer_qtype(question)
    date, operator = extract_temporal_constraint(question)
    slots.setdefault("date", date)
    slots.setdefault("temporal_operator", operator)
    slots.setdefault("relation", extract_relation_candidate(question))
    slots.setdefault("extreme", "first" if re.search(r"\b(first|earliest)\b", question, re.I) else "last")
    return {
        "question": question,
        "qtype": qtype,
        "template": template_for(qtype),
        "slots": slots,
        "slot_extraction_note": "Date/relation candidates are heuristic until replaced by trained span tagging and entity linking.",
    }


def question_to_sparql(question: str, slots: dict | None = None, answer_role: str = "actor", schema: dict | None = None) -> dict:
    prepared = prepare_question(question, slots)
    query = build_event_query(prepared["qtype"], prepared["slots"], answer_role, schema)
    prepared["sparql"] = query
    prepared["answer_role"] = answer_role
    return prepared
