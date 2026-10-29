"""Joint inference: BiLSTM question-type classifier + BiLSTM-CRF + template/SPARQL.

Example:
  python -m template_kgqa.predict_pipeline --question "When did John Garang praise Sudan People's Liberation Army in 2005?" --type-checkpoint template_based_kgqa/models/question_type_heldout.pt --ner-checkpoint template_based_kgqa/models/entity_relation_crf.pt

Optional local RDF execution:
  add --rdf data/events.ttl --schema-json data/event_schema.json
"""
import argparse
import json
import re
from pathlib import Path

import torch

from .dataset import tokenize as type_tokenize
from .models import BiLSTMCRF, QuestionTypeClassifier
from .sparql import build_event_query, execute_local_rdf
from .templates import template_for

# Must match the tokenizer used to create the BIO annotation CSV.
NER_TOKEN_RE = re.compile(
    r"\d{4}(?:-\d{1,2}(?:-\d{1,2})?)?"
    r"|[A-Za-zÀ-ÖØ-öø-ÿ]+(?:['’][A-Za-zÀ-ÖØ-öø-ÿ]+)*(?:[-/][A-Za-zÀ-ÖØ-öø-ÿ]+)*"
    r"|\d+(?:st|nd|rd|th)?|[^\w\s]",
    re.UNICODE,
)


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def encode(tokens, vocab):
    return torch.tensor([vocab.get(token, vocab.get(token.casefold(), vocab["<UNK>"])) for token in tokens],
                        dtype=torch.long)


def predict_question_type(question, checkpoint_path, device):
    checkpoint = load_checkpoint(checkpoint_path, device)
    vocab = checkpoint["vocab"]
    labels = checkpoint["labels"]
    config = checkpoint.get("config", {})
    model = QuestionTypeClassifier(
        len(vocab), len(labels),
        embedding_dim=int(config.get("embedding_dim", 96)),
        hidden_dim=int(config.get("hidden_dim", 128)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    tokens = type_tokenize(question)
    if not tokens:
        raise ValueError("Question contains no tokens.")
    token_ids = encode(tokens, vocab).unsqueeze(0).to(device)
    lengths = torch.tensor([len(tokens)], dtype=torch.long)
    with torch.no_grad():
        probabilities = torch.softmax(model(token_ids, lengths), dim=1)[0]
    index = int(probabilities.argmax().item())
    return labels[index], float(probabilities[index].item())


def repair_bio(tags):
    repaired = []
    previous_type = None
    for tag in tags:
        if tag == "O":
            repaired.append(tag)
            previous_type = None
            continue
        prefix, entity_type = tag.split("-", 1)
        if prefix == "I" and previous_type != entity_type:
            tag = "B-" + entity_type
        repaired.append(tag)
        previous_type = entity_type
    return repaired


def decode_spans(question, tokens, offsets, tags):
    tags = repair_bio(tags)
    spans = []
    start = None
    current_type = None

    def close(end_index):
        if current_type is None or start is None or end_index <= start:
            return
        char_start = offsets[start][0]
        char_end = offsets[end_index - 1][1]
        spans.append({
            "type": current_type,
            "text": question[char_start:char_end],
            "start": char_start,
            "end": char_end,
            "token_start": start,
            "token_end": end_index - 1,
        })

    for i, tag in enumerate(tags + ["O"]):
        if tag == "O":
            if current_type is not None:
                close(i)
                start, current_type = None, None
            continue
        prefix, entity_type = tag.split("-", 1)
        if prefix == "B" or entity_type != current_type:
            if current_type is not None:
                close(i)
            start, current_type = i, entity_type
    return spans, tags


def predict_ner(question, checkpoint_path, device):
    checkpoint = load_checkpoint(checkpoint_path, device)
    vocab, tag_names = checkpoint["vocab"], checkpoint["tags"]
    config = checkpoint.get("config", {})
    model = BiLSTMCRF(
        len(vocab), len(tag_names),
        embedding_dim=int(config.get("embedding_dim", 96)),
        hidden_dim=int(config.get("hidden_dim", 128)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    matches = list(NER_TOKEN_RE.finditer(question))
    tokens = [match.group() for match in matches]
    offsets = [(match.start(), match.end()) for match in matches]
    if not tokens:
        raise ValueError("Question contains no tokens.")
    token_ids = encode(tokens, vocab).unsqueeze(0).to(device)
    mask = torch.ones_like(token_ids, dtype=torch.bool)
    with torch.no_grad():
        ids = model.decode(token_ids, mask)[0]
    predicted_tags = [tag_names[index] for index in ids]
    spans, repaired_tags = decode_spans(question, tokens, offsets, predicted_tags)
    return tokens, repaired_tags, spans


def infer_answer_role(question, actor, target, requested):
    if requested != "auto":
        return requested
    q = question.casefold().strip()
    if re.match(r"^(when|at what time|what date|in what year|in which year|which year|in what month|in which month|which month)\b", q):
        return "date"
    if re.match(r"^(to|with|against|for)\s+whom\b", q) or re.match(r"^whom did\b", q):
        return "target"
    if re.match(r"^(who|which [\w -]+)\s+did\b", q) and actor:
        return "target"
    if re.match(r"^who was\b", q) and re.search(r"\bby\b", q):
        return "target"
    if re.match(r"^(who|which)\b", q):
        if actor and not target:
            return "target"
        return "actor"
    return "actor"


def make_slots(question, qtype, spans):
    entities = [s for s in spans if s["type"] == "ENTITY"]
    relations = [s for s in spans if s["type"] == "RELATION"]
    times = [s for s in spans if s["type"] == "TIME"]
    temporal = [s for s in spans if s["type"] == "TEMPORAL_OPERATOR"]
    relation = max(relations, key=lambda s: len(s["text"])) if relations else None

    anchor = None
    if qtype in {"before_last", "after_first"}:
        cue = next((s for s in temporal if re.search(r"\b(before|after)\b", s["text"], re.I)), None)
        if cue:
            after_cue = [s for s in entities if s["start"] >= cue["end"]]
            if after_cue:
                anchor = min(after_cue, key=lambda s: s["start"])

    usable_entities = [s for s in entities if s is not anchor]
    actor = target = None
    if relation:
        preceding = [s for s in usable_entities if s["end"] <= relation["start"]]
        following = [s for s in usable_entities if s["start"] >= relation["end"]]
        if preceding:
            actor = max(preceding, key=lambda s: s["end"])
        if following:
            target = min(following, key=lambda s: s["start"])

    temporal_text = " ".join(s["text"].casefold() for s in temporal)
    temporal_operator = "before" if "before" in temporal_text else (
        "after" if "after" in temporal_text else None
    )
    date = times[0]["text"] if times else None
    extreme = "first" if re.search(r"\b(first|earliest)\b", question, re.I) else "last"
    slots = {
        "actor": actor["text"] if actor else None,
        "target": target["text"] if target else None,
        "relation": relation["text"] if relation else None,
        "date": date,
        "temporal_operator": temporal_operator,
        "extreme": extreme,
        "anchor": anchor["text"] if anchor else None,
    }
    return slots, infer_answer_role(question, slots["actor"], slots["target"], "auto")


def run(question, type_checkpoint, ner_checkpoint, device, schema=None, rdf=None, answer_role="auto", force_query=False):
    qtype, qtype_confidence = predict_question_type(question, type_checkpoint, device)
    tokens, tags, spans = predict_ner(question, ner_checkpoint, device)
    slots, inferred_role = make_slots(question, qtype, spans)
    role = infer_answer_role(question, slots["actor"], slots["target"], answer_role)
    query_ready = bool(slots["relation"] and (slots["actor"] or slots["target"] or slots["anchor"] or slots["date"]))
    result = {
        "question": question,
        "question_type": {"label": qtype, "confidence": round(qtype_confidence, 6)},
        "template": template_for(qtype),
        "tokens": tokens,
        "bio_tags": tags,
        "spans": spans,
        "slots": slots,
        "answer_role": role,
        "query_ready": query_ready,
        "slot_assignment_note": (
            "Entity-role assignment is a transparent positional heuristic, not a trained relation-role classifier. "
            "Entity linking and schema/predicate mapping still require validation."
        ),
        "sparql": None,
    }
    if query_ready or force_query:
        result["sparql"] = build_event_query(qtype, slots, role, schema)
        if rdf:
            result["results"] = execute_local_rdf(rdf, result["sparql"])
    elif rdf:
        result["query_note"] = "Local RDF query was not run because required relation/entity evidence was not extracted."
    return result


def main():
    parser = argparse.ArgumentParser(description="Run trained classifier + BiLSTM-CRF + template/SPARQL pipeline")
    parser.add_argument("--question", required=True)
    parser.add_argument("--type-checkpoint", required=True)
    parser.add_argument("--ner-checkpoint", required=True)
    parser.add_argument("--schema-json")
    parser.add_argument("--rdf", help="Optional local RDF file; schema must match its predicates.")
    parser.add_argument("--answer-role", choices=["auto", "actor", "target", "relation", "date"], default="auto")
    parser.add_argument("--force-query", action="store_true", help="Generate SPARQL even when extracted slots are incomplete.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    schema = json.loads(Path(args.schema_json).read_text(encoding="utf-8")) if args.schema_json else None
    result = run(
        args.question, args.type_checkpoint, args.ner_checkpoint, device,
        schema=schema, rdf=args.rdf, answer_role=args.answer_role, force_query=args.force_query,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
