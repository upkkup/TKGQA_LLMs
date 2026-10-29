"""Dataset validation and tokenization utilities."""
import json
import re
from collections import Counter
from pathlib import Path

TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def load_dataset(path: str | Path) -> list[dict]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Expected a JSON list of question records.")
    seen = set()
    for i, row in enumerate(rows):
        if not isinstance(row.get("question"), str):
            raise ValueError(f"Row {i} has no question string.")
        if not isinstance(row.get("answers"), list) or not all(isinstance(a, str) for a in row["answers"]):
            raise ValueError(f"Row {i} must contain an answers list of strings.")
        if "qtype" not in row:
            raise ValueError(f"Row {i} has no qtype label; type-classifier training requires qtype.")
        key = row.get("quid", i)
        if key in seen:
            raise ValueError(f"Duplicate question ID: {key}")
        seen.add(key)
    return rows


def dataset_statistics(rows: list[dict]) -> dict:
    return {
        "rows": len(rows),
        "answer_type": dict(Counter(r.get("answer_type", "unknown") for r in rows)),
        "qtype": dict(Counter(r.get("qtype", "unknown") for r in rows)),
        "time_level": dict(Counter(r.get("time_level", "unknown") for r in rows)),
        "qlabel": dict(Counter(r.get("qlabel", "unknown") for r in rows)),
    }


def load_ner_annotations(path: str | Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        tokens, tags = item.get("tokens"), item.get("tags")
        if not isinstance(tokens, list) or not isinstance(tags, list) or len(tokens) != len(tags):
            raise ValueError(f"Line {line_no} must have equally sized tokens and tags lists.")
        if not all(isinstance(x, str) for x in tokens + tags):
            raise ValueError(f"Line {line_no} tokens and tags must be strings.")
        rows.append({"tokens": tokens, "tags": tags})
    if not rows:
        raise ValueError("No NER annotation records found.")
    return rows
