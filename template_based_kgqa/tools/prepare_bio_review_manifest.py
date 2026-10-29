"""Create a blank question-level review manifest from BIO annotation JSONL."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True, help="bio_annotations.jsonl")
    parser.add_argument("--output", default="bio_review_manifest.csv")
    args = parser.parse_args()

    questions = {}
    with Path(args.annotations).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            qid = row.get("quid")
            if qid is None:
                raise ValueError(f"Line {line_no} has no quid.")
            key = str(qid)
            question = row.get("question", "")
            if key in questions and questions[key] != question:
                raise ValueError(f"Question ID {key} maps to inconsistent question text.")
            questions[key] = question

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["quid", "question", "review_status", "review_notes"])
        writer.writeheader()
        for qid, question in sorted(questions.items(), key=lambda item: item[0]):
            writer.writerow({
                "quid": qid,
                "question": question,
                "review_status": "",
                "review_notes": "",
            })
    print(f"Created blank review manifest: {output} ({len(questions)} questions)")
    print("After reviewing every token label for a question, set that question's review_status to reviewed.")


if __name__ == "__main__":
    main()
