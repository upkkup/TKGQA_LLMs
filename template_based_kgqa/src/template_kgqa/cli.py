"""Command-line interface for template inspection and RDF queries."""
import argparse
import json
from pathlib import Path

from .dataset import load_dataset, dataset_statistics
from .pipeline import prepare_question, question_to_sparql
from .templates import induce_template_catalog


def main():
    parser = argparse.ArgumentParser(description="Template-based Knowledge Graph Q&A")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect")
    inspect.add_argument("--question", required=True)

    induce = sub.add_parser("induce")
    induce.add_argument("--dataset", required=True)
    induce.add_argument("--output", required=True)

    query = sub.add_parser("query")
    query.add_argument("--question", required=True)
    query.add_argument("--rdf")
    query.add_argument("--actor")
    query.add_argument("--relation")
    query.add_argument("--target")
    query.add_argument("--date")
    query.add_argument("--anchor")
    query.add_argument("--answer-role", choices=["actor", "target", "relation", "date"], default="actor")
    query.add_argument("--schema-json")

    args = parser.parse_args()
    if args.command == "inspect":
        print(json.dumps(prepare_question(args.question), ensure_ascii=False, indent=2))
        return
    if args.command == "induce":
        rows = load_dataset(args.dataset)
        result = induce_template_catalog(rows)
        result["dataset_statistics"] = dataset_statistics(rows)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {output} from {len(rows)} records")
        return

    slots = {k: getattr(args, k) for k in ("actor", "relation", "target", "date", "anchor") if getattr(args, k)}
    schema = json.loads(Path(args.schema_json).read_text(encoding="utf-8")) if args.schema_json else None
    prepared = question_to_sparql(args.question, slots, args.answer_role, schema)
    print(json.dumps(prepared, ensure_ascii=False, indent=2))
    if args.rdf:
        from .sparql import execute_local_rdf
        prepared["results"] = execute_local_rdf(args.rdf, prepared["sparql"])
        print(json.dumps({"results": prepared["results"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
