"""Train and evaluate BiLSTM-CRF using manually reviewed BIO annotations.

Expected annotation CSV columns:
  quid, question, token_index, token, auto_tag, reviewed_tag, review_notes

Expected manifest CSV columns:
  quid, review_status, review_notes

Only questions with review_status in {reviewed, approved, yes, true} are used.
The explicit review manifest prevents accidental training on silver labels.
"""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .models import BiLSTMCRF

PAD = "<PAD>"
UNK = "<UNK>"
TAG_NAMES = [
    "O",
    "B-ENTITY", "I-ENTITY",
    "B-RELATION", "I-RELATION",
    "B-TIME", "I-TIME",
    "B-TEMPORAL_OPERATOR", "I-TEMPORAL_OPERATOR",
]
APPROVED = {"reviewed", "approved", "yes", "true", "1"}


def read_manifest(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"quid", "review_status"}.issubset(reader.fieldnames):
            raise ValueError("Manifest needs columns: quid, review_status")
        status_by_id = {}
        for row in reader:
            qid = str(row.get("quid", "")).strip()
            if qid:
                status_by_id[qid] = str(row.get("review_status", "")).strip().casefold()
    return {qid for qid, status in status_by_id.items() if status in APPROVED}


def validate_bio(tags, qid):
    for i, tag in enumerate(tags):
        if tag not in TAG_NAMES:
            raise ValueError(f"Question {qid}: unsupported tag {tag!r} at token {i}")
        if tag.startswith("I-"):
            entity_type = tag[2:]
            if i == 0 or tags[i - 1] not in {f"B-{entity_type}", f"I-{entity_type}"}:
                raise ValueError(
                    f"Question {qid}: invalid BIO sequence at token {i} ({tag}); "
                    "I-TYPE must follow B-TYPE or I-TYPE. Correct reviewed_tag in the CSV."
                )


def load_reviewed_csv(annotation_path, manifest_path):
    approved_ids = read_manifest(manifest_path)
    if not approved_ids:
        raise ValueError(
            "No reviewed questions found. In the manifest, set review_status=reviewed "
            "only after checking every token label for that question."
        )

    grouped = defaultdict(list)
    with Path(annotation_path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"quid", "question", "token_index", "token", "reviewed_tag"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Annotation CSV must contain columns: {sorted(required)}")
        for row in reader:
            qid = str(row.get("quid", "")).strip()
            if qid not in approved_ids:
                continue
            token = row.get("token", "")
            tag = row.get("reviewed_tag", "").strip()
            if not token or not tag:
                raise ValueError(f"Question {qid}: empty token or reviewed_tag.")
            try:
                index = int(row["token_index"])
            except (TypeError, ValueError):
                raise ValueError(f"Question {qid}: token_index must be an integer.")
            grouped[qid].append({
                "index": index,
                "token": token,
                "tag": tag,
                "question": row.get("question", ""),
            })

    records = []
    for qid, items in grouped.items():
        items.sort(key=lambda x: x["index"])
        expected = list(range(len(items)))
        actual = [item["index"] for item in items]
        if actual != expected:
            raise ValueError(f"Question {qid}: token_index values must be continuous from 0.")
        questions = {item["question"] for item in items}
        if len(questions) != 1:
            raise ValueError(f"Question {qid}: inconsistent question text in annotation CSV.")
        tokens = [item["token"] for item in items]
        tags = [item["tag"] for item in items]
        validate_bio(tags, qid)
        records.append({"quid": qid, "question": items[0]["question"], "tokens": tokens, "tags": tags})

    if len(records) < 5:
        raise ValueError(
            f"Only {len(records)} reviewed questions are available. Review more questions "
            "before making a train/test split; at least 5 are required."
        )
    records.sort(key=lambda row: row["quid"])
    return records


def build_vocab(sequences):
    counts = Counter(token for sequence in sequences for token in sequence)
    vocab = {PAD: 0, UNK: 1}
    for token, count in counts.items():
        if count and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode(tokens, vocab):
    return torch.tensor([vocab.get(token, vocab[UNK]) for token in tokens], dtype=torch.long)


class SequenceDataset(Dataset):
    def __init__(self, rows, vocab, tag_to_id):
        self.items = [
            (encode(row["tokens"], vocab),
             torch.tensor([tag_to_id[tag] for tag in row["tags"]], dtype=torch.long))
            for row in rows
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate(batch):
    tokens, tags = zip(*batch)
    lengths = torch.tensor([len(item) for item in tokens], dtype=torch.long)
    token_pad = pad_sequence(tokens, batch_first=True, padding_value=0)
    # O has tag ID 0, so padded tags are harmless outside the mask.
    tag_pad = pad_sequence(tags, batch_first=True, padding_value=0)
    mask = torch.arange(token_pad.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
    return token_pad, tag_pad, mask


def split_records(rows, test_fraction, seed):
    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)
    test_n = max(1, round(len(shuffled) * test_fraction))
    test_n = min(test_n, len(shuffled) - 1)
    return shuffled[test_n:], shuffled[:test_n]


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
            tag = f"B-{entity_type}"
            prefix = "B"
        repaired.append(tag)
        previous_type = entity_type
    return repaired


def spans_from_tags(tags):
    tags = repair_bio(tags)
    spans = set()
    start = None
    current_type = None
    for i, tag in enumerate(tags + ["O"]):
        if tag == "O":
            if current_type is not None:
                spans.add((current_type, start, i))
                current_type, start = None, None
            continue
        prefix, entity_type = tag.split("-", 1)
        if prefix == "B" or entity_type != current_type:
            if current_type is not None:
                spans.add((current_type, start, i))
            current_type, start = entity_type, i
    return spans


def safe_div(a, b):
    return a / b if b else 0.0


def evaluate(model, loader, rows, tag_names, device):
    model.eval()
    true_tags, predicted_tags = [], []
    exact_gold, exact_pred = set(), set()
    row_offset = 0
    with torch.no_grad():
        for token_ids, tag_ids, mask in loader:
            token_ids, tag_ids, mask = token_ids.to(device), tag_ids.to(device), mask.to(device)
            paths = model.decode(token_ids, mask)
            lengths = mask.long().sum(dim=1).tolist()
            gold_batch = tag_ids.cpu().tolist()
            for b, length in enumerate(lengths):
                gold = [tag_names[i] for i in gold_batch[b][:length]]
                pred = [tag_names[i] for i in paths[b]]
                pred = repair_bio(pred)
                true_tags.extend(gold)
                predicted_tags.extend(pred)
                for span in spans_from_tags(gold):
                    exact_gold.add((row_offset + b, *span))
                for span in spans_from_tags(pred):
                    exact_pred.add((row_offset + b, *span))
            row_offset += len(lengths)

    labels = [tag for tag in tag_names if tag != "O"]
    per_tag = {}
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(true_tags, predicted_tags))
        fp = sum(g != label and p == label for g, p in zip(true_tags, predicted_tags))
        fn = sum(g == label and p != label for g, p in zip(true_tags, predicted_tags))
        precision, recall = safe_div(tp, tp + fp), safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        per_tag[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}

    correct = sum(g == p for g, p in zip(true_tags, predicted_tags))
    tp_span = len(exact_gold & exact_pred)
    span_precision = safe_div(tp_span, len(exact_pred))
    span_recall = safe_div(tp_span, len(exact_gold))
    span_f1 = safe_div(2 * span_precision * span_recall, span_precision + span_recall)
    return {
        "token_accuracy": safe_div(correct, len(true_tags)),
        "macro_f1_non_o": safe_div(sum(x["f1"] for x in per_tag.values()), len(per_tag)),
        "token_count": len(true_tags),
        "span_precision": span_precision,
        "span_recall": span_recall,
        "span_f1": span_f1,
        "gold_spans": len(exact_gold),
        "predicted_spans": len(exact_pred),
        "per_tag": per_tag,
    }


def main():
    parser = argparse.ArgumentParser(description="Train BiLSTM-CRF from reviewed BIO labels")
    parser.add_argument("--annotations", required=True, help="Token review CSV")
    parser.add_argument("--manifest", required=True, help="One row per question; mark reviewed questions")
    parser.add_argument("--output", default="template_based_kgqa/models/entity_relation_crf.pt")
    parser.add_argument("--metrics-output", default="template_based_kgqa/models/entity_relation_crf.metrics.json")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    if not 0.05 <= args.test_fraction <= 0.5:
        raise ValueError("--test-fraction must be between 0.05 and 0.5")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    rows = load_reviewed_csv(args.annotations, args.manifest)
    train_rows, test_rows = split_records(rows, args.test_fraction, args.seed)

    # Build vocabulary only from the training questions.
    vocab = build_vocab([row["tokens"] for row in train_rows])
    tag_names = TAG_NAMES[:]
    tag_to_id = {tag: i for i, tag in enumerate(tag_names)}
    train_ds = SequenceDataset(train_rows, vocab, tag_to_id)
    test_ds = SequenceDataset(test_rows, vocab, tag_to_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = BiLSTMCRF(len(vocab), len(tag_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"Reviewed questions: {len(rows)}")
    print(f"Training questions: {len(train_rows)}")
    print(f"Held-out questions: {len(test_rows)}")
    print(f"Training BIO tags: {dict(Counter(tag for row in train_rows for tag in row['tags']))}")

    for epoch in range(args.epochs):
        model.train()
        loss_sum, seen = 0.0, 0
        for token_ids, tag_ids, mask in train_loader:
            token_ids, tag_ids, mask = token_ids.to(device), tag_ids.to(device), mask.to(device)
            optimizer.zero_grad()
            loss = model.neg_log_likelihood(token_ids, tag_ids, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.item()) * token_ids.size(0)
            seen += token_ids.size(0)
        print(f"epoch={epoch + 1} train_loss={loss_sum / max(1, seen):.4f}")

    metrics = evaluate(model, test_loader, test_rows, tag_names, device)
    print("\nHELD-OUT BIO TEST METRICS")
    print(f"Token accuracy:  {metrics['token_accuracy']:.4f}")
    print(f"Macro F1 (non-O): {metrics['macro_f1_non_o']:.4f}")
    print(f"Exact-span P/R/F1: {metrics['span_precision']:.4f} / {metrics['span_recall']:.4f} / {metrics['span_f1']:.4f}")
    for tag, values in metrics["per_tag"].items():
        print(f"{tag}: P={values['precision']:.4f} R={values['recall']:.4f} F1={values['f1']:.4f} support={values['support']}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "task": "ner",
        "state_dict": model.cpu().state_dict(),
        "vocab": vocab,
        "tags": tag_names,
        "config": {"embedding_dim": 96, "hidden_dim": 128},
        "reviewed_question_ids": [row["quid"] for row in rows],
        "train_question_ids": [row["quid"] for row in train_rows],
        "test_question_ids": [row["quid"] for row in test_rows],
        "evaluation": metrics,
        "seed": args.seed,
        "note": "Model trained only on questions explicitly marked reviewed in the manifest.",
    }
    torch.save(checkpoint, output)
    metrics_path = Path(args.metrics_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved checkpoint: {output}")
    print(f"Saved metrics: {metrics_path}")
    print("Treat results as a pilot if the reviewed set is small; keep the held-out set untouched during tuning.")


if __name__ == "__main__":
    main()
