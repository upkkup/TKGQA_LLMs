"""Train question-type classifier or BiLSTM-CRF tagger."""
import argparse
import random
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .dataset import load_dataset, load_ner_annotations, tokenize
from .models import BiLSTMCRF, QuestionTypeClassifier


PAD = "<PAD>"
UNK = "<UNK>"


def build_vocab(sequences, min_freq=1):
    counts = Counter(token for seq in sequences for token in seq)
    vocab = {PAD: 0, UNK: 1}
    for token, count in counts.items():
        if count >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode(tokens, vocab):
    return torch.tensor([vocab.get(t, vocab[UNK]) for t in tokens], dtype=torch.long)


class TypeDataset(Dataset):
    def __init__(self, rows, vocab, labels):
        self.items = [(encode(tokenize(r["question"]), vocab), labels[r["qtype"]]) for r in rows]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def type_collate(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(x) for x in sequences], dtype=torch.long)
    return pad_sequence(sequences, batch_first=True, padding_value=0), lengths, torch.tensor(labels)


def train_type(args):
    rows = load_dataset(args.dataset)
    random.Random(args.seed).shuffle(rows)
    vocab = build_vocab([tokenize(r["question"]) for r in rows])
    label_names = sorted({r["qtype"] for r in rows})
    label_to_id = {label: i for i, label in enumerate(label_names)}
    ds = TypeDataset(rows, vocab, label_to_id)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=type_collate)
    model = QuestionTypeClassifier(len(vocab), len(label_names))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for token_ids, lengths, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(token_ids, lengths), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item())
        print(f"epoch={epoch + 1} mean_loss={total / max(1, len(loader)):.4f}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"task": "question_type", "state_dict": model.state_dict(), "vocab": vocab,
                "labels": label_names, "config": {"embedding_dim": 96, "hidden_dim": 128}}, output)
    print(f"saved={output} records={len(rows)} classes={len(label_names)}")
    print("Training loss is not held-out accuracy. Evaluate on a separate split before reporting performance.")


def ner_collate(batch):
    tokens, tags = zip(*batch)
    lengths = torch.tensor([len(x) for x in tokens], dtype=torch.long)
    token_pad = pad_sequence(tokens, batch_first=True, padding_value=0)
    tag_pad = pad_sequence(tags, batch_first=True, padding_value=0)
    mask = torch.arange(token_pad.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
    return token_pad, tag_pad, mask


class NERDataset(Dataset):
    def __init__(self, rows, vocab, tag_to_id):
        self.items = [(encode(r["tokens"], vocab),
                       torch.tensor([tag_to_id[t] for t in r["tags"]], dtype=torch.long))
                      for r in rows]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def train_ner(args):
    rows = load_ner_annotations(args.annotations)
    vocab = build_vocab([r["tokens"] for r in rows])
    tag_names = sorted({tag for row in rows for tag in row["tags"]}, key=lambda x: (x != "O", x))
    tag_to_id = {tag: i for i, tag in enumerate(tag_names)}
    ds = NERDataset(rows, vocab, tag_to_id)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=ner_collate)
    model = BiLSTMCRF(len(vocab), len(tag_names))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for token_ids, tags, mask in loader:
            optimizer.zero_grad()
            loss = model.neg_log_likelihood(token_ids, tags, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item())
        print(f"epoch={epoch + 1} mean_loss={total / max(1, len(loader)):.4f}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"task": "ner", "state_dict": model.state_dict(), "vocab": vocab,
                "tags": tag_names, "config": {"embedding_dim": 96, "hidden_dim": 128}}, output)
    print(f"saved={output} annotated_records={len(rows)} tags={len(tag_names)}")
    print("Use reviewed span annotations and evaluate token/span F1 on held-out data.")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="task", required=True)
    for task in ("type", "ner"):
        p = sub.add_parser(task)
        p.add_argument("--output", required=True)
        p.add_argument("--epochs", type=int, default=12)
        p.add_argument("--batch-size", type=int, default=32)
        p.add_argument("--lr", type=float, default=0.001)
        p.add_argument("--seed", type=int, default=42)
        if task == "type":
            p.add_argument("--dataset", required=True)
        else:
            p.add_argument("--annotations", required=True)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.task == "type":
        train_type(args)
    else:
        train_ner(args)


if __name__ == "__main__":
    main()
