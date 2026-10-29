"""PyTorch question classifier and BiLSTM-CRF sequence tagger."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class QuestionTypeClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, embedding_dim: int = 96, hidden_dim: int = 128, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, token_ids, lengths):
        embedded = self.embedding(token_ids)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.encoder(packed)
        features = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(self.dropout(features))


class BiLSTMCRF(nn.Module):
    """Linear-chain CRF with learned start/end scores and Viterbi decoding."""

    def __init__(self, vocab_size: int, num_tags: int, embedding_dim: int = 96, hidden_dim: int = 128, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.emission = nn.Linear(hidden_dim * 2, num_tags)
        self.transitions = nn.Parameter(torch.zeros(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.zeros(num_tags))
        self.end_transitions = nn.Parameter(torch.zeros(num_tags))

    def emissions(self, token_ids, mask):
        embedded = self.embedding(token_ids)
        lengths = mask.long().sum(dim=1).cpu()
        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        encoded, _ = self.encoder(packed)
        padded, _ = torch.nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True, total_length=token_ids.size(1))
        return self.emission(padded)

    def neg_log_likelihood(self, token_ids, tags, mask):
        emissions = self.emissions(token_ids, mask)
        return (self._log_partition(emissions, mask) - self._gold_score(emissions, tags, mask)).mean()

    def _log_partition(self, emissions, mask):
        alpha = self.start_transitions + emissions[:, 0]
        for t in range(1, emissions.size(1)):
            scores = alpha.unsqueeze(2) + self.transitions.unsqueeze(0) + emissions[:, t].unsqueeze(1)
            next_alpha = torch.logsumexp(scores, dim=1)
            alpha = torch.where(mask[:, t].unsqueeze(1), next_alpha, alpha)
        return torch.logsumexp(alpha + self.end_transitions, dim=1)

    def _gold_score(self, emissions, tags, mask):
        batch = torch.arange(tags.size(0), device=tags.device)
        score = self.start_transitions[tags[:, 0]] + emissions[batch, 0, tags[:, 0]]
        for t in range(1, tags.size(1)):
            transition = self.transitions[tags[:, t - 1], tags[:, t]]
            emission = emissions[batch, t, tags[:, t]]
            score = score + (transition + emission) * mask[:, t]
        lengths = mask.long().sum(dim=1) - 1
        last_tags = tags[batch, lengths]
        return score + self.end_transitions[last_tags]

    @torch.no_grad()
    def decode(self, token_ids, mask):
        emissions = self.emissions(token_ids, mask)
        paths = []
        for row in range(token_ids.size(0)):
            length = int(mask[row].sum().item())
            emit = emissions[row, :length]
            score = self.start_transitions + emit[0]
            backpointers = []
            for t in range(1, length):
                candidates = score.unsqueeze(1) + self.transitions
                best_score, best_previous = candidates.max(dim=0)
                score = best_score + emit[t]
                backpointers.append(best_previous)
            last = int((score + self.end_transitions).argmax().item())
            path = [last]
            for pointer in reversed(backpointers):
                last = int(pointer[last].item())
                path.append(last)
            paths.append(list(reversed(path)))
        return paths
