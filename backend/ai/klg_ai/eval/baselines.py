"""DKT — the sequence-model baseline for RQ1 (graph engine vs. sequence model).

Deep Knowledge Tracing (Piech et al., 2015): an LSTM over a learner's sequence
of (skill, correct) interactions that predicts the next answer's correctness. It
uses the *same* concept-node skill tags the engine does, but only their order —
no prerequisite graph, no forgetting curve. So engine-vs-DKT isolates what the
graph structure buys (RQ1).

Lightweight + self-contained (torch only, already a dep): a single padded-batch
forward pass, trained a few epochs on train next-steps, then the held-out items
are read off the same pass autoregressively (each eval item is predicted from the
state before it, then its true label is fed forward — standard causal KT eval).
Recent history is capped per learner to bound memory; the cap keeps all eval
items and trims only the oldest train.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .dataset import LearnerData

_UNK = "<unk>"


def _primary_skill(node_ids: tuple[str, ...]) -> str:
    """One skill per interaction for the sequence model (engine uses all nodes)."""
    return node_ids[0] if node_ids else _UNK


class _DKTNet(nn.Module):
    def __init__(self, n_skills: int, emb: int = 64, hidden: int = 64):
        super().__init__()
        self.n_skills = n_skills
        self.embed = nn.Embedding(2 * n_skills + 1, emb, padding_idx=2 * n_skills)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_skills)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(self.embed(x))
        return torch.sigmoid(self.out(h))  # [B, T, n_skills] = P(correct | next skill)


class DKTPredictor:
    """LSTM knowledge tracing over node-tagged interaction sequences."""
    name = "dkt"

    def __init__(self, *, epochs: int = 10, lr: float = 0.01, max_len: int = 400,
                 emb: int = 64, hidden: int = 64, seed: int = 0):
        self.epochs = epochs
        self.lr = lr
        self.max_len = max_len
        self.emb = emb
        self.hidden = hidden
        self.seed = seed

    def predict(self, learners: list[LearnerData]) -> list[float]:
        torch.manual_seed(self.seed)

        # Skill vocabulary from train; unseen eval skills fall back to <unk>.
        vocab: dict[str, int] = {_UNK: 0}
        for ld in learners:
            for it in ld.train:
                s = _primary_skill(it.node_ids)
                if s not in vocab:
                    vocab[s] = len(vocab)
        n_skills = len(vocab)
        pad = 2 * n_skills

        # Build padded sequences: recent train (trimmed) + all eval, in time order.
        seqs: list[tuple[list[int], list[int], list[float], int]] = []
        for ld in learners:
            keep_train = ld.train[-max(1, self.max_len - len(ld.evalset)):]
            items = keep_train + ld.evalset
            skills = [vocab.get(_primary_skill(it.node_ids), 0) for it in items]
            corrects = [1.0 if it.correct else 0.0 for it in items]
            inp = [s * 2 + int(c) for s, c in zip(skills, corrects)]
            seqs.append((inp, skills, corrects, len(keep_train)))

        batch = len(seqs)
        T = max(len(s[0]) for s in seqs)
        x = torch.full((batch, T), pad, dtype=torch.long)
        skill = torch.zeros((batch, T), dtype=torch.long)
        correct = torch.zeros((batch, T), dtype=torch.float32)
        is_train = torch.zeros((batch, T), dtype=torch.bool)
        length = torch.zeros(batch, dtype=torch.long)
        for i, (inp, sk, corr, n_tr) in enumerate(seqs):
            L = len(inp)
            x[i, :L] = torch.tensor(inp)
            skill[i, :L] = torch.tensor(sk)
            correct[i, :L] = torch.tensor(corr)
            is_train[i, :n_tr] = True
            length[i] = L

        # Predicting item t+1 from state after item t: target/skill shifted by one.
        next_skill = torch.zeros_like(skill)
        next_correct = torch.zeros_like(correct)
        next_skill[:, :-1] = skill[:, 1:]
        next_correct[:, :-1] = correct[:, 1:]
        valid_next = torch.zeros((batch, T), dtype=torch.bool)
        for i in range(batch):
            valid_next[i, : length[i] - 1] = True
        # Train loss only where the predicted (next) item is a train item.
        next_is_train = torch.zeros((batch, T), dtype=torch.bool)
        next_is_train[:, :-1] = is_train[:, 1:]
        train_mask = valid_next & next_is_train
        eval_mask = valid_next & (~next_is_train)

        model = _DKTNet(n_skills, self.emb, self.hidden)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.BCELoss(reduction="none")

        for _ in range(self.epochs):
            model.train()
            opt.zero_grad()
            out = model(x)  # [B, T, n_skills]
            pred = out.gather(2, next_skill.unsqueeze(2)).squeeze(2)  # [B, T]
            losses = loss_fn(pred.clamp(1e-6, 1 - 1e-6), next_correct)
            m = train_mask.float()
            loss = (losses * m).sum() / m.sum().clamp(min=1.0)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            out = model(x)
            pred = out.gather(2, next_skill.unsqueeze(2)).squeeze(2)  # P(correct)

        # Gather eval predictions in learner/evalset order (== all_eval_instances).
        preds: list[float] = []
        for i in range(batch):
            for t in range(T):
                if eval_mask[i, t]:
                    preds.append(1.0 - float(pred[i, t]))  # P(mistake)
        return preds
