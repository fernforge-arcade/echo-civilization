"""Learned guidance for the search (the "neural" half of the loop).

The brief's point 8: pure search doesn't scale, pure neural prediction is brittle;
the model should *guide* search toward promising ops rather than emit the answer.

This is a tiny per-op logistic regression, trained online (numpy SGD, no pretrained
anything) on (task features -> which ops appeared in the solution). At solve time it
scores every op in the current vocabulary and hands the searcher an ordering, so the
most likely-useful concepts (including freshly invented abstractions) are tried first.
It never produces a program; if its ranking is wrong the search still finds the answer,
just slower. That is the point -- guidance, not an oracle.
"""

from __future__ import annotations

import numpy as np


def task_features(examples) -> np.ndarray:
    """Cheap symbolic features of a task, from its input/output grids."""
    feats = []
    for inp, out in examples:
        ih, iw = inp.shape
        oh, ow = out.shape
        feats.append([
            oh / max(ih, 1) - 1.0,                       # height growth
            ow / max(iw, 1) - 1.0,                        # width growth
            (out.size - inp.size) / max(inp.size, 1),     # area change
            np.count_nonzero(out) / max(np.count_nonzero(inp), 1) - 1.0,
            float(np.array_equal(out, np.fliplr(out))),   # output h-symmetric
            float(np.array_equal(out, np.flipud(out))),   # output v-symmetric
            len(set(int(v) for v in out.flatten() if v)) -
            len(set(int(v) for v in inp.flatten() if v)),  # colour count delta
        ])
    v = np.mean(np.array(feats, dtype=np.float64), axis=0)
    return np.concatenate([v, [1.0]])   # + bias term


class Proposer:
    def __init__(self, n_features=8, lr=0.3):
        self.n_features = n_features
        self.lr = lr
        self.w: dict[str, np.ndarray] = {}   # op name -> weight vector

    def _wv(self, op):
        if op not in self.w:
            self.w[op] = np.zeros(self.n_features)
        return self.w[op]

    def score(self, op, x) -> float:
        return float(1.0 / (1.0 + np.exp(-self._wv(op) @ x)))

    def order(self, ops, examples, library):
        x = task_features(examples)
        # base ops keep their natural order as a tiebreak; sort by predicted usefulness
        return sorted(ops, key=lambda o: -self.score(o, x))

    def update(self, examples, solution, vocab):
        """One SGD step: ops used in `solution` are positives, the rest negatives.
        `solution` is expressed in library tokens (so it credits abstractions too)."""
        x = task_features(examples)
        used = set(solution)
        for op in vocab:
            y = 1.0 if op in used else 0.0
            p = self.score(op, x)
            self.w[op] = self._wv(op) - self.lr * (p - y) * x
