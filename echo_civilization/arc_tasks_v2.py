"""Task distribution v2 for the falsification experiment (Experiment N).

This tightens the Experiment-M distribution in the two ways the operator's brief
demands before the M result can be called a library-learning benchmark rather than a
designed demonstration:

  1. **Every task carries a held-out query grid.** Synthesis sees only the `train`
     I/O pairs. A program counts as a solve only if it *also* reproduces the unseen
     `query` output. This kills programs that fit the few demonstration pairs by luck
     but do not implement the intended function.

  2. **Train / eval are separated by hidden structure, not just by sampled program.**
     Each motif is an atom; a program is a chain of motifs, and every *adjacency*
     (consecutive motif pair) is either a TRAIN adjacency or a HELDOUT adjacency.
     Training programs use only TRAIN adjacencies; held-out programs must contain at
     least one HELDOUT adjacency. So a held-out task always exercises a motif
     *combination* that never occurred in training, even though every individual
     motif was seen. Generalisation therefore has to come from recombining learned
     concepts, not from reciting a memorised program.

The motifs themselves are identical to `arc_tasks.py` so the two experiments are
directly comparable; only the sampling constraints and the query grid are new.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from .gridworld_arc import BASE_OP_NAMES, BASE_OPS, run_program

# Same six length-2 motifs as Experiment M.
MOTIFS = [
    ("crop", "keep_largest"),      # 0
    ("gravity_d", "compress"),     # 1
    ("sym_h", "sym_v"),            # 2
    ("rot90", "flip_h"),           # 3
    ("tile_h", "tile_v"),          # 4
    ("color_cycle", "transpose"),  # 5
]

# Held-out adjacencies (unordered motif-index pairs). These motif *pairings* are
# forbidden in every training program and required in every held-out program. The
# individual motifs still appear in training via other adjacencies, so their concepts
# are learnable; only the combination is novel.
HELDOUT_ADJ = frozenset([
    frozenset((0, 2)),   # (crop,keep_largest) next to (sym_h,sym_v)
    frozenset((1, 4)),   # (gravity_d,compress) next to (tile_h,tile_v)
    frozenset((3, 0)),   # (rot90,flip_h) next to (crop,keep_largest)
    frozenset((5, 2)),   # (color_cycle,transpose) next to (sym_h,sym_v)
])
ALL_ADJ = frozenset(frozenset(p) for p in combinations(range(len(MOTIFS)), 2))
TRAIN_ADJ = ALL_ADJ - HELDOUT_ADJ


@dataclass
class TaskV2:
    train: list              # list[(in_grid, out_grid)] -- synthesis sees only these
    query: tuple             # (in_grid, out_grid) -- unseen; required for a solve
    program: tuple           # hidden ground-truth base-op program
    motif_idx: tuple         # the motif-index sequence
    depth: int               # number of motifs
    adjacencies: frozenset   # set of unordered consecutive motif-index pairs

    def all_pairs(self):
        """train pairs + query pair, for behavioural / oracle checks."""
        return list(self.train) + [self.query]


def _random_grid(rng) -> np.ndarray:
    H = int(rng.integers(4, 7))
    W = int(rng.integers(4, 7))
    g = np.zeros((H, W), dtype=np.int64)
    n = int(rng.integers(4, 8))
    for _ in range(n):
        r, c = int(rng.integers(0, H)), int(rng.integers(0, W))
        g[r, c] = int(rng.integers(1, 5))
    for _ in range(2):
        r, c = int(rng.integers(0, H - 1)), int(rng.integers(0, W))
        col = int(rng.integers(1, 5))
        g[r, c] = col
        g[r + 1, c] = col
    return g


def _program_from_motifs(motif_idx) -> tuple:
    prog = []
    for m in motif_idx:
        prog.extend(MOTIFS[m])
    return tuple(prog)


def _adjacencies(motif_idx) -> frozenset:
    return frozenset(frozenset((motif_idx[i], motif_idx[i + 1]))
                     for i in range(len(motif_idx) - 1)
                     if motif_idx[i] != motif_idx[i + 1])


def _sample_train_motifs(rng, depth, max_tries=200):
    """Chain of motifs whose every adjacency is a TRAIN adjacency."""
    for _ in range(max_tries):
        seq = [int(rng.integers(0, len(MOTIFS)))]
        ok = True
        while len(seq) < depth:
            cands = [m for m in range(len(MOTIFS))
                     if m == seq[-1] or frozenset((seq[-1], m)) in TRAIN_ADJ]
            if not cands:
                ok = False
                break
            seq.append(int(rng.choice(cands)))
        if ok and len(seq) == depth:
            return tuple(seq)
    return None


def _sample_heldout_motifs(rng, depth, max_tries=200):
    """Chain of `depth` motifs containing at least one HELDOUT adjacency."""
    for _ in range(max_tries):
        seq = tuple(int(rng.integers(0, len(MOTIFS))) for _ in range(depth))
        if _adjacencies(seq) & HELDOUT_ADJ:
            return seq
    return None


def _trivially_solvable(pairs) -> bool:
    for op in [None] + BASE_OP_NAMES:
        ok = True
        for inp, out in pairs:
            pred = inp if op is None else BASE_OPS[op](inp)
            if pred.shape != out.shape or not np.array_equal(pred, out):
                ok = False
                break
        if ok:
            return True
    return False


def _make_task(rng, motif_idx, n_train=5, tries=40):
    """Build a task with `n_train` demonstration pairs + 1 unseen query pair."""
    prog = _program_from_motifs(motif_idx)
    for _ in range(tries):
        pairs = []
        good = True
        outs = []
        for _ in range(n_train + 1):
            g = _random_grid(rng)
            out = run_program(prog, g)
            if out.size == 0 or out.size > 400 or np.count_nonzero(out) == 0:
                good = False
                break
            pairs.append((g, out))
            outs.append(out.tobytes())
        if not good:
            continue
        if len(set(outs)) < 2:
            continue
        if _trivially_solvable(pairs):
            continue
        return TaskV2(train=pairs[:n_train], query=pairs[n_train], program=prog,
                      motif_idx=motif_idx, depth=len(motif_idx),
                      adjacencies=_adjacencies(motif_idx))
    return None


def make_training_pool(rng, n=80):
    """Shallow tasks (1-2 motifs) built only from TRAIN adjacencies."""
    tasks = []
    guard = 0
    while len(tasks) < n and guard < n * 50:
        guard += 1
        depth = 1 if rng.random() < 0.45 else 2
        mi = _sample_train_motifs(rng, depth)
        if mi is None:
            continue
        t = _make_task(rng, mi)
        if t is not None:
            tasks.append(t)
    return tasks


def make_heldout(rng, n=40):
    """Deep tasks (3-4 motifs) that always contain a held-out motif pairing."""
    tasks = []
    guard = 0
    while len(tasks) < n and guard < n * 80:
        guard += 1
        depth = 3 if rng.random() < 0.5 else 4
        mi = _sample_heldout_motifs(rng, depth)
        if mi is None:
            continue
        t = _make_task(rng, mi)
        if t is not None:
            tasks.append(t)
    return tasks
