"""Task distribution for Echo-ARC.

Each task is a handful of input/output grid pairs produced by a hidden program. The
learner never sees the program -- only the examples -- and must synthesise a program
consistent with all of them (ARC's few-shot setup).

The distribution has deliberate, recurring compositional structure, because that is
what makes abstraction invention pay off:

  * `MOTIFS`  -- six length-2 base-op fragments. Every task's hidden program is a
                concatenation of motifs, so motifs recur across the whole corpus and
                the sleep phase can mine them out (Level-1 concepts).
  * `COMPOUNDS` -- a few favoured motif *pairs*, over-represented in the training
                pool. Once the motifs are named, these recur as motif-pairs and get
                abstracted into Level-2 concepts. (Some concept combinations are just
                common in a domain -- that is the cultural bias that builds hierarchy.)

Training tasks are shallow (1-2 motifs) so they are reachable from base ops; held-out
tasks are deeper (3-4 motifs) and use combinations that never appear as a training
program -- reachable only once the shallower concepts have been invented and reused.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gridworld_arc import BASE_OP_NAMES, BASE_OPS, run_program

MOTIFS = [
    ("crop", "keep_largest"),
    ("gravity_d", "compress"),
    ("sym_h", "sym_v"),
    ("rot90", "flip_h"),
    ("tile_h", "tile_v"),
    ("color_cycle", "transpose"),
]

# favoured 2-motif blocks -> recur -> become Level-2 abstractions
COMPOUNDS = [
    (MOTIFS[0], MOTIFS[2]),   # (crop,keep_largest) + (sym_h,sym_v)
    (MOTIFS[1], MOTIFS[4]),   # (gravity_d,compress) + (tile_h,tile_v)
    (MOTIFS[3], MOTIFS[0]),   # (rot90,flip_h) + (crop,keep_largest)
]


@dataclass
class Task:
    examples: list           # list[(in_grid, out_grid)]
    program: tuple           # hidden ground-truth (base ops)
    motifs: tuple            # the motif sequence it was built from
    depth: int               # number of motifs


def _random_grid(rng) -> np.ndarray:
    H = int(rng.integers(4, 7))
    W = int(rng.integers(4, 7))
    g = np.zeros((H, W), dtype=np.int64)
    n = int(rng.integers(4, 8))
    for _ in range(n):
        r, c = int(rng.integers(0, H)), int(rng.integers(0, W))
        g[r, c] = int(rng.integers(1, 5))
    # a couple of 2-cell blobs so "largest object" is meaningful
    for _ in range(2):
        r, c = int(rng.integers(0, H - 1)), int(rng.integers(0, W))
        col = int(rng.integers(1, 5))
        g[r, c] = col
        g[r + 1, c] = col
    return g


def _program_from_motifs(motif_seq) -> tuple:
    prog = []
    for m in motif_seq:
        prog.extend(m)
    return tuple(prog)


def _sample_motif_seq(rng, depth, use_compounds):
    seq = []
    while len(seq) < depth:
        if use_compounds and depth - len(seq) >= 2 and rng.random() < 0.6:
            comp = COMPOUNDS[int(rng.integers(0, len(COMPOUNDS)))]
            seq.extend(comp)
        else:
            seq.append(MOTIFS[int(rng.integers(0, len(MOTIFS)))])
    return tuple(seq[:depth])


def _trivially_solvable(examples) -> bool:
    """True if identity or any single base op already reproduces every example --
    such a task doesn't require any depth, so we discard it."""
    for op in [None] + BASE_OP_NAMES:
        ok = True
        for inp, out in examples:
            pred = inp if op is None else BASE_OPS[op](inp)
            if pred.shape != out.shape or not np.array_equal(pred, out):
                ok = False
                break
        if ok:
            return True
    return False


def _make_task(rng, depth, use_compounds, n_examples=5, tries=40):
    for _ in range(tries):
        motif_seq = _sample_motif_seq(rng, depth, use_compounds)
        prog = _program_from_motifs(motif_seq)
        examples = []
        good = True
        outs = []
        for _ in range(n_examples):
            g = _random_grid(rng)
            out = run_program(prog, g)
            if out.size == 0 or out.size > 400 or np.count_nonzero(out) == 0:
                good = False
                break
            examples.append((g, out))
            outs.append(out.tobytes())
        if not good:
            continue
        if len(set(outs)) < 2:          # outputs must vary across examples
            continue
        if _trivially_solvable(examples):
            continue
        return Task(examples=examples, program=prog, motifs=motif_seq, depth=depth)
    return None


def make_training_pool(rng, n=60):
    """Shallow tasks (1-2 motifs), reachable from base ops."""
    tasks = []
    while len(tasks) < n:
        depth = 1 if rng.random() < 0.45 else 2
        t = _make_task(rng, depth, use_compounds=(depth == 2))
        if t is not None:
            tasks.append(t)
    return tasks


def make_heldout(rng, n=40):
    """Deep tasks (3-4 motifs) with combinations never used as a training program.
    These are the accumulated-capability test: unreachable without invented concepts."""
    tasks = []
    while len(tasks) < n:
        depth = 3 if rng.random() < 0.5 else 4
        t = _make_task(rng, depth, use_compounds=True)
        if t is not None:
            tasks.append(t)
    return tasks
