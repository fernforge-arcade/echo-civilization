"""Environment 1 — Transformation World.

Harder, structured tasks (echo, reverse, count, caesar-shift, ...) and crucially
*compositions* of them (e.g. reverse-then-shift). The research point: composite
tasks are tractable only when the constituent skills are already known (inherited
from culture), so this world is where accumulated knowledge pays off.
"""

from __future__ import annotations

import numpy as np

from .base import StringEnvironment, Task

# Curriculum of rule families, ordered by composition depth.
PRIMITIVE_TASKS = [
    ("echo", ("copy",)),
    ("reverse", ("reverse",)),
    ("shift", ("inc1",)),
    ("count", ("count",)),
    ("last", ("last",)),
]

COMPOSITE_TASKS = [
    ("reverse_then_shift", ("reverse", "inc1")),
    ("shift_then_reverse", ("inc1", "reverse")),
    ("double_then_reverse", ("double", "reverse")),
    ("reverse_then_count", ("reverse", "count")),
    ("dedup_then_shift", ("dedup", "inc1")),
]

DEEP_TASKS = [
    ("reverse_shift_count", ("reverse", "inc1", "count")),
    ("double_reverse_shift", ("double", "reverse", "inc1")),
]


class TransformationWorld(StringEnvironment):
    name = "transformation"

    def __init__(self, rng: np.random.Generator, tier: str = "all"):
        super().__init__(rng)
        pool = []
        if tier in ("primitive", "all", "composite", "deep"):
            pool += PRIMITIVE_TASKS
        if tier in ("composite", "all", "deep"):
            pool += COMPOSITE_TASKS
        if tier in ("deep", "all"):
            pool += DEEP_TASKS
        self.pool = pool

    def sample_task(self) -> Task:
        idx = int(self.rng.integers(0, len(self.pool)))
        name, prog = self.pool[idx]
        return self.make_task(prog, name)

    def sample_by_difficulty(self, difficulty: int) -> Task:
        candidates = [(n, p) for (n, p) in self.pool if len(p) == difficulty]
        if not candidates:
            return self.sample_task()
        name, prog = candidates[int(self.rng.integers(0, len(candidates)))]
        return self.make_task(prog, name)
