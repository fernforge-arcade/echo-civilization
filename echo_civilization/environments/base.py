"""Base classes for the string-transformation worlds.

A `Task` hides a rule (a ground-truth program) and exposes a few demonstration
examples plus a held-out query. Agents never see the rule — they must infer it.
Reward is graded by character overlap so partial credit is available, exactly as
the brief specifies for Echo World.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..skills import ALPHABET, run_program


def grade(prediction: str, target: str) -> float:
    """Reward in [0,1]: 1.0 for exact match, otherwise partial character overlap."""
    if prediction == target:
        return 1.0
    if not target:
        return 1.0 if prediction == "" else 0.0
    # normalized positional match + length penalty
    n = max(len(prediction), len(target))
    matches = sum(1 for i in range(min(len(prediction), len(target)))
                  if prediction[i] == target[i])
    return matches / n


@dataclass
class Task:
    name: str                       # human label of the rule family
    program: tuple                  # ground-truth program (hidden from agent)
    examples: list                  # [(inp, out), ...] demonstrations
    query_input: str
    query_target: str
    difficulty: int = 1             # = program length (composition depth)

    def is_solved(self, prediction: str) -> bool:
        return prediction == self.query_target


class StringEnvironment:
    """Base for worlds that pose string tasks. Subclasses implement sample_task."""

    name = "string"

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def _rand_string(self, lo=3, hi=5) -> str:
        n = int(self.rng.integers(lo, hi + 1))
        return "".join(self.rng.choice(list(ALPHABET), size=n))

    def make_task(self, program, name, n_examples=3) -> Task:
        examples = []
        for _ in range(n_examples):
            inp = self._rand_string()
            examples.append((inp, run_program(program, inp)))
        q = self._rand_string()
        return Task(name=name, program=tuple(program), examples=examples,
                    query_input=q, query_target=run_program(program, q),
                    difficulty=len(program))

    def sample_task(self) -> Task:
        raise NotImplementedError
