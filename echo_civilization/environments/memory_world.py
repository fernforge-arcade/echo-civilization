"""Environment 2 — Memory World.

The agent is told a fact at time 1 ("the treasure is behind the blue door") and
quizzed at time 2 ("where is the treasure?"), possibly after a delay during which
unrelated experiences occur. This measures retention, forgetting, and — when
facts can be shared — knowledge transfer between agents.
"""

from __future__ import annotations

import numpy as np

from ..agent import Agent

FACTS = [
    ("treasure", "blue door"),
    ("key", "under the rug"),
    ("exit", "north tunnel"),
    ("water", "deep well"),
    ("food", "tall tree"),
    ("danger", "red cave"),
]


class MemoryWorld:
    name = "memory"

    def __init__(self, rng: np.random.Generator, delay: int = 5):
        self.rng = rng
        self.delay = delay  # interfering steps between telling and quizzing

    def run_episode(self, agent: Agent) -> dict:
        """Tell the agent a random fact, let time pass (forgetting), then quiz."""
        key, value = FACTS[int(self.rng.integers(0, len(FACTS)))]
        agent.long_term.remember_fact(key, value, strength=1.0)

        # interference: unrelated steps cause forgetting
        agent.long_term.step_forgetting(self.delay)

        recalled = agent.long_term.recall_fact(key)
        salience = agent.long_term.fact_salience(key)
        # recall is graded as correct if the value is retained with enough salience
        correct = (recalled == value) and (salience > 0.3)
        reward = 1.0 if correct else max(0.0, salience)
        agent.record_attempt(reward, correct)
        return {"key": key, "correct": correct, "salience": round(salience, 3),
                "reward": reward}

    def transfer_fact(self, teacher: Agent, student: Agent, key: str) -> bool:
        """Knowledge transfer: a teacher tells a student a fact it remembers."""
        val = teacher.long_term.recall_fact(key)
        if val is None:
            return False
        student.long_term.remember_fact(key, val, strength=0.9)
        return True
