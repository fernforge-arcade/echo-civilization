"""Agent memory: a short-term rolling buffer and a long-term episodic store.

Short-term memory holds the most recent observations/actions/rewards (a working
context). Long-term memory keeps important experiences and successful patterns,
with a simple salience-based forgetting curve so retention/forgetting can be
measured (Memory World).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Experience:
    t: int
    observation: Any
    action: Any
    reward: float
    salience: float = 1.0  # how strongly this is encoded (decays over time)


class ShortTermMemory:
    """A bounded rolling window of recent (obs, action, reward) triples."""

    def __init__(self, capacity: int = 8):
        self.capacity = capacity
        self.observations: deque = deque(maxlen=capacity)
        self.actions: deque = deque(maxlen=capacity)
        self.rewards: deque = deque(maxlen=capacity)

    def push(self, observation, action, reward):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)

    def recent_reward(self) -> float:
        return float(sum(self.rewards)) / len(self.rewards) if self.rewards else 0.0

    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()


class LongTermMemory:
    """Episodic + semantic memory with salience decay (forgetting).

    - episodes: notable experiences, each with a decaying salience.
    - facts:    key->value store used by Memory World (e.g. "treasure"->"blue door").
    - patterns: successful action patterns / strategies (free-form).
    """

    def __init__(self, decay: float = 0.02, capacity: int = 256):
        self.decay = decay
        self.capacity = capacity
        self.episodes: list[Experience] = []
        self.facts: dict[str, tuple[str, float]] = {}  # key -> (value, salience)
        self.patterns: dict[str, dict] = {}

    # --- episodic ----------------------------------------------------------
    def store_episode(self, exp: Experience):
        self.episodes.append(exp)
        if len(self.episodes) > self.capacity:
            # forget the least salient memory
            self.episodes.sort(key=lambda e: e.salience)
            self.episodes = self.episodes[1:]

    # --- semantic facts (for Memory World) ---------------------------------
    def remember_fact(self, key: str, value: str, strength: float = 1.0):
        self.facts[key] = (value, strength)

    def recall_fact(self, key: str) -> str | None:
        item = self.facts.get(key)
        if item is None:
            return None
        value, salience = item
        # recall succeeds probabilistically with salience, but here we return the
        # value and let the environment grade; salience drives forgetting metrics.
        return value if salience > 0.0 else None

    def fact_salience(self, key: str) -> float:
        item = self.facts.get(key)
        return item[1] if item else 0.0

    # --- patterns / strategies --------------------------------------------
    def store_pattern(self, name: str, info: dict):
        self.patterns[name] = info

    # --- forgetting curve --------------------------------------------------
    def step_forgetting(self, steps: int = 1):
        factor = (1.0 - self.decay) ** steps
        for e in self.episodes:
            e.salience *= factor
        for k, (v, s) in list(self.facts.items()):
            self.facts[k] = (v, s * factor)

    def reinforce_fact(self, key: str, amount: float = 0.5):
        if key in self.facts:
            v, s = self.facts[key]
            self.facts[key] = (v, min(1.0, s + amount))
