"""Learning algorithms behind a single swappable interface.

The brief asks to *start with Q-learning* and keep the architecture extendable to
policy gradients / evolutionary strategies / neural networks. We therefore define
a `Learner` protocol and provide:

- `QLearner`        : tabular Q-learning (the genuine experience->skill loop).
- `NeuralPolicy`    : a numpy MLP improved by evolutionary strategies (mutation +
                      selection across generations) — no gradient backprop needed.
- `RandomLearner`   : baseline / ablation control.

All learners expose act / observe / snapshot so environments and the evolution
engine can treat them uniformly and swap algorithms freely.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .neural import MLP


class Learner:
    """Interface. Subclasses implement act() and observe()."""

    def act(self, obs, valid_actions=None):
        raise NotImplementedError

    def observe(self, obs, action, reward, next_obs, done):
        pass

    def snapshot(self) -> dict:
        return {}


class RandomLearner(Learner):
    def __init__(self, n_actions: int, rng: np.random.Generator):
        self.n_actions = n_actions
        self.rng = rng

    def act(self, obs, valid_actions=None):
        choices = valid_actions if valid_actions is not None else range(self.n_actions)
        return int(self.rng.choice(list(choices)))


class QLearner(Learner):
    """Tabular Q-learning with epsilon-greedy exploration.

    State and action are hashable (we use ints / small tuples). Used to discover
    character-substitution mappings from reward in the Echo and Transformation
    worlds, and could equally drive a discretised grid world.
    """

    def __init__(self, n_actions: int, rng: np.random.Generator,
                 alpha: float = 0.4, gamma: float = 0.9,
                 epsilon: float = 0.3, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.02):
        self.n_actions = n_actions
        self.rng = rng
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q: dict = defaultdict(lambda: np.zeros(self.n_actions))

    def act(self, obs, valid_actions=None):
        if self.rng.random() < self.epsilon:
            choices = valid_actions if valid_actions is not None else range(self.n_actions)
            return int(self.rng.choice(list(choices)))
        qvals = self.q[obs]
        if valid_actions is not None:
            masked = np.full(self.n_actions, -np.inf)
            for a in valid_actions:
                masked[a] = qvals[a]
            return int(np.argmax(masked))
        return int(np.argmax(qvals))

    def observe(self, obs, action, reward, next_obs, done):
        best_next = 0.0 if done else float(np.max(self.q[next_obs]))
        target = reward + self.gamma * best_next
        self.q[obs][action] += self.alpha * (target - self.q[obs][action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def greedy_map(self):
        """Return the current greedy action for every seen state."""
        return {s: int(np.argmax(v)) for s, v in self.q.items()}

    def snapshot(self) -> dict:
        return {"states_seen": len(self.q), "epsilon": round(self.epsilon, 3)}

    # inheritance: a child can warm-start from a parent's Q-table
    def inherit_from(self, parent: "QLearner", fraction: float = 1.0):
        for s, v in parent.q.items():
            if self.rng.random() < fraction:
                self.q[s] = v.copy()


class NeuralPolicy(Learner):
    """MLP policy improved by evolutionary strategies (no backprop).

    During an episode it acts greedily from the network. Learning happens between
    generations: the evolution engine mutates weights and selects high-fitness
    networks. This demonstrates the "swap the learning algorithm" extensibility.
    """

    def __init__(self, layer_sizes, rng: np.random.Generator):
        self.net = MLP(layer_sizes, rng)
        self.rng = rng

    def act(self, obs, valid_actions=None):
        logits = self.net.forward(np.asarray(obs, dtype=float))
        if valid_actions is not None:
            masked = np.full_like(logits, -np.inf)
            for a in valid_actions:
                masked[a] = logits[a]
            return int(np.argmax(masked))
        return int(np.argmax(logits))

    def mutate(self, rate: float, scale: float):
        self.net.mutate(rate, scale)

    def clone(self) -> "NeuralPolicy":
        c = NeuralPolicy(self.net.layer_sizes, self.rng)
        c.net.set_params(self.net.get_params().copy())
        return c

    def snapshot(self) -> dict:
        return {"params": int(self.net.get_params().size)}
