"""A tiny, dependency-light neural network controller.

This is the agent's optional "decision module" for environments where a continuous
controller is appropriate (e.g. the grid world). It is intentionally small so its
weights can be inherited and mutated cheaply across generations (an evolutionary
strategy). It is NOT a language model and carries no pretrained knowledge.
"""

from __future__ import annotations

import numpy as np


def _xavier(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out))


class MLP:
    """A minimal fully-connected network: input -> hidden(s) -> output.

    Uses tanh hidden activations. Outputs are raw logits; callers apply argmax or
    softmax. Weights are flat-serialisable so evolution can mutate/crossover them.
    """

    def __init__(self, layer_sizes, rng: np.random.Generator | None = None):
        self.layer_sizes = list(layer_sizes)
        self.rng = rng or np.random.default_rng()
        self.weights = []
        self.biases = []
        for fan_in, fan_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            self.weights.append(_xavier(self.rng, fan_in, fan_out))
            self.biases.append(np.zeros(fan_out))

    def forward(self, x: np.ndarray) -> np.ndarray:
        a = np.asarray(x, dtype=float)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ w + b
            a = np.tanh(z) if i < len(self.weights) - 1 else z
        return a

    def act(self, x: np.ndarray) -> int:
        return int(np.argmax(self.forward(x)))

    # --- evolution support -------------------------------------------------
    def get_params(self) -> np.ndarray:
        parts = [w.ravel() for w in self.weights] + [b.ravel() for b in self.biases]
        return np.concatenate(parts)

    def set_params(self, flat: np.ndarray) -> None:
        idx = 0
        for i, w in enumerate(self.weights):
            n = w.size
            self.weights[i] = flat[idx:idx + n].reshape(w.shape)
            idx += n
        for i, b in enumerate(self.biases):
            n = b.size
            self.biases[i] = flat[idx:idx + n].reshape(b.shape)
            idx += n

    def mutate(self, rate: float, scale: float) -> None:
        flat = self.get_params()
        mask = self.rng.random(flat.shape) < rate
        flat = flat + mask * self.rng.normal(0.0, scale, size=flat.shape)
        self.set_params(flat)

    def clone(self) -> "MLP":
        child = MLP(self.layer_sizes, self.rng)
        child.set_params(self.get_params().copy())
        return child

    @staticmethod
    def crossover(a: "MLP", b: "MLP", rng: np.random.Generator) -> "MLP":
        child = a.clone()
        pa, pb = a.get_params(), b.get_params()
        mask = rng.random(pa.shape) < 0.5
        child.set_params(np.where(mask, pa, pb))
        return child
