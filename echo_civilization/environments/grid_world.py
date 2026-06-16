"""Environment 3 — Grid World (a physical simulation).

Agents move on a 2D grid with limited vision and finite energy. They collect
resources (+energy, +reward), avoid hazards (energy loss), and explore. The
policy here is a small numpy neural network (NeuralPolicy) improved across
generations by evolutionary strategies — demonstrating a *different* learning
algorithm behind the same agent abstraction.

Observation (per step), all normalised:
  [energy, dx_to_nearest_resource, dy_to_nearest_resource,
   hazard_N, hazard_S, hazard_E, hazard_W, frac_explored]
Actions: 0=N 1=S 2=E 3=W 4=stay
"""

from __future__ import annotations

import numpy as np

OBS_DIM = 8
N_ACTIONS = 5
MOVES = {0: (0, -1), 1: (0, 1), 2: (1, 0), 3: (-1, 0), 4: (0, 0)}


class GridWorld:
    name = "grid"

    def __init__(self, rng: np.random.Generator, size: int = 8,
                 n_resources: int = 6, n_hazards: int = 5,
                 max_steps: int = 60, start_energy: float = 10.0):
        self.rng = rng
        self.size = size
        self.n_resources = n_resources
        self.n_hazards = n_hazards
        self.max_steps = max_steps
        self.start_energy = start_energy

    def _reset(self):
        cells = [(x, y) for x in range(self.size) for y in range(self.size)]
        self.rng.shuffle(cells)
        self.pos = cells.pop()
        self.resources = set(cells[:self.n_resources])
        del cells[:self.n_resources]
        self.hazards = set(cells[:self.n_hazards])
        self.energy = self.start_energy
        self.visited = {self.pos}
        self.collected = 0

    def _nearest_resource_delta(self):
        if not self.resources:
            return 0.0, 0.0
        px, py = self.pos
        best = min(self.resources, key=lambda r: abs(r[0] - px) + abs(r[1] - py))
        dx = (best[0] - px) / self.size
        dy = (best[1] - py) / self.size
        return dx, dy

    def _hazard_adjacency(self):
        px, py = self.pos
        adj = []
        for d in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
            cell = (px + d[0], py + d[1])
            adj.append(1.0 if cell in self.hazards else 0.0)
        return adj

    def _obs(self):
        dx, dy = self._nearest_resource_delta()
        haz = self._hazard_adjacency()
        explored = len(self.visited) / (self.size * self.size)
        return np.array([self.energy / self.start_energy, dx, dy, *haz, explored])

    def run_episode(self, policy, agent=None) -> dict:
        """Run one life. `policy` exposes act(obs)->action."""
        self._reset()
        total_reward = 0.0
        steps = 0
        for steps in range(1, self.max_steps + 1):
            obs = self._obs()
            action = policy.act(obs)
            dx, dy = MOVES[action]
            nx = min(self.size - 1, max(0, self.pos[0] + dx))
            ny = min(self.size - 1, max(0, self.pos[1] + dy))
            self.pos = (nx, ny)
            self.energy -= 0.3  # cost of living / moving

            if self.pos not in self.visited:
                self.visited.add(self.pos)
                total_reward += 0.05  # exploration bonus

            if self.pos in self.resources:
                self.resources.remove(self.pos)
                self.energy += 4.0
                self.collected += 1
                total_reward += 1.0

            if self.pos in self.hazards:
                self.energy -= 4.0
                total_reward -= 0.5

            if self.energy <= 0:
                break

        survived = self.energy > 0
        result = {
            "reward": total_reward,
            "collected": self.collected,
            "steps_survived": steps,
            "explored": len(self.visited) / (self.size * self.size),
            "survived": survived,
        }
        if agent is not None:
            agent.record_attempt(total_reward, survived)
        return result
