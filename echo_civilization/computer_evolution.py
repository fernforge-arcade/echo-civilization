"""Computer-World civilization with an auto-curriculum.

This is the extension that pushes the project toward *more capable, tool-using*
agents. A population operates the simulated computer (`ComputerWorld`) under an
**auto-curriculum**: the difficulty offered tracks a moving *frontier* level, and
the frontier advances whenever the population masters it. Mastering level *k* and
then reaching level *k+1* requires composing/modifying macros accumulated at
lower levels — so a civilization that shares and inherits macros keeps climbing
the ladder, while one that does not stalls near the bottom.

The headline metric is therefore not a fixed-task success rate but **how high up
an open-ended ladder of task sophistication the civilization can climb over
generations** — i.e. whether the agents *evolve to match increasingly
sophisticated tasks*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .agent import Agent
from .culture import CulturalMemory
from .environments.computer_world import (CURRICULUM, MAX_LEVEL, ComputerWorld)
from .skills import Skill


@dataclass
class ComputerConfig:
    name: str = "E_computer_civilization"
    population_size: int = 24
    generations: int = 30
    tasks_per_agent: int = 10
    budget: int = 150
    advance_threshold: float = 0.45   # frontier solve-rate needed to advance
    use_culture: bool = True
    use_inheritance: bool = True
    use_teaching: bool = True
    use_reputation: bool = True
    elite_fraction: float = 0.3
    inherit_skill_fraction: float = 0.9
    cultural_seed_top: int = 20
    seed: int = 0

    def as_dict(self):
        return self.__dict__.copy()


class ComputerCivilization:
    def __init__(self, config: ComputerConfig, db=None):
        self.cfg = config
        self.db = db
        self.rng = np.random.default_rng(config.seed)
        self.world = ComputerWorld(self.rng)
        self.culture = CulturalMemory()
        self.population: list[Agent] = []
        self.frontier = 1            # current curriculum level offered
        self.history: list[dict] = []

    # --------------------------------------------------------- population
    def _spawn_initial(self):
        self.population = [Agent(0, self.rng) for _ in range(self.cfg.population_size)]

    def _seed_from_culture(self, agent: Agent):
        if not self.cfg.use_culture:
            return
        for skill in self.culture.top_skills(self.cfg.cultural_seed_top):
            if self.rng.random() < max(0.5, agent.preferences["imitation"]):
                if agent.learn_computer_skill(Skill(
                        name=skill.name, program=skill.program, creator=skill.creator,
                        generation=skill.generation,
                        preconditions=list(skill.preconditions),
                        examples=list(skill.examples))):
                    skill.adoption += 1

    def _reproduce(self, gen: int):
        ranked = sorted(self.population, key=lambda a: a.fitness, reverse=True)
        n_elite = max(1, int(self.cfg.elite_fraction * len(ranked)))
        elites = ranked[:n_elite]
        children = []
        for _ in range(self.cfg.population_size):
            if self.cfg.use_reputation:
                weights = np.array([max(0.01, a.fitness) for a in elites])
                parent = elites[int(self.rng.choice(len(elites), p=weights / weights.sum()))]
            else:
                parent = elites[int(self.rng.integers(0, len(elites)))]
            child = Agent(gen, self.rng, parents=[parent.id])
            for k in child.preferences:
                child.preferences[k] = float(np.clip(
                    parent.preferences[k] + self.rng.normal(0, 0.05), 0.0, 1.0))
            if self.cfg.use_inheritance:
                for prog, skill in parent.computer_skills.items():
                    if self.rng.random() < self.cfg.inherit_skill_fraction:
                        child.learn_computer_skill(Skill(
                            name=skill.name, program=skill.program,
                            creator=skill.creator, generation=skill.generation,
                            preconditions=list(skill.preconditions),
                            examples=list(skill.examples)))
            self._seed_from_culture(child)
            children.append(child)
        return children

    # ----------------------------------------------------------- teaching
    def _teach_round(self, gen: int) -> int:
        transfers = 0
        n = len(self.population)
        for i in range(n):
            teacher = self.population[i]
            if not teacher.computer_skills:
                continue
            skill = max(teacher.computer_skills.values(),
                        key=lambda s: (s.reputation, s.complexity()))
            j = int(self.rng.integers(0, n))
            if j == i:
                continue
            student = self.population[j]
            if student.preferences["imitation"] < self.rng.uniform(0, 1):
                continue
            if student.learn_computer_skill(Skill(
                    name=skill.name, program=skill.program, creator=skill.creator,
                    generation=skill.generation,
                    preconditions=list(skill.preconditions),
                    examples=list(skill.examples))):
                skill.adoption += 1
                teacher.taught.append(skill.name)
                teacher.reputation += 0.5
                teacher.befriend(student.id, 0.2)
                student.befriend(teacher.id, 0.1)
                self.culture.record_propagation(skill.program, teacher.id,
                                                student.id, gen)
                transfers += 1
        return transfers

    # -------------------------------------------------------- generation
    def _offered_levels(self):
        """Task mix: mostly the frontier, some review, some stretch (frontier+1)."""
        f = self.frontier
        levels = [f] * 5
        if f > 1:
            levels += [max(1, f - 1)] * 2  # review
        if f < MAX_LEVEL:
            levels += [f + 1] * 2          # stretch toward the next rung
        return levels

    def run_generation(self, gen: int) -> dict:
        levels = self._offered_levels()
        solved_at = {l: 0 for l in range(1, MAX_LEVEL + 1)}
        attempt_at = {l: 0 for l in range(1, MAX_LEVEL + 1)}
        contributions = 0

        for agent in self.population:
            for _ in range(self.cfg.tasks_per_agent):
                lvl = int(self.rng.choice(levels))
                task = self.world.sample(lvl)
                res, disc = agent.solve_computer_task(
                    task, budget=self.cfg.budget, generation=gen)
                reward = res.score
                agent.record_attempt(reward, res.solved)
                attempt_at[lvl] += 1
                solved_at[lvl] += int(res.solved)
                if disc is not None and self.cfg.use_culture:
                    canonical = self.culture.contribute(disc)
                    agent.contributions.append(canonical.name)
                    agent.reputation += 0.3 * canonical.complexity()
                    contributions += 1
                if self.db is not None:
                    self.db.log_reward(self.cfg.name, gen, agent.id, task.name,
                                       lvl, reward, res.solved)

        transfers = self._teach_round(gen) if self.cfg.use_teaching else 0
        if self.cfg.use_culture:
            self.culture.decay(0.02)

        # current-frontier solve rate
        f = self.frontier
        frontier_rate = (solved_at[f] / attempt_at[f]) if attempt_at[f] else 0.0

        # "mastered level": highest level the population solves >= threshold
        mastered = 0
        for l in range(1, MAX_LEVEL + 1):
            if attempt_at[l] and solved_at[l] / attempt_at[l] >= self.cfg.advance_threshold:
                mastered = l

        # Advance the curriculum frontier (monotone, never regresses) when the
        # population masters the current frontier OR is already mastering a higher
        # (stretch) level. The frontier is the highest difficulty the civilization
        # has unlocked, so it only ever ratchets upward.
        if self.frontier < MAX_LEVEL:
            if frontier_rate >= self.cfg.advance_threshold:
                self.frontier = min(MAX_LEVEL, self.frontier + 1)
            if mastered >= self.frontier:
                self.frontier = min(MAX_LEVEL, mastered + 1)
        rate_by_level = {l: (solved_at[l] / attempt_at[l]) if attempt_at[l] else None
                         for l in range(1, MAX_LEVEL + 1)}

        stats = {
            "generation": gen,
            "frontier": f,
            "frontier_rate": frontier_rate,
            "mastered_level": mastered,
            "max_level": MAX_LEVEL,
            "culture_size": self.culture.size(),
            "avg_macros": float(np.mean([len(a.computer_skills) for a in self.population])),
            "avg_solved": float(np.mean([a.tasks_solved / max(1, a.tasks_attempted)
                                         for a in self.population])),
            "avg_fitness": float(np.mean([a.fitness for a in self.population])),
            "best_fitness": float(np.max([a.fitness for a in self.population])),
            "teaching_transfers": transfers,
            "contributions": contributions,
            "solve_rate_by_level": rate_by_level,
            "avg_difficulty_solved": float(np.mean(
                [l for l in range(1, MAX_LEVEL + 1) for _ in range(solved_at[l])]) or 0.0)
                if any(solved_at.values()) else 0.0,
            "n_propagation_events": len(self.culture.propagation_log),
        }
        self.history.append(stats)
        self._persist(gen, stats)
        return stats

    def _persist(self, gen, stats):
        if self.db is None:
            return
        self.db.log_generation(self.cfg.name, gen, stats)
        for a in self.population:
            self.db.log_agent(self.cfg.name, a.to_row())
        for sk in self.culture.skills.values():
            self.db.log_skill(self.cfg.name, gen, sk.to_row())
        if self.cfg.use_teaching:
            for prog_name, frm, to, g in self.culture.propagation_log:
                if g == gen:
                    self.db.log_propagation(self.cfg.name, gen, prog_name, frm, to)
        self.db.commit()

    def run(self):
        if self.db is not None:
            self.db.log_experiment(self.cfg.name, self.cfg.as_dict())
        self._spawn_initial()
        for gen in range(self.cfg.generations):
            self.run_generation(gen)
            if gen < self.cfg.generations - 1:
                self.population = self._reproduce(gen + 1)
        return self.history


def run_computer_experiments(db=None, generations=30, population_size=24,
                             budget=150, tasks_per_agent=10, seed=0):
    """Full Computer-Civilization vs a no-sharing control."""
    base = dict(generations=generations, population_size=population_size,
                budget=budget, tasks_per_agent=tasks_per_agent, seed=seed)
    full_cfg = ComputerConfig(name="E_computer_civilization", **base)
    ctrl_cfg = ComputerConfig(name="E_computer_nosharing", use_culture=False,
                              use_inheritance=False, use_teaching=False,
                              use_reputation=False, **base)
    full = ComputerCivilization(full_cfg, db=db)
    ctrl = ComputerCivilization(ctrl_cfg, db=db)
    full.run()
    ctrl.run()
    return {"full": full, "control": ctrl}
