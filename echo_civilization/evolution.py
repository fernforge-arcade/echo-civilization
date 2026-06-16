"""Generations & evolution — the civilization loop.

A `Civilization` runs a population through generations on the Transformation
World (the main capability-accumulation testbed). Feature flags let us turn the
civilization machinery on and off to run the baseline experiments A/B/C/D:

    use_culture     skills enter a shared repository
    use_inheritance children inherit parent + cultural skills (vertical transfer)
    use_teaching    peers share skills within a generation (horizontal transfer)
    use_reputation  selection & skill-priority weighted by reputation

Each generation:
  1. build population (gen 0 from scratch; later gens via selection+mutation,
     inheriting skills/culture if enabled),
  2. agents attempt tasks under a fixed per-task evaluation BUDGET,
  3. discovered skills are abstracted; if culture is on they are contributed,
  4. optional peer teaching,
  5. fitness measured, stats recorded,
  6. cultural reputation decays (failed skills fade).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .agent import Agent
from .culture import CulturalMemory
from .skills import Skill
from .teaching import share_round


@dataclass
class CivConfig:
    name: str = "D_full"
    population_size: int = 24
    generations: int = 30
    tasks_per_agent: int = 12
    budget: int = 45                 # per-task evaluation budget (the key knob)
    use_culture: bool = True
    use_inheritance: bool = True
    use_teaching: bool = True
    use_reputation: bool = True
    elite_fraction: float = 0.3
    inherit_skill_fraction: float = 0.85
    cultural_seed_top: int = 16      # how many top cultural skills new agents see
    mutation_rate: float = 0.1       # chance to drop/forget an inherited skill
    seed: int = 0

    def as_dict(self):
        return self.__dict__.copy()


class Civilization:
    def __init__(self, config: CivConfig, world, db=None):
        self.cfg = config
        self.world = world
        self.db = db
        self.rng = np.random.default_rng(config.seed)
        self.culture = CulturalMemory()
        self.population: list[Agent] = []
        self.history: list[dict] = []

    # ---------------------------------------------------------- population
    def _spawn_initial(self):
        self.population = [Agent(0, self.rng) for _ in range(self.cfg.population_size)]

    def _seed_from_culture(self, agent: Agent):
        if not self.cfg.use_culture:
            return
        for skill in self.culture.top_skills(self.cfg.cultural_seed_top):
            # adopt according to imitation preference (and skill standing)
            if self.rng.random() < agent.preferences["imitation"]:
                if agent.learn_skill(Skill(
                        name=skill.name, program=skill.program,
                        creator=skill.creator, generation=skill.generation,
                        preconditions=list(skill.preconditions),
                        examples=list(skill.examples))):
                    skill.adoption += 1

    def _reproduce(self, gen: int) -> list[Agent]:
        """Selection + mutation + inheritance to build the next generation."""
        ranked = sorted(self.population, key=lambda a: a.fitness, reverse=True)
        n_elite = max(1, int(self.cfg.elite_fraction * len(ranked)))
        elites = ranked[:n_elite]
        children = []
        for _ in range(self.cfg.population_size):
            if self.cfg.use_reputation:
                # fitness-weighted parent selection (high performers reproduce more)
                weights = np.array([max(0.01, a.fitness) for a in elites])
                p = weights / weights.sum()
                parent = elites[int(self.rng.choice(len(elites), p=p))]
            else:
                parent = elites[int(self.rng.integers(0, len(elites)))]

            child = Agent(gen, self.rng, parents=[parent.id])
            # mutate inherited preferences
            for k in child.preferences:
                child.preferences[k] = float(np.clip(
                    parent.preferences[k] + self.rng.normal(0, 0.05), 0.0, 1.0))

            # vertical inheritance of skills
            if self.cfg.use_inheritance:
                for prog, skill in parent.known_skills.items():
                    if self.rng.random() < self.cfg.inherit_skill_fraction \
                            and self.rng.random() > self.cfg.mutation_rate:
                        child.learn_skill(Skill(
                            name=skill.name, program=skill.program,
                            creator=skill.creator, generation=skill.generation,
                            preconditions=list(skill.preconditions),
                            examples=list(skill.examples)))
            # cultural inheritance (shared repository)
            self._seed_from_culture(child)
            children.append(child)
        return children

    # ---------------------------------------------------------- one generation
    def run_generation(self, gen: int) -> dict:
        difficulties_solved = []
        per_diff_attempts = {}
        per_diff_solved = {}
        contributions_this_gen = 0

        for agent in self.population:
            for _ in range(self.cfg.tasks_per_agent):
                task = self.world.sample_task()
                pred, prog, evals, disc = agent.solve_task(
                    task.examples, task.query_input,
                    budget=self.cfg.budget, generation=gen)
                from .environments.base import grade
                reward = grade(pred, task.query_target)
                solved = task.is_solved(pred)
                agent.record_attempt(reward, solved)

                d = task.difficulty
                per_diff_attempts[d] = per_diff_attempts.get(d, 0) + 1
                per_diff_solved[d] = per_diff_solved.get(d, 0) + int(solved)
                if solved:
                    difficulties_solved.append(d)

                # discovered a new skill -> contribute to culture
                if disc is not None and self.cfg.use_culture:
                    canonical = self.culture.contribute(disc)
                    agent.contributions.append(canonical.name)
                    agent.reputation += 0.3 * canonical.complexity()
                    contributions_this_gen += 1

                if self.db is not None:
                    self.db.log_reward(self.cfg.name, gen, agent.id, task.name,
                                       d, reward, solved)

        # horizontal sharing (teaching)
        transfers = 0
        if self.cfg.use_teaching:
            transfers = share_round(self.population, self.culture, gen, self.rng)
            # log propagation events accumulated in culture this generation
            if self.db is not None:
                for prog_name, frm, to, g in self.culture.propagation_log:
                    if g == gen:
                        self.db.log_propagation(self.cfg.name, gen, prog_name, frm, to)

        # cultural reputation decay (failed/unused skills fade)
        if self.cfg.use_culture:
            self.culture.decay(0.03)

        stats = self._collect_stats(gen, per_diff_attempts, per_diff_solved,
                                     difficulties_solved, transfers,
                                     contributions_this_gen)
        self.history.append(stats)
        self._persist(gen, stats)
        return stats

    def _collect_stats(self, gen, attempts, solved, diffs, transfers, contribs):
        fitnesses = [a.fitness for a in self.population]
        solve_rates = [a.tasks_solved / max(1, a.tasks_attempted)
                       for a in self.population]
        per_diff_rate = {d: solved.get(d, 0) / attempts[d] for d in attempts}
        max_diff_solved = max(diffs) if diffs else 0
        return {
            "generation": gen,
            "avg_fitness": float(np.mean(fitnesses)),
            "best_fitness": float(np.max(fitnesses)),
            "avg_solved": float(np.mean(solve_rates)),
            "culture_size": self.culture.size(),
            "avg_skills": float(np.mean([len(a.known_skills) for a in self.population])),
            "avg_difficulty_solved": float(np.mean(diffs)) if diffs else 0.0,
            "max_difficulty_solved": max_diff_solved,
            "solve_rate_by_difficulty": per_diff_rate,
            "teaching_transfers": transfers,
            "contributions": contribs,
            "n_propagation_events": len(self.culture.propagation_log),
        }

    def _persist(self, gen, stats):
        if self.db is None:
            return
        self.db.log_generation(self.cfg.name, gen, stats)
        for a in self.population:
            self.db.log_agent(self.cfg.name, a.to_row())
        for sk in self.culture.skills.values():
            self.db.log_skill(self.cfg.name, gen, sk.to_row())
        self.db.commit()

    # ---------------------------------------------------------- full run
    def run(self) -> list[dict]:
        if self.db is not None:
            self.db.log_experiment(self.cfg.name, self.cfg.as_dict())
        self._spawn_initial()
        for gen in range(self.cfg.generations):
            self.run_generation(gen)
            if gen < self.cfg.generations - 1:
                self.population = self._reproduce(gen + 1)
        return self.history
