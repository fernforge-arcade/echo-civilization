"""Environment 7 / Experiment G — Autonomous Operation World ("run a business").

The highest level of abstraction in the project: instead of being handed one task,
a *firm* of agents pursues a long-lived, never-terminating objective — fulfilling a
continuous stream of customer **orders** — and is judged on sustained outcome
(cumulative profit, throughput, the sophistication of orders it can handle) rather
than single-task success.

This layer adds, on top of the civilization machinery already built, the three
ingredients that distinguish "running an operation" from "solving a task":

  1. **Hierarchical goal decomposition** — an order is a bundle of sub-tasks
     (reusing the Computer World jobs) at a spread of difficulty levels.
  2. **Delegation & emergent specialization** — a manager routes each sub-task to
     the best-suited specialist; agents accumulate macros in their specialty,
     reinforcing a division of labour.
  3. **An economy with continuous evaluation** — completed work earns revenue,
     wages are paid, the treasury compounds, weak performers are replaced by new
     hires that inherit the firm's shared knowledge base, and the firm's ambition
     (max order level) ratchets up as it succeeds — forever.

The ablation (firm WITH a shared knowledge base + inheritance vs WITHOUT) shows
that only the culture-bearing firm grows more capable and profitable over time;
the knowledge-less firm stalls and bleeds money. Same thesis, top of the ladder.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .agent import Agent
from .culture import CulturalMemory
from .environments.computer_world import CURRICULUM, MAX_LEVEL, ComputerWorld
from .skills import Skill

# revenue earned per fulfilled sub-task, by difficulty level (deeper = worth more)
TASK_VALUE = {1: 1.0, 2: 2.5, 3: 5.0, 4: 8.0, 5: 13.0}
WAGE = 0.45               # paid per agent per day (cost of doing business)


@dataclass
class EnterpriseConfig:
    name: str = "G_autonomous_firm"
    n_agents: int = 10
    days: int = 120
    tasks_per_order: int = 6
    budget: int = 150
    use_knowledge_base: bool = True   # shared culture + inheritance on hire
    use_delegation: bool = True       # route to specialists (vs random assignment)
    tenure: int = 7                   # workers retire after this many days (churn)
    ambition_threshold: float = 0.7   # fulfilment rate needed to raise max level
    seed: int = 0

    def as_dict(self):
        return self.__dict__.copy()


class AutonomousFirm:
    def __init__(self, config: EnterpriseConfig, db=None):
        self.cfg = config
        self.db = db
        self.rng = np.random.default_rng(config.seed)
        self.world = ComputerWorld(self.rng)
        self.kb = CulturalMemory()          # the firm's shared knowledge base
        self.treasury = 0.0
        self.max_order_level = 1            # ambition: hardest level the firm sells
        self.day = 0
        self.history: list[dict] = []
        # workers, each with an emergent specialty (a preferred difficulty band)
        self.agents = []
        for i in range(config.n_agents):
            a = Agent(0, self.rng)
            a.specialization = f"L{1 + (i % MAX_LEVEL)}"  # assigned specialty band
            # stagger initial ages so retirements spread out over time
            a._hire_day = -int(self.rng.integers(0, config.tenure))
            self.agents.append(a)
        self.specialty_jobs = {a.id: {} for a in self.agents}  # id -> {level: count}

    # ----------------------------------------------------- hiring / culture
    def _seed_new_hire(self, agent: Agent):
        if not self.cfg.use_knowledge_base:
            return
        for skill in self.kb.top_skills(MAX_LEVEL * 4):
            agent.learn_computer_skill(Skill(
                name=skill.name, program=skill.program, creator=skill.creator,
                generation=skill.generation, preconditions=list(skill.preconditions),
                examples=list(skill.examples)))

    def _hire_replacement(self, specialty):
        new = Agent(self.day, self.rng)
        new.specialization = specialty
        new._hire_day = self.day
        self._seed_new_hire(new)
        self.specialty_jobs[new.id] = {}
        return new

    # ----------------------------------------------------- delegation
    def _route(self, level: int, load: dict) -> Agent:
        """Route a sub-task to a specialist, balancing the day's workload.

        Prefer an agent whose specialty band covers this level (closest match),
        then least-loaded, then most reputable. Without delegation, assign at
        random — so no division of labour forms."""
        if not self.cfg.use_delegation:
            return self.agents[int(self.rng.integers(0, len(self.agents)))]

        def band(a):
            try:
                return int(a.specialization[1:])
            except (TypeError, ValueError):
                return 1

        def score(a):
            covers = 0 if band(a) >= level else 1      # specialists-first
            distance = abs(band(a) - level)
            return (covers, distance, load.get(a.id, 0), -a.reputation)

        return min(self.agents, key=score)

    # ----------------------------------------------------- one business day
    def run_day(self, day: int) -> dict:
        self.day = day
        order_levels = [int(self.rng.integers(1, self.max_order_level + 1))
                        for _ in range(self.cfg.tasks_per_order)]
        revenue = 0.0
        fulfilled = 0
        levels_done = []
        load: dict = {}
        for level in order_levels:
            worker = self._route(level, load)
            load[worker.id] = load.get(worker.id, 0) + 1
            task = self.world.sample(level)
            res, disc = worker.solve_computer_task(
                task, budget=self.cfg.budget, generation=day)
            if res.solved:
                revenue += TASK_VALUE.get(level, level)
                fulfilled += 1
                levels_done.append(level)
                worker.reputation += 0.2 * level
                worker.lifetime_reward += TASK_VALUE.get(level, level)
                self.specialty_jobs[worker.id][level] = \
                    self.specialty_jobs[worker.id].get(level, 0) + 1
                worker.record_attempt(1.0, True)
                # capture the WORKING procedure into the firm's knowledge base —
                # institutional memory that outlives the individual who found it.
                if self.cfg.use_knowledge_base and res.program:
                    canon = self.kb.contribute(Skill(
                        name=task.name, program=tuple(res.program),
                        creator=worker.id, generation=day))
                    canon.reputation += 0.1 * level
                    worker.contributions.append(canon.name)
            else:
                worker.record_attempt(res.score, False)
            if self.db is not None:
                self.db.log_reward(self.cfg.name, day, worker.id, task.name,
                                   level, res.score, res.solved)

        wages = WAGE * len(self.agents)
        profit = revenue - wages
        self.treasury += profit

        # ambition: raise the hardest order level the firm sells once it reliably
        # fulfils current orders (never-ending escalation of sophistication)
        fulfil_rate = fulfilled / len(order_levels)
        if fulfil_rate >= self.cfg.ambition_threshold and self.max_order_level < MAX_LEVEL:
            self.max_order_level += 1

        # workforce turnover: workers RETIRE after a bounded tenure (taking their
        # PRIVATE skill with them); replacements inherit only the firm's shared
        # knowledge base. Bounded tenure keeps the workforce young, so the firm's
        # capability is carried by institutional memory, not individual veterans —
        # without a KB, every retirement is a permanent loss of expertise.
        for i, a in enumerate(self.agents):
            if day - getattr(a, "_hire_day", 0) >= self.cfg.tenure:
                self.agents[i] = self._hire_replacement(a.specialization)

        n_specialists = len({a.specialization for a in self.agents
                             if a.specialization})
        stats = {
            "generation": day, "day": day,
            "revenue": revenue, "profit": profit,
            "cum_profit": self.treasury,
            "fulfil_rate": fulfil_rate,
            "max_order_level": self.max_order_level,
            "max_level_done": max(levels_done) if levels_done else 0,
            "kb_size": self.kb.size(),
            "avg_macros": float(np.mean([len(a.computer_skills) for a in self.agents])),
            "distinct_specialties": n_specialists,
            "avg_fitness": float(np.mean([a.fitness for a in self.agents])),
            "best_fitness": float(np.max([a.fitness for a in self.agents])),
        }
        self.history.append(stats)
        if self.db is not None:
            self.db.log_generation(self.cfg.name, day, stats)
            for a in self.agents:
                self.db.log_agent(self.cfg.name, a.to_row())
            for sk in self.kb.skills.values():
                self.db.log_skill(self.cfg.name, day, sk.to_row())
            self.db.commit()
        return stats

    def run(self):
        if self.db is not None:
            self.db.log_experiment(self.cfg.name, self.cfg.as_dict())
        for day in range(self.cfg.days):
            self.run_day(day)
        return self.history


def run_enterprise_experiments(db=None, days=120, n_agents=10, seed=0):
    """Autonomous firm WITH a shared knowledge base vs an otherwise-identical firm
    WITHOUT one (no inheritance on hire, random assignment)."""
    base = dict(days=days, n_agents=n_agents, seed=seed)
    full = AutonomousFirm(EnterpriseConfig(name="G_autonomous_firm", **base), db=db)
    ctrl = AutonomousFirm(EnterpriseConfig(
        name="G_firm_no_knowledge", use_knowledge_base=False,
        use_delegation=False, **base), db=db)
    full.run()
    ctrl.run()
    return {"full": full, "control": ctrl}
