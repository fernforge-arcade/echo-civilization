"""Evaluation framework + the four baseline experiments (A/B/C/D).

We hold the world, seed, budget and per-agent task count fixed across all four
conditions and vary ONLY the civilization machinery, so any divergence in
capability is attributable to knowledge accumulation rather than luck or compute.

A held-out suite of hard (composite + deep) tasks is evaluated against each
generation's population to give a clean "capability over generations" curve.
"""

from __future__ import annotations

import numpy as np

from .environments import TransformationWorld
from .environments.base import grade
from .environments.transformation_world import COMPOSITE_TASKS, DEEP_TASKS
from .evolution import Civilization, CivConfig


EXPERIMENTS = {
    "A_single": dict(population_size=1, use_culture=False, use_inheritance=False,
                     use_teaching=False, use_reputation=False),
    "B_population_nosharing": dict(use_culture=False, use_inheritance=False,
                                   use_teaching=False, use_reputation=False),
    "C_population_memorysharing": dict(use_culture=True, use_inheritance=True,
                                       use_teaching=False, use_reputation=False),
    "D_full_civilization": dict(use_culture=True, use_inheritance=True,
                                use_teaching=True, use_reputation=True),
}

EXPERIMENT_LABELS = {
    "A_single": "A: single agent, no memory/culture",
    "B_population_nosharing": "B: population, no sharing",
    "C_population_memorysharing": "C: population + memory/skill sharing",
    "D_full_civilization": "D: full civilization",
}


def _make_eval_suite(seed: int = 999, n_each: int = 8):
    """A fixed held-out suite of hard tasks (never used for learning)."""
    rng = np.random.default_rng(seed)
    world = TransformationWorld(rng, tier="all")
    suite = []
    for name, prog in COMPOSITE_TASKS + DEEP_TASKS:
        for _ in range(n_each):
            suite.append(world.make_task(prog, name))
    return suite


def evaluate_population(population, suite, budget: int):
    """Average per-agent fraction of held-out hard tasks solvable using ONLY
    accumulated knowledge (recall + recombination of known skills, no fresh blind
    search). This isolates cultural accumulation from raw per-trial brute force,
    so larger populations get no unfair advantage."""
    if not population:
        return 0.0
    rates = []
    for agent in population:
        solved = 0
        for task in suite:
            pred, prog, evals, disc = agent.solve_task(
                task.examples, task.query_input, budget=budget,
                generation=0, allow_discovery=False)
            solved += int(task.is_solved(pred))
        rates.append(solved / len(suite))
    return float(np.mean(rates))


# Capability is graded with a generous, fixed budget (identical across all four
# conditions) so it measures *what agents know* — recall + recombination of their
# accumulated skills — rather than how fast they can search. A tight budget would
# unfairly penalise agents with larger inherited libraries (more pairs to check).
EVAL_BUDGET = 150


def run_experiment(name: str, base_cfg: dict, db=None, eval_suite=None):
    overrides = EXPERIMENTS[name]
    cfg = CivConfig(name=name, **{**base_cfg, **overrides})
    rng = np.random.default_rng(cfg.seed)
    world = TransformationWorld(rng, tier="all")
    civ = Civilization(cfg, world, db=db)

    if db is not None:
        db.log_experiment(cfg.name, cfg.as_dict())
    civ._spawn_initial()
    capability_curve = []
    for gen in range(cfg.generations):
        civ.run_generation(gen)
        if eval_suite is not None:
            cap = evaluate_population(civ.population, eval_suite, EVAL_BUDGET)
            capability_curve.append(cap)
        if gen < cfg.generations - 1:
            civ.population = civ._reproduce(gen + 1)

    return {
        "name": name,
        "label": EXPERIMENT_LABELS[name],
        "config": cfg.as_dict(),
        "history": civ.history,
        "capability_curve": capability_curve,
        "final_capability": capability_curve[-1] if capability_curve else 0.0,
        "culture": civ.culture,
        "population": civ.population,
        "civ": civ,
    }


def run_all_experiments(db=None, generations: int = 30, population_size: int = 24,
                        budget: int = 35, tasks_per_agent: int = 8, seed: int = 0):
    base_cfg = dict(generations=generations, population_size=population_size,
                    budget=budget, tasks_per_agent=tasks_per_agent, seed=seed)
    suite = _make_eval_suite()
    results = {}
    for name in EXPERIMENTS:
        cfg = dict(base_cfg)
        # experiment A is a single agent; keep its other knobs identical
        results[name] = run_experiment(name, cfg, db=db, eval_suite=suite)
    return results
