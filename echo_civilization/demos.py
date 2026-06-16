"""Standalone demonstrations of the individual subsystems, used to populate the
research report with concrete evidence that each mechanism works:

  - Echo World  : tabular Q-learning genuinely learns to copy (individual learning)
  - Memory World: retention vs. forgetting, and fact transfer between agents
  - Grid World  : a neural-net policy is improved by evolution across generations
  - Social World: a shared communication protocol emerges from meaningless symbols
"""

from __future__ import annotations

import numpy as np

from .agent import Agent
from .environments import GridWorld, MemoryWorld, SocialWorld
from .environments.grid_world import N_ACTIONS, OBS_DIM
from .learning import NeuralPolicy, QLearner
from .skills import ALPHABET


def echo_qlearning_demo(seed: int = 0, episodes: int = 80):
    """A single agent learns the identity (copy) character map from reward.

    The plotted curve is the *greedy* (exploitation) accuracy after each training
    episode, so it reflects what the agent has actually learned rather than its
    exploration noise."""
    rng = np.random.default_rng(seed)
    ql = QLearner(n_actions=len(ALPHABET), rng=rng, epsilon=0.5,
                  epsilon_decay=0.93, alpha=0.5)
    curve = []
    target_map = {c: c for c in ALPHABET}  # echo == identity
    for _ in range(episodes):
        # one training pass (with exploration)
        for ch in ALPHABET:
            s = ALPHABET.index(ch)
            a = ql.act(s)
            r = 1.0 if ALPHABET[a] == target_map[ch] else -0.1
            ql.observe(s, a, r, s, True)
        ql.decay_epsilon()
        # measure greedy accuracy
        greedy = ql.greedy_map()
        correct = sum(1 for ch in ALPHABET
                      if ALPHABET[greedy.get(ALPHABET.index(ch), -1)] == target_map[ch])
        curve.append(correct / len(ALPHABET))
    return {"curve": curve, "final_accuracy": curve[-1],
            "episodes_to_mastery": next((i for i, v in enumerate(curve) if v >= 1.0), None)}


def memory_demo(seed: int = 0, delays=(0, 5, 10, 20, 40, 60), trials: int = 60):
    """Retention as a function of delay (forgetting curve), plus a transfer test."""
    rng = np.random.default_rng(seed)
    retention = {}
    for delay in delays:
        world = MemoryWorld(rng, delay=delay)
        scores = []
        for _ in range(trials):
            agent = Agent(0, rng)
            agent.long_term.decay = 0.04  # forgetting rate for the curve
            res = world.run_episode(agent)
            scores.append(res["salience"])  # graded retention strength
        retention[delay] = float(np.mean(scores))

    # knowledge transfer: a knower teaches a naive agent a fact
    world = MemoryWorld(rng, delay=2)
    teacher = Agent(0, rng)
    teacher.long_term.remember_fact("treasure", "blue door", 1.0)
    student = Agent(0, rng)
    before = student.long_term.recall_fact("treasure")
    world.transfer_fact(teacher, student, "treasure")
    after = student.long_term.recall_fact("treasure")
    transfer_ok = (before is None) and (after == "blue door")
    return {"retention_by_delay": retention, "transfer_ok": transfer_ok}


def grid_evolution_demo(seed: int = 0, generations: int = 30, pop: int = 40,
                        elite_frac: float = 0.25, lives: int = 5):
    """Evolve a population of MLP policies on the grid world (ES, no backprop).

    Uses elitism (the best policies survive unmutated) plus mutated offspring, and
    averages several lives per evaluation to cut the noise of random maps."""
    rng = np.random.default_rng(seed)
    world = GridWorld(rng, n_hazards=4, max_steps=70)
    layers = [OBS_DIM, 12, N_ACTIONS]
    pop_nets = [NeuralPolicy(layers, rng) for _ in range(pop)]

    def evaluate(net):
        return float(np.mean([world.run_episode(net)["reward"] for _ in range(lives)]))

    curve = []
    n_elite = max(2, int(elite_frac * pop))
    for g in range(generations):
        scores = np.array([evaluate(net) for net in pop_nets])
        order = np.argsort(scores)[::-1]
        curve.append({"avg": float(scores.mean()), "best": float(scores.max())})
        elites = [pop_nets[i] for i in order[:n_elite]]
        children = [e.clone() for e in elites]  # elitism: carry survivors over
        while len(children) < pop:
            parent = elites[int(rng.integers(0, len(elites)))]
            child = parent.clone()
            child.mutate(rate=0.15, scale=0.12)
            children.append(child)
        pop_nets = children
    return {"curve": curve, "final_best": curve[-1]["best"],
            "initial_best": curve[0]["best"]}


def computer_trace_demo(seed: int = 3):
    """Show an agent that has inherited basic macros solving a real multi-step
    computer task, returning a human-readable trace for the report."""
    from .environments.computer_world import (CURRICULUM, ComputerWorld,
                                               run_computer_program)
    from .skills import Skill

    rng = np.random.default_rng(seed)
    world = ComputerWorld(rng)
    agent = Agent(0, rng)
    # give it the lower-level macros a mature civilization would have inherited
    for lvl in range(1, 4):
        for name, prog in CURRICULUM[lvl]:
            agent.learn_computer_skill(Skill(name=name, program=prog,
                                             creator="culture", generation=0))
    task = world.sample(4)  # a level-4 pipeline task
    res, disc = agent.solve_computer_task(task, budget=200)
    final = run_computer_program(res.program, task.machine, task.ctx)
    return {
        "task_name": task.name,
        "level": task.level,
        "keyword": task.ctx.keyword,
        "input_file": task.ctx.input_file,
        "files": {k: v for k, v in list(task.machine.files.items())[:3]},
        "discovered_program": " -> ".join(res.program),
        "canonical_program": " -> ".join(task.canonical),
        "solved": res.solved,
        "via_composition": res.via_composition,
        "output": final.files.get(task.ctx.output_file, ""),
        "expected": task.expected_output,
    }


def social_demo(seed: int = 0, n_agents: int = 6, rounds: int = 200,
                n_concepts: int = 3):
    """Run the Lewis signalling game; a protocol should emerge from scratch."""
    rng = np.random.default_rng(seed)
    world = SocialWorld(rng, n_concepts=n_concepts, n_symbols=n_concepts)
    agents = [Agent(0, rng) for _ in range(n_agents)]
    return world.run(agents, rounds=rounds)


def real_os_demo(seed: int = 5, fresh_budget: int = 30):
    """Genuine sandboxed-shell demonstration (Environment 6 / Experiment F).

    For each curriculum level we compare two agents on REAL tasks executed by bash
    in a temp sandbox:
      - a 'cultured' agent that inherited the lower-level macros, and
      - a 'fresh' agent with an empty library and a bounded real-execution budget.
    We measure whether each solves the task and how many real shell commands it
    had to run. Culture should make real computer use cheap; from scratch it is
    prohibitively expensive. Returns per-level data + one real command trace.
    """
    from .environments.computer_world import CURRICULUM, MAX_LEVEL
    from .environments.real_computer_world import ALLOWED_OPS, RealComputerWorld
    from .synthesis import synthesize

    rng = np.random.default_rng(seed)
    world = RealComputerWorld(rng)
    primitives = sorted(ALLOWED_OPS)
    rows = []
    trace = None

    def solve(known_programs, task, budget):
        calls = {"n": 0}
        def evaluate(program):
            ok, score, out, c, log = world.grade(program, task)
            calls["n"] += c
            return ok, score
        res = synthesize(known_programs, primitives, evaluate, budget, rng,
                         max_depth=3, discovery_sample=60)
        return res, calls["n"]

    for lvl in range(1, MAX_LEVEL + 1):
        task = world.make_task(lvl)
        # cultured agent: inherited macros up to lvl-1 (so it must compose/recall)
        known = []
        for L in range(1, lvl + 1):
            for nm, prog in CURRICULUM[L]:
                known.append(prog)
        cres, ccalls = solve(known, task, budget=60)
        fres, fcalls = solve([], task, budget=fresh_budget)
        if lvl == MAX_LEVEL and trace is None:
            ok, score, out, c, log = world.grade(cres.program, task)
            trace = {"level": lvl, "name": task.name, "keyword": task.keyword,
                     "input_file": task.input_file, "commands": log,
                     "output": out, "expected": task.expected_output}
        rows.append({
            "level": lvl, "name": task.name,
            "cultured_solved": cres.solved, "cultured_shell_calls": ccalls,
            "fresh_solved": fres.solved, "fresh_shell_calls": fcalls,
        })
        task.cleanup()
    return {"rows": rows, "trace": trace, "fresh_budget": fresh_budget}
