"""Compositional-generalization experiment (the test that can fail).

The headline A/B/C/D result cannot distinguish *memorization* from
*generalization*: training (tier="all") and the eval suite are drawn from the
SAME composite programs — only the input strings differ. So "capability" might
just be culture caching the exact compositions it trained on.

This module builds the version that can fail:

  * TRAIN on all primitive tasks + a SUBSET of depth-2 composites.
  * TEST on a DISJOINT, never-trained set of composites built from the same
    primitives, stratified by depth (depth-2 and depth-3).

Why depth-3 is the real test. Recombination in `Agent.solve_task` is *pairwise*
only (`product(known, known)`), so a depth-3 target P=(x,y,z) is reachable at
eval ONLY if the agent holds a depth-2 building block, e.g. (x,y), and the
primitive (z). The held-out depth-3 tasks are constructed so their depth-2
sub-program IS in the training set. Solving them therefore measures whether the
civilization accumulated and redeployed *intermediate abstractions* (the
DreamCoder question), not merely primitives.

Predicted outcomes:
  * C/D win at depth-2 AND depth-3  -> real compositional generalization.
  * win at depth-2 only             -> culture spreads primitives, not abstractions.
  * C/D approx A/B on novel tasks   -> the headline was mostly memorization.

Methodology guards (see `audit_split`):
  * eval is frozen: allow_discovery=False AND learn_at_solve=False (no test-time
    learning), with a generous budget so pairwise search is never the bottleneck;
  * every held-out task is oracle-checked solvable-in-principle before use, so a
    null cannot be blamed on impossible tasks;
  * the final culture is dumped and checked to confirm no held-out program leaked
    in during training;
  * identical hyperparameters to the headline; multi-seed; nothing tuned to make
    culture win.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .agent import Agent
from .environments.base import StringEnvironment
from .evaluation import EXPERIMENTS
from .evolution import Civilization, CivConfig
from .skills import PRIMITIVES, Skill, program_name, run_program

# All primitive ops that exist.
ALL_OPS = list(PRIMITIVES)  # copy reverse inc1 inc2 dec1 count first last double dedup

# Primitives that are presented as STANDALONE depth-1 training tasks (so they
# become individually-known skills). `double` and `dedup` are deliberately NOT
# here — per the brief they are only ever introduced *inside* depth-2 composites,
# which is the only way they enter the culture.
PRIMITIVE_OP_TASKS = ["copy", "reverse", "inc1", "inc2", "dec1", "count",
                      "first", "last"]

# A generous, fixed eval budget so pairwise recombination over even a large
# inherited library is never the limiting factor. This is a *measurement* budget
# (eval is run rarely), applied identically to every condition; a larger budget
# can only help the weaker (no-culture) conditions, so it is the conservative,
# non-culture-favouring choice.
GEN_EVAL_BUDGET = 4000


# --------------------------------------------------------------------- battery
def build_battery(seed: int = 12345):
    """A fixed, varied set of probe strings used to compute behavioural
    signatures (so we can tell programs apart and detect degeneracy)."""
    rng = np.random.default_rng(seed)
    alpha = list("abcdefgh")
    strings = []
    for _ in range(40):
        n = int(rng.integers(2, 8))
        strings.append("".join(rng.choice(alpha, size=n)))
    # add strings with deliberate repeats (exercise dedup) and edge cases
    strings += ["aab", "aabb", "abba", "aaa", "abcabc", "hh", " a"[1:], "abcd",
                "aabbcc", "ababab", "deed", "gggg"]
    # dedup duplicates in the battery itself
    seen, out = set(), []
    for s in strings:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def signature(program, battery):
    return tuple(run_program(program, s) for s in battery)


# --------------------------------------------------------------- program universe
def build_universe(battery):
    """Return (depth1_sigs, depth2_progs, depth2_sigs_by_prog, depth_leq2_sigs).

    depth2_progs: non-degenerate, signature-deduped depth-2 programs (a program is
    non-degenerate iff its behaviour differs from every depth<=1 program)."""
    depth1_sigs = {signature((op,), battery) for op in ALL_OPS}

    depth2_progs = []
    seen_sigs = set()
    for a, b in itertools.product(ALL_OPS, repeat=2):
        prog = (a, b)
        sig = signature(prog, battery)
        if sig in depth1_sigs:      # collapses to a primitive -> degenerate
            continue
        if sig in seen_sigs:        # behaviourally identical to a kept depth-2
            continue
        seen_sigs.add(sig)
        depth2_progs.append(prog)

    depth_leq2_sigs = set(depth1_sigs) | seen_sigs
    return depth1_sigs, depth2_progs, depth_leq2_sigs


# ------------------------------------------------------------------------ split
@dataclass
class Split:
    battery: list
    primitives: list          # depth-1 programs presented as training tasks
    train2: list              # depth-2 programs in the training set
    held2: list               # depth-2 programs held out (disjoint from train2)
    held3: list               # depth-3 programs held out (decompose via train2)
    notes: dict = field(default_factory=dict)


def make_split(split_seed: int = 7, train_frac: float = 0.5,
               max_held3: int = 24):
    """Construct a fixed train/held-out split.

    depth-2 non-degenerate composites are split (seeded) into train2 / held2.
    depth-3 held-outs are built as (train2 prefix)+(primitive) or
    (primitive)+(train2 prefix), kept only if non-degenerate at depth 3 and
    behaviourally novel vs. everything trained."""
    battery = build_battery()
    depth1_sigs, depth2_progs, depth_leq2_sigs = build_universe(battery)

    rng = np.random.default_rng(split_seed)
    order = list(range(len(depth2_progs)))
    rng.shuffle(order)
    n_train = int(train_frac * len(depth2_progs))
    train2 = [depth2_progs[i] for i in order[:n_train]]
    held2 = [depth2_progs[i] for i in order[n_train:]]

    trained_sigs = set(depth1_sigs)
    trained_sigs |= {signature(p, battery) for p in train2}
    held2_sigs = {signature(p, battery) for p in held2}

    # build depth-3 held-outs whose depth-2 building block is in train2
    held3, held3_sigs = [], set()
    candidates = []
    for a in train2:
        for r in PRIMITIVE_OP_TASKS:
            candidates.append(a + (r,))        # suffix route: (x,y)+(z)
            candidates.append((r,) + a)        # prefix route: (z)+(x,y)
    rng.shuffle(candidates)
    for prog in candidates:
        if len(held3) >= max_held3:
            break
        sig = signature(prog, battery)
        if sig in depth_leq2_sigs:             # secretly depth<=2 -> reject
            continue
        if sig in trained_sigs or sig in held2_sigs:  # not novel -> reject
            continue
        if sig in held3_sigs:                  # dup -> reject
            continue
        held3_sigs.add(sig)
        held3.append(prog)

    primitives = [(op,) for op in PRIMITIVE_OP_TASKS]
    notes = {
        "n_depth2_universe": len(depth2_progs),
        "n_train2": len(train2),
        "n_held2": len(held2),
        "n_held3": len(held3),
        "split_seed": split_seed,
        "train_frac": train_frac,
    }
    return Split(battery=battery, primitives=primitives, train2=train2,
                 held2=held2, held3=held3, notes=notes)


# ------------------------------------------------------------------ training world
class SplitWorld(StringEnvironment):
    """Trains on primitive tasks + the TRAIN-2 composite subset only."""
    name = "generalization_train"

    def __init__(self, rng, split: Split):
        super().__init__(rng)
        self.pool = [(f"prim::{p[0]}", p) for p in split.primitives] + \
                    [(f"train2::{program_name(p)}", p) for p in split.train2]

    def sample_task(self):
        name, prog = self.pool[int(self.rng.integers(0, len(self.pool)))]
        return self.make_task(prog, name)


# ---------------------------------------------------------------- eval task suites
def make_eval_tasks(programs, eval_seed=999, n_each=5, n_examples=6):
    """Build held-out evaluation tasks (more examples than training to make a
    spuriously-consistent wrong program vanishingly unlikely)."""
    rng = np.random.default_rng(eval_seed)
    world = StringEnvironment(rng)
    tasks = []
    for prog in programs:
        for _ in range(n_each):
            tasks.append(world.make_task(prog, program_name(prog),
                                         n_examples=n_examples))
    return tasks


def _oracle_agent(split: Split, rng):
    """An agent that KNOWS all primitives + all train-2 composites — used to
    confirm held-out tasks are solvable-in-principle by recombination."""
    ag = Agent(0, rng)
    for p in split.primitives:
        ag.learn_skill(Skill(name=program_name(p), program=p, creator="oracle",
                             generation=0))
    for p in split.train2:
        ag.learn_skill(Skill(name=program_name(p), program=p, creator="oracle",
                             generation=0))
    return ag


def frozen_solve_rate(agents, tasks, budget=GEN_EVAL_BUDGET):
    """Average per-agent fraction of `tasks` solved with FROZEN knowledge:
    no discovery, no test-time learning. The held-out query (not just the
    examples) decides correctness, so spurious example-consistency cannot inflate
    the score."""
    if not agents or not tasks:
        return 0.0
    rates = []
    for ag in agents:
        solved = 0
        for t in tasks:
            pred, prog, ev, disc = ag.solve_task(
                t.examples, t.query_input, budget=budget, generation=0,
                allow_discovery=False, learn_at_solve=False)
            solved += int(t.is_solved(pred))
        rates.append(solved / len(tasks))
    return float(np.mean(rates))


# ------------------------------------------------------------------- audit
def audit_split(split: Split, held2_tasks, held3_tasks):
    """Oracle-check that held-out tasks are solvable-in-principle, and filter out
    any that aren't (so a null result can't be blamed on impossible tasks)."""
    rng = np.random.default_rng(0)
    oracle = _oracle_agent(split, rng)

    def keep_solvable(tasks):
        good = []
        for t in tasks:
            pred, prog, ev, disc = oracle.solve_task(
                t.examples, t.query_input, budget=GEN_EVAL_BUDGET, generation=0,
                allow_discovery=False, learn_at_solve=False)
            if t.is_solved(pred):
                good.append(t)
        return good

    h2 = keep_solvable(held2_tasks)
    h3 = keep_solvable(held3_tasks)
    return {
        "held2_total": len(held2_tasks), "held2_solvable": len(h2),
        "held3_total": len(held3_tasks), "held3_solvable": len(h3),
        "oracle_held2_rate": frozen_solve_rate([oracle], held2_tasks),
        "oracle_held3_rate": frozen_solve_rate([oracle], held3_tasks),
    }, h2, h3


def culture_leak_check(culture, split: Split):
    """Confirm no held-out FUNCTION (by behavioural signature) was stored in the
    final culture during training, stratified by held-out depth.

    The depth-3 stratum is the one that matters: a stored depth-3 equivalent would
    let an agent solve a held-out depth-3 by plain recall instead of recombining an
    intermediate abstraction. (Depth-2 'leaks' are typically spurious commutative
    twins of junk skills the 3-example training occasionally stores — they only
    touch the easy stratum and are reported for transparency.)"""
    cult_sigs = {signature(prog, split.battery) for prog in culture.skills}
    held2_sigs = {signature(p, split.battery): program_name(p) for p in split.held2}
    held3_sigs = {signature(p, split.battery): program_name(p) for p in split.held3}
    return {
        "held2": sorted(nm for sig, nm in held2_sigs.items() if sig in cult_sigs),
        "held3": sorted(nm for sig, nm in held3_sigs.items() if sig in cult_sigs),
    }


# ------------------------------------------------------------------- experiment
# Headline hyperparameters (identical to run_all_experiments): nothing here is
# tuned to make culture win.
HEADLINE = dict(generations=30, population_size=24, budget=35, tasks_per_agent=8)


def run_condition(name: str, split: Split, suites: dict, seed: int,
                  curve_suite: str | None = None, curve_every: int = 5):
    """Train one condition on the SPLIT world, then frozen-eval on each suite.

    Returns final per-suite solve rates, an optional per-generation curve for one
    suite, the culture-leak result and culture size."""
    cfg = CivConfig(name=name, **{**HEADLINE, **EXPERIMENTS[name], "seed": seed})
    world = SplitWorld(np.random.default_rng(1000 + seed), split)
    civ = Civilization(cfg, world, db=None)
    civ._spawn_initial()

    curve = []
    for gen in range(cfg.generations):
        civ.run_generation(gen)
        if curve_suite is not None and (gen % curve_every == 0
                                        or gen == cfg.generations - 1):
            curve.append((gen, frozen_solve_rate(civ.population,
                                                 suites[curve_suite])))
        if gen < cfg.generations - 1:
            civ.population = civ._reproduce(gen + 1)

    final = {s: frozen_solve_rate(civ.population, tasks)
             for s, tasks in suites.items()}
    leaks = culture_leak_check(civ.culture, split)
    return {
        "name": name, "seed": seed, "final": final, "curve": curve,
        "culture_size": civ.culture.size(), "leaks": leaks,
        "avg_known": float(np.mean([len(a.known_skills) for a in civ.population])),
    }


def run_generalization(seeds=(0, 1, 2), split_seed: int = 7,
                       curve_suite: str = "held3"):
    """Full multi-seed compositional-generalization experiment."""
    split = make_split(split_seed=split_seed)

    # eval suites (held-out programs + in-distribution controls)
    held2_tasks = make_eval_tasks(split.held2, eval_seed=2002)
    held3_tasks = make_eval_tasks(split.held3, eval_seed=3003)
    audit, held2_tasks, held3_tasks = audit_split(split, held2_tasks, held3_tasks)
    suites = {
        # in-distribution control: trained programs, BRAND-NEW input strings.
        # (This is what the original headline measured — memorization is enough.)
        "train2_newinputs": make_eval_tasks(split.train2, eval_seed=4004),
        "held2": held2_tasks,     # novel depth-2 (needs only primitives)
        "held3": held3_tasks,     # novel depth-3 (needs an intermediate abstraction)
    }

    per_condition = {n: [] for n in EXPERIMENTS}
    curves = {}
    for seed in seeds:
        for name in EXPERIMENTS:
            res = run_condition(name, split, suites, seed,
                                curve_suite=(curve_suite if seed == seeds[0] else None))
            per_condition[name].append(res)
            if seed == seeds[0]:
                curves[name] = res["curve"]

    # aggregate mean/std across seeds
    summary = {}
    for name, runs in per_condition.items():
        agg = {}
        for suite in suites:
            vals = [r["final"][suite] for r in runs]
            agg[suite] = (float(np.mean(vals)), float(np.std(vals)))
        agg["culture_size"] = float(np.mean([r["culture_size"] for r in runs]))
        agg["avg_known"] = float(np.mean([r["avg_known"] for r in runs]))
        agg["leaks_held2"] = sorted({l for r in runs for l in r["leaks"]["held2"]})
        agg["leaks_held3"] = sorted({l for r in runs for l in r["leaks"]["held3"]})
        summary[name] = agg

    return {
        "split": split, "audit": audit, "suites": {k: len(v) for k, v in suites.items()},
        "summary": summary, "curves": curves, "seeds": list(seeds),
        "per_condition": per_condition,
    }
