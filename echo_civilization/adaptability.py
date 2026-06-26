"""Experiment H — Adaptability to a genuinely NOVEL task FAMILY.

The compositional-generalization study (generalization.py) already showed that
culture lets agents solve *novel compositions* of trained primitives. But every
held-out task there was still the same KIND of task the population trained on:
"apply a whole-string transform to a whole string." The split was at the level of
*which* composition, not *what kind of task*.

The operator's steer is harder: can the civilization handle tasks **never seen in
their entirety** — a task whose very STRUCTURE is new? This module builds that test.

THE NOVEL FAMILY: HIGHER-ORDER COMBINATORS
------------------------------------------
Training (the normal Transformation-World civilization) only ever applies a
program f to a whole string. The eval family introduces a layer the population has
NEVER trained on: a *combinator* C that decides HOW an inner transform f is applied
across a multi-token input. Examples:

    input:  "abc de fgh"      inner f = (reverse, inc1)   ["reverse then shift"]
    map_each(f):   "deb fe ihg"       (f on each token, order kept)
    map_reversed(f): "ihg fe deb"     (reshape: token order reversed, then f each)
    first_only(f): "deb de fgh"       (f on the first token only)
    last_only(f):  "abc de ihg"       (f on the last token only)
    map_evens(f):  "deb de ihg"       (f on even-indexed tokens only)

To solve a held-out task an agent must recover BOTH:
  * the combinator C (NOVEL to everyone — nobody trained on any combinator), and
  * the inner program f (a depth-2 composite).

Why this isolates ADAPTABILITY. Neither cultured nor fresh agents have ever seen a
combinator, so the combinator itself confers no inherited advantage — it must be
discovered at eval time by both. The ONLY thing a cultured agent brings is its
inherited library of inner abstractions f. So the experiment asks precisely:

    "Does carrying a rich library of learned abstractions let an agent ADAPT to a
     structurally unfamiliar task type faster than an agent starting fresh?"

The asymmetry that makes it work: the inner f's are depth-2 composites the cultured
population accumulated during ordinary training. A cultured agent recalls f and only
has to search the tiny combinator axis (stage 1). A fresh agent must discover the
combinator AND rediscover a depth-2 f from scratch — under a tight budget it cannot
even get past the single-op inner candidates, so it never reaches the depth-2 f the
task needs. Identical task, identical budget; only the inherited library differs.

Guards (so the result can genuinely FAIL):
  * eval is frozen — agents do not learn/store anything while solving (no test-time
    leakage of the freshly-discovered combinator across tasks);
  * an ORACLE that knows the inner f's is confirmed to solve 100% of held-out tasks
    (so a null can never be blamed on impossible tasks);
  * correctness is judged on a HELD-OUT query, not the demonstration examples, so a
    spuriously example-consistent hypothesis does not score;
  * two budget regimes — GENEROUS (did the ceiling move? can both reach it?) and
    TIGHT (does culture still decide?);
  * multiple seeds; identical headline hyperparameters; nothing tuned to win.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .agent import Agent
from .environments.transformation_world import COMPOSITE_TASKS
from .evaluation import EXPERIMENTS
from .evolution import Civilization, CivConfig
from .skills import PRIMITIVES, Skill, program_name, run_program


# --------------------------------------------------------------- combinators
def _toks(s: str):
    return s.split(" ")


# Each combinator: (inner_fn, string) -> string. NONE of these appear anywhere in
# training; they are the novel structural axis both agent types meet for the first
# time at evaluation.
COMBINATORS = {
    "map_each":     lambda f, s: " ".join(f(t) for t in _toks(s)),
    "map_reversed": lambda f, s: " ".join(f(t) for t in reversed(_toks(s))),
    "first_only":   lambda f, s: " ".join([f(_toks(s)[0])] + _toks(s)[1:]),
    "last_only":    lambda f, s: " ".join(_toks(s)[:-1] + [f(_toks(s)[-1])]),
    "map_evens":    lambda f, s: " ".join(f(t) if i % 2 == 0 else t
                                          for i, t in enumerate(_toks(s))),
}
COMBINATOR_NAMES = list(COMBINATORS.keys())

# The inner programs the held-out family is built from: the SAME depth-2 composites
# the Transformation-World civilization trains on (so a cultured population holds
# them as inherited abstractions, while a fresh agent must rediscover them).
INNER_PROGRAMS = [prog for _, prog in COMPOSITE_TASKS]


def apply_family(combinator: str, inner: tuple, s: str) -> str:
    f = lambda x: run_program(inner, x)
    return COMBINATORS[combinator](f, s)


# ------------------------------------------------------------------- tasks
@dataclass
class NovelTask:
    combinator: str
    inner: tuple
    examples: list           # [(inp, out), ...]
    query_input: str
    query_target: str

    @property
    def name(self) -> str:
        return f"{self.combinator}({program_name(self.inner)})"

    def is_solved(self, prediction: str) -> bool:
        return prediction == self.query_target


def _rand_multitoken(rng, lo_tok=2, hi_tok=4, lo_len=2, hi_len=4) -> str:
    alpha = list("abcdefgh")
    n = int(rng.integers(lo_tok, hi_tok + 1))
    toks = []
    for _ in range(n):
        m = int(rng.integers(lo_len, hi_len + 1))
        toks.append("".join(rng.choice(alpha, size=m)))
    return " ".join(toks)


def make_novel_tasks(seed: int = 555, n_each: int = 4, n_examples: int = 5):
    """Build the held-out novel-family suite: every (combinator, inner) pair, each
    instantiated with fresh multi-token strings. Demonstration examples and the
    graded query are disjoint draws."""
    rng = np.random.default_rng(seed)
    tasks = []
    for comb in COMBINATOR_NAMES:
        for inner in INNER_PROGRAMS:
            for _ in range(n_each):
                exs = []
                for _ in range(n_examples):
                    inp = _rand_multitoken(rng)
                    exs.append((inp, apply_family(comb, inner, inp)))
                q = _rand_multitoken(rng)
                tasks.append(NovelTask(
                    combinator=comb, inner=inner, examples=exs,
                    query_input=q, query_target=apply_family(comb, inner, q)))
    return tasks


# ------------------------------------------------------------- synthesiser
@dataclass
class AdaptResult:
    hypothesis: tuple | None   # (combinator, inner) chosen, or None
    evals: int
    via_known: bool            # solved using an inherited inner abstraction
    discovered: bool           # inner found by from-scratch discovery


def synthesize_adaptive(known_programs, primitives, examples, budget, rng,
                        allow_discovery: bool = True):
    """Search the (combinator x inner) hypothesis space for one consistent with the
    examples. Cultured agents win because the inner f is already in `known_programs`
    (stage 1); fresh agents must reach the depth-2 inner via from-scratch discovery
    (stage 3), which a tight budget forbids.

    Returns the first example-consistent hypothesis (held-out query decides whether
    that hypothesis is actually correct — done by the caller)."""
    evals = 0

    def consistent(comb, inner):
        nonlocal evals
        evals += 1
        try:
            return all(apply_family(comb, inner, i) == o for i, o in examples)
        except Exception:
            return False

    # 1) known inner abstractions x every combinator (the cultural shortcut) -----
    for inner in known_programs:
        for comb in COMBINATOR_NAMES:
            if evals >= budget:
                return AdaptResult(None, evals, False, False)
            if consistent(comb, inner):
                return AdaptResult((comb, tuple(inner)), evals, True, False)

    # 2) recombine known inner programs (pairwise) x every combinator ------------
    for a in known_programs:
        for b in known_programs:
            inner = tuple(a) + tuple(b)
            for comb in COMBINATOR_NAMES:
                if evals >= budget:
                    return AdaptResult(None, evals, False, False)
                if consistent(comb, inner):
                    return AdaptResult((comb, inner), evals, True, False)

    if not allow_discovery:
        return AdaptResult(None, evals, False, False)

    # 3) discover the inner from scratch: single ops first (cheap), then SHUFFLED
    #    depth-2 composites. A fresh agent cannot clear the singles within a tight
    #    budget, so it never reaches the depth-2 inner the novel task requires. ---
    singles = [(op,) for op in primitives]
    pairs = [(a, b) for a in primitives for b in primitives]
    rng.shuffle(pairs)
    for inner in singles + pairs:
        for comb in COMBINATOR_NAMES:
            if evals >= budget:
                return AdaptResult(None, evals, False, True)
            if consistent(comb, inner):
                return AdaptResult((comb, inner), evals, evals == 0, True)
    return AdaptResult(None, evals, False, True)


def agent_solve_novel(agent: Agent, task: NovelTask, budget: int,
                      allow_discovery: bool = True):
    """Have an agent attempt one novel-family task with a FROZEN library (it never
    stores the discovered combinator/inner). Returns (solved, AdaptResult)."""
    known = agent._priority_known_programs()
    primitives = list(PRIMITIVES.keys())
    res = synthesize_adaptive(known, primitives, task.examples, budget, agent.rng,
                              allow_discovery=allow_discovery)
    if res.hypothesis is None:
        return False, res
    comb, inner = res.hypothesis
    pred = apply_family(comb, inner, task.query_input)
    return task.is_solved(pred), res


def novel_solve_rate(agents, tasks, budget):
    if not agents or not tasks:
        return 0.0
    rates = []
    for ag in agents:
        solved = sum(int(agent_solve_novel(ag, t, budget)[0]) for t in tasks)
        rates.append(solved / len(tasks))
    return float(np.mean(rates))


# ----------------------------------------------------------------- oracle / fresh
def oracle_population(rng, n: int = 4):
    """Agents that already KNOW every inner abstraction (but no combinator). Used to
    prove the held-out suite is solvable-in-principle."""
    pop = []
    for _ in range(n):
        ag = Agent(0, rng)
        for inner in INNER_PROGRAMS:
            ag.learn_skill(Skill(name=program_name(inner), program=tuple(inner),
                                 creator="oracle", generation=0))
        pop.append(ag)
    return pop


def fresh_population(rng, n: int = 24):
    """Brand-new gen-0 agents with empty libraries (the no-accumulation baseline)."""
    return [Agent(0, rng) for _ in range(n)]


# ------------------------------------------------------------------ experiment
HEADLINE = dict(generations=30, population_size=24, budget=35, tasks_per_agent=8)


def train_condition(name: str, seed: int):
    """Train one A/B/C/D condition on the ordinary Transformation World and return
    its final population (identical machinery to the headline experiment)."""
    from .environments import TransformationWorld
    cfg = CivConfig(name=name, **{**HEADLINE, **EXPERIMENTS[name], "seed": seed})
    world = TransformationWorld(np.random.default_rng(1000 + seed), tier="all")
    civ = Civilization(cfg, world, db=None)
    civ._spawn_initial()
    for gen in range(cfg.generations):
        civ.run_generation(gen)
        if gen < cfg.generations - 1:
            civ.population = civ._reproduce(gen + 1)
    return civ


# Budget regimes (measured: cultured solves in < ~75 checks via stage 1; fresh needs
# > ~50 just to clear the single-op inner candidates before it can reach a depth-2
# inner). TIGHT sits below that wall for fresh but above it for cultured.
GENEROUS_BUDGET = 4000
TIGHT_BUDGET = 45


def run_adaptability(seeds=(0, 1, 2), curve_condition="D_full_civilization"):
    """Full experiment: train A/B/C/D, then frozen-eval each final population (plus
    fresh + oracle baselines) on the novel combinator family at both budgets."""
    novel = make_novel_tasks()

    # oracle audit: confirm every held-out task is solvable-in-principle
    oracle = oracle_population(np.random.default_rng(0))
    oracle_rate = novel_solve_rate(oracle, novel, GENEROUS_BUDGET)

    conditions = list(EXPERIMENTS.keys())
    per = {c: {"tight": [], "generous": [], "culture_known": []} for c in conditions}
    per["FRESH"] = {"tight": [], "generous": [], "culture_known": []}

    curve_budgets = [10, 20, 30, 45, 60, 100, 200, 500, 1500, 4000]
    curves = {}  # condition -> list[(budget, rate)] for seed[0]

    for si, seed in enumerate(seeds):
        # cultured conditions
        for c in conditions:
            civ = train_condition(c, seed)
            pop = civ.population
            per[c]["tight"].append(novel_solve_rate(pop, novel, TIGHT_BUDGET))
            per[c]["generous"].append(novel_solve_rate(pop, novel, GENEROUS_BUDGET))
            per[c]["culture_known"].append(
                float(np.mean([len(a.known_skills) for a in pop])))
            if si == 0 and c in (curve_condition,):
                curves[c] = [(b, novel_solve_rate(pop, novel, b))
                             for b in curve_budgets]
        # fresh baseline
        fresh = fresh_population(np.random.default_rng(7000 + seed))
        per["FRESH"]["tight"].append(novel_solve_rate(fresh, novel, TIGHT_BUDGET))
        per["FRESH"]["generous"].append(novel_solve_rate(fresh, novel, GENEROUS_BUDGET))
        per["FRESH"]["culture_known"].append(0.0)
        if si == 0:
            curves["FRESH"] = [(b, novel_solve_rate(fresh, novel, b))
                               for b in curve_budgets]

    summary = {}
    for c, d in per.items():
        summary[c] = {
            "tight": (float(np.mean(d["tight"])), float(np.std(d["tight"]))),
            "generous": (float(np.mean(d["generous"])), float(np.std(d["generous"]))),
            "avg_known": float(np.mean(d["culture_known"])),
        }

    return {
        "summary": summary,
        "curves": curves,
        "curve_budgets": curve_budgets,
        "oracle_rate": oracle_rate,
        "tight_budget": TIGHT_BUDGET,
        "generous_budget": GENEROUS_BUDGET,
        "n_tasks": len(novel),
        "n_combinators": len(COMBINATOR_NAMES),
        "combinators": COMBINATOR_NAMES,
        "n_inner": len(INNER_PROGRAMS),
        "seeds": list(seeds),
    }


# ----------------------------------------------------------------- worked trace
def capture_trace(seed: int = 0):
    """Capture a concrete worked example: a single held-out novel task, what a
    cultured agent (condition D) does with it vs. a fresh agent, at the tight
    budget. Returned as a JSON-able dict for the report."""
    novel = make_novel_tasks(seed=556)
    # pick a representative task whose inner is a genuine depth-2 composite
    task = next(t for t in novel if t.combinator == "map_reversed")

    civ = train_condition("D_full_civilization", seed)
    cultured = max(civ.population, key=lambda a: len(a.known_skills))
    fresh = Agent(0, np.random.default_rng(99))

    c_solved, c_res = agent_solve_novel(cultured, task, TIGHT_BUDGET)
    f_solved, f_res = agent_solve_novel(fresh, task, TIGHT_BUDGET)

    def hyp(res):
        if res.hypothesis is None:
            return None
        comb, inner = res.hypothesis
        return f"{comb}({program_name(inner)})"

    return {
        "task_name": task.name,
        "true_rule": f"{task.combinator}({program_name(task.inner)})",
        "examples": task.examples,
        "query_input": task.query_input,
        "query_target": task.query_target,
        "tight_budget": TIGHT_BUDGET,
        "cultured": {
            "library_size": len(cultured.known_skills),
            "knows_inner": tuple(task.inner) in cultured.known_skills,
            "solved": c_solved,
            "hypothesis": hyp(c_res),
            "prediction": (apply_family(*c_res.hypothesis, task.query_input)
                           if c_res.hypothesis else "(gave up)"),
            "evals_used": c_res.evals,
            "via_known": c_res.via_known,
        },
        "fresh": {
            "library_size": len(fresh.known_skills),
            "solved": f_solved,
            "hypothesis": hyp(f_res),
            "prediction": (apply_family(*f_res.hypothesis, task.query_input)
                           if f_res.hypothesis else "(gave up — budget exhausted on "
                           "single-op inner candidates)"),
            "evals_used": f_res.evals,
        },
    }
