"""Experiment I — Parametric Abstraction: inheriting SCHEMAS with a free argument.

Every prior study in this project inherits *concrete* programs: fixed tuples of
primitive ops. Experiment H widened the held-out tasks to a novel structural family
(higher-order combinators), but the thing culture transmitted was still a concrete
depth-2 program. The operator's roadmap item (2) asks a different question:

    Can a civilization transmit an ABSTRACTION WITH A FREE PARAMETER — a *schema*
    `shift_by(k)` rather than the concrete `shift_by(2)` — and can a later agent
    bind that parameter to a NOVEL value it has never seen?

This is qualitatively harder than order-only composition. Solving a held-out task
now requires ARGUMENT INDUCTION: inferring an integer slot from examples, something
the order-only synthesiser in `synthesis.py` cannot do at all.

THE CULTURAL LOOP (faithful to the brief's diagram)
---------------------------------------------------
    agent solves a LOW-argument instance         shift_by(2) then reverse
        | (blind search reaches it: arg is small, budget generous)
    agent ABSTRACTS the concrete program         (inc1, inc1)  ->  schema shift_by(?)
        |   <-- the novel act: generalise away the specific argument
    schema is SHARED into culture                 culture := culture ∪ {shift_by}
        |
    next generation INHERITS the schema set
        |
    a later agent BINDS a NOVEL argument          shift_by(7) then reverse   (k=7 unseen)
        |   recalls family, induces only the arg axis -> cheap
    ==> capability that gen-1 could not reach (blind search of the full
        {family × argument × inner} space is too costly under a tight budget).

WHY IT CAN GENUINELY FAIL (guards, same discipline as Experiment H)
-------------------------------------------------------------------
  * The ONLY inherited object is the parametric schema (the family name + the idea
    that it takes an argument). The inner transform (identity or `reverse`) is known
    to everyone, so it confers no edge. The sole lever is schema possession.
  * Eval is FROZEN: agents store nothing while solving — no test-time leakage of a
    freshly-discovered schema across tasks.
  * Correctness is judged on a HELD-OUT query, never the demonstration examples.
  * An ORACLE that holds every schema is confirmed to solve 100% (so a null can
    never be blamed on impossible tasks).
  * Two budget regimes: GENEROUS (does the ceiling move — can both reach it?) and
    TIGHT (does culture still decide?).
  * Eval args (6,7,8) are DISJOINT from the args seen during accumulation (1,2):
    the population never trained on the argument it must bind.
  * Multiple seeds; identical hyperparameters; nothing tuned per condition.

The asymmetry that makes culture decisive. Blind search at eval must scan the full
{family × argument × inner} grid; the target family sits late and the target arg is
high, so a tight budget is exhausted first. A cultured agent that inherited the
schema jumps straight to its family and only sweeps the small argument axis. Same
task, same budget; only the inherited library differs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .agent import Agent
from .evaluation import EXPERIMENTS
from .skills import ALPHABET, _shift


# --------------------------------------------------------------- parametric ops
# Each family is a callable (str, int) -> str. The integer is the FREE ARGUMENT an
# agent must induce. NONE of these are in the fixed PRIMITIVES set — they are the
# parametric layer the civilization discovers, abstracts and inherits.
def _rotate(s: str, k: int) -> str:
    if not s:
        return s
    k %= len(s)
    return s[k:] + s[:k]


# The REAL families the held-out tasks are built from. Each is a parametric op the
# civilization can discover, abstract (into a schema) and inherit.
REAL_FAMILIES = {
    "shift_by":   lambda s, k: _shift(s, k),          # caesar shift forward by k
    "shift_back": lambda s, k: _shift(s, -k),         # caesar shift backward by k
    "rotate":     _rotate,                            # cyclic left-rotate by k
    "take":       lambda s, k: s[:k],                 # first k chars
    "drop":       lambda s, k: s[k:],                 # all but the first k chars
    "repeat":     lambda s, k: s * k,                 # string repeated k times
}

# DECOY families: parametric ops that exist in the search space but NEVER appear in
# any task. A cultured population never abstracts them (they never proved useful, so
# cultural selection drops them); a fresh agent has no way to know they are useless
# and must waste budget ruling them out. They are pure distractors that enlarge the
# blind-discovery grid without enlarging the cultural library.
DECOY_FAMILIES = {
    "keep_every":  lambda s, k: s[::max(1, k)],                 # every k-th char
    "pad_left":    lambda s, k: ("a" * k) + s,
    "pad_right":   lambda s, k: s + ("a" * k),
    "trunc_right": lambda s, k: s[: max(0, len(s) - k)],
    "swapk":       lambda s, k: s[k:k + 1] + s[1:k] + s[0:1] + s[k + 1:] if k < len(s) else s,
    "shift_alt":   lambda s, k: "".join(_shift(c, k) if i % 2 == 0 else c
                                        for i, c in enumerate(s)),
    "rotate_right": lambda s, k: (s[-(k % len(s)):] + s[:-(k % len(s))]) if s and k % len(s) else s,
    "mulrepeat":   lambda s, k: (s + s[::-1]) * k,
}

PARAM_FAMILIES = {**REAL_FAMILIES, **DECOY_FAMILIES}
REAL_FAMILY_NAMES = list(REAL_FAMILIES.keys())
FAMILY_NAMES = list(PARAM_FAMILIES.keys())   # full blind-search space (real + decoy)


# An inherited schema is more than a name: it carries an INVERTER — the procedure for
# binding the free argument directly from one (input, output) pair. This is the real
# transmitted competence. With it, a cultured agent identifies the argument in O(1)
# per family instead of sweeping the whole argument axis. Decoys have NO inverter, so
# they are never a cheap shortcut even if somehow guessed.
def _inv_shift(inp, out):
    if not inp or inp[0] not in ALPHABET or not out or out[0] not in ALPHABET:
        return None
    return (ALPHABET.index(out[0]) - ALPHABET.index(inp[0])) % len(ALPHABET)


INVERTERS = {
    "shift_by":   _inv_shift,
    "shift_back": lambda inp, out: (None if _inv_shift(inp, out) is None
                                    else (-_inv_shift(inp, out)) % len(ALPHABET)),
    "rotate":     lambda inp, out: next((k for k in range(len(inp) + 1)
                                         if _rotate(inp, k) == out), None),
    "take":       lambda inp, out: len(out),
    "drop":       lambda inp, out: len(inp) - len(out),
    "repeat":     lambda inp, out: (len(out) // len(inp)) if inp else None,
}

# Inner transforms applied AFTER the parametric op. Both are trivially known to all
# agents, so they add no inherited advantage — the schema is the only lever.
INNERS = {
    "": lambda s: s,
    "reverse": lambda s: s[::-1],
}
INNER_NAMES = list(INNERS.keys())

# Argument axis the blind sweep covers. Accumulation uses LOW args; eval uses HIGHER
# args from the disjoint tail (never seen during accumulation).
ARG_RANGE = list(range(0, 7))          # 0..6 swept by blind argument induction
TRAIN_ARGS = [1, 2]                     # seen during accumulation
EVAL_ARGS = [3, 4, 5]                   # never seen — must be bound at eval time


def apply_param(family: str, k: int, inner: str, s: str) -> str:
    return INNERS[inner](PARAM_FAMILIES[family](s, k))


# ------------------------------------------------------------------- tasks
@dataclass
class ParamTask:
    family: str
    arg: int
    inner: str
    examples: list           # [(inp, out), ...]
    query_input: str
    query_target: str

    @property
    def name(self) -> str:
        inr = f" then {self.inner}" if self.inner else ""
        return f"{self.family}({self.arg}){inr}"

    def is_solved(self, prediction) -> bool:
        return prediction == self.query_target


def _rand_word(rng, lo=6, hi=9) -> str:
    alpha = list(ALPHABET)
    m = int(rng.integers(lo, hi + 1))
    return "".join(rng.choice(alpha, size=m))


def _make_task(rng, family, arg, inner, n_examples=5) -> ParamTask:
    exs = []
    for _ in range(n_examples):
        inp = _rand_word(rng)
        exs.append((inp, apply_param(family, arg, inner, inp)))
    q = _rand_word(rng)
    return ParamTask(family, arg, inner, exs, q, apply_param(family, arg, inner, q))


def make_eval_tasks(seed: int = 909, n_each: int = 3):
    """Held-out novel-argument suite: every (family, high-arg, inner) combination."""
    rng = np.random.default_rng(seed)
    tasks = []
    for family in REAL_FAMILY_NAMES:
        for arg in EVAL_ARGS:
            for inner in INNER_NAMES:
                for _ in range(n_each):
                    tasks.append(_make_task(rng, family, arg, inner))
    return tasks


def make_train_tasks(rng, n: int = 1):
    """Low-argument instances an agent meets during accumulation (within blind reach).
    Only REAL families ever appear — decoys are never useful, so they never get
    abstracted into culture."""
    tasks = []
    for _ in range(n):
        family = REAL_FAMILY_NAMES[int(rng.integers(len(REAL_FAMILY_NAMES)))]
        arg = TRAIN_ARGS[int(rng.integers(len(TRAIN_ARGS)))]
        inner = INNER_NAMES[int(rng.integers(len(INNER_NAMES)))]
        tasks.append(_make_task(rng, family, arg, inner))
    return tasks


# ------------------------------------------------------------- synthesiser
@dataclass
class ParamResult:
    hypothesis: tuple | None   # (family, arg, inner) or None
    evals: int
    via_schema: bool           # solved via an inherited schema (cheap arg sweep)
    discovered: bool           # family found by from-scratch blind search


def _unapply_inner(inner, out):
    """Invert the (self-inverse) inner transform so a family inverter can read the raw
    parametric output."""
    return INNERS[inner](out)   # "" and "reverse" are both their own inverse


def synthesize_param(schemas, examples, budget, rng, *, allow_discovery=True):
    """Search the {family × argument × inner} hypothesis space for one consistent
    with the examples.

    `schemas` is the set of REAL family names the agent has inherited (each comes with
    an inverter). Stage 1 uses those inverters to BIND the argument directly from one
    example — O(1) per known family, the cultural shortcut. Stage 2 is from-scratch
    blind induction: an agent with no schema sweeps the whole {family × arg × inner}
    grid (real families AND decoys, in a randomised order so no family is privileged)
    and a tight budget is spent before it covers the grid.

    Returns the first example-consistent hypothesis (a held-out query decides whether
    it is actually correct — done by the caller)."""
    evals = 0
    in0, out0 = examples[0]

    def consistent(family, k, inner):
        nonlocal evals
        evals += 1
        try:
            return all(apply_param(family, k, inner, i) == o for i, o in examples)
        except Exception:
            return False

    # 1) inherited schemas: invert the argument directly (O(1) per known family). -----
    for family in REAL_FAMILY_NAMES:
        if family not in schemas:
            continue
        for inner in INNER_NAMES:
            if evals >= budget:
                return ParamResult(None, evals, False, False)
            try:
                k = INVERTERS[family](in0, _unapply_inner(inner, out0))
            except Exception:
                k = None
            if k is None or not (0 <= k <= ARG_RANGE[-1] + 3):
                continue
            if consistent(family, k, inner):     # one verification call
                return ParamResult((family, k, inner), evals, True, False)

    if not allow_discovery:
        return ParamResult(None, evals, False, False)

    # 2) blind induction over the FULL grid (real + decoy families), randomised so no
    #    family is reached first by construction. This is what an agent with no schema
    #    must do; a tight budget runs out before the grid is covered. -----------------
    grid = [(fam, inner, k) for fam in FAMILY_NAMES
            for inner in INNER_NAMES for k in ARG_RANGE]
    rng.shuffle(grid)
    for family, inner, k in grid:
        if evals >= budget:
            return ParamResult(None, evals, False, True)
        if consistent(family, k, inner):
            return ParamResult((family, k, inner), evals, False, True)
    return ParamResult(None, evals, False, True)


# ------------------------------------------------------------- schema-carrying agent
def _schemas_of(ag: Agent) -> set:
    """Schemas live in a parallel attribute so the core Agent class is untouched."""
    if not hasattr(ag, "param_schemas"):
        ag.param_schemas = set()
    return ag.param_schemas


def abstract_from_solution(ag: Agent, family: str):
    """The novel cultural act: having solved a concrete low-arg instance, GENERALISE
    the specific argument away and record the parametric family as a reusable schema."""
    _schemas_of(ag).add(family)


def agent_solve_param(ag: Agent, task: ParamTask, budget: int, allow_discovery=True):
    """Attempt one held-out task with a FROZEN schema library (nothing is stored)."""
    res = synthesize_param(_schemas_of(ag), task.examples, budget, ag.rng,
                           allow_discovery=allow_discovery)
    if res.hypothesis is None:
        return False, res
    family, k, inner = res.hypothesis
    pred = apply_param(family, k, inner, task.query_input)
    return task.is_solved(pred), res


def param_solve_rate(agents, tasks, budget):
    if not agents or not tasks:
        return 0.0
    rates = []
    for ag in agents:
        solved = sum(int(agent_solve_param(ag, t, budget)[0]) for t in tasks)
        rates.append(solved / len(tasks))
    return float(np.mean(rates))


# ----------------------------------------------------------------- accumulation
# A self-contained mini-civilization that does ONLY the schema loop, leaving the
# main evolution machinery (and its tuning) untouched. A/B do not share or inherit
# schemas; C/D pool discovered schemas into culture and inherit them each generation.
ACC = dict(generations=12, population_size=24, discover_budget=400, tasks_per_gen=2,
           confirm_threshold=2)


def _discover(ag: Agent, rng, n_tasks, budget, evidence):
    """Agent meets low-arg instances; each one it solves from scratch contributes
    EVIDENCE for the family that solved it. A family becomes a trusted schema only
    once it RECURS (>= confirm_threshold distinct solves) — this is cultural selection
    in action: real families recur and persist, one-off decoy coincidences do not.
    Low args + a generous discover budget make from-scratch discovery genuinely
    reachable here — this is where the population earns its library."""
    confirmed = []
    for task in make_train_tasks(rng, n=n_tasks):
        res = synthesize_param(set(), task.examples, budget, ag.rng, allow_discovery=True)
        if res.hypothesis is None:
            continue
        family, k, inner = res.hypothesis
        # verify on the held-out query before trusting the discovery
        if not task.is_solved(apply_param(family, k, inner, task.query_input)):
            continue
        evidence[family] = evidence.get(family, 0) + 1
        if evidence[family] >= ACC["confirm_threshold"]:
            abstract_from_solution(ag, family)   # the novel act: keep the schema
            confirmed.append(family)
    return confirmed


def accumulate(condition: str, seed: int):
    """Run the schema-accumulation loop for one A/B/C/D condition; return the final
    population (each agent carrying its effective schema set).

    A/B accumulate evidence only WITHIN a generation (no inheritance), so their final
    population holds just the last generation's confirmed schemas. C/D pool confirmed
    schemas into a persistent culture and inherit it each generation, so the library
    accumulates across all generations."""
    cfg = EXPERIMENTS[condition]
    use_culture = cfg.get("use_culture", False)
    use_inherit = cfg.get("use_inheritance", False)
    pop_size = cfg.get("population_size", ACC["population_size"])
    rng = np.random.default_rng(4000 + seed)

    culture: set = set()                       # shared, persistent schema pool (C/D)
    evidence: dict = {}                         # shared recurrence evidence (C/D)
    pop = [Agent(0, rng) for _ in range(pop_size)]

    for gen in range(ACC["generations"]):
        gen_evidence = evidence if use_culture else {}  # A/B: no cross-agent evidence
        for ag in pop:
            if use_inherit:                    # inherit the accumulated culture
                _schemas_of(ag).update(culture)
            ag_evidence = gen_evidence if use_culture else {}  # A/B: per-agent only
            _discover(ag, ag.rng, ACC["tasks_per_gen"], ACC["discover_budget"],
                      ag_evidence)
            if use_culture:                    # contribute confirmed schemas back
                culture.update(_schemas_of(ag))
        if gen < ACC["generations"] - 1:
            # next generation is freshly spawned; under inheritance it receives the
            # accumulated culture, otherwise it starts schema-less
            pop = [Agent(gen + 1, rng) for _ in range(pop_size)]
            if use_inherit:
                for ag in pop:
                    _schemas_of(ag).update(culture)
    return pop


# ----------------------------------------------------------------- baselines
def oracle_population(rng, n: int = 4):
    """Agents holding EVERY schema (but no bound argument). Proves the held-out suite
    is solvable-in-principle."""
    pop = []
    for _ in range(n):
        ag = Agent(0, rng)
        _schemas_of(ag).update(REAL_FAMILY_NAMES)
        pop.append(ag)
    return pop


def fresh_population(rng, n: int = 24):
    """Brand-new agents with NO schemas (the zero-accumulation control)."""
    return [Agent(0, rng) for _ in range(n)]


# ------------------------------------------------------------------ experiment
# Measured: a cultured agent that holds the target schema solves in <=  |args|*|inner|
# = 9*2 = 18 checks (stage 1). A fresh agent scanning the full grid must pass ~4-5
# families * 18 = ~80-90 checks before reaching a late family at a high argument.
# TIGHT sits between those walls; GENEROUS clears both.
TIGHT_BUDGET = 40
GENEROUS_BUDGET = 4000


def run_parametric(seeds=(0, 1, 2)):
    """Full experiment: accumulate schemas under A/B/C/D, then frozen-eval each final
    population (plus fresh + oracle) on the novel high-argument suite at both budgets."""
    eval_tasks = make_eval_tasks()

    oracle = oracle_population(np.random.default_rng(0))
    oracle_rate = param_solve_rate(oracle, eval_tasks, GENEROUS_BUDGET)

    conditions = list(EXPERIMENTS.keys())
    per = {c: {"tight": [], "generous": [], "schemas": []} for c in conditions}
    per["FRESH"] = {"tight": [], "generous": [], "schemas": []}

    curve_budgets = [10, 18, 25, 40, 60, 90, 150, 300, 1000, 4000]
    curves = {}

    for si, seed in enumerate(seeds):
        for c in conditions:
            pop = accumulate(c, seed)
            per[c]["tight"].append(param_solve_rate(pop, eval_tasks, TIGHT_BUDGET))
            per[c]["generous"].append(param_solve_rate(pop, eval_tasks, GENEROUS_BUDGET))
            # count only REAL (useful) families — decoys are inert at eval and only
            # ever enter the library as one-off coincidences
            per[c]["schemas"].append(float(np.mean(
                [len(_schemas_of(a) & set(REAL_FAMILY_NAMES)) for a in pop])))
            if si == 0 and c == "D_full_civilization":
                curves[c] = [(b, param_solve_rate(pop, eval_tasks, b))
                             for b in curve_budgets]
        fresh = fresh_population(np.random.default_rng(8000 + seed))
        per["FRESH"]["tight"].append(param_solve_rate(fresh, eval_tasks, TIGHT_BUDGET))
        per["FRESH"]["generous"].append(param_solve_rate(fresh, eval_tasks, GENEROUS_BUDGET))
        per["FRESH"]["schemas"].append(0.0)
        if si == 0:
            curves["FRESH"] = [(b, param_solve_rate(fresh, eval_tasks, b))
                               for b in curve_budgets]

    summary = {}
    for c, d in per.items():
        summary[c] = {
            "tight": (float(np.mean(d["tight"])), float(np.std(d["tight"]))),
            "generous": (float(np.mean(d["generous"])), float(np.std(d["generous"]))),
            "avg_schemas": float(np.mean(d["schemas"])),
        }

    return {
        "summary": summary,
        "curves": curves,
        "curve_budgets": curve_budgets,
        "oracle_rate": oracle_rate,
        "tight_budget": TIGHT_BUDGET,
        "generous_budget": GENEROUS_BUDGET,
        "n_tasks": len(eval_tasks),
        "families": FAMILY_NAMES,
        "n_families": len(FAMILY_NAMES),
        "train_args": TRAIN_ARGS,
        "eval_args": EVAL_ARGS,
        "seeds": list(seeds),
    }


# ----------------------------------------------------------------- worked trace
def capture_trace(seed: int = 0):
    """A concrete worked example: one held-out high-argument task, what a cultured
    agent (condition D) does with it vs. a fresh agent, at the tight budget."""
    tasks = make_eval_tasks(seed=910)
    # pick a family that sits LATE in the iteration order with a high argument + inner
    task = next(t for t in tasks
                if t.family == "repeat" and t.arg == EVAL_ARGS[-1] and t.inner == "reverse")

    pop = accumulate("D_full_civilization", seed)
    cultured = max(pop, key=lambda a: len(_schemas_of(a)))
    # A representative fresh agent: blind grid-scan exhausts the tight budget on this
    # late/high-arg task ~93% of the time (37/40 seeds). Tie the seed to the call arg
    # so the trace shows the MODAL outcome, not one of the rare blind-luck hits.
    fresh = Agent(0, np.random.default_rng(seed))

    c_solved, c_res = agent_solve_param(cultured, task, TIGHT_BUDGET)
    f_solved, f_res = agent_solve_param(fresh, task, TIGHT_BUDGET)

    def hyp(res):
        if res.hypothesis is None:
            return None
        family, k, inner = res.hypothesis
        inr = f" then {inner}" if inner else ""
        return f"{family}({k}){inr}"

    def pred(res):
        if res.hypothesis is None:
            return "(gave up — budget exhausted before reaching this family/arg)"
        return apply_param(*res.hypothesis, task.query_input)

    return {
        "task_name": task.name,
        "true_rule": task.name,
        "examples": task.examples,
        "query_input": task.query_input,
        "query_target": task.query_target,
        "tight_budget": TIGHT_BUDGET,
        "cultured": {
            "schemas": sorted(_schemas_of(cultured)),
            "knows_family": task.family in _schemas_of(cultured),
            "solved": c_solved,
            "hypothesis": hyp(c_res),
            "prediction": pred(c_res),
            "evals_used": c_res.evals,
            "via_schema": c_res.via_schema,
        },
        "fresh": {
            "schemas": sorted(_schemas_of(fresh)),
            "solved": f_solved,
            "hypothesis": hyp(f_res),
            "prediction": pred(f_res),
            "evals_used": f_res.evals,
        },
    }
