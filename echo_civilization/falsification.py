"""Experiment N -- falsifying (or defending) the Experiment-M abstraction result.

Experiment M showed a from-scratch grid solver climb from 2.5% to 85% (97.5% with a
proposer) once it was allowed to invent, name and stack its own concepts. That is a
striking curve, and striking curves in a self-designed benchmark deserve suspicion.
This module is the adversarial follow-up. It re-runs the same mechanism against a
battery of controls whose whole purpose is to make the library look useless:

  fresh_flat      from-scratch search, deployment budget                (M's FLAT)
  generous_flat   from-scratch search, huge budget + length oracle      (is the gap
                  just a stingy budget?)
  cache_only      a persistent cache of previously-solved *whole programs*, retried
                  against new tasks -- no recombination                  (is the
                  library just memorising solved programs?)
  invent          the learned fragment library, cumulative corpus, MDL/held-out
                  promotion                                              (the claim)
  invent_guide    invent + the numpy proposer                           (the claim)
  oracle_lib      library pre-seeded with the six *true* motifs          (upper bound)
  shuffled_lib    a size-matched library of *useless* random fragments   (does content
                  matter, or just having more/longer tokens?)

Everything runs on the SAME frozen held-out stream, every task scored on an UNSEEN
query grid, multi-seed. No pretrained models; numpy + stdlib only.

The honest questions it answers:
  * Does query-grid scoring collapse the M numbers?  (over-fitting the demo pairs)
  * Can a cache of solved programs do the same job?   (memorisation vs recombination)
  * Can a generous flat solver reach the ceiling?     (budget artefact vs real gap)
  * Does a size-matched useless library help at all?  (branching vs content)
  * How much does an irrelevant-concept load cost?    (negative transfer)
"""

from __future__ import annotations

import numpy as np

from .abstraction import Abstraction, Library, mine_abstractions, solve_task
from .arc_tasks_v2 import MOTIFS, make_heldout, make_training_pool
from .gridworld_arc import BASE_OP_NAMES, grids_equal, run_program
from .proposer import Proposer


# --------------------------------------------------------------------------- #
# Query-aware solve + evaluation
# --------------------------------------------------------------------------- #
def solve_with_query(task, library, budget, max_len, proposer=None):
    """Synthesise from the demonstration pairs only, then score on the unseen query.

    Returns (result, train_consistent, query_correct). Deployment protocol: we take
    the FIRST (shortest, iterative-deepening) program consistent with the demos and
    check it against the held-out query -- exactly one submission, no peeking.
    """
    r = solve_task(task.train, library, budget, max_len, proposer=proposer)
    if not r.solved:
        return r, False, False
    pred = run_program(r.program, task.query[0], library)
    return r, True, grids_equal(pred, task.query[1])


def eval_heldout(held, library, budget, max_len, proposer=None):
    train_ok = q_ok = 0
    evals, toks, blen = [], [], []
    used_ops = set()
    for t in held:
        r, tc, qc = solve_with_query(t, library, budget, max_len, proposer)
        evals.append(r.evals)
        if tc:
            train_ok += 1
        if qc:
            q_ok += 1
            toks.append(len(r.program))
            blen.append(r.base_length)
            used_ops.update(o for o in r.program if o in library.ops)
    n = len(held)
    return {
        "acc": q_ok / n,                       # the real number: query-correct
        "train_consistent_acc": train_ok / n,  # looser: fits the demos (overfit gap)
        "mean_evals": float(np.mean(evals)),
        "total_eval_cost": int(np.sum(evals)),
        "mean_tokens": float(np.mean(toks)) if toks else 0.0,
        "mean_base_len": float(np.mean(blen)) if blen else 0.0,
        "n_concepts_used": len(used_ops),
    }


# --------------------------------------------------------------------------- #
# Improved, measured promotion (brief point 6)
# --------------------------------------------------------------------------- #
def mine_with_utility(corpus, library, invented_round, val_tasks, budget, max_len,
                      top_k=3):
    """Promote a mined fragment only if it *measurably* helps a held-out validation
    slice more than it costs.

    MDL alone (Experiment M) scores a fragment by compression of the corpus. That
    ignores two real costs the brief calls out: the branching a new op adds to every
    future search, and negative transfer on tasks the concept does not fit. Here we
    trial each MDL-positive candidate on `val_tasks` and keep it only when

        evals_saved (with the candidate)  -  definition_cost  >  0

    measured, not assumed. Candidates that merely enlarge the branching factor lose
    the trial and are dropped. Legible and directly grounded in search cost.
    """
    # candidate fragments = what plain MDL would consider, but don't commit yet
    trial = Library()
    trial.ops = dict(library.ops)
    trial._counter = library._counter
    candidates = mine_abstractions(corpus, trial, invented_round, top_k=top_k * 3)
    if not candidates:
        return []

    # baseline validation cost with the current library
    base_cost = sum(solve_task(t.train, library, budget, max_len).evals
                    for t in val_tasks)

    kept = []
    for cand in candidates:
        probe = Library()
        probe.ops = dict(library.ops)
        probe._counter = library._counter
        probe.ops[cand.name] = cand         # add just this candidate
        cost = sum(solve_task(t.train, probe, budget, max_len).evals for t in val_tasks)
        def_cost = len(cand.expansion) + 1
        utility = (base_cost - cost) - def_cost
        cand.mdl_value = float(utility)     # store the measured utility
        if utility > 0:
            kept.append(cand)
        if len(kept) >= top_k:
            break

    # actually commit the survivors to the real library (fresh names/levels)
    committed = []
    for cand in kept:
        if any(a.expansion == cand.expansion for a in library.ops.values()):
            continue
        a = library.add(cand.expansion, invented_round)
        a.mdl_value = cand.mdl_value
        a.uses = cand.uses
        committed.append(a)
    return committed


# --------------------------------------------------------------------------- #
# The invention learner (cumulative corpus -- brief point 6)
# --------------------------------------------------------------------------- #
def run_invent(seed, n_rounds, held, cfg, guide=False, promotion="mdl"):
    """Single persistent lifelong learner. Retains a cumulative corpus of every
    solved program and re-encodes it against the current library each round before
    mining (M discarded each round's solutions -- this is the fix)."""
    rng = np.random.default_rng(seed)
    lib = Library()
    prop = Proposer() if guide else None
    corpus_base = []            # cumulative solved programs, stored in BASE ops
    per_round = []
    lifetime_evals = 0
    for rnd in range(1, n_rounds + 1):
        train = make_training_pool(rng, cfg["train_per_round"])
        for t in train:
            r = solve_task(t.train, lib, cfg["budget"], cfg["max_len"], proposer=prop)
            lifetime_evals += r.evals
            if r.solved:
                corpus_base.append(lib.expand_to_base(r.program))
                if prop is not None:
                    prop.update(t.train, lib.encode(r.program), lib.op_names())
        # re-encode the WHOLE cumulative corpus in current library tokens, then mine
        corpus_enc = [lib.encode(p) for p in corpus_base]
        if promotion == "utility":
            val = train[: min(12, len(train))]
            invented = mine_with_utility(corpus_enc, lib, rnd, val,
                                         cfg["budget"], cfg["max_len"])
        else:
            invented = mine_abstractions(corpus_enc, lib, rnd)
        h = eval_heldout(held, lib, cfg["budget"], cfg["max_len"], proposer=prop)
        h.update({"round": rnd, "n_abstractions": len(lib.ops),
                  "max_level": lib.max_level(), "levels": lib.levels(),
                  "corpus_size": len(corpus_base),
                  "lifetime_train_evals": lifetime_evals,
                  "invented": [a.name for a in invented]})
        per_round.append(h)
    return per_round, lib


# --------------------------------------------------------------------------- #
# Flat controls
# --------------------------------------------------------------------------- #
def run_flat(seed, n_rounds, held, cfg, budget=None, max_len=None):
    """From-scratch search, no invention. `budget`/`max_len` override lets us run
    both the deployment flat and the generous-oracle flat with the same code."""
    b = budget or cfg["budget"]
    ml = max_len or cfg["max_len"]
    empty = Library()
    h = eval_heldout(held, empty, b, ml)
    h.update({"budget": b, "max_len": ml})
    return [dict(h, round=r) for r in range(1, n_rounds + 1)]


# --------------------------------------------------------------------------- #
# Cache-only control: persistent store of solved WHOLE programs, retried verbatim
# --------------------------------------------------------------------------- #
def run_cache_only(seed, n_rounds, held, cfg):
    """Accumulate solved base-op programs from training and, at eval, retry each
    cached program against the new task's demos (never by task id or grid). This is
    pure memorisation with zero recombination -- the control that isolates whether
    the library's gains come from *composing* fragments or merely from *recalling*
    whole solutions."""
    rng = np.random.default_rng(seed)
    empty = Library()
    cache = []
    seen = set()
    per_round = []
    for rnd in range(1, n_rounds + 1):
        train = make_training_pool(rng, cfg["train_per_round"])
        for t in train:
            r = solve_task(t.train, empty, cfg["budget"], cfg["max_len"])
            if r.solved:
                key = r.program
                if key not in seen:
                    seen.add(key)
                    cache.append(r.program)
        # eval: try each cached program against demos, then query
        q_ok = tc = 0
        evals = []
        for t in held:
            hit = None
            used = 0
            for prog in cache:
                used += 1
                if used > cfg["budget"]:
                    break
                ok = all(grids_equal(run_program(prog, i), o) for i, o in t.train)
                if ok:
                    hit = prog
                    break
            evals.append(used)
            if hit is not None:
                tc += 1
                if grids_equal(run_program(hit, t.query[0]), t.query[1]):
                    q_ok += 1
        n = len(held)
        per_round.append({"round": rnd, "acc": q_ok / n,
                          "train_consistent_acc": tc / n,
                          "mean_evals": float(np.mean(evals)),
                          "cache_size": len(cache)})
    return per_round


# --------------------------------------------------------------------------- #
# Oracle & shuffled-library controls
# --------------------------------------------------------------------------- #
def _seed_library(expansions):
    lib = Library()
    for exp in expansions:
        lib.add(tuple(exp), invented_round=0)
    return lib


def oracle_library():
    """The six *true* motifs handed in as level-1 concepts. This is a LOWER bound,
    not an upper bound: a depth-4 held-out task is four motif tokens, and searching
    length-4 over ~23 ops blows the budget. Motifs alone don't make deep tasks
    reachable -- you need the hierarchical L2 compounds the invention arm builds.
    Reported to show the L1 vocabulary is necessary but not sufficient."""
    return _seed_library(MOTIFS)


def full_oracle_library():
    """The true UPPER bound: the six motifs as L1 concepts PLUS every ordered pair of
    motifs as an L2 concept. This is the maximal *correct* library a perfect inventor
    could build from this grammar. With it, any depth-3/4 held-out task collapses to
    two L2 tokens (or one L2 + one L1), so it is reachable by a length-2 search --
    the honest ceiling for the invention mechanism. We add the motifs first as named
    L1 ops, then build the compounds on top of those names so the hierarchy is real
    (L2 expansions reference L1 op names, exactly like a mined library)."""
    lib = Library()
    motif_names = []
    for m in MOTIFS:
        a = lib.add(tuple(m), invented_round=0)   # L1
        motif_names.append(a.name)
    for i in range(len(motif_names)):
        for j in range(len(motif_names)):
            lib.add((motif_names[i], motif_names[j]), invented_round=0)  # L2
    return lib


def shuffled_library(rng, n_ops):
    """`n_ops` USELESS length-2 fragments: random base-op pairs that are not motifs.
    Size-matched negative control -- same vocabulary growth as INVENT, zero signal."""
    motif_set = {tuple(m) for m in MOTIFS}
    exps = []
    while len(exps) < n_ops:
        pair = (str(rng.choice(BASE_OP_NAMES)), str(rng.choice(BASE_OP_NAMES)))
        if pair not in motif_set:
            exps.append(pair)
    return _seed_library(exps)


def run_static_library(lib, n_rounds, held, cfg):
    h = eval_heldout(held, lib, cfg["budget"], cfg["max_len"])
    h.update({"n_abstractions": len(lib.ops)})
    return [dict(h, round=r) for r in range(1, n_rounds + 1)]


# --------------------------------------------------------------------------- #
# Negative transfer: load the oracle library with irrelevant concepts (brief pt 5)
# --------------------------------------------------------------------------- #
def negative_transfer(seed, held, cfg, loads=(0, 2, 4, 8, 16)):
    """Start from the *useful* oracle library and progressively bolt on useless
    concepts. Measures the pure cost of vocabulary bloat: accuracy, mean search
    cost, and how many concepts the held-out tasks actually use vs how many sit in
    the vocabulary raising the branching factor."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in loads:
        lib = oracle_library()
        base_n = len(lib.ops)
        for exp in shuffled_library(rng, k).ops.values():
            lib.add(exp.expansion, invented_round=0)
        h = eval_heldout(held, lib, cfg["budget"], cfg["max_len"])
        rows.append({"useless_added": k, "vocab_size": len(BASE_OP_NAMES) + len(lib.ops),
                     "n_abstractions": len(lib.ops), "useful_abstractions": base_n,
                     "acc": h["acc"], "mean_evals": h["mean_evals"],
                     "concepts_used": h["n_concepts_used"]})
    return rows


# --------------------------------------------------------------------------- #
# Budget sweep: is the flat gap a stingy-budget artefact? (brief point 4)
# --------------------------------------------------------------------------- #
def budget_sweep(seed, held, cfg, budgets, max_len):
    """Give flat search ever more compute (and length headroom) and watch whether it
    can climb toward the invention ceiling. Reports per-task deployment cost so the
    exponential wall is visible."""
    rows = []
    empty = Library()
    for b in budgets:
        h = eval_heldout(held, empty, b, max_len)
        rows.append({"budget": b, "max_len": max_len, "acc": h["acc"],
                     "mean_evals": h["mean_evals"]})
    return rows


# --------------------------------------------------------------------------- #
# Civilization vs lifelong single learner (brief point 7)
# --------------------------------------------------------------------------- #
def _add_base_fragment(lib, base_exp, rnd, known):
    """Add a base-op fragment to `lib` as an abstraction, re-encoded against the
    library's current concepts so hierarchy can form. Deduplicated by base expansion."""
    key = tuple(base_exp)
    if key in known:
        return None
    enc = lib.encode(base_exp)
    if len(enc) < 2:            # collapses to one existing op -> nothing new
        return None
    known.add(key)
    return lib.add(enc, rnd)


def run_population(seed, n_gens, agents_per_gen, per_gen, held, cfg):
    """A generational population that shares discoveries as culture.

    With `agents_per_gen=1` this reduces to a single lifelong learner that sees every
    task -- the matched control. With `agents_per_gen=P` the same total task budget is
    split across P agents that discover concepts independently and pool them into a
    shared, inherited culture. The comparison isolates the *civilization* effect
    (parallel independent discovery + cultural sharing) from plain individual lifelong
    library learning. Total training solves are identical for every P.
    """
    rng = np.random.default_rng(seed)
    shared = Library()
    known = set()
    per_gen_metrics = []
    lifetime_evals = 0
    for gen in range(1, n_gens + 1):
        gen_tasks = make_training_pool(rng, per_gen)
        slices = [list(s) for s in np.array_split(np.array(gen_tasks, dtype=object),
                                                  agents_per_gen)]
        discovered = []                      # base-op expansions found this gen
        for sl in slices:
            agent_lib = Library()
            agent_lib.ops = dict(shared.ops)
            agent_lib._counter = shared._counter
            corpus = []
            for t in sl:
                r = solve_task(t.train, agent_lib, cfg["budget"], cfg["max_len"])
                lifetime_evals += r.evals
                if r.solved:
                    corpus.append(agent_lib.expand_to_base(r.program))
            enc = [agent_lib.encode(p) for p in corpus]
            for a in mine_abstractions(enc, agent_lib, gen):
                discovered.append(agent_lib.expand_to_base(a.expansion))
        # merge discoveries into shared culture (dedup, re-encode for hierarchy)
        for base_exp in discovered:
            _add_base_fragment(shared, base_exp, gen, known)
        h = eval_heldout(held, shared, cfg["budget"], cfg["max_len"])
        per_gen_metrics.append({"gen": gen, "acc": h["acc"],
                                "n_abstractions": len(shared.ops),
                                "max_level": shared.max_level(),
                                "mean_evals": h["mean_evals"],
                                "lifetime_train_evals": lifetime_evals})
    return per_gen_metrics, shared


# --------------------------------------------------------------------------- #
# Integrity checks: oracle solvability + behavioural leakage
# --------------------------------------------------------------------------- #
def collect_training_programs(seed, cfg, n_pool=None):
    """Every distinct base-op program a flat search finds on a training pool. Used
    both by the behavioural-leak filter and the leak integrity gate."""
    rng = np.random.default_rng(seed)
    train = make_training_pool(rng, n_pool or cfg["train_per_round"])
    empty = Library()
    solved = set()
    for t in train:
        r = solve_task(t.train, empty, cfg["budget"], cfg["max_len"])
        if r.solved:
            solved.add(r.program)
    return solved


def filter_behavioural_leaks(held, cfg, leak_seeds=(0, 1, 2)):
    """Drop held-out tasks that are behaviourally reachable from training.

    A held-out program is *supposed* to require a novel motif combination, but two
    different motif chains can be behaviourally identical (many ops are idempotent or
    commute), so a task can leak: a program discovered on a shallow training task also
    reproduces the held-out task's demos AND its unseen query. Those tasks don't test
    recombination -- they were already reachable. We pool training solutions across a
    few seeds, drop any held-out task any of them satisfies, and report the count. The
    surviving set is genuinely disjoint from training behaviour, not just syntactically.
    """
    solved = set()
    for s in leak_seeds:
        solved |= collect_training_programs(s, cfg)
    clean, dropped = [], 0
    for t in held:
        leaked = any(
            all(grids_equal(run_program(p, i), o) for i, o in t.train)
            and grids_equal(run_program(p, t.query[0]), t.query[1])
            for p in solved)
        if leaked:
            dropped += 1
        else:
            clean.append(t)
    return clean, dropped, len(solved)


def integrity_checks(held, seed, cfg):
    """Three sanity gates the brief requires before trusting any accuracy number.

      ground_truth_query : the hidden program reproduces the unseen query (the suite
                           is well-posed).
      oracle_solvable    : the true-motif library solves the task within budget on
                           demos AND query (the suite is not accidentally impossible).
      behavioural_leak   : NO program that solves a training task also reproduces a
                           held-out task's demos+query (train/eval really are disjoint
                           behaviourally, not just syntactically).
    """
    gt_ok = 0
    for t in held:
        if grids_equal(run_program(t.program, t.query[0]), t.query[1]):
            gt_ok += 1

    orc = oracle_library()
    orc_ok = 0
    for t in held:
        _, _, qc = solve_with_query(t, orc, cfg["budget"], cfg["max_len"])
        if qc:
            orc_ok += 1

    # collect solved training programs, then test them against held-out behaviour
    rng = np.random.default_rng(seed)
    train = make_training_pool(rng, cfg["train_per_round"])
    empty = Library()
    solved = set()
    for t in train:
        r = solve_task(t.train, empty, cfg["budget"], cfg["max_len"])
        if r.solved:
            solved.add(r.program)
    leaks = 0
    for t in held:
        for prog in solved:
            if all(grids_equal(run_program(prog, i), o) for i, o in t.train) and \
               grids_equal(run_program(prog, t.query[0]), t.query[1]):
                leaks += 1
                break
    return {"n_heldout": len(held), "ground_truth_query_ok": gt_ok,
            "oracle_solvable": orc_ok, "behavioural_leaks": leaks,
            "n_train_programs_tested": len(solved)}
