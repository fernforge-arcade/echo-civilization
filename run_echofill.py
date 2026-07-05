"""Echo Civilization — the evolved agents do LLM-style data wrangling at ~$0.

Three arms, scored on the SAME held-out suite:

  A. NAIVE agent   — empty skill library, must synthesise every task from scratch.
  B. CULTURED agent— inherited the skill library the population accumulated while
                     solving the *training* tasks; solves held-out tasks by
                     recombining inherited pieces.
  C. REAL LLM      — what you'd do today: call an LLM per row. Accuracy assumed
                     high; cost computed from token accounting (Haiku 4.5 pricing).

The headline is the gap between A and B on the composite held-out tasks: those need
two parametric ops chained, which the from-scratch search cannot compose, but the
cultured agent reaches by recombination. That gap is the accumulated culture doing
real capability lifting — on work an LLM is paid to do per row.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from echo_civilization.echofill import apply_program, program_str
from echo_civilization.echofill_civ import (EchofillAgent, WrangleCulture,
                                            WrangleSkill)
from echo_civilization.wrangle_suite import HELDOUT, TRAIN

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

# Budget: candidate programs an agent may test per task. Generous enough that the
# from-scratch search runs to its natural ceiling (it exhausts ~820 candidates and
# still can't compose two parametric ops), so the gap is a capability gap, not a
# budget artefact.
BUDGET = 2000
MAX_DEPTH = 3

# Haiku 4.5 list pricing (2026-01), used for the cost baseline.
HAIKU_IN_PER_M = 1.00     # $ / 1M input tokens
HAIKU_OUT_PER_M = 5.00    # $ / 1M output tokens


# --------------------------------------------------------------------------- #
# Phase 1 — the population learns the TRAINING tasks; winning pieces -> culture.
# --------------------------------------------------------------------------- #

def build_culture():
    culture = WrangleCulture()
    # a small founder population each takes a slice of the training tasks
    founders = [EchofillAgent(generation=0) for _ in range(3)]
    for i, task in enumerate(TRAIN):
        agent = founders[i % len(founders)]
        # solve from scratch (empty library at first); contribute the piece
        pred, res = agent.solve(task.train, task.test[0][0], budget=BUDGET,
                                generation=0, max_depth=MAX_DEPTH)
        if res.solved:
            skill = WrangleSkill(program=res.program, creator=agent.id,
                                 generation=0, examples=task.train[:2])
            culture.contribute(skill)
    return culture


def make_cultured_agent(culture, generation=1):
    """A gen-1 child inheriting the accumulated cultural pieces."""
    child = EchofillAgent(generation=generation)
    for skill in culture.top():
        child.learn(WrangleSkill(program=skill.program, creator=skill.creator,
                                 generation=skill.generation,
                                 examples=skill.examples,
                                 reputation=skill.reputation))
    return child


# --------------------------------------------------------------------------- #
# Scoring one arm on the held-out suite.
# --------------------------------------------------------------------------- #

def score_agent(make_agent, culture=None):
    rows = []
    for task in HELDOUT:
        agent = make_agent()
        # learn the rule from the few-shot demo, predict the FIRST held-out row
        first_in = task.test[0][0]
        t0 = time.perf_counter()
        pred, res = agent.solve(task.train, first_in, budget=BUDGET,
                                generation=1, max_depth=MAX_DEPTH,
                                learn_at_solve=False)
        infer_us = (time.perf_counter() - t0) * 1e6
        # apply the learned program to ALL held-out rows and score exact-match
        if res.solved:
            prog = list(res.program)
            correct = sum(apply_program(prog, i) == o for i, o in task.test)
        else:
            correct = 0
        rows.append({
            "task": task.name, "kind": task.kind,
            "solved": res.solved, "via": res.via,
            "n_test": len(task.test), "correct": correct,
            "accuracy": correct / len(task.test),
            "evals": res.evals, "infer_us": infer_us,
            "program": program_str(list(res.program)) if res.solved else None,
            "note": task.note,
        })
    return rows


# --------------------------------------------------------------------------- #
# Arm C — the LLM cost model. We do NOT fabricate accuracy: an LLM handles these
# trivially, so we credit it 100%. The honest, checkable number is the COST, from
# a conservative per-row token estimate.
# --------------------------------------------------------------------------- #

def llm_cost_model(n_rows, tokens_in_per_row=90, tokens_out_per_row=8):
    """Per-row prompt: a short instruction + a few-shot demo + the input cell.
    ~90 input tokens and ~8 output tokens per row is conservative for a batched
    'apply this transform to this cell' call; a naive one-call-per-row workflow
    without prompt caching is far higher."""
    tin = n_rows * tokens_in_per_row
    tout = n_rows * tokens_out_per_row
    cost = tin / 1e6 * HAIKU_IN_PER_M + tout / 1e6 * HAIKU_OUT_PER_M
    return {"tokens_in": tin, "tokens_out": tout, "cost_usd": cost,
            "tokens_in_per_row": tokens_in_per_row,
            "tokens_out_per_row": tokens_out_per_row}


def main():
    print("Echo Civilization — wrangling benchmark (naive vs cultured vs LLM)\n")

    culture = build_culture()
    print(f"Culture accumulated {culture.size()} pieces from {len(TRAIN)} "
          f"training tasks:")
    for s in culture.top():
        print(f"   {program_str(list(s.program)):40}  (creator {s.creator})")
    print()

    naive = score_agent(lambda: EchofillAgent(generation=1))
    cultured = score_agent(lambda: make_cultured_agent(culture))

    # --- per-task table ---
    hdr = f"{'held-out task':22} {'kind':10} {'naive':>7} {'cultured':>9}  via(cultured)"
    print(hdr)
    print("-" * len(hdr))
    naive_solved = cultured_solved = 0
    for n, c in zip(naive, cultured):
        naive_solved += n["solved"]
        cultured_solved += c["solved"]
        print(f"{n['task']:22} {n['kind']:10} "
              f"{n['accuracy']*100:6.0f}% {c['accuracy']*100:8.0f}%  {c['via']}")
    print()

    # --- aggregate + cost ---
    n_tasks = len(HELDOUT)
    total_test_rows = sum(r["n_test"] for r in cultured)
    naive_acc = sum(r["correct"] for r in naive) / total_test_rows
    cult_acc = sum(r["correct"] for r in cultured) / total_test_rows
    avg_us = sum(r["infer_us"] for r in cultured) / len(cultured)

    # cost projected to a realistic column size
    N = 100_000
    llm = llm_cost_model(N)
    print(f"Held-out tasks solved:   naive {naive_solved}/{n_tasks}   "
          f"cultured {cultured_solved}/{n_tasks}")
    print(f"Held-out row accuracy:   naive {naive_acc*100:.0f}%   "
          f"cultured {cult_acc*100:.0f}%   (LLM ~100%)")
    print(f"Cultured inference:      ~{avg_us:.1f} µs/row, $0 marginal, "
          f"deterministic")
    print(f"LLM on {N:,} rows:       ${llm['cost_usd']:.2f} "
          f"(~{llm['tokens_in_per_row']} in / {llm['tokens_out_per_row']} out "
          f"tokens per row, Haiku 4.5)")
    print(f"Cultured on {N:,} rows:  $0.00")

    out = {
        "culture": [program_str(list(s.program)) for s in culture.top()],
        "naive": naive, "cultured": cultured,
        "summary": {
            "n_heldout_tasks": n_tasks,
            "naive_tasks_solved": naive_solved,
            "cultured_tasks_solved": cultured_solved,
            "naive_row_accuracy": naive_acc,
            "cultured_row_accuracy": cult_acc,
            "avg_infer_us": avg_us,
            "llm_cost_100k": llm["cost_usd"],
            "llm_tokens_per_row": [llm["tokens_in_per_row"], llm["tokens_out_per_row"]],
        },
    }
    (RESULTS / "echofill_bench.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote results/echofill_bench.json")
    return out


if __name__ == "__main__":
    main()
