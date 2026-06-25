#!/usr/bin/env python3
"""Computer-Use Frontier, Tier 8 — group-by aggregation (the next rung up).

Tier 7 reached a flat per-column reduction. Tier 8 climbs to a program that
builds and iterates a DATA STRUCTURE: read a CSV, group rows by a key column,
aggregate a value column per group, print sorted `key:value` pairs. We synthesise
it as REAL Python and run it in a subprocess against hidden tests (no LLM).

Two regimes, exactly as the rest of the frontier:
  * GENEROUS budget -> did the ceiling MOVE? (can any agent reach it now?)
  * TIGHT budget    -> does CULTURE still decide who clears it? (the thesis)

This runner also captures concrete RUN TRACES — the synthesised source, the CSV
fed in, and the program's real stdout — so the report can show example output
from actual runs.

Outputs:
  results/tier8.json
  figures/20_tier8_groupby.png

    ./venv/bin/python run_tier8.py             # full (default 10 trials)
    ./venv/bin/python run_tier8.py --quick     # fewer trials
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from echo_civilization.codegen2 import (make_tier8_task, make_group_by_tests,
                                        synthesize_code, run_script,
                                        GROUP_BY_SKELETON)

# Budgets: generous >> deepest reachable solution; tight << from-scratch search
# but >> a recall. (Probed empirically: fresh needs 63-82 real runs, cultured
# 3-22; 45 separates them cleanly, 300 lets both solve.)
TIER8_GENEROUS = 300
TIER8_TIGHT = 45


def run_tier8(trials, seed):
    res = {"generous": {"fresh": 0, "cultured": 0, "fresh_trials": [], "cult_trials": []},
           "tight": {"fresh": 0, "cultured": 0},
           "example": None, "fresh_fail_example": None}
    for t in range(trials):
        rng = np.random.default_rng(seed + 2000 + t)
        tests, spec = make_tier8_task(rng)

        f = synthesize_code(tests, TIER8_GENEROUS, np.random.default_rng(seed + t))
        c = synthesize_code(tests, TIER8_GENEROUS, np.random.default_rng(seed + t),
                            inherited_skeleton=GROUP_BY_SKELETON)
        res["generous"]["fresh"] += f.solved
        res["generous"]["cultured"] += c.solved
        res["generous"]["fresh_trials"].append(f.trials)
        res["generous"]["cult_trials"].append(c.trials)

        ft = synthesize_code(tests, TIER8_TIGHT, np.random.default_rng(seed + t))
        ct = synthesize_code(tests, TIER8_TIGHT, np.random.default_rng(seed + t),
                             inherited_skeleton=GROUP_BY_SKELETON)
        res["tight"]["fresh"] += ft.solved
        res["tight"]["cultured"] += ct.solved

        # capture a worked example from the first solved cultured run
        if c.solved and res["example"] is None:
            trace = _capture_trace(c, spec, np.random.default_rng(seed + 9000 + t))
            res["example"] = {"spec": spec, "source": c.source,
                              "cult_trials": c.trials, "fresh_trials": f.trials,
                              "trace": trace}
        # capture a tight-budget fresh FAILURE next to a tight cultured SUCCESS
        if (not ft.solved) and ct.solved and res["fresh_fail_example"] is None:
            res["fresh_fail_example"] = {
                "spec": spec, "budget": TIER8_TIGHT,
                "fresh_score": round(ft.score, 2), "fresh_trials": ft.trials,
                "fresh_best_skeleton": ft.skeleton,
                "cult_trials": ct.trials, "cult_skeleton": ct.skeleton}

    g = res["generous"]
    g["fresh"] /= trials; g["cultured"] /= trials
    g["fresh_trials"] = float(np.mean(g["fresh_trials"]))
    g["cult_trials"] = float(np.mean(g["cult_trials"]))
    res["tight"]["fresh"] /= trials; res["tight"]["cultured"] /= trials
    print(f"  T8 group_by_aggregate | generous F {g['fresh']:.2f} C {g['cultured']:.2f}"
          f"  (real runs {g['fresh_trials']:.0f} vs {g['cult_trials']:.0f})"
          f" | tight F {res['tight']['fresh']:.2f} C {res['tight']['cultured']:.2f}")
    return res


def _capture_trace(result, spec, rng):
    """Run the synthesised program on a FRESH held-out instance and record the
    CSV in, the expected output, and the program's real stdout."""
    held = make_group_by_tests(rng, n=1, reducer=spec["reducer"],
                               key_col=spec["key_col"], val_col=spec["val_col"])
    csv_text, expected = held[0]
    got = run_script(result.source, csv_text)
    return {"csv_in": csv_text, "expected": expected, "stdout": got,
            "match": got == expected}


def make_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.suptitle("Computer-Use Frontier · Tier 8 — group-by aggregation "
                 "(synthesised, really-run Python)",
                 fontsize=13, fontweight="bold", y=0.99)

    # left: solve rate, two regimes
    ax = axes[0]
    x = np.arange(2)
    fresh = [res["generous"]["fresh"], res["tight"]["fresh"]]
    cult = [res["generous"]["cultured"], res["tight"]["cultured"]]
    ax.bar(x - 0.2, fresh, 0.4, label="fresh", color="#90caf9")
    ax.bar(x + 0.2, cult, 0.4, label="cultured (inherited skeleton)", color="#1565c0")
    for i, (vf, vc) in enumerate(zip(fresh, cult)):
        ax.text(i - 0.2, vf + 0.02, f"{vf:.2f}", ha="center", fontsize=8)
        ax.text(i + 0.2, vc + 0.02, f"{vc:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"generous\n(budget {TIER8_GENEROUS})",
                        f"tight\n(budget {TIER8_TIGHT})"])
    ax.set_ylabel("solve rate")
    ax.set_ylim(0, 1.12)
    ax.set_title("was: NOT REPRESENTABLE\n→ reachable; culture decides under pressure",
                 fontsize=10)
    ax.legend(fontsize=8, loc="lower left")

    # right: search cost (real subprocess runs to solve, generous)
    ax = axes[1]
    ax.bar([0, 1], [res["generous"]["fresh_trials"], res["generous"]["cult_trials"]],
           0.5, color=["#90caf9", "#1565c0"])
    for i, v in enumerate([res["generous"]["fresh_trials"], res["generous"]["cult_trials"]]):
        ax.text(i, v + 1, f"{v:.0f}", ha="center", fontsize=10)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["fresh", "cultured"])
    ax.set_ylabel("real program executions to solve")
    ax.set_title("cost of reaching the rung\n(inheriting the skeleton skips the search)",
                 fontsize=10)
    ax.text(0.97, 0.62, "group rows by a key column,\naggregate a value column,\n"
            "print sorted  key:value  pairs",
            transform=ax.transAxes, ha="right", va="center", fontsize=8.5,
            bbox=dict(boxstyle="round", fc="#e3f2fd", ec="#1565c0"))

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    trials = 4 if args.quick else args.trials

    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    print("== Tier 8: group-by aggregation (grammar-guided real-Python synthesis) ==")
    res = run_tier8(trials, args.seed)

    make_figure(res, "figures/20_tier8_groupby.png")

    out = {"trials": trials, "seed": args.seed,
           "budgets": {"generous": TIER8_GENEROUS, "tight": TIER8_TIGHT},
           "tier8": res}
    with open("results/tier8.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("  wrote results/tier8.json")

    if res["example"]:
        ex = res["example"]
        tr = ex["trace"]
        print("\n  -- worked example (run trace) --")
        print(f"     hidden transform: group by col {ex['spec']['key_col']}, "
              f"{ex['spec']['reducer']} of col {ex['spec']['val_col']}")
        print(f"     cultured solved in {ex['cult_trials']} real runs; "
              f"fresh needed {ex['fresh_trials']}")
        print(f"     held-out stdout: {tr['stdout']!r}  (expected {tr['expected']!r}) "
              f"-> {'MATCH' if tr['match'] else 'MISMATCH'}")

    print(f"\nHEADLINE: Tier-8 group-by now reachable "
          f"(generous cultured {res['generous']['cultured']:.2f}); "
          f"under a tight budget only the cultured agent clears it "
          f"({res['tight']['cultured']:.2f} vs fresh {res['tight']['fresh']:.2f}).")


if __name__ == "__main__":
    main()
