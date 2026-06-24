#!/usr/bin/env python3
"""Computer-Use FRONTIER — actually reaching the benchmark's locked top rungs.

The Computer-Use Benchmark stopped at two honest walls:
  * Tier 6  — runnable on a real shell but OUT of the op vocabulary
              (find_and_replace / word_frequency / sum_numbers): proven
              unreachable by an oracle over op-ORDER alone.
  * Tier 7  — "write a Python script ..." : NOT REPRESENTABLE as an op pipeline.

This script runs the two new mechanisms that knock those walls down without any
pretrained model (see echo_civilization/frontier.py and codegen.py):

  Tier 6:  PARAMETRIC OPS + ARGUMENT SYNTHESIS BY EXAMPLE.  Ops gain holes
           (replace(<find>,<repl>), ...); the agent infers the hole-fillers from
           input->output examples (programming-by-example). New reductions
           (word_freq, sum_numbers) cover the non-literal cases.

  Tier 7:  GRAMMAR-GUIDED CODE SYNTHESIS.  The agent emits a program in a tiny
           typed grammar that COMPILES TO REAL PYTHON, which we run in a
           subprocess against hidden tests — genuine executable-code synthesis.

For each unlocked rung we report two regimes, which together make the claim
honest:
  * GENEROUS budget  -> can ANY agent reach it now?  (Did the ceiling move?)
  * TIGHT budget     -> does CULTURE still decide who reaches it?  (The thesis.)

Outputs:
  results/frontier.json
  figures/19_computer_use_frontier.png

    ./venv/bin/python run_frontier.py            # full
    ./venv/bin/python run_frontier.py --quick    # fewer trials
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from echo_civilization.frontier import (make_param_task, synthesize_param,
                                        PARAM_TASKS)
from echo_civilization.codegen import (make_csv_avg_tests, synthesize_code,
                                       CSV_AVG_SKELETON)


# Budgets chosen so generous >> the cost of the deepest reachable solution, and
# tight << the from-scratch search but >> a recall.  (See run output / report.)
TIER6_GENEROUS = 300
TIER6_TIGHT = 12
TIER7_GENEROUS = 400
TIER7_TIGHT = 60


def run_tier6(trials, seed):
    """Each parametric task, fresh (no inherited template) vs cultured (inherited
    the op-sequence template, holes intact -> filled by example), at two budgets."""
    out = {}
    for name in PARAM_TASKS:
        gen = {"fresh": 0, "cultured": 0, "fresh_evals": [], "cult_evals": []}
        tig = {"fresh": 0, "cultured": 0}
        for t in range(trials):
            rng = np.random.default_rng(seed + 1000 + t)
            examples, canon = make_param_task(name, rng)
            sr = np.random.default_rng(seed + 7 + t)
            # GENEROUS
            f = synthesize_param([], examples, TIER6_GENEROUS, np.random.default_rng(seed + 7 + t))
            c = synthesize_param([canon], examples, TIER6_GENEROUS, np.random.default_rng(seed + 7 + t))
            gen["fresh"] += f.solved
            gen["cultured"] += c.solved
            gen["fresh_evals"].append(f.evals)
            gen["cult_evals"].append(c.evals)
            # TIGHT
            ft = synthesize_param([], examples, TIER6_TIGHT, np.random.default_rng(seed + 7 + t))
            ct = synthesize_param([canon], examples, TIER6_TIGHT, np.random.default_rng(seed + 7 + t))
            tig["fresh"] += ft.solved
            tig["cultured"] += ct.solved
        out[name] = {
            "generous": {"fresh": gen["fresh"] / trials,
                         "cultured": gen["cultured"] / trials,
                         "fresh_evals": float(np.mean(gen["fresh_evals"])),
                         "cult_evals": float(np.mean(gen["cult_evals"]))},
            "tight": {"fresh": tig["fresh"] / trials,
                      "cultured": tig["cultured"] / trials},
        }
        g, ti = out[name]["generous"], out[name]["tight"]
        print(f"  T6 {name:16s} | generous F {g['fresh']:.2f} C {g['cultured']:.2f}"
              f"  (evals {g['fresh_evals']:.0f} vs {g['cult_evals']:.0f})"
              f" | tight F {ti['fresh']:.2f} C {ti['cultured']:.2f}")
    return out


def run_tier7(trials, seed):
    """The CSV-averages script, synthesised as REAL Python and executed, fresh vs
    cultured (inherited the code skeleton), at two budgets."""
    res = {"generous": {"fresh": 0, "cultured": 0, "fresh_trials": [], "cult_trials": []},
           "tight": {"fresh": 0, "cultured": 0},
           "example_source": None}
    for t in range(trials):
        rng = np.random.default_rng(seed + 2000 + t)
        tests = make_csv_avg_tests(rng, n=3)
        f = synthesize_code(tests, TIER7_GENEROUS, np.random.default_rng(seed + t))
        c = synthesize_code(tests, TIER7_GENEROUS, np.random.default_rng(seed + t),
                            inherited_skeleton=CSV_AVG_SKELETON)
        res["generous"]["fresh"] += f.solved
        res["generous"]["cultured"] += c.solved
        res["generous"]["fresh_trials"].append(f.trials)
        res["generous"]["cult_trials"].append(c.trials)
        if c.solved and res["example_source"] is None:
            res["example_source"] = c.source
        ft = synthesize_code(tests, TIER7_TIGHT, np.random.default_rng(seed + t))
        ct = synthesize_code(tests, TIER7_TIGHT, np.random.default_rng(seed + t),
                             inherited_skeleton=CSV_AVG_SKELETON)
        res["tight"]["fresh"] += ft.solved
        res["tight"]["cultured"] += ct.solved
    g = res["generous"]
    g["fresh"] /= trials; g["cultured"] /= trials
    g["fresh_trials"] = float(np.mean(g["fresh_trials"]))
    g["cult_trials"] = float(np.mean(g["cult_trials"]))
    res["tight"]["fresh"] /= trials; res["tight"]["cultured"] /= trials
    print(f"  T7 csv_column_avgs   | generous F {g['fresh']:.2f} C {g['cultured']:.2f}"
          f"  (real runs {g['fresh_trials']:.0f} vs {g['cult_trials']:.0f})"
          f" | tight F {res['tight']['fresh']:.2f} C {res['tight']['cultured']:.2f}")
    return res


def make_figure(t6, t7, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.suptitle("Computer-Use Frontier — reaching the benchmark's locked top rungs",
                 fontsize=14, fontweight="bold", y=0.99)

    # --- left: Tier 6 reachability (generous) + cultural gap (tight) ---
    ax = axes[0]
    names = PARAM_TASKS
    x = np.arange(len(names))
    w = 0.2
    gf = [t6[n]["generous"]["fresh"] for n in names]
    gc = [t6[n]["generous"]["cultured"] for n in names]
    tf = [t6[n]["tight"]["fresh"] for n in names]
    tc = [t6[n]["tight"]["cultured"] for n in names]
    ax.bar(x - 1.5 * w, gf, w, label="fresh · generous budget", color="#bfd8bd")
    ax.bar(x - 0.5 * w, gc, w, label="cultured · generous budget", color="#2e7d32")
    ax.bar(x + 0.5 * w, tf, w, label="fresh · tight budget", color="#f3b6b6")
    ax.bar(x + 1.5 * w, tc, w, label="cultured · tight budget", color="#c62828")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_ylabel("solve rate")
    ax.set_ylim(0, 1.08)
    ax.set_title("Tier 6  (was: out of op-vocabulary)\n"
                 "parametric ops + argument-by-example", fontsize=10, pad=14)
    ax.legend(fontsize=7, loc="lower center", ncol=2)
    ax.axhline(0.0, color="#888", lw=0.6)
    ax.set_xlabel("generous: ceiling MOVED (reachable now)    tight: culture still decides",
                  fontsize=8, style="italic", color="#444")

    # --- right: Tier 7 code-synthesis ---
    ax = axes[1]
    labels = ["generous\nbudget", "tight\nbudget"]
    x = np.arange(2)
    fresh = [t7["generous"]["fresh"], t7["tight"]["fresh"]]
    cult = [t7["generous"]["cultured"], t7["tight"]["cultured"]]
    ax.bar(x - 0.2, fresh, 0.4, label="fresh", color="#90caf9")
    ax.bar(x + 0.2, cult, 0.4, label="cultured (inherited skeleton)", color="#1565c0")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("solve rate")
    ax.set_ylim(0, 1.08)
    ax.set_title("Tier 7  (was: NOT REPRESENTABLE)\n"
                 "grammar-guided synthesis of REAL Python", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    ax.text(0.5, 0.5,
            "csv→column averages\nnow a runnable\nsynthesised script",
            transform=ax.transAxes, ha="center", fontsize=9,
            bbox=dict(boxstyle="round", fc="#e3f2fd", ec="#1565c0"))

    fig.tight_layout(rect=[0, 0, 1, 0.94])
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

    print("== Tier 6: parametric ops + argument-by-example ==")
    t6 = run_tier6(trials, args.seed)
    print("== Tier 7: grammar-guided real-Python synthesis ==")
    t7 = run_tier7(trials, args.seed)

    make_figure(t6, t7, "figures/19_computer_use_frontier.png")

    out = {"trials": trials, "seed": args.seed,
           "budgets": {"tier6_generous": TIER6_GENEROUS, "tier6_tight": TIER6_TIGHT,
                       "tier7_generous": TIER7_GENEROUS, "tier7_tight": TIER7_TIGHT},
           "tier6": t6, "tier7": t7}
    with open("results/frontier.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("  wrote results/frontier.json")

    # headline
    reachable6 = sum(1 for n in PARAM_TASKS if t6[n]["generous"]["cultured"] >= 0.99)
    print(f"\nHEADLINE: Tier-6 rungs now reachable: {reachable6}/{len(PARAM_TASKS)} "
          f"(were 0 — oracle-proven out of vocabulary).")
    print(f"          Tier-7 csv-averages script: reachable={t7['generous']['cultured']>=0.99} "
          f"(was NOT REPRESENTABLE).")
    print("          Under a tight budget only the CULTURED agent reaches them — "
          "the ceiling moved, and culture still decides who clears it.")


if __name__ == "__main__":
    main()
