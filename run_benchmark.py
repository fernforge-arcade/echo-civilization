#!/usr/bin/env python3
"""Run the Computer-Use Benchmark and emit results + a figure.

Answers the operator's question — "do they actually become computer-use agents?"
— by running a CULTURED agent (carrying a civilization's accumulated macro
library) and a FRESH gen-0 agent against a graded ladder of real computer
projects, from "move this file" to "write a web app", grading the reachable
rungs by EXECUTING the agents' programs as real shell commands.

    ./venv/bin/python run_benchmark.py            # full
    ./venv/bin/python run_benchmark.py --quick    # smaller/faster

Outputs:
    results/benchmark.json
    figures/18_computer_use_benchmark.png
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from echo_civilization.computer_use_benchmark import (
    LADDER, BEYOND, OPEN_ENDED, BENCH_PRIMITIVES, make_instance,
    sim_evaluate_factory, grade_on_real_shell, oracle_best_score)
from echo_civilization.computer_evolution import (ComputerCivilization,
                                                  ComputerConfig)
from echo_civilization.environments.real_computer_world import RealComputerWorld
from echo_civilization.synthesis import synthesize


def train_culture(seed=0, generations=20, quick=False):
    """Run a Computer-World civilization and return its accumulated macro library
    (list of op-programs), best-first by reputation/complexity."""
    if quick:
        generations = 10
    cfg = ComputerConfig(name="bench_culture", generations=generations,
                         population_size=20, budget=150, tasks_per_agent=10,
                         seed=seed)
    civ = ComputerCivilization(cfg)
    civ.run()
    skills = sorted(civ.culture.skills.values(),
                    key=lambda s: (-s.reputation, s.complexity()))
    macros = [tuple(s.program) for s in skills]
    return macros, civ


def run_rung(rung, cultured_macros, world, rng, n_trials, budget, max_depth):
    """Run both agents on a runnable rung; grade on the real shell."""
    res = {"cultured": [], "fresh": [], "cultured_calls": [], "fresh_calls": []}
    for _ in range(n_trials):
        inst = make_instance(rung, rng)
        evaluate = sim_evaluate_factory(inst)
        # CULTURED: knows the civilization's macro library
        sc = synthesize(cultured_macros, BENCH_PRIMITIVES, evaluate, budget, rng,
                        max_depth=max_depth, discovery_sample=400)
        solved_c, calls_c = grade_on_real_shell(sc.program, inst, world)
        # FRESH: empty library
        sf = synthesize([], BENCH_PRIMITIVES, evaluate, budget, rng,
                        max_depth=max_depth, discovery_sample=400)
        solved_f, calls_f = grade_on_real_shell(sf.program, inst, world)
        res["cultured"].append(int(solved_c))
        res["fresh"].append(int(solved_f))
        if solved_c:
            res["cultured_calls"].append(calls_c)
        if solved_f:
            res["fresh_calls"].append(calls_f)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=8)
    args = ap.parse_args()

    n_trials = 4 if args.quick else args.trials
    budget = 4000
    max_depth = 5

    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    rng = np.random.default_rng(args.seed)
    world = RealComputerWorld(rng)

    print("== Training a Computer-World civilization to build culture ==")
    cultured_macros, civ = train_culture(seed=args.seed, quick=args.quick)
    frontier = civ.history[-1]["frontier"] if civ.history else 0
    print(f"   culture size={len(cultured_macros)} macros; "
          f"final curriculum frontier reached = L{frontier}")

    report = {"meta": {"seed": args.seed, "trials": n_trials, "budget": budget,
                       "max_depth": max_depth, "n_macros": len(cultured_macros),
                       "frontier_reached": frontier},
              "rungs": []}

    print("\n== Reachable rungs (graded on the REAL shell) ==")
    for rung in LADDER + BEYOND:
        r = run_rung(rung, cultured_macros, world, rng, n_trials, budget, max_depth)
        cult = float(np.mean(r["cultured"]))
        fresh = float(np.mean(r["fresh"]))
        # Ladder rungs are reachable by construction (a canonical program exists);
        # for the BEYOND rungs we PROVE unreachability with a bounded oracle search
        # over the entire op vocabulary.
        if rung.canonical is not None:
            ceil = 1.0
        else:
            ceil, _ = oracle_best_score(rung, world, n_tasks=6, max_depth=4, rng=rng)
        entry = {
            "tier": rung.tier, "name": rung.name, "blurb": rung.blurb,
            "cultured_solve": cult, "fresh_solve": fresh,
            "cultured_calls": (float(np.mean(r["cultured_calls"]))
                               if r["cultured_calls"] else None),
            "fresh_calls": (float(np.mean(r["fresh_calls"]))
                            if r["fresh_calls"] else None),
            "reachable": ceil >= 0.999, "oracle_ceiling": round(ceil, 3),
            "runnable": True,
        }
        report["rungs"].append(entry)
        flag = "" if entry["reachable"] else "   <- UNREACHABLE (out of op-vocabulary)"
        print(f"  T{rung.tier} {rung.name:18s} cultured={cult:.2f} fresh={fresh:.2f} "
              f"ceiling={ceil:.2f}{flag}")

    print("\n== Open-ended software projects (not executable in this world) ==")
    for rung in OPEN_ENDED:
        entry = {"tier": rung.tier, "name": rung.name, "blurb": rung.blurb,
                 "cultured_solve": 0.0, "fresh_solve": 0.0,
                 "cultured_calls": None, "fresh_calls": None,
                 "reachable": False, "oracle_ceiling": 0.0, "runnable": False}
        report["rungs"].append(entry)
        print(f"  T{rung.tier} {rung.name:20s} NOT REPRESENTABLE — needs open-ended code generation")

    with open("results/benchmark.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print("\nwrote results/benchmark.json")

    make_figure(report)
    print("wrote figures/18_computer_use_benchmark.png")


def make_figure(report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rungs = report["rungs"]
    labels = [f"T{r['tier']}\n{r['name']}" for r in rungs]
    cult = [r["cultured_solve"] for r in rungs]
    fresh = [r["fresh_solve"] for r in rungs]
    x = np.arange(len(rungs))
    w = 0.4

    fig, ax = plt.subplots(figsize=(15, 6))
    b1 = ax.bar(x - w / 2, cult, w, label="Cultured agent (inherited macro library)",
                color="#2a7ae2")
    b2 = ax.bar(x + w / 2, fresh, w, label="Fresh gen-0 agent (no culture)",
                color="#e2862a")
    ax.set_ylabel("Solve rate (graded on real shell)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Computer-Use Benchmark — how far up the ladder do the agents get?")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=0)

    # shade the unreachable region
    first_unreach = next((i for i, r in enumerate(rungs) if not r["reachable"]), len(rungs))
    if first_unreach < len(rungs):
        ax.axvspan(first_unreach - 0.5, len(rungs) - 0.5, color="#cccccc", alpha=0.35)
        ax.text((first_unreach + len(rungs) - 1) / 2, 0.55,
                "CAPABILITY CEILING\n(no op-program can express these —\n"
                "needs open-ended code generation)",
                ha="center", va="center", fontsize=9, color="#555555")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig("figures/18_computer_use_benchmark.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
