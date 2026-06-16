#!/usr/bin/env python3
"""Echo Civilization — run the full research pipeline.

Runs the four baseline experiments (A/B/C/D), the four subsystem demos
(Echo/Memory/Grid/Social), logs everything to SQLite, renders all figures, and
generates research_report.md.

Usage:
    python run_experiments.py [--generations N] [--population N] [--budget N]
                              [--quick]
"""

from __future__ import annotations

import argparse
import time

from echo_civilization import demos, visualization as viz
from echo_civilization.database import Database
from echo_civilization.evaluation import run_all_experiments
from echo_civilization.report import generate_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=30)
    ap.add_argument("--population", type=int, default=24)
    ap.add_argument("--budget", type=int, default=35)
    ap.add_argument("--tasks-per-agent", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--db", default="results/echo_civilization.db")
    ap.add_argument("--quick", action="store_true",
                    help="small/fast run for smoke-testing")
    args = ap.parse_args()

    if args.quick:
        args.generations, args.population = 12, 14

    t0 = time.time()
    print("=" * 70)
    print("ECHO CIVILIZATION — artificial civilization laboratory")
    print("=" * 70)

    db = Database(args.db)

    # 1. baseline experiments A/B/C/D ------------------------------------
    print("\n[1/4] Running baseline experiments A/B/C/D ...")
    results = run_all_experiments(
        db=db, generations=args.generations, population_size=args.population,
        budget=args.budget, tasks_per_agent=args.tasks_per_agent, seed=args.seed)
    for name, r in results.items():
        cc = r["capability_curve"]
        print(f"    {name:30s} capability {cc[0]:.2f} -> {cc[-1]:.2f}"
              f"  (culture={r['culture'].size()})")

    # 2. subsystem demos -------------------------------------------------
    print("\n[2/4] Running subsystem demos (Echo / Memory / Grid / Social) ...")
    demos_out = {
        "echo": demos.echo_qlearning_demo(),
        "memory": demos.memory_demo(),
        "grid": demos.grid_evolution_demo(),
        "social": demos.social_demo(),
    }
    print(f"    echo mastery@episode {demos_out['echo']['episodes_to_mastery']}, "
          f"grid {demos_out['grid']['initial_best']:.1f}->"
          f"{demos_out['grid']['final_best']:.1f}, "
          f"social acc {demos_out['social']['final_accuracy']:.2f}")

    # 3. figures ---------------------------------------------------------
    print("\n[3/4] Rendering figures ...")
    D = results["D_full_civilization"]
    figures = {}
    figures["avg"] = viz.plot_average_intelligence(results, "figures/01_average_intelligence.png")
    figures["best"] = viz.plot_best_performance(results, "figures/02_best_performance.png")
    figures["prop"] = viz.plot_skill_propagation(D["culture"], "figures/03_skill_propagation.png")
    figures["rel"] = viz.plot_relationship_network(D["population"], "figures/04_relationship_network.png")
    figures["complexity"] = viz.plot_complexity_over_time(results, "figures/05_complexity_over_time.png")
    figures["culture"] = viz.plot_culture_growth(results, "figures/06_culture_growth.png")
    figures["echo"] = viz.plot_echo_learning(demos_out["echo"]["curve"], "figures/07_echo_learning.png")
    figures["mem"] = plot_memory(demos_out["memory"], "figures/08_memory_forgetting.png")
    figures["grid"] = viz.plot_grid_evolution(demos_out["grid"]["curve"], "figures/09_grid_evolution.png")
    figures["social"] = viz.plot_social_emergence(demos_out["social"], "figures/10_social_emergence.png")
    figures["diff"] = viz.plot_difficulty_breakdown(D, "figures/11_difficulty_breakdown.png")
    print(f"    wrote {len(figures)} figures to figures/")

    # 4. report ----------------------------------------------------------
    print("\n[4/4] Generating research_report.md ...")
    path = generate_report(results, demos_out, figures, "research_report.md")
    db.close()

    print(f"\nDone in {time.time() - t0:.1f}s.")
    print(f"  report : {path}")
    print(f"  data   : {args.db}")
    print(f"  figures: figures/")


def plot_memory(mem, path):
    """Small inline helper for the forgetting-curve figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ret = mem["retention_by_delay"]
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = sorted(ret)
    ax.plot(xs, [ret[x] for x in xs], marker="o", color="#2a9d8f")
    ax.set_title("Memory World: forgetting curve (retention vs. interference delay)")
    ax.set_xlabel("interfering steps before recall")
    ax.set_ylabel("mean recall strength")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
