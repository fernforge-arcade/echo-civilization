"""Experiment N -- falsifying (or defending) the Experiment-M abstraction result.

Experiment M reported a from-scratch grid solver climbing 2.5% -> 85% -> 97.5% once it
could invent, name and stack its own concepts. That curve was measured on a benchmark
Echo designed for itself, scored on the same demonstration pairs the synthesis used. A
striking curve on a self-designed, self-scored benchmark deserves an adversarial audit.

This runner puts the M mechanism through a control battery whose whole purpose is to make
the invented library look unnecessary, on a HARDER task distribution than M:

  * every task carries an UNSEEN query grid -- synthesis sees only the demo pairs, and a
    "solve" requires the found program to also reproduce the held-out query (kills demo
    over-fitting);
  * train/eval are separated by held-out motif *pairings*, not sampled programs, and
    behaviourally-leaked held-out tasks are dropped (genuine disjointness);
  * matched conditions on ONE frozen held-out stream, multi-seed:
        fresh_flat      from-scratch search at the deployment budget
        generous_flat   from-scratch search, big budget + length oracle
        cache_only      persistent cache of solved WHOLE programs, retried verbatim
        invent          the learned fragment library (cumulative corpus)
        invent_guide    invent + numpy proposer
        motif_oracle    library = the 6 true motifs (LOWER bound: L1 not enough)
        full_oracle     motifs + all compounds as L2 (true UPPER bound / ceiling)
        shuffled_lib    size-matched USELESS random library (content vs branching)
  * a budget sweep (is the flat gap just a stingy budget?),
  * a negative-transfer sweep (cost of irrelevant concepts),
  * a civilization vs single-learner control (P=1 vs P=4, matched total solves),
  * promotion by MDL vs by measured held-out utility.

Raw numbers -> results/falsification.json (git-ignored). Figures 33-37. No pretrained
models; numpy + stdlib only.

    ./venv/bin/python run_falsification.py            # canonical (seeds 0,1,2)
    ./venv/bin/python run_falsification.py --quick    # 1 seed, fewer rounds, faster
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from echo_civilization import falsification as F
from echo_civilization.abstraction import Library, solve_task
from echo_civilization.arc_tasks_v2 import make_heldout
from echo_civilization.gridworld_arc import BASE_OP_NAMES

HELDOUT_SEED = 12345
RESULTS = "results/falsification.json"


def _agg_acc(runs):
    """Average per-round acc / train_consistent / mean_evals across seed runs."""
    n = len(runs[0])
    out = []
    for i in range(n):
        accs = [r[i]["acc"] for r in runs]
        tcs = [r[i].get("train_consistent_acc", r[i]["acc"]) for r in runs]
        evs = [r[i]["mean_evals"] for r in runs]
        row = {"round": i + 1,
               "acc_mean": float(np.mean(accs)), "acc_sd": float(np.std(accs)),
               "train_consistent_mean": float(np.mean(tcs)),
               "mean_evals": float(np.mean(evs))}
        for k in ("n_abstractions", "max_level", "corpus_size",
                  "lifetime_train_evals", "cache_size"):
            if k in runs[0][i]:
                row[k] = float(np.mean([r[i][k] for r in runs]))
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    seeds = [0] if args.quick else args.seeds
    n_rounds = 3 if args.quick else args.rounds
    n_heldout = 20 if args.quick else 40
    train_per_round = 60 if args.quick else 80
    generous_budget = 30000 if args.quick else 60000
    generous_len = 8
    cfg = {"budget": 3000, "max_len": 6, "train_per_round": train_per_round}

    t0 = time.time()
    raw_held = make_heldout(np.random.default_rng(HELDOUT_SEED), n_heldout)
    held, dropped, n_leak_progs = F.filter_behavioural_leaks(raw_held, cfg)
    print(f"held-out: {len(raw_held)} generated, {dropped} behaviourally-leaked dropped, "
          f"{len(held)} clean (depths {sorted(set(t.depth for t in held))})")
    print(f"base vocabulary: {len(BASE_OP_NAMES)} primitives; "
          f"seeds {seeds}; rounds {n_rounds}\n")

    data = {"config": {**cfg, "n_heldout_generated": n_heldout,
                       "n_heldout_clean": len(held), "leaked_dropped": dropped,
                       "seeds": seeds, "rounds": n_rounds,
                       "generous_budget": generous_budget,
                       "generous_len": generous_len},
            "conditions": {}}

    # ---- integrity gates (frozen held-out) --------------------------------- #
    data["integrity"] = F.integrity_checks(held, seeds[0], cfg)
    data["integrity"]["leak_programs_pooled"] = n_leak_progs
    print("integrity:", data["integrity"])

    # ---- seed-independent conditions (no training; deterministic) ---------- #
    def static(lib, budget=None, ml=None):
        h = F.eval_heldout(held, lib, budget or cfg["budget"], ml or cfg["max_len"])
        return [dict(h, round=r + 1) for r in range(n_rounds)]

    print("running deterministic conditions ...")
    det = {
        "fresh_flat": static(Library()),
        "generous_flat": static(Library(), generous_budget, generous_len),
        "motif_oracle": static(F.oracle_library()),
        "full_oracle": static(F.full_oracle_library()),
    }
    for k, v in det.items():
        data["conditions"][k] = {"per_round": v, "final_acc": v[-1]["acc"],
                                 "final_train_consistent": v[-1]["train_consistent_acc"]}
        print(f"  {k:16s} acc={v[-1]['acc']:.3f}  "
              f"train_consistent={v[-1]['train_consistent_acc']:.3f}  "
              f"mean_evals={v[-1]['mean_evals']:.0f}")

    # ---- seed-dependent conditions ----------------------------------------- #
    print("running seed-dependent conditions ...")
    seedcond = {"cache_only": [], "invent": [], "invent_guide": [],
                "invent_utility": [], "shuffled_lib": []}
    for s in seeds:
        seedcond["cache_only"].append(F.run_cache_only(s, n_rounds, held, cfg))
        seedcond["invent"].append(F.run_invent(s, n_rounds, held, cfg)[0])
        seedcond["invent_guide"].append(
            F.run_invent(s, n_rounds, held, cfg, guide=True)[0])
        seedcond["invent_utility"].append(
            F.run_invent(s, n_rounds, held, cfg, promotion="utility")[0])
        rng = np.random.default_rng(1000 + s)
        shuf = F.shuffled_library(rng, n_ops=9)   # size-matched to invent's ~9 concepts
        seedcond["shuffled_lib"].append(F.run_static_library(shuf, n_rounds, held, cfg))
        print(f"  seed {s} done ({time.time() - t0:.0f}s)")

    for k, runs in seedcond.items():
        agg = _agg_acc(runs)
        data["conditions"][k] = {"per_round": agg, "final_acc": agg[-1]["acc_mean"],
                                 "final_acc_sd": agg[-1]["acc_sd"],
                                 "final_train_consistent": agg[-1]["train_consistent_mean"],
                                 "seed0_detail": runs[0]}
        print(f"  {k:16s} acc={agg[-1]['acc_mean']:.3f} (sd {agg[-1]['acc_sd']:.3f})  "
              f"mean_evals={agg[-1]['mean_evals']:.0f}")

    # ---- budget sweep: is the flat gap a budget artefact? ------------------ #
    print("budget sweep ...")
    sweep_held = held[: min(20, len(held))]
    budgets = [500, 1500, 3000, 10000, 30000, 100000]
    data["budget_sweep"] = {
        "held_n": len(sweep_held),
        "len6": F.budget_sweep(0, sweep_held, cfg, budgets, max_len=6),
        "len8": F.budget_sweep(0, sweep_held, cfg, budgets, max_len=8),
    }
    for row in data["budget_sweep"]["len8"]:
        print(f"  flat len8 budget={row['budget']:>6d}  acc={row['acc']:.3f}  "
              f"mean_evals={row['mean_evals']:.0f}")

    # ---- negative transfer: cost of irrelevant concepts -------------------- #
    print("negative transfer ...")
    nt = [F.negative_transfer(s, sweep_held, cfg) for s in seeds]
    loads = [r["useless_added"] for r in nt[0]]
    data["negative_transfer"] = [
        {"useless_added": loads[i],
         "acc": float(np.mean([nt[s][i]["acc"] for s in range(len(seeds))])),
         "mean_evals": float(np.mean([nt[s][i]["mean_evals"] for s in range(len(seeds))])),
         "concepts_used": float(np.mean([nt[s][i]["concepts_used"] for s in range(len(seeds))])),
         "vocab_size": nt[0][i]["vocab_size"]}
        for i in range(len(loads))]
    for row in data["negative_transfer"]:
        print(f"  +{row['useless_added']:>2d} useless  acc={row['acc']:.3f}  "
              f"mean_evals={row['mean_evals']:.0f}  concepts_used={row['concepts_used']:.1f}")

    # ---- civilization vs single lifelong learner --------------------------- #
    print("civilization control (P=1 vs P=4) ...")
    per_gen = train_per_round
    n_gens = n_rounds
    pop = {}
    for P in (1, 4):
        runs = [F.run_population(s, n_gens, P, per_gen, held, cfg)[0] for s in seeds]
        n = len(runs[0])
        pop[f"P{P}"] = [
            {"gen": i + 1,
             "acc": float(np.mean([runs[s][i]["acc"] for s in range(len(seeds))])),
             "acc_sd": float(np.std([runs[s][i]["acc"] for s in range(len(seeds))])),
             "n_abstractions": float(np.mean([runs[s][i]["n_abstractions"] for s in range(len(seeds))])),
             "max_level": float(np.mean([runs[s][i]["max_level"] for s in range(len(seeds))])),
             "lifetime_train_evals": float(np.mean([runs[s][i]["lifetime_train_evals"] for s in range(len(seeds))]))}
            for i in range(n)]
        print(f"  P={P}: final acc {pop[f'P{P}'][-1]['acc']:.3f}, "
              f"abstractions {pop[f'P{P}'][-1]['n_abstractions']:.1f}")
    data["population"] = pop

    # ---- worked traces ----------------------------------------------------- #
    _, lib = F.run_invent(seeds[0], n_rounds, held, cfg, guide=True)
    traces = []
    for t in held:
        if t.depth < 4:
            continue
        r, tc, qc = F.solve_with_query(t, lib, cfg["budget"], cfg["max_len"])
        if qc:
            traces.append({
                "hidden_program": list(t.program),
                "hidden_depth_base_ops": len(t.program),
                "motif_idx": list(t.motif_idx),
                "solution_tokens": list(r.program),
                "solution_n_tokens": len(r.program),
                "evals_used": r.evals,
                "expands_to": [list(lib.expand_to_base((op,))) if op in lib.ops
                               else [op] for op in r.program]})
        if len(traces) >= 3:
            break
    data["worked_traces"] = traces

    os.makedirs("results", exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nwrote {RESULTS}  ({time.time() - t0:.0f}s total)")

    make_figures(data)


def make_figures(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C = data["conditions"]

    # Fig 33: all conditions, final query-correct acc (with overfit gap) ------ #
    order = ["fresh_flat", "cache_only", "shuffled_lib", "generous_flat",
             "motif_oracle", "invent", "invent_utility", "invent_guide", "full_oracle"]
    labels = {"fresh_flat": "flat\n(deploy)", "cache_only": "cache\nonly",
              "shuffled_lib": "shuffled\nlib", "generous_flat": "generous\nflat",
              "motif_oracle": "motif\noracle (L1)", "invent": "INVENT",
              "invent_utility": "invent\n(utility)", "invent_guide": "invent\n+guide",
              "full_oracle": "full\noracle"}
    accs = [C[k]["final_acc"] for k in order]
    tcs = [C[k]["final_train_consistent"] for k in order]
    colors = ["#999999", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
              "#1f77b4", "#17becf", "#d62728", "#2ca02c"]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(order))
    ax.bar(x, tcs, width=0.62, color="#dddddd", label="fits demos (train-consistent)")
    ax.bar(x, accs, width=0.62, color=colors, label="query-correct (real solve)")
    for xi, a in zip(x, accs):
        ax.text(xi, a + 0.02, f"{a:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[k] for k in order], fontsize=8)
    ax.set_ylabel("held-out solve rate (unseen query grid)")
    ax.set_ylim(0, 1.08)
    ax.set_title("Experiment N: does the invented library survive query scoring and controls?")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig("figures/33_falsification_conditions.png", dpi=130)
    plt.close(fig)

    # Fig 34: budget sweep --------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for key, mk, lab in [("len6", "-o", "flat, max_len 6"),
                         ("len8", "-s", "flat, max_len 8 (length oracle)")]:
        rows = data["budget_sweep"][key]
        ax.plot([r["budget"] for r in rows], [r["acc"] for r in rows], mk, label=lab)
    ax.axhline(C["invent_guide"]["final_acc"], color="#d62728", ls="--",
               label=f"invent+guide ({C['invent_guide']['final_acc']:.2f}, budget 3000)")
    ax.axhline(C["full_oracle"]["final_acc"], color="#2ca02c", ls=":",
               label=f"full oracle ceiling ({C['full_oracle']['final_acc']:.2f})")
    ax.set_xscale("log")
    ax.set_xlabel("flat search budget (evaluations per task, log scale)")
    ax.set_ylabel("held-out solve rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Can a generous flat solver reach the ceiling? (it hits a wall)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/34_falsification_budget.png", dpi=130)
    plt.close(fig)

    # Fig 35: negative transfer ---------------------------------------------- #
    nt = data["negative_transfer"]
    fig, ax1 = plt.subplots(figsize=(7.5, 4.8))
    xs = [r["useless_added"] for r in nt]
    ax1.plot(xs, [r["acc"] for r in nt], "-o", color="#1f77b4", label="accuracy")
    ax1.set_xlabel("irrelevant concepts bolted onto the oracle library")
    ax1.set_ylabel("held-out accuracy", color="#1f77b4")
    ax1.set_ylim(0, 1.05)
    ax2 = ax1.twinx()
    ax2.plot(xs, [r["mean_evals"] for r in nt], "-s", color="#ff7f0e",
             label="mean search evals")
    ax2.set_ylabel("mean search evals per task", color="#ff7f0e")
    ax1.set_title("Negative transfer: useless concepts raise search cost")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/35_falsification_negtransfer.png", dpi=130)
    plt.close(fig)

    # Fig 36: civilization vs single learner --------------------------------- #
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for P, col in [("P1", "#9467bd"), ("P4", "#2ca02c")]:
        rows = data["population"][P]
        lab = "single lifelong learner (P=1)" if P == "P1" else "civilization (P=4, shared culture)"
        ax.plot([r["gen"] for r in rows], [r["acc"] for r in rows], "-o", color=col, label=lab)
    ax.set_xlabel("generation (matched total training solves)")
    ax.set_ylabel("held-out solve rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Is it civilization or just individual lifelong learning?")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/36_falsification_population.png", dpi=130)
    plt.close(fig)

    # Fig 37: MDL vs measured-utility promotion ------------------------------ #
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for k, col, lab in [("invent", "#1f77b4", "promotion = MDL"),
                        ("invent_utility", "#17becf", "promotion = measured utility")]:
        pr = C[k]["per_round"]
        ax.plot([p["round"] for p in pr], [p["acc_mean"] for p in pr], "-o",
                color=col, label=lab)
    ax.set_xlabel("round of experience")
    ax.set_ylabel("held-out solve rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Promotion rule: compression vs measured held-out utility")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/37_falsification_promotion.png", dpi=130)
    plt.close(fig)

    print("wrote figures/33..37")


if __name__ == "__main__":
    main()
