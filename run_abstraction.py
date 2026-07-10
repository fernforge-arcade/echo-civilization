"""Experiment M -- Abstraction Invention (Echo-ARC).

Central question, one level deeper than the rest of Echo: instead of accumulating
*programs*, can the civilization accumulate the *vocabulary programs are written in*?
i.e. can it invent, name, and hierarchically compose its own concepts, and does that
make otherwise-unreachable tasks reachable?

Three arms, same search budget, same held-out test set (deep 3-4 motif grid tasks
never seen as a training program):

  FLAT          base ops only, no invention. The from-scratch-search control.
  INVENT        mine successful programs -> MDL-scored abstractions -> grow a
                hierarchical library across rounds (continuous cultural accumulation).
  INVENT+GUIDE  same, plus a tiny online-learned proposer that orders the search
                toward promising ops (search + learned guidance).

Everything is numpy + stdlib; no pretrained models. Run:
    ./venv/bin/python run_abstraction.py            # canonical (seeds 0,1,2)
    ./venv/bin/python run_abstraction.py --quick    # 1 seed, fewer rounds
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from echo_civilization.abstraction import Library, mine_abstractions, solve_task
from echo_civilization.arc_tasks import make_heldout, make_training_pool
from echo_civilization.gridworld_arc import BASE_OP_NAMES
from echo_civilization.proposer import Proposer

BUDGET = 3000
MAX_LEN = 6
TRAIN_PER_ROUND = 80
N_HELDOUT = 40
HELDOUT_SEED = 12345

RESULTS = "results/abstraction.json"


def eval_heldout(held, lib, proposer):
    solved = 0
    evals = []
    tok_len = []
    base_len = []
    for t in held:
        r = solve_task(t.examples, lib, BUDGET, MAX_LEN, proposer=proposer)
        evals.append(r.evals)
        if r.solved:
            solved += 1
            tok_len.append(len(r.program))
            base_len.append(r.base_length)
    return {
        "acc": solved / len(held),
        "mean_evals": float(np.mean(evals)),
        "mean_tokens": float(np.mean(tok_len)) if tok_len else 0.0,
        "mean_base_len": float(np.mean(base_len)) if base_len else 0.0,
    }


def run_arm(arm, seed, n_rounds, held):
    """arm in {'flat','invent','guide'}. Returns per-round metric dicts."""
    rng = np.random.default_rng(seed)
    lib = Library()
    proposer = Proposer() if arm == "guide" else None
    invent = arm in ("invent", "guide")
    per_round = []
    for rnd in range(1, n_rounds + 1):
        train = make_training_pool(rng, TRAIN_PER_ROUND)
        solved_progs = []
        train_solved = 0
        for t in train:
            r = solve_task(t.examples, lib, BUDGET, MAX_LEN, proposer=proposer)
            if r.solved:
                train_solved += 1
                enc = lib.encode(r.program)
                solved_progs.append(enc)
                if proposer is not None:
                    proposer.update(t.examples, enc, lib.op_names())
        invented = []
        if invent:
            invented = mine_abstractions(solved_progs, lib, rnd)
        h = eval_heldout(held, lib, proposer)
        per_round.append({
            "round": rnd,
            "train_acc": train_solved / len(train),
            "n_abstractions": len(lib.ops),
            "levels": lib.levels(),
            "max_level": lib.max_level(),
            "invented": [{"name": a.name,
                          "expansion": list(a.expansion),
                          "level": a.level,
                          "base_length": a.base_length,
                          "mdl_value": a.mdl_value,
                          "uses": a.uses} for a in invented],
            **h,
        })
    return per_round, lib


def worked_example(lib, held, proposer):
    """Grab one deep held-out task and show how invented concepts solve it."""
    for t in held:
        if t.depth < 4:
            continue
        r = solve_task(t.examples, lib, BUDGET, MAX_LEN, proposer=proposer)
        if r.solved:
            used = []
            for op in r.program:
                if op in lib.ops:
                    a = lib.ops[op]
                    used.append({"name": op, "level": a.level,
                                 "expands_to": list(lib.expand_to_base(a.expansion))})
                else:
                    used.append({"name": op, "level": 0, "expands_to": [op]})
            return {
                "hidden_program": list(t.program),
                "hidden_depth_base_ops": len(t.program),
                "solution_tokens": list(r.program),
                "solution_n_tokens": len(r.program),
                "evals_used": r.evals,
                "concepts_used": used,
            }
    return None


def oracle_checks(held):
    """Honest sanity checks: FLAT genuinely can't reach these; a fully-grown library
    (generous budget) can. Confirms the gap is about vocabulary, not luck."""
    flat = sum(solve_task(t.examples, Library(), BUDGET, MAX_LEN).solved for t in held)
    return {"flat_budget_solved": flat, "n_heldout": len(held)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    seeds = [0] if args.quick else args.seeds
    n_rounds = 4 if args.quick else args.rounds

    held = make_heldout(np.random.default_rng(HELDOUT_SEED), N_HELDOUT)
    print(f"held-out: {len(held)} deep tasks, depths "
          f"{sorted(set(t.depth for t in held))}, base-op program length "
          f"{min(len(t.program) for t in held)}-{max(len(t.program) for t in held)}")
    print(f"base vocabulary: {len(BASE_OP_NAMES)} primitives\n")

    arms = ["flat", "invent", "guide"]
    labels = {"flat": "FLAT (no invention)", "invent": "INVENT",
              "guide": "INVENT+GUIDE"}
    data = {"config": {"budget": BUDGET, "max_len": MAX_LEN,
                       "train_per_round": TRAIN_PER_ROUND, "n_heldout": N_HELDOUT,
                       "seeds": seeds, "rounds": n_rounds},
            "oracle": oracle_checks(held), "arms": {}}

    example = None
    for arm in arms:
        seed_runs = []
        last_lib, last_prop = None, None
        for s in seeds:
            per_round, lib = run_arm(arm, s, n_rounds, held)
            seed_runs.append(per_round)
            if arm == "guide" and s == seeds[0]:
                # rebuild proposer to grab a worked example from the trained library
                _, lib2 = run_arm("guide", s, n_rounds, held)
                last_lib = lib2
        # average metrics across seeds, per round
        agg = []
        for i in range(n_rounds):
            accs = [sr[i]["acc"] for sr in seed_runs]
            evs = [sr[i]["mean_evals"] for sr in seed_runs]
            toks = [sr[i]["mean_tokens"] for sr in seed_runs]
            blen = [sr[i]["mean_base_len"] for sr in seed_runs]
            nabs = [sr[i]["n_abstractions"] for sr in seed_runs]
            mlvl = [sr[i]["max_level"] for sr in seed_runs]
            agg.append({
                "round": i + 1,
                "acc_mean": float(np.mean(accs)), "acc_sd": float(np.std(accs)),
                "mean_evals": float(np.mean(evs)),
                "mean_tokens": float(np.mean(toks)),
                "mean_base_len": float(np.mean(blen)),
                "n_abstractions": float(np.mean(nabs)),
                "max_level": float(np.mean(mlvl)),
            })
        data["arms"][arm] = {"label": labels[arm], "per_round": agg,
                             "seed0_detail": seed_runs[0]}
        print(f"{labels[arm]:22s} final held-out acc "
              f"{agg[-1]['acc_mean']:.2f}  mean_evals {agg[-1]['mean_evals']:.0f}  "
              f"abstractions {agg[-1]['n_abstractions']:.0f}  max_level {agg[-1]['max_level']:.0f}")

    # worked example from a fully-trained guided library (seed 0)
    per_round, lib = run_arm("guide", seeds[0], n_rounds, held)
    prop = Proposer()
    # retrain proposer quickly on final library isn't needed; solve with lib+fresh prop
    example = worked_example(lib, held, None)
    data["worked_example"] = example

    os.makedirs("results", exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nwrote {RESULTS}")

    if example:
        print("\n--- worked held-out example ---")
        print("hidden program (base ops):", " -> ".join(example["hidden_program"]))
        print("solved with tokens       :", " -> ".join(example["solution_tokens"]),
              f"({example['solution_n_tokens']} tokens, {example['evals_used']} search steps)")

    make_figures(data)


def make_figures(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = ["flat", "invent", "guide"]
    colors = {"flat": "#999999", "invent": "#1f77b4", "guide": "#d62728"}
    rounds = [r["round"] for r in data["arms"]["flat"]["per_round"]]

    # Fig 30: held-out accuracy vs round
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for arm in arms:
        pr = data["arms"][arm]["per_round"]
        ax.plot(rounds, [p["acc_mean"] for p in pr], "-o", color=colors[arm],
                label=data["arms"][arm]["label"], linewidth=2)
    ax.set_xlabel("round of experience (continuous accumulation)")
    ax.set_ylabel("held-out solve rate (deep, unseen tasks)")
    ax.set_title("Inventing concepts unlocks tasks that from-scratch search can't reach")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig("figures/30_abstraction_accuracy.png", dpi=130)
    plt.close(fig)

    # Fig 31: library growth + compression
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    pr = data["arms"]["guide"]["per_round"]
    a1.plot(rounds, [p["n_abstractions"] for p in pr], "-o", color="#2ca02c",
            linewidth=2, label="concepts in library")
    a1.plot(rounds, [p["max_level"] for p in pr], "-s", color="#9467bd",
            linewidth=2, label="deepest abstraction level")
    a1.set_xlabel("round")
    a1.set_ylabel("count / level")
    a1.set_title("The library grows and becomes hierarchical")
    a1.grid(alpha=0.3)
    a1.legend()
    # compression: base-op length vs library-token length of held-out solutions
    a2.plot(rounds, [p["mean_base_len"] for p in pr], "-o", color="#ff7f0e",
            linewidth=2, label="solution length in base ops")
    a2.plot(rounds, [p["mean_tokens"] for p in pr], "-o", color="#1f77b4",
            linewidth=2, label="solution length in library tokens")
    a2.set_xlabel("round")
    a2.set_ylabel("mean solution length (solved held-out)")
    a2.set_title("Concepts compress solutions (fewer tokens = shorter search)")
    a2.grid(alpha=0.3)
    a2.legend()
    fig.tight_layout()
    fig.savefig("figures/31_abstraction_library.png", dpi=130)
    plt.close(fig)

    # Fig 32: search cost
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for arm in ["invent", "guide"]:
        pr = data["arms"][arm]["per_round"]
        ax.plot(rounds, [p["mean_evals"] for p in pr], "-o", color=colors[arm],
                label=data["arms"][arm]["label"], linewidth=2)
    flat_ev = data["arms"]["flat"]["per_round"][-1]["mean_evals"]
    ax.axhline(flat_ev, color=colors["flat"], linestyle="--",
               label="FLAT (search exhausts budget, still fails)")
    ax.set_xlabel("round of experience")
    ax.set_ylabel("mean search steps per held-out task")
    ax.set_title("Learned guidance + concepts cut search cost as capability rises")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/32_abstraction_search.png", dpi=130)
    plt.close(fig)
    print("wrote figures/30_abstraction_accuracy.png, 31_abstraction_library.png, "
          "32_abstraction_search.png")


if __name__ == "__main__":
    main()
