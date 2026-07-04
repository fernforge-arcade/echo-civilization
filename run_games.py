#!/usr/bin/env python3
"""Experiment L — the Game World ladder.

Runs the four rungs (Tic-Tac-Toe -> Connect Four -> Los Alamos minichess -> EchoCraft), each
under three matched conditions:

  SOLO  one lifelong learner (individual learning, no population, no culture)
  POP   a mortal population, each generation born naive (population, no culture)
  CIV   a mortal population born into a shared, accumulating culture (the full civilization)

The comparison of interest is POP vs CIV: identical mechanics and mortality, differing only in
whether knowledge is inherited. The prediction is that culture's advantage (CIV - POP) *grows*
as the environment gets more open-ended and noisy, from ~0 on solved Tic-Tac-Toe to large on
open-ended EchoCraft where the tech tree cannot be rediscovered from scratch each generation.

Writes results/games.json.  Usage: python run_games.py [--seeds 0 1 2] [--rung NAME] [--quick]
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from echo_civilization.games.harness import run_condition, SOLO, POP, CIV
from echo_civilization.games.tictactoe import TicTacToeRung
from echo_civilization.games.connect4 import Connect4Rung
from echo_civilization.games.minichess import MiniChessRung
from echo_civilization.games.craft import CraftRung

# per-rung training budgets (gens, population, episodes/generation). Kept modest; the deeper
# rungs cost more per episode so they run fewer, wider generations.
CONFIG = {
    "tictactoe": dict(rung=TicTacToeRung(), gens=8,  pop=5, eps=200),
    "connect4":  dict(rung=Connect4Rung(),  gens=8,  pop=4, eps=90),
    "minichess": dict(rung=MiniChessRung(),  gens=6,  pop=3, eps=30),
    "echocraft": dict(rung=CraftRung(),      gens=10, pop=6, eps=10),
}
QUICK = {
    "tictactoe": dict(gens=3, pop=3, eps=60),
    "connect4":  dict(gens=3, pop=3, eps=40),
    "minichess": dict(gens=2, pop=2, eps=12),
    "echocraft": dict(gens=4, pop=4, eps=8),
}
ORDER = ["tictactoe", "connect4", "minichess", "echocraft"]


def run_rung(name, cfg, seeds):
    rung = cfg["rung"]
    out = {"name": name, "complexity": rung.complexity,
           "gens": cfg["gens"], "pop": cfg["pop"], "eps": cfg["eps"],
           "conditions": {}}
    for cond in (SOLO, POP, CIV):
        curves_mean, curves_best, extras = [], [], []
        for s in seeds:
            recs = run_condition(rung, cond, seed=s, gens=cfg["gens"], pop=cfg["pop"],
                                 episodes_per_gen=cfg["eps"], renew_pop=True)
            curves_mean.append([r["mean_score"] for r in recs])
            curves_best.append([r["best_score"] for r in recs])
            # keep every best_* extra series for the figures
            keys = [k for k in recs[-1] if k.startswith("best_") and k != "best_score"]
            extras.append({k: [r.get(k) for r in recs] for k in keys})
        cm = np.array(curves_mean)
        cb = np.array(curves_best)
        merged_extra = {}
        for k in extras[0]:
            try:
                merged_extra[k] = np.array([e[k] for e in extras], dtype=float).mean(0).tolist()
            except (TypeError, ValueError):
                merged_extra[k] = extras[0][k]
        out["conditions"][cond] = {
            "mean_curve": cm.mean(0).tolist(),
            "mean_curve_std": cm.std(0).tolist(),
            "best_curve": cb.mean(0).tolist(),
            "final_mean": float(cm.mean(0)[-1]),
            "final_best": float(cb.mean(0)[-1]),
            "extras": merged_extra,
        }
    pop_f = out["conditions"][POP]["final_mean"]
    civ_f = out["conditions"][CIV]["final_mean"]
    out["culture_advantage"] = civ_f - pop_f
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--rung", type=str, default=None, help="run only this rung")
    ap.add_argument("--quick", action="store_true", help="tiny budgets for a smoke test")
    args = ap.parse_args()

    names = [args.rung] if args.rung else ORDER
    results = {"seeds": args.seeds, "rungs": {}}
    for name in names:
        cfg = dict(CONFIG[name])
        if args.quick:
            cfg.update(QUICK[name])
        t = time.time()
        print(f"[{name}] complexity={cfg['rung'].complexity} "
              f"gens={cfg['gens']} pop={cfg['pop']} eps={cfg['eps']} seeds={args.seeds} ...",
              flush=True)
        res = run_rung(name, cfg, args.seeds)
        results["rungs"][name] = res
        adv = res["culture_advantage"]
        cs = res["conditions"]
        print(f"  SOLO={cs[SOLO]['final_mean']:.3f}  POP={cs[POP]['final_mean']:.3f}  "
              f"CIV={cs[CIV]['final_mean']:.3f}  culture_advantage(CIV-POP)={adv:+.3f}  "
              f"({time.time()-t:.0f}s)", flush=True)

    os.makedirs("results", exist_ok=True)
    with open("results/games.json", "w") as f:
        json.dump(results, f, indent=2)
    print("wrote results/games.json", flush=True)


if __name__ == "__main__":
    main()
