# Echo Civilization — Progress

## Goal
Research simulation testing whether complex intelligence emerges from simple agents
via learning, memory, communication, specialization, competition, cooperation,
cultural inheritance and generations. Central question: *Can a population of simple
learning agents accumulate knowledge and become more capable over generations
through a civilization-like process?* No pretrained LLMs — pure Python + numpy +
sqlite3 + matplotlib + networkx.

Operator's latest steer: **push the frontier on ADAPTABILITY — solving genuinely
NEW, never-before-seen tasks** (not just deeper rungs of known task families). Keep
giving example output from runs in the reports.

## Current state — COMPLETE & WORKING (extending on frontier)
Package `echo_civilization/` is fully built and verified end-to-end. Eight
experiments A–G plus a computer-use frontier (Tiers 6/7/8), a compositional-
generalization study, and an ADAPTABILITY study (Experiment H — novel task family)
all run, log to sqlite, and emit figures (22 PNGs) + reports.

Headline results (all reproduced, multi-seed where noted):
- Cultural accumulation: held-out HARD tasks A/B ~0.5 (flat) vs C/D ~0.97 (climbs).
- Compositional generalization (DISJOINT held-out composites): novel depth-3
  A .13/B .06 vs C .60/D .62; depth-3 leaks NONE; oracle 100%.
- Computer-use: cultured agent clears 13 real-shell rungs @1.00; fresh 0.52.
  Frontier moved 2+ tiers (parametric ops, code synthesis, group-by). Under TIGHT
  budget culture still decides (fresh ~0 vs cult ~1).
- Grid NN evolves 1.6->6.1; social signaling protocol emerges to 100%; firm with
  shared KB profits +426 vs no-KB -92.

## Modules (modular, swappable learners)
neural / memory / learning (QLearner, NeuralPolicy-ES, Random) / skills / culture /
teaching / agent / environments/{echo,transformation,memory,grid,social,computer,
real_computer} / evolution / computer_evolution / synthesis / frontier / codegen /
codegen2 / database / evaluation / visualization / report. Generalization study in
generalization.py. Runners: run_experiments / run_generalization / run_benchmark /
run_frontier / run_tier8.

## What's left (operator roadmap — adaptability frontier)
1. **Experiment H — Adaptability: DONE & COMMITTED.** Module `adaptability.py`,
   runner `run_adaptability.py`, viz fns `plot_adaptability_bars`/
   `plot_adaptability_curve` (figs 21/22), REPORT.md §7 (+renumbered §8–10 +
   conclusion bullet 10), flagship `ADAPTABILITY_FINDINGS.md`, README section all
   shipped. Multi-seed canonical numbers below. Nothing left here.
2. Learned ARGUMENTS + agent-proposed sub-tasks (order-only synthesis today).
3. Agent-proposed goals; multi-firm economy; unbounded runs.

## Next concrete step
Experiment H complete. Next frontier item is (2): learned ARGUMENTS / agent-proposed
sub-tasks, OR (3) agent-proposed goals + multi-firm economy. Pick one and design a
test-that-can-fail in the same style (oracle audit + frozen eval + culture-decides
under tight budget). No mechanical work pending.

CANONICAL NUMBERS (run_adaptability.py --seeds 0 1 2, committed):
  ORACLE 1.00 (suite solvable-in-principle ✓). Generous budget: ALL = 1.00.
  TIGHT budget (45): A 0.55±0.25 (1 noisy agent) / B 0.47 / C 0.90 / D 0.91 /
  FRESH 0.22±0.00. Headline = best-cult 0.91 vs FRESH 0.22 = +0.69 (pure inherited-
  library value; combinator novel to all). FRESH is the clean zero-accumulation
  control; A is high-variance single agent (error bars shown in fig 21).
DESIGN of Experiment H (so a cold resume understands it): novel task family =
HIGHER-ORDER COMBINATORS (map_each / map_reversed / first_only / last_only /
map_evens) wrapped around an inner depth-2 transform. NOBODY trains on combinators,
so the combinator confers no inherited edge — it's discovered at eval by both. The
ONLY cultured advantage is the inherited inner abstractions (the 5 COMPOSITE_TASKS
depth-2 progs). Fresh must rediscover a depth-2 inner from scratch but a tight budget
(45) is exhausted clearing the 10 single-op inner candidates (10×5 combinators=50>45)
before it can reach depth-2 → structural wall. Eval is FROZEN (no test-time learning),
query-judged, oracle-audited. Constants: TIGHT_BUDGET=45, GENEROUS_BUDGET=4000,
make_novel_tasks → 100 tasks (5 comb × 5 inner × 4 each). Do NOT lower tight budget
much or cultured also fails; do NOT raise past ~50 or fresh starts clearing singles.

## Key decisions & why (do NOT revert)
- Two solving channels: tabular Q-learning DISCOVERS char-substitution primitives
  from reward; program/code synthesis COMPOSES known skills for structural tasks
  under a fixed budget. Culture supplies primitives -> composites become a cheap
  search; without culture the same search exceeds budget and fails. That asymmetry
  (expensive-to-discover / cheap-to-inherit) is the whole thesis — preserve it.
- Tuning that must stay: train budget=35; EVAL_BUDGET=150; cultural_seed_top=16;
  inherit_skill_fraction=0.85; discovery pool = singles-first then SHUFFLED
  pairs/triples; social game n_concepts=3/n_agents=6/rounds=200. Computer civ:
  budget=150, advance_threshold=0.45, tasks_per_agent=10 (don't drop below ~140).
  Frontier budgets TIER6 12/300, TIER7 60/400, TIER8 45/300; skeleton ORDER (correct
  skeleton LAST for fresh / FIRST for cultured) is the cultural lever — don't reorder.
- Generalization eval is frozen: allow_discovery=False AND learn_at_solve=False so
  no test-time skill abstraction leaks across tasks. GEN_EVAL_BUDGET=4000.

## Gotchas
- Workspace is a CASE-INSENSITIVE bind-mount: RESEARCH_REPORT.md == research_report.md
  (same file). Flagship is REPORT.md (distinct) so the auto-generated
  research_report.md never clobbers it. Don't create a name that lowercases to a clash.
- Git: repo dir is root-owned; each fresh container needs
  `git config --global --add safe.directory /home/node/workspace` before git works.
- Use `./venv/bin/python` (note: BOTH venv/ and .venv/ exist; venv/ is the real one).
- run_tier8.py / background runners buffer stdout until exit — read the .output file
  or wait for the task notification. --quick OVERWRITES json/figs with fewer trials;
  regenerate canonical numbers with --trials 10 before committing.

## How to run / test
`./venv/bin/python run_experiments.py` (full pipeline ~60–75s; `--quick` faster) ->
research_report.md, figures/*.png, results/echo_civilization.db.
Other studies: `run_generalization.py --seeds 0 1 2`, `run_benchmark.py --trials 10`,
`run_frontier.py --trials 10`, `run_tier8.py`.

## Log
- 2026-06-26: trimmed PROGRESS.md (history -> .cb/log/progress-archive-20260626.md);
  starting adaptability/cross-family frontier (Experiment H).
