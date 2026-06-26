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
generalization study, an ADAPTABILITY study (Experiment H — novel task family) and a
PARAMETRIC-ABSTRACTION study (Experiment I — inherited schema + novel argument) all
run, log to sqlite, and emit figures (24 PNGs) + reports.

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
1. **Experiment H — Adaptability: DONE & COMMITTED.**
2. **Experiment I — Parametric Abstraction (learned ARGUMENTS): DONE & COMMITTED.**
   parametric.py + run_parametric.py + viz (plot_parametric_bars/curve, figs 23/24) +
   REPORT §8 (renumbered to §11 Reproduce) + PARAMETRIC_FINDINGS.md + README + exec
   summary table rows H/I. Verified seeds 0 1 2: C/D 1.00 vs FRESH 0.25 (+0.75),
   oracle 1.00, generous 1.00 all. Worked trace shows MODAL fresh-fails outcome
   (fresh seed tied to call arg; 37/40 seeds fail on repeat(5)+reverse).
3. Still open (future): agent-proposed sub-tasks/goals; multi-firm economy;
   unbounded runs. Learned-arguments roadmap item is now complete.

## Next concrete step
Nothing pending. If extending further, the next operator-roadmap rung is
agent-proposed sub-tasks/goals or a multi-firm economy (both unbuilt).
DESIGN of Experiment I (so a cold resume understands it): NEW axis = ARGUMENT BINDING,
not order. A schema = a parametric family (shift_by/shift_back/rotate/take/drop/repeat)
PLUS an INVERTER (binds the integer arg from one (in,out) pair in O(1)). Cultural loop:
discover a LOW-arg instance (args 1,2 — blind-reachable) -> ABSTRACT into a schema ->
share -> inherit -> at eval BIND a NOVEL HIGH arg (3,4,5). Cultured agent inverts the
arg per known family (additive cost); FRESH must blind-sweep the full {14 families ×
7 args × 2 inners} grid (real 6 + 8 DECOY distractor families that never appear in
tasks, so never abstracted) — TIGHT budget 40 is exhausted first. Recurrence filter
(confirm_threshold=2) prunes one-off decoy coincidences so only real families persist
as USEFUL schemas; decoys, even if inherited, are INERT (stage 1 only iterates real
families). Constants in parametric.py: TIGHT_BUDGET=40, GENEROUS_BUDGET=4000,
ARG_RANGE 0..6, TRAIN_ARGS=[1,2], EVAL_ARGS=[3,4,5], words len 6-9, ACC=
generations12/pop24/discover_budget400/tasks_per_gen2/confirm_threshold2; eval suite
make_eval_tasks -> 108 tasks (6 real fam × 3 eval args × 2 inners × 3 each). Do NOT
raise TIGHT past ~60 (fresh starts covering the grid) or below ~20 (cultured arg-sweep
margin). Eval is FROZEN (no schema stored at solve), query-judged, oracle-audited.

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
- 2026-06-26: Experiment I (parametric abstraction) fully wired & committed —
  run_parametric.py, viz figs 23/24, REPORT §8, PARAMETRIC_FINDINGS.md, README,
  exec-summary rows. Verified seeds 0 1 2. Roadmap learned-arguments item complete.
