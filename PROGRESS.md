# Echo Civilization — Progress

## Goal
Research sim: can a population of simple learning agents (NO pretrained LLMs; pure
Python + numpy + sqlite3 + matplotlib + networkx) accumulate knowledge and become
more capable over generations through a civilization-like process? Operator's LATEST
steer (DONE): move toward what larger models do — take a vague task ("build a website
that does X") and actually BUILD a working app, using task decomposition. Ultimate
goal: agents genuinely capable of building NEW apps. Write up a report.

## Current state — COMPLETE & PUBLISHED. Public repo live on GitHub.
The whole project (Experiments A–J) is finished, committed, and PUBLISHED to a public
GitHub repo: https://github.com/fernforge/echo-civilization (default branch `main`).
Local branch was renamed master->main. figures/ are now committed (un-ignored) so the
README renders on GitHub; results/ (9.5M DB) stays git-ignored. Builder World (Exp J) is
the operator's final steer realized: agents emit REAL JS, executed in Node vs hidden
tests; an app is "built" only if every requirement passes. Five openable apps in output_apps/.

Builder canonical numbers (verified): frontier over 8 gens — A monolithic=0 (builds
nothing); B decomposed/no-culture flat ~4 (flukes never compound); C decomposed+culture
climbs 4.7→6.0 by gen2 and holds (library 13–14→16). Fresh→cultured build rate:
counter .81→1.0, tip .52→1.0, todo .85→1.0, cart .33→1.0, notes(6) .07→1.0. 3325 real
Node runs. Thesis: decomposition makes building possible (additive vs multiplicative
search); culture makes the frontier climb & hold.

## What's left
NOTHING required. Project is complete and committed. Optional future work (operator's
roadmap, NOT requested): free-form codegen from a blank file (vs fixed 22-component
library); agent-proposed goals; multi-firm economy. Do not start without a steer.

## Key decisions & why (do NOT revert)
- Two solving channels across the project: tabular Q-learning DISCOVERS primitives from
  reward; program/code synthesis COMPOSES known skills under a fixed budget. Culture
  supplies primitives so composites become cheap search; without culture the same search
  exceeds budget. That asymmetry (expensive-to-discover / cheap-to-inherit) is the thesis.
- Builder: a component is the unit discovered/graded/shared/inherited. Decoys discriminate
  (non-zero remove index, price string-coercion, .done checks) so each subtask has a
  UNIQUE correct component. Don't raise BUDGET past ~60 or below ~20 (margin gone at both ends).
- Tuning that must stay (other experiments): train budget=35; EVAL_BUDGET=150;
  cultural_seed_top=16; inherit_skill_fraction=0.85; generalization eval frozen
  (allow_discovery=False, learn_at_solve=False, GEN_EVAL_BUDGET=4000).

## Gotchas
- Workspace is a CASE-INSENSITIVE bind-mount: RESEARCH_REPORT.md == research_report.md.
  Flagship is REPORT.md (distinct). Don't create a name that lowercases to a clash.
- Git: repo dir root-owned; each fresh container needs
  `git config --global --add safe.directory /home/node/workspace` first.
- Use `./venv/bin/python` (BOTH venv/ and .venv/ exist; venv/ is the real one).
- Builder needs `node` on PATH; re-run regenerates everything: `run_builder.py --seeds 0 1 2`.
- Background runners buffer stdout until exit — read the .output file or wait for notify.

## How to run / test
`./venv/bin/python run_experiments.py` (~60–75s; `--quick`) -> research_report.md,
figures/*.png, results/echo_civilization.db. Other runners: run_generalization.py /
run_benchmark.py / run_frontier.py / run_tier8.py / run_adaptability.py /
run_parametric.py / run_builder.py — each `--seeds 0 1 2` or `--trials 10`.
Flagship doc: REPORT.md (§1–§12). Per-experiment flagships: *_FINDINGS.md.

## Log
- 2026-06-26: finished Exp J report; project complete & committed.
- 2026-06-27: published to public GitHub repo fernforge/echo-civilization (branch main).
  Committed figures/ (un-ignored). Remote uses token-less URL; auth via $GITHUB_TOKEN.
