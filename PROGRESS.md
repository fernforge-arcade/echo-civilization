# Echo Civilization — Progress

## Goal
Research sim: can a population of simple learning agents (NO pretrained LLMs; pure
Python + numpy + sqlite3 + matplotlib + networkx) accumulate knowledge and become
more capable over generations through a civilization-like process? Operator's LATEST
steer (DONE): move toward what larger models do — take a vague task ("build a website
that does X") and actually BUILD a working app, using task decomposition. Ultimate
goal: agents genuinely capable of building NEW apps. Write up a report.

## Experiment K — Stack World: COMPLETE & PUBLISHED (2026-06-27)
README "Stack World" section (§10) added with stack_frontier.png + stack_app.png and the
honest framing; Roadmap + Running list updated. Committed & pushed to main (output_apps/
stack apps, stack figures, stack.py, run_stack.py, STACK_FINDINGS.md, REPORT §10). results/
and .cb/ stay git-ignored. NOTHING left on this steer.

## (history) ACTIVE STEER (2026-06-27): Experiment K — Stack World (full-stack + resilient)
Operator: "they needed a heavy harness and it hardly worked — make the agents more
resilient and able to actually make bigger projects across the entire dev stack."
`echo_civilization/stack.py` (DONE, ~760 lines) + `run_stack.py` (DONE) both written.
Agents build REAL multi-file Node projects: db.js + validate.js + app.js (HTTP router
exporting handle(method,path,body)) + server.js (bootable http server) +
public/index.html (fetch frontend) + package.json. Unit = a REST endpoint (create/list/
read/update/delete), graded by REAL Node execution: status codes (201/200/404/400/204),
validation, 404 semantics, persistence. 4 specs rising: task_api(5) blog(10) shop(15)
platform(20). Each handler is a CONFIG of flags; each flag has its OWN hidden test →
smooth repair landscape. Endpoint TYPES transfer across resources (culture masters 5
types once, then scales resources for free).

### DONE THIS RUN (the build_project fix + runner + tuning, all VERIFIED):
- FIXED build_project: RESILIENT now = take FIRST candidate (inherited if cultured else
  first shuffled preset), grade once, else `_repair` hill-climb to CORRECT (additive).
  BRITTLE = blind-enumerate presets (multiplicative). repair_budget default raised 6→30.
- Enlarged FLAGS pools (create status now 13 codes → pool 78; others 18-24). This is the
  lever: blind create-search burns the whole project budget (multiplicative), repair
  fixes the status flag deterministically and moves on (additive). render handles extra
  status codes generically (just integers), so no handler changes needed.
- run_stack.py written: 4 conditions BRITTLE / RESILIENT / BRITTLE+CULTURE /
  RESILIENT+CULTURE, memoising cache wraps S.GRADE. Defaults budget=90, repair_budget=45,
  gens 8 × pop5 × seeds[0,1,2]. --emit flag emits+boot-probes real apps to output_apps/.

VERIFIED RESULT (budget 90, gens5 pop4 seeds[0-3]): BRITTLE frontier ~2.7 flat (chokes on
create), RESILIENT ~5 flat (repairs create, ~2x, recovery~0.9), BOTH+CULTURE climb gen0→
gen1 from ~3 to 20.0 and HOLD (culture=5 types). Honest read: culture drives the frontier
CLIMB (5→20-endpoint platform); resilience drives per-project reliability (recovery ~90%,
2x no-culture frontier, graceful degradation — partial projects still boot). Both compose
to reliably build 20-endpoint full-stack apps. Note: brittle+culture also climbs because a
POPULATION collectively discovers the 5-type vocabulary; the resilience win is per-agent
reliability/recovery + the no-culture frontier, NOT the asymptote.

### DONE THIS RUN (Experiment K fully built & verified; ONLY README + commit/push left):
- Canonical run COMPLETE (~4 min, not 20-45 — RESILIENT just prints late): `run_stack.py
  --emit --seeds 0 1 2`, budget90 repair45 gens8 pop5, 3666 real node runs of 101283 grades.
  results/stack.json written. Final-gen frontier: BRITTLE 3.0, RESILIENT 5.3, both+CULTURE 20.0.
- FIXED a real boot-probe bug in stack.py boot_and_probe: it POSTed only the FIRST field, so
  multi-field resources (blog posts = [title,body]) got a correct 400 and boot_ok=False.
  Now posts ALL required fields. Re-emitted + re-probed: ALL FOUR apps boot_ok=True
  (task_api 5/5, blog_api 10/10, shop_api 15/15, platform_api 20/20). stack_probes.json current.
- Figures DONE (gen_stack_figures.py + plot_stack_* in visualization.py):
  figures/stack_frontier.png, stack_reliability.png, stack_culture_growth.png.
- Live-app screenshot DONE: shot_stack_app.py boots task_api/server.js, seeds 3 records via
  real HTTP, Playwright screenshots -> figures/stack_app.png. NOTE: pip playwright 1.60 in
  venv expects chromium-1223 but only chromium-1228 is installed, so launch() MUST pass
  executable_path="/ms-playwright/chromium-1228/chrome-linux64/chrome" (already in the script),
  and set PLAYWRIGHT_BROWSERS_PATH=/ms-playwright (NOT exported by default in the shell).
- STACK_FINDINGS.md WRITTEN (full flagship, mirrors BUILDER_FINDINGS format, real numbers).
- REPORT.md UPDATED: new "## 10. Building bigger: resilient full-stack apps" inserted before
  Conclusions; Conclusions->11, Limitations->12, Reproducibility->13; ToC updated to match.
  All 4 stack figures referenced.

KEY HONEST FRAMING (keep in any README text): frontier climb to 20 = CULTURE's win (a
population discovers the 5-type vocab regardless of repair). Resilience's win = per-agent
reliability: pass 0.45->0.61, completion 0.15->0.27, no-culture frontier 3.0->5.3, recovery
0.97 (debugging not luck) + graceful degradation. Together they reliably build 20-endpoint
bootable apps. Don't credit the asymptote to resilience.

### EXACT NEXT STEP (only this remains):
1. Add a "Stack World" section to README.md (mirror its existing Builder World section;
   embed figures/stack_frontier.png + stack_app.png; use the honest framing above). Check
   how README references the other experiments and match that style/heading level.
2. `git add -A && git commit` (end message with the Co-Authored-By line for Claude Opus
   4.8) and `git push` (branch main; remote is token-less URL, auth via $GITHUB_TOKEN —
   see Log section / prior commits for the exact push incantation). figures/ is un-ignored
   so the new PNGs commit; results/ stays git-ignored (so results/stack.json is local-only —
   that's fine, the numbers live in the docs).
3. Done. Update this PROGRESS to "Experiment K COMPLETE & PUBLISHED" and let the run finish.

GOTCHA: run_stack.py loads stack.specs fresh each call; the cache is process-wide keyed on
(spec.name, bindings_json, testmap_json) — correct (identical executions only). Background
runners buffer stdout; read the .output file. A full 4-cond run is heavy on node — caching
makes node_runs ~1k not ~18k.

## Prior state — COMPLETE & PUBLISHED (Experiments A–J). Public repo live on GitHub.
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
