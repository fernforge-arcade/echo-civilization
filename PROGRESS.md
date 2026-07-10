# Echo Civilization — Progress

## Goal
Research sim: can simple learning agents accumulate knowledge/capability over generations
via a civilization process? NO pretrained LLMs; numpy + stdlib only.

## STATUS: FROZEN after Experiment N (2026-07-10)
Echo is complete and frozen at the Experiment N checkpoint. Experiments A–L, §11b (echofill),
§11c (Experiment M, abstraction invention), and §11d (Experiment N, falsification) are all
published on GitHub (fernforge-arcade/echo-civilization, main). REPORT.md is the flagship.
The object-centered DSL / Mini-ARC direction has been **moved to a separate project** — do NOT
build it inside Echo. Do NOT retune/regress any prior experiment.

## Experiment N — Falsifying M (final checkpoint). COMPLETE.
Turns M from a designed macro-invention demo into a defensible library-learning benchmark via
an adversarial control battery. Ran 3 seeds × 6 rounds × 38 clean held-out (~19 min).
- Code: `echo_civilization/falsification.py`, `echo_civilization/arc_tasks_v2.py`,
  `run_falsification.py`. Figures 33–37. Findings: `FALSIFICATION_FINDINGS.md`.
- Raw: `results/falsification.json` (git-ignored; numbers live in the findings doc).
- Result — M SURVIVES as synthetic hierarchical macro invention. Defensible number under
  unseen-query scoring: invent_guide **0.667** vs full-oracle reachability ceiling 1.000.
  Every non-oracle control fails: fresh_flat 0.0, generous_flat 0.316 (@53k evals),
  cache_only 0.0, shuffled_lib 0.0, motif_oracle 0.316.
- Honest negatives KEPT: (a) reusable unit is the L2 motif *pairing*, atoms alone give 0.316;
  (b) flat wall is a compute wall (flat 0.40 @ ~80k evals) not an impossibility, and the
  per-task gap is NOT equal-accuracy; (c) P=4 civilization 0.254 LOSES to P=1 lifelong 0.456
  while spending MORE lifetime compute → Echo supports lifelong individual abstraction
  learning, NOT a civilization advantage; (d) neg-transfer = ~4% branching tax.

## This run (2026-07-10) — reconciliation + freeze. DONE.
Operator changed the research boundary: stop Mini-ARC/object-DSL inside Echo, finish as an N
checkpoint. Actions taken:
1. Reconciled `FALSIFICATION_FINDINGS.md` with 7 caveats, all verified against code:
   - `integrity.oracle_solvable` uses `oracle_library()` (motif-only, 6 L1), NOT full oracle
     — relabeled as a lower-bound solvability gate.
   - utility promotion slice is `val = train[:12]` = in-sample CURRENT-round training, not a
     frozen validation stream — relabeled.
   - full_oracle = reachability ceiling (hand-inserted L2 pairs), not learned generalization.
   - behavioural-leak audit is meaningful but non-exhaustive (only tests found programs).
   - lifetime training cost (161k–270k evals) now reported SEPARATELY from deploy cost.
   - removed equal-accuracy framing of the ~60× gap (flat 0.40 vs guided 0.667).
   - fixed motif wording: motifs are length-2, held-out = 3–4 motifs = 6–8 base ops.
2. Preserved P=4 < P=1 negative; conclusion = lifelong individual learning, not civilization.
3. Removed just-started Mini-ARC files (object_dsl.py, miniarc_tasks.py, run_miniarc.py,
   figs 38–40, results/miniarc.json). Preserved all Experiment N + prior work.
4. Added §11d to REPORT.md and a falsification block to README.md (both honest, measured).
5. Focused checks PASSED: all 34 echo_civilization modules import; `run_abstraction.py
   --quick` reproduces (INVENT+GUIDE 0.95); N modules import clean.

## What is left
Nothing on Echo. Commit + push the N checkpoint, cb note the commit hash, stop.
(Mini-ARC is a separate project — not started here.)

## Key decisions & why
- oracle_solvable is a LOWER-bound solvability gate (motif-only), not a ceiling; full_oracle
  is the reachability ceiling. Kept distinct in prose.
- P=1 IS the matched single-learner; P>1 pools inherited culture. P=4<P=1 kept as the load-
  bearing honest negative — a single persistent library is NOT "culture."

## Gotchas
- `./venv/bin/python`. Case-INSENSITIVE bind-mount. python stdout BUFFERED (poll JSON).
- Git: `git config --global --add safe.directory /home/node/workspace`.
  Push: `git push "https://x-access-token:${GITHUB_TOKEN}@github.com/fernforge-arcade/echo-civilization.git" main`.
  figures/ committed; results/ git-ignored (findings docs hold the numbers).

## How to run / test
`./venv/bin/python run_falsification.py` (canonical N, ~19 min; `--quick` for smoke).
`./venv/bin/python run_abstraction.py` (canonical M). Earlier: see README run block.

## Log
- 2026-07-10: Reconciled N findings (7 caveats), added §11d + README block, removed Mini-ARC
  files, froze Echo after N. Focused checks passed. Committing the N checkpoint.
