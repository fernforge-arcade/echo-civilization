# Echo Civilization — Progress

## Goal
Research sim: can simple learning agents accumulate knowledge over generations via a
civilization process? (NO pretrained LLMs; numpy + stdlib only.) Experiments A–L PUBLISHED
(GitHub fernforge-arcade/echo-civilization, main). REPORT.md is the flagship. Don't touch
the delicate tuning of A–L or the echofill §11b work (all committed at 2a229d8).

## CURRENT STEER (2026-07-10): ABSTRACTION INVENTION — the DreamCoder/ARC direction
Operator's new brief: move Echo "from learning programs to learning the *vocabulary* in
which programs are written." Invent+name+hierarchically-compose reusable concepts (not just
reuse fixed skills); MDL-based abstraction selection; richer grid DSL + perception (ARC
bridge); continuous cultural accumulation (one lifelong library, not generations); search +
learned guidance. = new self-contained "Experiment M / Echo-ARC" (does NOT touch A–L).

## Current state — EXPERIMENT M COMPLETE & COMMITTED. Engine + docs + report wiring all done.
ABSTRACTION_FINDINGS.md written; REPORT.md §11c + conclusion #13 + §14 repro added; README.md
section + doc-list + run cmd added. Committed & pushed to main. Nothing outstanding on this steer.
New files (all self-contained, no edits to A–L classes):
- `echo_civilization/gridworld_arc.py` — Echo-ARC substrate: perception (connected
  components/objects/attributes/symmetry = pixels→objects→relations bridge) + a 17-op total
  grid→grid DSL. Programs = tuples of op names (same shape as string world → mining/search reuse).
- `echo_civilization/abstraction.py` — THE HEART. `Library` (hierarchical invented-op store,
  encode/expand/levels), `solve_task` (iterative-deepening search over base+invented vocab,
  budget-bounded), `mine_abstractions` (MDL: value = occ*(len-1) - (len+1); promote positive-
  value recurring fragments → named ops; re-encode enables Level-2+ hierarchy).
- `echo_civilization/arc_tasks.py` — task generator: 6 length-2 MOTIFS + favoured COMPOUNDS
  (→ Level-2). Training shallow (1-2 motifs, base-reachable); held-out deep (3-4 motifs,
  disjoint, reachable only via invented concepts). 5 examples/task; discards trivially-solvable.
- `echo_civilization/proposer.py` — learned guidance: tiny per-op numpy logistic regression on
  task features → orders search (guides, never solves). Point 8 of brief.
- `run_abstraction.py` — 3 arms (FLAT/INVENT/INVENT+GUIDE) × seeds × rounds; writes
  results/abstraction.json + figures/30,31,32. Worked-example trace + oracle checks included.

RESULT (seeds 0,1,2, 6 rounds, budget 3000): FLAT flat at 0.03 (oracle: solves 1/40, exhausts
2930 evals). INVENT climbs 0.14→0.85 (9 concepts, max level 2). INVENT+GUIDE 0.28→0.97, 320
evals (~3× cheaper than INVENT's 900, ~9× under FLAT's wall). Worked example: hidden 8-base-op
task solved in 2 tokens (a Level-2 + a Level-1 concept). Hierarchy (L1 then L2) forms from
accumulation. This demonstrates every point of the brief honestly.

## What is left
Nothing on the abstraction-invention steer — done and pushed. If the operator wants MORE on
this thread, natural extensions (NOT started): Level-3 concepts (needs a deeper task grammar),
a real ARC-subset task loader, or richer DSL ops (containment/counting). Otherwise await next steer.

## Next concrete step
Await operator's next direction. Experiment M shipped (engine + ABSTRACTION_FINDINGS.md +
REPORT §11c + README + figs 30-32, committed & pushed).

## Key decisions & why
- New self-contained module (not edits to skills.py/culture.py/synthesis.py) → zero regression
  risk to published A–L, mirrors how echofill §11b was added.
- Motifs are DISCOVERED by mining, NOT handed in: training solutions found by base search, then
  mined → the honest DreamCoder loop. Held-out is deeper + combinatorially disjoint = real
  accumulation test, not memorization (mirrors the existing generalization guard).
- Continuous accumulation across "rounds" with a fresh training sample each round = one lifelong
  library (brief point 7); the depth curriculum EMERGES from library growth (not hand-sequenced).
- budget=3000, max_len=6 (tokens), 5 examples/task: tuned so base search reaches depth≤2 only,
  deeper tasks need invented concepts, and 5 examples kill spurious short solutions (so mined
  fragments ≈ true motifs). Proposer is genuinely needed for the deepest (depth-4) tasks.

## Gotchas
- Use `./venv/bin/python`. Case-INSENSITIVE bind-mount: don't collide filenames on lowercase.
- Long runs: python stdout is BUFFERED → the tail/output file stays empty until the end; poll
  results/abstraction.json existence, not stdout. `ps`/`pgrep` NOT available in this container.
- Git: `git config --global --add safe.directory /home/node/workspace` on fresh container.
  Push: `git push "https://x-access-token:${GITHUB_TOKEN}@github.com/fernforge-arcade/echo-civilization.git" main`.
  figures/ is un-ignored (commit PNGs); results/ is git-ignored.

## How to run / test
`./venv/bin/python run_abstraction.py` (canonical, ~40s) or `--quick` (1 seed, 4 rounds).
Older experiments: run_experiments.py, run_echofill.py, run_games.py, etc.

## Log
- 2026-07-10: Experiment M (abstraction invention / Echo-ARC) SHIPPED — engine, 3-arm result,
  ABSTRACTION_FINDINGS.md, REPORT §11c + conclusion #13, README section, figs 30-32, committed
  & pushed to main. Steer complete. Older history in .cb/log/ (write-only, don't read back).
