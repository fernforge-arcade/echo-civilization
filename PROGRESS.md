# Echo Civilization — Progress

## Goal
Research sim: can a population of simple learning agents (NO pretrained LLMs; pure Python +
numpy + sqlite3 + matplotlib + networkx) accumulate knowledge and become more capable over
generations through a civilization-like process? Experiments A–K are COMPLETE & PUBLISHED
(GitHub fernforge-arcade/echo-civilization, main). REPORT.md is the flagship (§1–§13); don't
touch the delicate tuning of A–K.

## Current state — Experiment L (Game World ladder): COMPLETE & PUBLISHED
DONE 2026-07-04: figures (figures/games_*.png), GAMES_FINDINGS.md, REPORT.md §11 (renumbered
Conclusions/Limitations/Repro → §12/13/14), README Game World section. Committed b965994 and
PUSHED to fernforge-arcade/echo-civilization main. Nothing outstanding for Experiment L.
Below is the result record (kept for reference); "What's left" is now empty.

## Result record — Game World ladder
Operator steer: push toward usefulness via games (tic-tac-toe → chess-like → open-ended RPG/
Minecraft); test culture on increasingly open-ended/noisy environments; report honestly.
Plan in `GAMES_PLAN.md`. Four rungs under `echo_civilization/games/` all built, tested, and
run for real (seeds 0,1,2). `results/games.json` (42KB) is COMPLETE. Runner `run_games.py`.

FINAL numbers (culture_advantage = CIV−POP final_mean; also see per-gen curves in json):
- tictactoe   cx5   SOLO .704  POP .349  CIV .557  → adv +0.208 (SOLO lifelong is ceiling)
- connect4    cx21  SOLO .706  POP .891  CIV .763  → adv −0.128 (COUNTEREXAMPLE, see below)
- minichess   cx60  SOLO .323  POP .333  CIV .395  → adv +0.061 (modest/noisy middle rung)
- echocraft   cx100 SOLO .758  POP .656  CIV .854  → adv +0.197 (HEADLINE)

THE FINDING (honest, richer than a monotone line — report this): culture's payoff is NOT a
simple function of game-tree complexity. It is governed by (1) whether the skill is
re-discoverable within ONE lifetime budget, and (2) whether culture is stored LOSSLESSLY.
- connect4 is the instructive counterexample: a lone agent masters it in one lifetime AND its
  culture = averaged weight vector (lossy, averaging continuous policies degrades them) → POP
  beats CIV.
- echocraft is where both align: deep tech tree NOT re-discoverable in a lifetime + culture =
  discrete recipe SET (lossless union) → CIV climbs the tree over generations and POP stays
  flat. Money result (from json extras, CIV vs POP across 10 gens):
    CIV  max_depth 5.3→8.0 (bottom of tree), crafter 33→99, achievements 9.2→12.9
    POP  max_depth ~5.5 FLAT, crafter ~33→42 FLAT, achievements ~9 FLAT
    SOLO max_depth →7.3, crafter ~77 (lifelong solo gets close but never reaches depth 8)
  => generation N reaches tech depth gen 1 / the no-sharing population never reach. The thesis.

## What's left
Nothing for Experiment L — shipped. Possible FUTURE directions if the operator steers again:
harder rungs (real minichess opening book / lossless board-game culture to flip connect4's
sign), or a larger EchoCraft map. Not started; no action pending.

## Next concrete step
None pending. Await operator steer.

## Key decisions & why
- Self-contained under games/ so A–K tuning untouched. Reuse the culture pattern, not classes.
- Report the NON-monotone result honestly (operator asked for honesty + best judgement). The
  connect4 negative is a feature, not a bug: it isolates WHEN culture helps.
- civ ablation needs generational MORTALITY (renew_pop=True): POP children born naive → knowledge
  dies with them → POP flat; CIV children inherit culture → ratchet. run_games passes it to all.
- No pretrained models; numpy + stdlib only.

## Gotchas
- Use `./venv/bin/python` (both venv/ and .venv/ exist; venv/ is real).
- Case-INSENSITIVE bind-mount: REPORT.md is the flagship; don't create a name that clashes on
  lowercase (RESEARCH_REPORT.md == research_report.md).
- Git: `git config --global --add safe.directory /home/node/workspace` on fresh container.
  Push: `git push "https://x-access-token:${GITHUB_TOKEN}@github.com/fernforge-arcade/echo-civilization.git" main`.
  figures/ un-ignored (commit PNGs); results/ git-ignored. End commit msgs w/ Co-Authored-By: Claude Opus 4.8.

## How to run / test
Game ladder: `./venv/bin/python run_games.py --seeds 0 1 2` (--quick for smoke, --rung NAME to
isolate) → results/games.json. Original suite: `./venv/bin/python run_experiments.py`.

## Log
- 2026-07-04: Experiment L built + run complete (see above). Full history in
  .cb/log/progress-archive-20260704.md (write-only, don't read back). Next: figures + writeup.
