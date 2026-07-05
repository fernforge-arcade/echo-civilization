# Echo Civilization — Progress

## Goal
Research sim: can simple learning agents accumulate knowledge over generations via a
civilization process? (NO pretrained LLMs; numpy + stdlib only.) Experiments A–L COMPLETE &
PUBLISHED (GitHub fernforge-arcade/echo-civilization, main). REPORT.md is the flagship.
Don't touch the delicate tuning of A–L.

## CURRENT STEER (2026-07-05): agents do REAL LLM-work cheaper, USING OUR RESEARCH — DONE
Operator: "push agents into useful stuff, a lot cheaper, using OUR research (the evolved
civilization + accumulated culture), not a random unrelated tool." Plus: "make the final
report show *why* this is actually new and not something that already exists."

## Current state — EXPERIMENT §11b BUILT, TESTED, WIRED IN. Ready to commit/push.
The "echofill" wrangling experiment is complete and self-contained (does NOT touch A–L):
- `echo_civilization/echofill.py` — synthesis engine (staged search + FlashFill-style arg
  induction). Public API: learn(examples)->Rule; Rule.apply/apply_column/describe.
- `echo_civilization/echofill_civ.py` — the CIVILIZATION wiring: EchofillAgent with
  recall→recombine→modify→discover solve loop + WrangleCulture (shared skill library). This
  is what makes it "the agents/culture doing it", mirroring agent.solve_task/synthesis.py.
- `echo_civilization/wrangle_suite.py` — TRAIN tasks (culture builds from these) + HELDOUT
  (composites = concat of two train pieces; + single-op controls).
- `run_echofill.py` — 3 arms on same held-out suite: naive / cultured / real-LLM(Haiku 4.5
  cost model). RESULT: cultured 6/6, naive 3/6; 100% vs 46% row acc; ~530 ns/row, $0 vs
  ~$13/100k rows. Writes results/echofill_bench.json.
- `gen_echofill_figures.py` — figures/28_echofill_arms.png, 29_echofill_cost.png (regen'd).
- `echofill_cli.py` — CLI (--ex "in=>out" repeatable, --csv/--col or stdin, --demo). Works.
- `ECHOFILL_FINDINGS.md` — full write-up incl. HONEST "Is this actually new?" section
  (FlashFill/PROSE exist; novelty = the civilization/cultural-accumulation result + $0
  deterministic inference). Named ceiling honestly.
- REPORT.md §11b + README section added, both link ECHOFILL_FINDINGS.md and show the table.

KEY HONEST MECHANISM (the "why it's new + why culture matters"): naive from-scratch search
provably CANNOT compose two parametric ops (e.g. split '@' then split '.'), halts after ~820
candidates → composites fail. Cultured agent inherited the two single-op pieces and solves by
RECOMBINING them. Same synthesiser+budget; only difference is the inherited library.

## What is left (in order)
1. COMMIT + PUSH everything (see Gotchas for the push URL). New/changed files:
   echo_civilization/{echofill.py(untracked),echofill_civ.py,wrangle_suite.py},
   run_echofill.py, gen_echofill_figures.py, echofill_cli.py, ECHOFILL_FINDINGS.md,
   figures/28_echofill_arms.png, figures/29_echofill_cost.png, REPORT.md, README.md,
   PROGRESS.md. (results/ is git-ignored — fine.)
2. (optional polish) eyeball figures/29 once; verify links render. Nothing blocking.

## Next concrete step
Run: `cd /home/node/workspace && git add -A && git commit` with a clear message, then push
via the tokenized URL in Gotchas. Then the steer is DONE — post a cb note summarizing.

## Key decisions & why
- Reused the civilization PATTERN (recall→recombine→modify→discover + shared culture) via a
  NEW self-contained module, not by editing A–L classes → no regression risk.
- Held-out composites are genuine capability gaps (param+param unreachable from scratch), not
  budget artefacts — budget 2000, search self-halts at ~820. This is the honest core.
- LLM cost is ESTIMATED (no API key in sandbox): credited 100% acc, cost from 90-in/8-out
  tok/row @ Haiku 4.5 $1/$5 per 1M. Stated as estimate in the report.

## Gotchas
- Use `./venv/bin/python`. Case-INSENSITIVE bind-mount: don't collide names on lowercase.
- Git: `git config --global --add safe.directory /home/node/workspace` on fresh container.
  Push: `git push "https://x-access-token:${GITHUB_TOKEN}@github.com/fernforge-arcade/echo-civilization.git" main`.
  figures/ un-ignored (commit PNGs); results/ git-ignored. End commit msgs w/ Co-Authored-By: Claude Opus 4.8.

## How to run / test
`./venv/bin/python run_echofill.py` ; `./venv/bin/python echofill_cli.py --demo`.
Original suite: `run_experiments.py`; games `run_games.py`.

## Log
- 2026-07-05: §11b echofill built+tested+wired (report/readme). Left: commit+push.
  Older history in .cb/log/ (write-only, don't read back).
