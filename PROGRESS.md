# Echo Civilization — Progress

## Goal
Research sim: can simple learning agents accumulate knowledge/capability over generations
via a civilization process? NO pretrained LLMs; numpy + stdlib only.
Experiments A–O are FROZEN and published (fernforge-arcade/echo-civilization). Do NOT retune.

## Current state — DONE
Operator task (2026-07-14) "make Echo play Neural MMO 2.0" = **Experiment O**, COMPLETE.
Echo's civilization machinery (skill DAG, proficiency/practice valley, cultural ratchet,
teaching, generational evolution) runs inside a live `nmmo==2.1.1`. Same thesis, same 4
conditions (A/B/C/D), scored by real nmmo rollouts. Full run done, figures + findings +
report + README written, committed & pushed to `main`.

## RESULT (3 seeds, 50 gens, pop12, steps180) — thesis reproduces in real MMO
final maxFrontier / meanCapability:
  A isolated:      0.0 / 3.6   (never leaves floor)
  B inheritance:   1.7 / 11.1  (stalls shallow)
  C + ratchet:     3.0 / 32.6  (crosses valley, ~3x B)
  D full civ:      3.0 / 32.6  (== C)
Ladder monotone: move 2.1 -> explore 2.4 -> catalog 14.9 -> forage 28.3 -> harvest 119.7 (57x).
C≈D reported HONESTLY: when discovery is the bottleneck, the ratchet already carries culture's
value; teaching only speeds practice of an already-discovered rung. Not retuned to force D>C.

## Files (all committed)
- `run_nmmo.py` — driver: run_condition A/B/C/D x seeds[0,1,2] x 50 gens, sqlite + 4 figures + findings.
- `echo_civilization/nmmo_{world,primitives,agent,civ}.py` — MiniNMMO config, skill LIBRARY,
  NMMOAgent w/ proficiency, NMMOCivilization + CONDITIONS.
- `NMMO_FINDINGS.md`, REPORT.md §11e, README §"Experiment O", figures/nmmo_0{1,2,3,4}_*.png.
- results/nmmo_civ.db (gitignored). maps/ = nmmo cache (gitignored).

## nmmo env
Real `nmmo==2.1.1` in SEPARATE venv `/home/node/nmmoenv` (numpy 1.23.3 + matplotlib 3.6.3 +
contourpy<1.1). `/home/node/nmmoenv/bin/python -c "import nmmo"` works. Echo's own ./venv is
numpy2 — do NOT import nmmo there; do NOT add `import nmmo` to shared __init__.

## Nothing left. If reopened: task is done; only re-run `/home/node/nmmoenv/bin/python run_nmmo.py`
to regenerate results.

## Gotchas
- rollout seed masked with &0x3FFFFFFF (<2^32). pop<=16/episode. nmmo stdout is buffered &
  noisy (filter WARNING|UserWarning|pufferlib|deprecat).

## How to run / test
`/home/node/nmmoenv/bin/python -c "import nmmo"` then `/home/node/nmmoenv/bin/python run_nmmo.py`.

## Log
- 2026-07-14: Built & ran Experiment O in real Neural MMO 2.0. A<B<C≈D cleanly across 3 seeds.
  Wrote §11e + README + NMMO_FINDINGS.md. Committed & pushed. Task complete.
