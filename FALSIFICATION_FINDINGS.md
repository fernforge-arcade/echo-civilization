# Experiment N — Falsifying Experiment M

Experiment M (§11c) claimed that Echo invents a hierarchical macro vocabulary and
climbs from a 2.5% flat baseline to 97.5% on held-out synthesis tasks. That number was
measured on *sampled* held-out programs, scored on the training demonstrations, with the
library trained on the same task stream. Three things could inflate it: the flat baseline
could be under-resourced rather than genuinely stuck, "held-out" tasks could be
behaviourally reachable from training, and success could be scored on grids the solver
already saw. Experiment N re-runs the whole thing under adversarial controls to find out
which parts survive.

Raw numbers live in `results/falsification.json` (git-ignored). Figures 33–37. Command:
`./venv/bin/python run_falsification.py` (canonical: 3 seeds, 6 rounds, 38 clean held-out
tasks, ~19 min). `--quick` for a 1-seed smoke test.

## What changed versus M

- **Unseen query grid per task.** Every task now carries a query grid never shown during
  synthesis. Synthesis may read only the training pairs; a program counts as solved only
  if it reproduces the exact query output. This is the single biggest change — it removes
  "fit the demos, get scored on the demos."
- **Behavioural leak filter.** Held-out tasks that any pooled training program reproduces
  on demos *and* query are dropped before scoring. Of 40 generated held-out tasks, 2 were
  behaviourally reachable from training and removed; 38 clean remain (`behavioural_leaks:
  0` after filtering).
- **Motif-pairing holdout.** Held-out programs use motif adjacencies (`HELDOUT_ADJ`) never
  present in any training program, so the library cannot have memorised the exact
  composition. Each motif is a fixed length-2 base-op pair; held-out programs are 3–4
  motifs deep, i.e. **6–8 base ops**. The held-out unit is the *pairing* between two
  motifs (an adjacency), not any single motif — every individual motif is seen in training
  through other adjacencies.
- **Cumulative solved-program corpus.** M discarded each round's `solved_progs`. N retains
  every solved program and re-encodes the corpus whenever the library changes, which is
  what the M prose described but the code did not do.

## Conditions and results (3 seeds, 6 rounds, 38 held-out, query-scored)

The `Deploy evals/task` column is the *per-task* search cost at evaluation (deployment).
It does **not** include the lifetime cost of building the library — training solves,
mining, and proposer updates — which is reported separately below.

| Condition        | Held-out acc | Deploy evals/task | What it tests |
|------------------|-------------:|------------------:|---------------|
| fresh_flat       | 0.000        | 3000              | 17 primitives, len ≤6, 3k budget — the M baseline |
| shuffled_lib     | 0.000        | 3000              | size-matched library of useless concepts |
| cache_only       | 0.000        | 10                | persistent exact-solution cache, no ID/grid retrieval |
| motif_oracle     | 0.316        | 2317              | library = the 6 true L1 motifs only |
| generous_flat    | 0.316        | 53118             | flat, len ≤8, 60k budget |
| invent_utility   | 0.482        | 1913              | learned library, in-sample utility promotion |
| invent           | 0.544        | 1798              | learned library, MDL promotion |
| invent_guide     | 0.667        | 1360              | learned library + proposer guidance |
| full_oracle      | 1.000        | 1259              | 6 motifs + all 36 ordered L2 pairs (reachability ceiling) |

Standard deviations across seeds are small (invent ±0.012, invent_guide ±0.033,
invent_utility ±0.081). The deterministic conditions have no seed variance.

### Lifetime cost, reported separately

The learned-library conditions pay an up-front training cost that the deploy column hides.
Total training-search evaluations accumulated over all 6 rounds (mean across seeds):

| Condition      | Final acc | Deploy evals/task | Lifetime train evals |
|----------------|----------:|------------------:|---------------------:|
| invent         | 0.544     | 1798              | 266,120              |
| invent_guide   | 0.667     | 1360              | 161,129              |
| invent_utility | 0.482     | 1913              | 269,771              |

So invent_guide reaches the best accuracy *and* the lowest lifetime cost (guidance makes
training cheaper too). The cheap per-task deploy number is only meaningful once ~160k–270k
evaluations have already been spent building the vocabulary. Any "≈60× cheaper than flat"
claim below is about *deploy* cost after that investment, never total cost.

## What survives

**Invention beats every non-oracle control under query scoring.** The three learned-library
conditions (0.48–0.67) all clear the fresh flat baseline (0.00), the shuffled library
(0.00), the cache (0.00), and — this is the load-bearing one — the *generous* flat solver
(0.316). The M claim that "learning a vocabulary helps" is not an artifact of demo scoring;
it holds when the solver must produce an unseen query output.

**The flat wall is real, not an under-resourcing artifact.** This was the sharpest
objection to M: the 2.5% baseline used a 3k-evaluation budget over 17 primitives, which
exhausts after length 2 plus part of length 3, so it never reaches the 6–8-op held-out
programs. N answers it directly with a budget sweep at len ≤8 (the held-out depth):

| Budget   | 500 | 1500 | 3000 | 10000 | 30000 | 100000 |
|----------|----:|-----:|-----:|------:|------:|-------:|
| acc      | 0.0 | 0.0  | 0.0  | 0.0   | 0.0   | 0.40   |

Flat search reaches 0.40 at ~80k evaluations per task (budget 100k, mean 80.3k actual).
A sufficiently generous flat solver *does* climb (0.316 at 53k evals, 0.40 at 80k), so the
wall is a compute wall, not a representational impossibility.

**The compute-gap number is not an equal-accuracy comparison.** Flat at 80k evals/task
reaches only 0.40; invent_guide at 1360 deploy evals/task reaches 0.667. These are the two
best points each method hits, but they are at *different accuracies* — the library is both
cheaper per task *and* more accurate here, so quoting a single "≈60×" ratio (80k/1360)
conflates two axes. Read it as: at the flat solver's best measured accuracy (0.40) it spends
~80k deploy evals/task, whereas the library beats that accuracy (0.667) at ~1.4k deploy
evals/task — after paying a 161k lifetime training cost. It is not "the same result 60×
cheaper"; it is a better result at far lower deploy cost, amortised over an up-front
investment. The honest core claim is narrower: flat search is not representationally blocked,
but within any budget we swept it never matches the learned library's accuracy, and the
library's per-task deploy cost is one to two orders of magnitude lower.

**Hierarchy is necessary, not decorative.** motif_oracle (the 6 true L1 motifs only)
reaches 0.316 — the same ceiling as generous flat. full_oracle (motifs *plus* all 36 L2
pairs) reaches 1.000. The held-out programs are 3–4 motifs deep; a flat-over-motifs search
still can't compose them within budget, but a library that already contains the L2 pairings
solves everything. Invention's job is exactly to discover those L2 compositions, and it
recovers 0.48–0.67 of the 1.0 that the full oracle contains.

full_oracle is a **reachability ceiling, not a learned-generalization result.** It is not an
agent that generalized; it is a hand-built library into which we inserted every L2 pairing,
so 1.000 only says "the held-out tasks are solvable within budget once the right
compositions are present." It bounds what invention *could* reach, and the gap between
invent_guide (0.667) and full_oracle (1.000) is exactly the fraction of useful compositions
the learner has not yet discovered. Do not read full_oracle as evidence that anything
generalized.

**Guidance and MDL both help; utility promotion is competitive but not better here.**
invent_guide (0.667) > invent (0.544) > invent_utility (0.482). The utility-based promotion
trials each MDL-positive candidate and keeps it when (evals-saved − definition-cost) > 0.
Important caveat on what "evals-saved" is measured against: the trial set is the *current
round's own training tasks* (`val = train[:12]` in `run_invent`), i.e. an **in-sample /
current-training slice, not a separate frozen validation stream**. It is legible and
negative-transfer-aware but it is not held-out utility, and it did not beat MDL at this
scale; MDL's description-length prior is already a decent proxy for reuse. Reported plainly
rather than tuned to win (figure 37).

## What is nuanced or negative

**Motif-only oracle is insufficient — you need the L2 layer.** motif_oracle at 0.316 means
handing the agent the correct primitives is not enough; the reusable unit is the *pairing*,
and that has to be discovered. This bounds the M story: the value is in the composition, not
the atoms.

**Negative transfer is real but small at this scale.** Adding 0…16 useless concepts to the
library (vocab 23→39) leaves accuracy flat at 0.30 but raises mean evals from 2245 to 2342
— a ~4% branching-cost tax with zero benefit. `concepts_used` stays at 5.0 throughout: the
solver never adopts the junk, it just pays to skip past it. So negative transfer shows up as
search cost, not accuracy loss, and a promotion rule that prices in branching cost would
suppress it. It does not explode at these vocabulary sizes, but it is monotonic and would
matter as the library grows.

**Population "civilization" LOSES to a single lifelong learner at this scale.** This is the
most important negative result. With matched total compute, a single persistent learner
(P=1) reaches 0.456 final accuracy and 12.7 abstractions over 6 generations. Splitting the
same task budget across 4 agents that pool discoveries into shared culture and inherit them
(P=4) reaches only 0.254 and 5.7 abstractions. Each P=4 agent sees a quarter of the tasks,
discovers less, and the shared culture does not recover the lost individual depth (P=4 tops
out at max abstraction level 1; P=1 reaches level 2). And P=4 is not even cheaper: its mean
lifetime training cost is 444k evals versus P=1's 285k, so the population spends *more*
compute for *less* accuracy. At this scale, cultural inheritance is strictly worse than one
agent thinking longer. We do not spin this: a single persistent library is lifelong
individual learning, and it should not be called "culture." Calling M a civilization result
would be unsupported by this control. Civilization may win at larger scale or with genuine
task specialisation, but we did not measure that here, and nothing in Echo currently
demonstrates a civilization advantage.

## Integrity checks

- `ground_truth_query_ok: 38/38` — every held-out task's hidden program reproduces its own
  query output (the tasks are solvable as posed).
- `oracle_solvable: 12/38` — this gate uses the **motif-only oracle** (`oracle_library()`,
  the 6 L1 motifs), *not* the full oracle. It is a lower-bound solvability sanity check:
  the motif-only library solves 12 of 38 within the 3k budget used for the gate. It is not a
  ceiling and should not be confused with full_oracle (which reaches 1.000 at its own budget
  because it also contains the L2 pairings). The gate's job is only to confirm the suite is
  not accidentally impossible, and 12/38 already clears that.
- `behavioural_leaks: 0` after dropping 2 of 40 generated tasks; 10 pooled training
  solutions tested against every held-out task on demos and query. **This audit is
  meaningful but non-exhaustive:** it only tests the specific programs that our training
  solver actually found (pooled over 3 seeds), against demos + the single query grid. A
  held-out task reproducible by some *other* program the solver never found, or on some grid
  we never posed, would not be caught. It rules out the leaks we can enumerate, not all
  possible behavioural overlap.
- cache_only cannot retrieve by task ID or exact grid; it stores solution programs keyed by
  behaviour and reaches 0.000 on held-out, confirming no memorisation shortcut.

## Worked trace (one held-out task)

Hidden program (8 base ops): `color_cycle · transpose · sym_h · sym_v · sym_h · sym_v ·
crop · keep_largest`, built from motifs `[5,2,2,0]`. Invented solution: 2 tokens `C4_L1 ·
C3_L1`, where `C4_L1 → transpose·color_cycle` and `C3_L1 → sym_h·sym_v`. Solved in 791
evaluations. Flat search over 17 primitives would need to find an 8-op sequence — outside
the 3k budget, consistent with fresh_flat 0.000. This is the mechanism the aggregate
numbers describe: the library compresses an 8-op search into a 2-token search.

## Verdict

Experiment M survives the falsification battery as **synthetic hierarchical macro
invention**: under unseen-query scoring, behavioural-leak filtering, and motif-pairing
holdout, learning a compositional vocabulary beats a fresh flat solver, a generous flat
solver, an exact-solution cache, and a size-matched useless library. The learned library is
both more accurate and one to two orders of magnitude cheaper *per deployment task* — but
that per-task figure is not equal-accuracy and does not include the 161k–270k-evaluation
lifetime cost of building the vocabulary; total-compute framing must keep both in view.
Three claims are trimmed:

1. The flat baseline is not representationally blocked — a generous flat solver reaches
   ~0.40 at ~80k evals/task, not zero — so the library's advantage is per-task deploy cost
   plus a modest accuracy edge, not an impossibility result, and the two methods are being
   compared at different accuracies.
2. The reusable unit is the L2 motif *pairing*, not the atoms: the motif-only oracle gives
   only 0.316, and full_oracle's 1.000 is a reachability ceiling from a hand-inserted
   library, not learned generalization.
3. The population "civilization" condition (P=4) underperforms a single lifelong learner
   (P=1) — 0.254 vs 0.456 — while spending more lifetime compute, so **Echo currently
   supports lifelong individual abstraction learning, not a civilization advantage.**

The canonical 97.5% from §11c was inflated by demo scoring and non-adversarial holdout; the
defensible number under unseen-query scoring is invent_guide 0.667 against a full-oracle
reachability ceiling of 1.0. This is a library-learning benchmark result on a synthetic grid
DSL. It is not ARC, and Echo is frozen here — the object-centered / Mini-ARC direction has
been moved to a separate project rather than expanded inside Echo.
