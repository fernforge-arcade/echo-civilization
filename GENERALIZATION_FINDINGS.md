# Echo Civilization — Does Culture *Generalize*, or Just Memorize?

### A falsification study of cultural accumulation

*Companion to [`REPORT.md`](REPORT.md). This report covers only the new
compositional-generalization experiment: a test deliberately built so the
headline result could fail.*

**Run:** 3 seeds (0, 1, 2) · identical hyperparameters to the headline ·
nothing tuned to make culture win · **no pretrained models.**
Reproduce: `./venv/bin/python run_generalization.py --seeds 0 1 2` (~45 s).
Raw data: [`results/generalization.json`](results/generalization.json) ·
terse auto-log: [`GENERALIZATION_REPORT.md`](GENERALIZATION_REPORT.md).

---

## TL;DR

The original headline — *culture-sharing populations reach ~0.97 on hard tasks
while solo/no-sharing agents stay near ~0.5* — had a hole big enough to sink it:
**training and evaluation drew from the same composite programs.** "Capability"
could simply be culture caching the exact compositions it had already trained on.
That is memorization, not intelligence accumulating.

So we built the version that can fail, and ran it. The verdict:

> **Outcome 1 — real compositional generalization.** On composites the
> civilization *never trained on*, sharing populations solved **62 %** of
> never-seen **depth-3** programs versus **6–13 %** for the no-sharing
> baselines — a **+0.49** gap — with **zero leakage** of held-out functions into
> the culture, every test verified solvable, and the effect **absent at
> generation 0** and emerging only as intermediate abstractions accumulate.

The civilization is not memorizing compositions. It is accumulating reusable
**intermediate abstractions** and recombining them to solve problems it has never
encountered. This is the DreamCoder question — *do you build a growing library of
reusable concepts?* — answered, for this system, *yes.*

---

## 1. Why the headline was not enough

In the headline experiment the training world (`tier="all"`) and the held-out
evaluation suite were both drawn from the **same** set of composite programs
(`reverse∘inc`, `double∘reverse`, …); only the random input *strings* differed.

That design cannot separate two very different stories:

| Story | What "0.97 capability" would mean |
|---|---|
| **Memorization** | Culture caches the exact composite programs it trained on, and replays them on new inputs. Impressive bookkeeping; not accumulating intelligence. |
| **Generalization** | Culture accumulates *parts* and recombines them on demand, solving compositions it never saw. Genuine cumulative capability. |

Both predict ~0.97 on the headline suite. To tell them apart you must test on
compositions the system **was never trained on.**

---

## 2. A test designed to fail

### 2.1 The split

We separate the program space into trained and never-trained sets:

- **Train** on all primitive tasks (`copy, reverse, inc1, inc2, dec1, count,
  first, last`) **+ a random 50 % subset of the depth-2 composites.**
- **Test** on a **disjoint, never-trained** set of composites, **stratified by
  depth**: novel depth-2 and novel depth-3.

From a universe of **29 non-degenerate depth-2 composites** (each behaviourally
distinct from every primitive, verified on a 40-string probe battery), the split
was **14 trained / 15 held out**, plus **24 held-out depth-3** composites.

<details><summary><b>The 14 trained depth-2 abstractions</b></summary>

`dec1 then double`, `dec1 then dedup`, `inc2 then dedup`, `reverse then dedup`,
`inc2 then double`, `double then dedup`, `last then double`, `reverse then
double`, `dedup then count`, `dec1 then last`, `inc2 then last`,
**`reverse then inc1`**, `double then count`, `first then double`

</details>

<details><summary><b>The 24 never-trained depth-3 test programs</b></summary>

`double then count then reverse`, `reverse then inc2 then dedup`, `dec1 then
reverse then dedup`, `inc1 then inc2 then double`, `last then double then inc2`,
`double then count then count`, `first then double then dec1`, `first then double
then inc1`, `inc2 then last then inc2`, `last then double then count`, `last then
double then inc1`, `dec1 then double then reverse`, `inc2 then inc2 then double`,
**`inc2 then double then reverse`**, `double then dedup then count`, `double then
dedup then reverse`, **`inc2 then reverse then inc1`**, `inc2 then dedup then
inc1`, `dec1 then dec1 then last`, `inc2 then double then dedup`, `inc1 then
reverse then double`, `double then dedup then dec1`, `dec1 then dec1 then double`,
`dec1 then dec1 then dedup`

</details>

### 2.2 Why depth-3 is the decisive test

An agent's recombination at evaluation time is **pairwise only** — it concatenates
two programs it already knows (`product(known, known)` in `solve_task`). The
consequences are sharp:

- A **non-degenerate depth-3** program *cannot* be reached by gluing two
  primitives together (that only yields depth-2). It can be reached **only** if
  the agent already holds a **depth-2 building block** and bolts a primitive onto
  it.
- We constructed every held-out depth-3 task so its depth-2 sub-program **is in
  the training set.** For example, the novel `inc2 then reverse then inc1`
  decomposes as the primitive `inc2` + the *trained* abstraction
  `reverse then inc1`.

So solving a novel depth-3 task is a direct measurement of whether the
civilization **kept a useful intermediate abstraction around and redeployed it.**
Primitives alone are not enough. This is exactly the capability the brief asks
about: *"does generation N have capabilities generation 1 could not, because
knowledge accumulated?"*

(Depth-2 held-outs only require recombining two primitives, so even no-sharing
agents that personally rediscovered the primitives can sometimes pass them — we
expected, and saw, a smaller gap there.)

`double` and `dedup` are deliberately **never** standalone primitive tasks; they
appear only *inside* trained depth-2 composites. The only way to ever use them is
to inherit an abstraction that contains them.

### 2.3 Methodology guards (so a null would be trustworthy)

| Guard | What it rules out |
|---|---|
| **Frozen evaluation** — `allow_discovery=False` *and* `learn_at_solve=False` | No fresh search and no test-time learning: an agent can't learn a held-out depth-2 mid-eval and reuse it. Measures only knowledge accumulated during *training*. |
| **Generous eval budget** (4000) | Pairwise search is never the bottleneck. A bigger budget can only *help the weaker, no-culture conditions*, so it is the conservative choice. |
| **Held-out *query* decides correctness** (not just the demo examples) | A program that matches the few examples by luck but is wrong still scores 0. |
| **Oracle check** | An agent that knows all primitives + all trained depth-2 solves **100 %** of held-2 and **100 %** of held-3. Every scored task is solvable-in-principle, so a null can't be blamed on impossible tasks. |
| **Behavioural leak check** | The final culture is dumped and compared *by behaviour* to every held-out program — confirming the answers weren't memorized during training. |
| **Multi-seed, fixed hyperparameters** | The split is a fixed benchmark; seeds vary only training stochasticity; nothing was tuned to make culture win. |

---

## 3. A worked example

A real trace: a culture-bearing agent facing the **never-trained** depth-3 task
`inc2 then reverse then inc1`, shown only four input→output demonstrations.

```
Demonstrations the agent sees:
   'gfgbb' -> 'eebab'
   'bedcg' -> 'bfghe'
   'fahg'  -> 'bcda'
   'fdc'   -> 'fga'
Held-out query:  'deaa'   (true answer: 'ddhg')

Agent's frozen solution (no search, no learning):
   inc2('deaa')          = 'fgcc'
   reverse('fgcc')       = 'ccgf'
   inc1('ccgf')          = 'ddhg'   ✓ correct
```

The agent never trained on `inc2 then reverse then inc1`. It solved it by composing
the **primitive `inc2`** with the **inherited depth-2 abstraction `reverse then
inc1`** — a part it had picked up from the culture. That is generalization by
recombination, made concrete.

---

## 4. Results

**Frozen solve rate, mean ± SD over seeds {0, 1, 2}:**

| Condition | trained depth-2 (new inputs) | novel depth-2 | **novel depth-3** | culture | skills/agent |
|---|---|---|---|---|---|
| A — single agent | 0.20 ± 0.04 | 0.11 ± 0.04 | **0.13 ± 0.04** | 0 | 4.0 |
| B — population, no sharing | 0.13 ± 0.01 | 0.20 ± 0.00 | **0.06 ± 0.01** | 0 | 4.1 |
| **C — skill sharing/inheritance** | 0.64 ± 0.01 | 0.86 ± 0.02 | **0.60 ± 0.06** | 47 | 17.3 |
| **D — full civilization** | 0.63 ± 0.02 | 0.85 ± 0.02 | **0.62 ± 0.02** | 46 | 16.8 |

**Culture − baseline gap** (best of C/D minus best of A/B):
**+0.66** on novel depth-2, **+0.49** on novel depth-3.

![Compositional generalization by depth](figures/16_generalization_bars.png)

*Figure A — Frozen solve rate by condition, on three strata. Left: trained
programs on **new inputs** (the in-distribution control — what the headline
measured). Middle: **novel depth-2** (needs only primitives). Right: **novel
depth-3** (needs an inherited intermediate abstraction). Culture wins everywhere,
but the depth-3 group is the one that proves abstraction, not caching.*

### 4.1 Generalization is *built*, not present at birth

![Accumulation of generalization over generations](figures/17_generalization_curve.png)

*Figure B — Solve rate on never-trained depth-3 programs, generation by generation
(seed 0). At generation 0 every condition is near **0.05** — nobody has a culture
yet. The sharing conditions then climb to **~0.66 by generation 5** and hold it,
exactly as depth-2 abstractions enter the shared repository and are inherited. The
no-sharing baselines never leave the floor.*

This curve is the strongest single piece of evidence. The depth-3 capability is
**absent when culture is empty** and **emerges as culture accumulates** — it is a
property of the *civilization*, not of any individual agent.

### 4.2 No leakage, every task solvable

- **Oracle solvability:** held-2 75/75, held-3 120/120 (100 %). No null could be
  blamed on impossible tasks.
- **Depth-3 leaks: NONE** in any condition. No held-out depth-3 function was ever
  stored in the culture, so every depth-3 success genuinely routed through
  pairwise recombination of an inherited abstraction.
- **One depth-2 leak** (`inc1 then first`, in C/D): a *commutative twin* of a junk
  skill that 3-example training occasionally stores (`first∘inc1 ≡ inc1∘first`).
  It touches only the easy depth-2 stratum, not the decisive depth-3 result, and
  is reported here for full transparency — it also shows the leak detector works.

---

## 5. Interpretation

**The headline survives, upgraded.** What looked like it *might* be cultural
memorization is, on a test built to expose memorization, **genuine compositional
generalization.** Three facts pin this down:

1. **The test programs were never trained** (disjoint split, behaviourally
   verified) and **never entered the culture** (leak check).
2. **A non-degenerate depth-3 program is unreachable from primitives alone** under
   pairwise recombination — so a depth-3 success is *mechanically forced* to use a
   stored depth-2 abstraction.
3. **The capability appears only as culture accumulates** (Fig. B), and is **+0.49
   above** identical agents that lack the sharing channel.

The no-sharing baselines fail depth-3 not from lack of compute (the eval budget is
generous and frozen) but because each agent only ever personally rediscovers a few
depth-2 composites in its own lifetime — and that knowledge **dies with it.**
Culture is precisely the mechanism that makes intermediate abstractions *persist
across agents and generations* and *recombine* into capabilities no single agent
ever built. That is the civilization-scale answer the project set out to test.

---

## 6. Honest caveats

- **One fixed split.** Seeds vary training stochasticity, not the benchmark
  itself; a multi-split study would harden the claim further.
- **Partial, not perfect.** Depth-3 generalization is ~0.6, not ~1.0, because not
  every trained abstraction is reliably re-discovered and retained in every run.
  The scientific signal is the **gap to the baselines**, not the absolute level.
- **Training noise.** Tasks use 3 examples, so the culture accrues some
  behaviourally-redundant "junk" skills (the source of the single depth-2 leak).
  This is identical to the headline setup and cannot affect the frozen,
  query-checked evaluation.
- **Narrow domain.** String transforms. The mechanism is general, but replication
  in richer domains (the computer / firm worlds in `REPORT.md` are first steps)
  would strengthen external validity.

---

## 7. How to reproduce

```bash
./venv/bin/python run_generalization.py --seeds 0 1 2
```

Writes `figures/16_generalization_bars.png`,
`figures/17_generalization_curve.png`, `results/generalization.json`, and the
auto-log `GENERALIZATION_REPORT.md`. The split, oracle check, and leak check live
in `echo_civilization/generalization.py`; the frozen-eval flag (`learn_at_solve`)
is in `Agent.solve_task`.

*A clean null here would have been reported just as loudly — the whole point was a
test that could fail. It didn't.*
