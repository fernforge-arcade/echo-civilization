# Echo Civilization — Compositional Generalization Experiment
*Can the civilization’s “capability” survive a test designed to fail — novel compositions it never trained on?*
> Multi-seed run (seeds [0, 1, 2]); identical hyperparameters to the headline A/B/C/D; nothing tuned to make culture win.

## 1. Why the headline was not enough
The headline result (C/D ≈ 0.97 vs A/B ≈ 0.5 on hard tasks) is real and reproducible, but it **cannot distinguish memorization from generalization**: training (`tier="all"`) and the eval suite were drawn from the *same* composite programs — only the input strings differed. So the measured "capability" could just be culture caching the exact compositions it trained on.

## 2. A test that can fail
- **Train** on all primitive tasks + a SUBSET of depth-2 composites.
- **Test** on a DISJOINT, never-trained set of composites, stratified by depth.
- **Depth-3 is the real test.** Recombination in `solve_task` is *pairwise* (`product(known, known)`), so a depth-3 target is reachable at eval only if the agent holds a depth-2 *building block*. The held-out depth-3 tasks are built so their depth-2 sub-program is in the training set. Solving them measures whether culture accumulates and redeploys **intermediate abstractions** (the DreamCoder question), not just primitives.
- `double` and `dedup` are never standalone primitive tasks — they appear only *inside* trained depth-2 composites, so the only way to use them is via an inherited abstraction.

### Methodology guards
- **Frozen eval:** `allow_discovery=False` AND `learn_at_solve=False` (no test-time learning), with a generous budget so pairwise search is never the bottleneck (a bigger budget can only *help the no-culture baselines*).
- **Correctness on the held-out query**, not just the demo examples, decides each task — spurious example-consistency cannot inflate scores.
- **Oracle check:** an agent that knows all primitives + all train-2 composites solves **100%** of depth-2 and **100%** of depth-3 held-outs, so every scored task is solvable-in-principle (unsolvable ones were dropped: held-2 75/75, held-3 120/120).
- **Leak check (stratified):** the final culture was dumped and compared *by behaviour* against every held-out program. **Depth-3 leaks (the stratum that matters): NONE** — so no held-out depth-3 function was memorized; every depth-3 success must route through recombination of an inherited intermediate abstraction. Depth-2 leaks: `inc1 then first` (a spurious commutative twin of a junk skill the 3-example training stored; it touches only the easy depth-2 stratum).

### The split
From a universe of **29** non-degenerate depth-2 composites (behaviourally distinct from every primitive): **14 trained**, **15 held out**; plus **24 held-out depth-3** composites whose depth-2 sub-program is in the training set.

<details><summary>Trained depth-2 composites</summary>

`dec1 then double`, `dec1 then dedup`, `inc2 then dedup`, `reverse then dedup`, `inc2 then double`, `double then dedup`, `last then double`, `reverse then double`, `dedup then count`, `dec1 then last`, `inc2 then last`, `reverse then inc1`, `double then count`, `first then double`

</details>

<details><summary>Held-out depth-3 composites (never trained)</summary>

`double then count then reverse`, `reverse then inc2 then dedup`, `dec1 then reverse then dedup`, `inc1 then inc2 then double`, `last then double then inc2`, `double then count then count`, `first then double then dec1`, `first then double then inc1`, `inc2 then last then inc2`, `last then double then count`, `last then double then inc1`, `dec1 then double then reverse`, `inc2 then inc2 then double`, `inc2 then double then reverse`, `double then dedup then count`, `double then dedup then reverse`, `inc2 then reverse then inc1`, `inc2 then dedup then inc1`, `dec1 then dec1 then last`, `inc2 then double then dedup`, `inc1 then reverse then double`, `double then dedup then dec1`, `dec1 then dec1 then double`, `dec1 then dec1 then dedup`

</details>

## 3. Results
Frozen solve rate (mean ± SD over seeds [0, 1, 2]), per condition:

| Condition | trained depth-2 (new inputs) | NOVEL depth-2 | **NOVEL depth-3** | culture | avg skills/agent |
|---|---|---|---|---|---|
| A — single agent (no memory/culture) | 0.20 ± 0.04 | 0.11 ± 0.04 | **0.13 ± 0.04** | 0 | 4.0 |
| B — population, no sharing | 0.13 ± 0.01 | 0.20 ± 0.00 | **0.06 ± 0.01** | 0 | 4.1 |
| C — population + skill sharing/inheritance | 0.64 ± 0.01 | 0.86 ± 0.02 | **0.60 ± 0.06** | 47 | 17.3 |
| D — full civilization | 0.63 ± 0.02 | 0.85 ± 0.02 | **0.62 ± 0.02** | 46 | 16.8 |

Culture−baseline separation (best of C/D minus best of A/B): **+0.66** at novel depth-2, **+0.49** at novel depth-3.

![Generalization by depth](figures/16_generalization_bars.png)

![Accumulation of generalization over generations](figures/17_generalization_curve.png)

## 4. Interpretation
**Outcome 1 — real compositional generalization.** Culture wins on BOTH novel depth-2 and novel depth-3 held-outs. Since the depth-3 tasks were never trained and pairwise recombination can only reach them via an inherited depth-2 building block, this is genuine evidence that the civilization accumulates and redeploys *intermediate abstractions*, not just primitives.

Mechanistic note: because the held-out depth-3 programs are behaviourally novel (confirmed non-degenerate and absent from the culture), and pairwise recombination cannot reach a non-degenerate depth-3 from two primitives, any depth-3 success **must** route through a known depth-2 program — an inherited intermediate abstraction. The no-sharing baselines fail depth-3 not for lack of search budget (the eval budget is generous and frozen) but because each agent only personally discovered a few depth-2 composites in its own lifetime, and that knowledge dies with it. Culture is what makes the depth-2 abstractions persist and recombine.

## 5. Honest caveats
- The split is fixed (one benchmark); seeds vary only training stochasticity. A multi-split study would further harden the claim.
- Depth-3 success is partial (not ~100%) because not every trained depth-2 abstraction is reliably re-discovered and retained each run; the signal is the *gap* to the baselines, not the absolute level.
- Training still uses 3-example tasks, so the culture contains some behaviourally-redundant junk skills; this is identical to the headline setup and does not affect the frozen, query-checked eval.
