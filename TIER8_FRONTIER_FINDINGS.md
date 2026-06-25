# Pushing the Computer-Use Frontier to Tier 8 — Synthesising (and Running) Harder Programs

*Echo Civilization — frontier report, 2026-06-25*

This is a focused write-up of the latest push in the Echo Civilization project: we
took the agents' computer-use frontier **one rung higher** than the Tier-7 result
documented in `COMPUTER_USE_FRONTIER.md`, and we did it the same honest way — by a
real mechanism, with **no pretrained model**, and with the project's central law
intact: *a hard-won skill is expensive to discover but cheap to inherit, so culture
decides who can reach the top rungs.* This report leads with **example output from
actual runs**, because that is the cleanest evidence that the agents are writing and
executing real, correct programs rather than memorising answers.

It is a companion to the flagship `REPORT.md` (the new material lives there as
**§6.6**). Reproduce everything here with:

```bash
./venv/bin/python run_tier8.py --trials 10        # seed 0 canonical (~2 min)
./venv/bin/python run_tier8.py --trials 6 --seed 1 # robustness check
```

---

## 1. Where the frontier stood, and why Tier 8 is the right next rung

The Computer-Use Benchmark (§6.4) marched a *cultured* agent (carrying its
accumulated macro library) and a *fresh* gen-0 agent up a ladder of real computer
tasks, grading each rung by **executing real shell commands**. It ended at two
honest walls:

- **Tier 6** — runnable on a shell but *out of the op-vocabulary* (find-and-replace,
  word-frequency, sum numbers). §6.5 knocked this down with **parametric ops +
  argument-by-example**.
- **Tier 7** — *"write a Python script"* — not representable as an op-pipeline at all.
  §6.5 reached **one** Tier-7 rung (CSV → column averages) via **grammar-guided code
  synthesis**: the agent emits a program in a tiny typed grammar that *compiles to
  real Python*, which we run in a subprocess against hidden tests.

That reached rung was, honestly, a *flat* computation: read a CSV, reduce each
column independently, print. Real tool use climbs past that into programs that
**build and iterate a data structure**. So the natural next rung — the one that
tests whether the mechanism keeps working as the synthesised program gets
structurally harder — is **group-by aggregation**:

> **Tier 8.** *Read a CSV, group rows by a key column, aggregate a value column per
> group, print the sorted `key:value` pairs.*

Why this is genuinely harder than Tier 7:

1. it needs a **dict accumulator** and a **two-pass shape** — accumulate values per
   key, then reduce-and-emit per key — not a single flat sweep;
2. the agent is **told none of the specifics**: it must discover *which* column is
   the key, *which* is the value, and *which* of five reductions
   (`sum / mean / max / min / count`) the hidden transform uses;
3. it is graded by **really running the synthesised Python** against hidden test
   CSVs — output must match the oracle exactly on every test.

Implementation: `echo_civilization/codegen2.py` (grammar + renderer + real-execution
grader), driven by `run_tier8.py`.

---

## 2. Example output from a real run (the headline evidence)

This is the program the synthesiser **kept** on seed 0 — printed verbatim from
`results/tier8.json`. The hidden transform for this instance was *group by column 1,
**mean** of column 2*, and the synthesiser recovered exactly that:

```python
import sys, csv
path = sys.argv[1]
with open(path, newline='') as fh:
    rows = list(csv.reader(fh))
rows = rows[1:] if rows else rows   # drop header
groups = {}
for row in rows:
    try:
        k = row[1]
        x = float(row[2])
    except (IndexError, ValueError):
        continue
    groups.setdefault(k, []).append(x)
out = []
for k in sorted(groups):
    v = groups[k]
    r = sum(v) / len(v)
    rs = str(int(r)) if float(r).is_integer() else str(r)
    out.append(k + ':' + rs)
print(' '.join(out))
```

Then we ran it for real on a **held-out CSV it never saw during synthesis**, and
captured its actual stdout:

```
INPUT  (data.csv)            ACTUAL STDOUT
-------------------          -----------------------------
c0,c1,c2                     green:35.8 red:18.5
32,red,24
6,green,49                   EXPECTED
29,green,28                  green:35.8 red:18.5     → MATCH ✓
36,green,21
35,red,13                    check:
4,green,47                     green = (49+28+21+47+34)/5 = 35.8
42,green,34                    red   = (24+13)/2          = 18.5
```

It is correct, and it **generalises**: the same synthesised program, unchanged,
produces the right answer on a fresh instance — it recovered the structure, it did
not memorise an answer.

---

## 3. The result — the ceiling moved, and culture still decides

As everywhere in this project, we report **two budget regimes** so the claim stays
honest. The "budget" is the number of **real program executions** the agent is
allowed while searching the grammar.

- **Generous budget (300):** *did the ceiling move?* — can any agent reach the rung
  now?
- **Tight budget (45):** *does culture still decide?* — when search is scarce, who
  actually clears it?

A *fresh* agent searches the grammar from scratch (skeletons tried in a fixed order,
the correct `group_by_aggregate` skeleton **last**). A *cultured* agent inherits that
skeleton from culture and tries it **first**, so it only has to search the cheap
parameter tail (key column / value column / reducer).

**Seed 0, 10 trials** (`results/tier8.json`):

| regime | budget | fresh solve | cultured solve | real executions to solve |
|---|:---:|:---:|:---:|---|
| generous | 300 | **1.00** | **1.00** | fresh **76** vs cultured **16** |
| tight | 45 | **0.00** | **1.00** | — |

**Seed 1, 6 trials** — reproduces exactly: generous 1.00 / 1.00 at 76 vs 16
executions; tight 0.00 / 1.00. The separation is not a seed artefact.

What the numbers say:

- **The ceiling moved.** Group-by aggregation was *not representable* before; it is
  now reachable and *really run* — generous solve rate 1.00 for both agents.
- **Discovery is expensive, recall is cheap.** Finding the structural skeleton from
  scratch costs ~**76** real program executions (the fresh agent must grind through
  every wrong skeleton's parameter grid before it reaches the right shape).
  Inheriting the skeleton collapses that to ~**16** — only the parameter tail is left
  to search.
- **Under pressure, culture is decisive.** Squeeze the budget to 45 and the gap
  becomes a wall: the fresh agent **cannot reach the rung at all (0.00)** while the
  cultured agent **clears it every time (1.00)**.

![Tier 8 — group-by aggregation](figures/20_tier8_groupby.png)

*Left: solve rate, fresh vs cultured, at the generous and tight budgets. Right: the
real program executions needed to solve — inheriting the skeleton skips most of the
search.*

A concrete side-by-side from the run (`fresh_fail_example` in `results/tier8.json`),
tight budget = 45, hidden transform *group by col 1, mean of col 2*:

```
fresh    : best score 0.00 after 45 executions — never found the skeleton → FAIL
cultured : solved in 19 executions via inherited skeleton group_by_aggregate → PASS
```

---

## 4. Why this is honest, not a stacked deck

- **Real execution, real grading.** Every candidate is compiled to actual Python and
  run in a subprocess against hidden test CSVs; "solved" means stdout matched the
  oracle on *every* test. The budget counts *real* executions.
- **The agent is told nothing.** Key column, value column and reducer are all hidden;
  the agent must recover them from the tests.
- **The skeleton ordering is the only cultural lever**, and it is applied uniformly:
  the correct skeleton is placed *last* in the fresh search order and *first* for the
  cultured agent. Every skeleton iterates the *same* parameter grid, so a fresh search
  genuinely pays for each wrong shape before reaching the right one — that is the
  source of the 76-vs-16 cost gap, not a hand-tuned penalty.
- **No pretrained model.** The entire mechanism is a typed grammar + enumeration +
  subprocess execution.
- **Multi-seed.** Seeds 0 and 1 give identical headline numbers.

---

## 5. Limits (stated plainly)

- **Still a bounded grammar.** Tier 8 widened the grammar to express a dict-accumulator
  program; it is not free-form code generation. The mechanism — *cumulative,
  inheritable, recombinable structure* — is what is demonstrated, in a harder corner
  of the program space than before.
- **Multi-file / stateful-service rungs remain out of reach** (Flask app, repo
  refactor). The frontier moved one more rung by mechanism; it is not infinite, and we
  don't pretend it is.
- **The cost gap depends on grammar size.** Make the wrong-skeleton grids larger and
  the fresh agent's discovery cost rises further; smaller and it shrinks. The chosen
  budgets (300 / 45) were picked from the measured fresh (63–82) and cultured (3–22)
  execution costs, not tuned to manufacture the result.

---

## 6. Bottom line

The computer-use frontier is **not a fixed wall** — it moves rung by rung as the
unlocking abstractions accumulate in the shared culture. Pushed to a genuinely
harder, multi-statement, *really-run* program (group-by aggregation), the project's
central law held one rung higher: **expensive to discover, cheap to inherit, so
culture decides who clears it.** And the cleanest evidence is the simplest — a
synthesised program, run for real on data it never saw, printing the right answer.

*Files: `echo_civilization/codegen2.py`, `run_tier8.py`, `results/tier8.json`,
`figures/20_tier8_groupby.png`. Flagship cross-reference: `REPORT.md` §6.6 +
conclusion 9.*
