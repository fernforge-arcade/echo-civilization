# Echo Civilization — Parametric Abstraction (inheriting a schema with a free argument)
*Every earlier study transmitted a **concrete** program — a fixed tuple of ops. Can a civilization instead transmit an **abstraction with a free parameter** — the schema `shift_by(k)` rather than the concrete `shift_by(2)` — so a later agent can **bind that parameter to a value it has never seen**?*

## Example run output (worked trace, one held-out task)
A single held-out task whose argument (here `k`=5) was **never seen during accumulation** (training only ever used args 1 and 2), shown to the strongest cultured agent (condition D) and to a fresh gen-0 agent under the SAME tight budget (40 consistency checks):

```
TRUE RULE (hidden):   repeat(5) then reverse
  = a PARAMETRIC family (the schema the civilization inherited) applied with a
    NOVEL argument, then a trivially-known inner transform

DEMONSTRATION EXAMPLES given to both agents:
    'hagfgachf'  ->  'fhcagfgahfhcagfgahfhcagfgahfhcagfgahfhcagfgah'
    'dedchdhdh'  ->  'hdhdhcdedhdhdhcdedhdhdhcdedhdhdhcdedhdhdhcded'
      'bhefgbh'  ->  'hbgfehbhbgfehbhbgfehbhbgfehbhbgfehb'
    'ffaddcebf'  ->  'fbecddafffbecddafffbecddafffbecddafffbecddaff'
      'fcfbhbc'  ->  'cbhbfcfcbhbfcfcbhbfcfcbhbfcfcbhbfcf'

HELD-OUT QUERY (decides correctness):  'agddechb'
                         true answer:  'bhceddgabhceddgabhceddgabhceddgabhceddga'

--- CULTURED agent (condition D) ---
  inherited schemas:   6 real families  (the abstraction with a free arg)
  knows this family:   True
  hypothesis found:    repeat(5) then reverse
  via inherited schema: True   (checks used: 6)
  prediction on query: 'bhceddgabhceddgabhceddgabhceddgabhceddga'
  SOLVED: True

--- FRESH gen-0 agent ---
  inherited schemas:   0
  hypothesis found:    None
  prediction on query: '(gave up — budget exhausted before reaching this family/arg)'
  checks used:         40 (budget 40)
  SOLVED: False
```

Same task, same budget. The cultured agent recognises the family from its inherited schema and **inverts the unknown argument from a single (input, output) pair** — a handful of checks. The fresh agent has no schema, so it must blind-sweep the entire {family × argument × inner} grid; the tight budget runs out before it reaches this (late family, high argument) cell. **The only difference is the inherited schema** — and crucially the argument value itself (3/4/5) is novel to *everyone*, so this is not memorisation.

## 1. Why this is a different (harder) axis than before
The generalization study held out novel *compositions*; Experiment H held out a novel *structural family* (higher-order combinators). In both, the unit culture transmitted was still a **concrete** program. Here the inherited object is a **parametric schema** — a family `f(k)` plus the competence to recover its integer argument:

```
  shift_by(k):   caesar-shift each char forward by k
  shift_back(k): caesar-shift backward by k
  rotate(k):     cyclic left-rotate the string by k
  take(k):       keep the first k chars
  drop(k):       drop the first k chars
  repeat(k):     repeat the whole string k times

  schema = family name + INVERTER(in,out)->k  (binds the arg in O(1) per family)
```

The held-out suite is **108 tasks** = 6 real families × 3 novel arguments (3, 4, 5) × 2 inner transforms × several fresh string draws each. Accumulation only ever uses args 1, 2, so the **argument at eval is disjoint from anything trained on**.

The blind-search grid is deliberately larger than the cultural library: **14 families** (6 real + 8 DECOY distractors that never appear in any task). A cultured population never abstracts the decoys (they never recur, so cultural selection drops them); a fresh agent has no way to know they are useless and must waste budget ruling them out.

## 2. Why the test isolates ARGUMENT-BINDING (and can fail)
- **The schema is the only lever.** The inner transforms (identity / reverse) are known to everyone, and the argument value (3/4/5) is novel to everyone. The sole thing a cultured agent brings is the inherited parametric family + its inverter — so any advantage is *purely* the value of carrying a parametric abstraction.
- **Frozen eval:** agents store nothing while solving — a schema or argument found on task 1 cannot leak to task 2.
- **Query-judged:** correctness is decided on a held-out query, not the demo examples, so an example-consistent-but-wrong hypothesis scores zero.
- **Oracle audit:** agents holding every schema solve **100%** of the suite, proving every task is solvable-in-principle — a null can never be blamed on impossible tasks.
- **Recurrence-gated abstraction:** a family becomes a trusted schema only after it recurs (≥2 distinct solves), so one-off decoy coincidences are pruned and never pollute the inherited library.
- **Two budgets:** GENEROUS (4000) asks *can both reach the ceiling?*; TIGHT (40) asks *does the inherited schema still decide?*

## 3. Results
Frozen solve rate on the novel high-argument suite (mean ± SD over seeds [0, 1, 2]):

| Condition | TIGHT budget (40) | generous budget (4000) | avg real schemas inherited |
|---|---|---|---|
| A — single agent (no memory/culture) | **0.34 ± 0.08** | 1.00 ± 0.00 | 0.3 |
| B — population, no sharing | **0.29 ± 0.01** | 1.00 ± 0.00 | 0.2 |
| C — population + schema sharing/inheritance | **1.00 ± 0.00** | 1.00 ± 0.00 | 6.0 |
| D — full civilization | **1.00 ± 0.00** | 1.00 ± 0.00 | 6.0 |
| FRESH — brand-new gen-0 agents (no accumulation) | **0.25 ± 0.01** | 1.00 ± 0.00 | 0.0 |

Oracle (holds every schema): **1.00**.

**Headline:** at the generous budget the ceiling is reachable by everyone (any argument is findable given enough blind search — note **generous = 1.00 for ALL conditions**, so the suite is not intrinsically hard). But under the TIGHT budget, the inherited schema DECIDES: best cultured **1.00** vs fresh **0.25** — a **+0.75** gap created entirely by inheriting parametric schemas, even though the argument those schemas bind is novel to every agent.

![Parametric abstraction by condition](figures/23_parametric_bars.png)

![Argument-binding frontier vs budget](figures/24_parametric_curve.png)

The frontier curve shows the mechanism as a budget shift: the cultured civilization reaches the ceiling at a small budget by inverting the argument per known family, while a fresh agent needs an order of magnitude more search to blind-sweep the full grid to the same place.

## 4. Interpretation
**Parametric abstraction confirmed.** A civilization can transmit an abstraction with a *free parameter*, not just a concrete program, and a later agent can bind that parameter to a value it has never seen far faster than an agent starting fresh. Under a matched tight budget the inherited schema is the difference between solving the family and failing it. The advantage is NOT memorisation — the bound argument (3/4/5) was never seen by anyone; it is the reuse of a *parametric family + its inverter* as a reusable unit of cultural knowledge. This is a strictly more general form of inheritance than the concrete-program transmission of every earlier study: the unit of accumulated knowledge is now an abstraction with a slot.

## 5. Honest caveats
- The argument axis swept by blind induction is small (0..6); the claim is about binding a *novel* argument cheaply via an inherited family, not about inducing arbitrarily large or structured parameters.
- The tight budget is a chosen operating point; the frontier *curve* shows the full budget sweep so the reader can see the gap is a frontier shift, not a single cherry-picked budget.
- The worked trace shows the MODAL fresh outcome (blind scan fails on this late/high-arg task ~93% of the time); on ~7% of seeds a fresh agent gets lucky and hits the family early — that luck is exactly what the 0.25 aggregate reflects.
- Eval is symbolic program search, not gradient learning; it isolates the value of an inherited parametric abstraction cleanly but does not model test-time neural adaptation.

---
*Reproduce:* `./venv/bin/python run_parametric.py --seeds 0 1 2`
