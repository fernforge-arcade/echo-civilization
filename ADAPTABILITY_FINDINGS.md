# Echo Civilization — Adaptability to a Novel Task Family
*Can a civilization that accumulated a library of abstractions ADAPT to a task type it has **never seen in its entirety** — faster than an agent starting fresh?*

## Example run output (worked trace, one held-out task)
A single novel task, drawn from a family **no agent trained on**, shown to the strongest cultured agent (condition D) and to a fresh gen-0 agent under the SAME tight budget (45 consistency checks):

```
TRUE RULE (hidden):   map_reversed(reverse then inc1)
  = apply a higher-order COMBINATOR (novel to everyone) around an inner
    depth-2 transform (an abstraction the cultured population accumulated)

DEMONSTRATION EXAMPLES given to both agents:
               'ab gg ed'  ->  'ef hh cb'
                'ehf hee'  ->  'ffa gaf'
               'ccba beh'  ->  'afc bcdd'
        'adb hgb gfaa fa'  ->  'bg bbgh cha ceb'
                 'had bh'  ->  'ac eba'

HELD-OUT QUERY (decides correctness):  'hfg ef ced'
                         true answer:  'efd gf hga'

--- CULTURED agent (condition D) ---
  inherited library:   14 known programs
  already knows inner: True
  hypothesis found:    map_reversed(reverse then inc1)
  via known abstraction: True   (checks used: 27)
  prediction on query: 'efd gf hga'
  SOLVED: True

--- FRESH gen-0 agent ---
  inherited library:   0 known programs
  hypothesis found:    None
  prediction on query: '(gave up — budget exhausted on single-op inner candidates)'
  checks used:         45 (budget 45)
  SOLVED: False
```

Same task, same budget. The cultured agent recalls the inner abstraction and only has to search the tiny novel-combinator axis; the fresh agent spends its whole budget on single-op inner candidates and never reaches the depth-2 inner the task needs. **The only difference is the inherited library.**

## 1. Why this is harder than compositional generalization
The generalization study held out novel *compositions*, but every test task was still the same KIND of task the population trained on: apply one program to one string. Here the eval family adds a structural layer **nobody ever trained on** — a higher-order *combinator* `C` that decides HOW an inner transform `f` is mapped across a multi-token input:

```
  input:  "abc de fgh"      inner f = (reverse, inc1)
  map_each(f):     "deb fe ihg"      f on each token, order kept
  map_reversed(f): "ihg fe deb"      token order reversed, then f each
  first_only(f):   "deb de fgh"      f on the first token only
  last_only(f):    "abc de ihg"      f on the last token only
  map_evens(f):    "deb de ihg"      f on even-indexed tokens only
```

The suite is **100 tasks** = 5 combinators × 5 inner programs × several fresh string draws each. Combinators: `map_each`, `map_reversed`, `first_only`, `last_only`, `map_evens`.

## 2. Why the test isolates ADAPTABILITY (and can fail)
- **The combinator confers no inherited edge.** Neither cultured nor fresh agents have ever seen a combinator, so both must discover it at eval time. The ONLY thing a cultured agent brings is its library of inner abstractions `f`. So any cultured advantage is *purely* the value of carrying abstractions into an unfamiliar problem.
- **Frozen eval:** agents store nothing while solving — the combinator discovered on task 1 cannot leak to task 2.
- **Query-judged:** correctness is decided on a held-out query, not the demo examples, so an example-consistent-but-wrong hypothesis scores zero.
- **Oracle audit:** agents that already know the inner `f`'s (but no combinator) solve **100%** of the suite, proving every task is solvable-in-principle — a null can never be blamed on impossible tasks.
- **Two budgets:** GENEROUS (4000) asks *did the ceiling move and can both reach it?*; TIGHT (45) asks *does culture still decide?*

## 3. Results
Frozen solve rate on the novel family (mean ± SD over seeds [0, 1, 2]):

| Condition | TIGHT budget (45) | generous budget (4000) | avg known skills |
|---|---|---|---|
| A — single agent (no memory/culture) | **0.55 ± 0.25** | 1.00 ± 0.00 | 4.0 |
| B — population, no sharing | **0.47 ± 0.03** | 1.00 ± 0.00 | 3.8 |
| C — population + skill sharing/inheritance | **0.90 ± 0.02** | 1.00 ± 0.00 | 15.0 |
| D — full civilization | **0.91 ± 0.02** | 1.00 ± 0.00 | 14.6 |
| FRESH — brand-new gen-0 agents (no accumulation) | **0.22 ± 0.00** | 1.00 ± 0.00 | 0.0 |

Oracle (knows inner `f`'s): **1.00**.

**Headline:** at the generous budget the ceiling is reachable by everyone (the novel combinator is findable given enough search). But under the TIGHT budget, culture DECIDES adaptation: best cultured **0.91** vs fresh **0.22** — a **+0.69** gap created entirely by the inherited library of inner abstractions.

![Adaptability by condition](figures/21_adaptability_bars.png)

![Adaptation curve vs budget](figures/22_adaptation_curve.png)

The adaptation curve shows the same story as a frontier shift: the cultured civilization reaches the ceiling at a small budget, while a fresh agent needs an order of magnitude more search to get there — carrying abstractions moves the budget frontier for a problem type the abstractions were never built for.

## 4. Interpretation
**Adaptability confirmed.** A civilization that accumulated a library of intermediate abstractions adapts to a structurally novel task family far faster than agents starting fresh — under a matched budget the inherited library is the difference between solving the family and failing it. Crucially the advantage is NOT memorization of the new task type (nobody saw a combinator); it is the *reuse of old abstractions as building blocks inside a new control structure discovered on the spot*. This is the strongest form of cultural accumulation the project set out to test: knowledge accumulated for one purpose pays off on problems it was never collected for.

## 5. Honest caveats
- The combinator search axis is small by construction (5 combinators); the claim is about reusing *inner* abstractions, not about discovering arbitrarily complex new control structures.
- The tight budget is a chosen operating point; the adaptation *curve* (fig below) shows the full budget sweep so the reader can see the gap is a frontier shift, not a single cherry-picked budget.
- Eval is symbolic program search, not gradient learning; it isolates the *value of inherited abstractions* cleanly but does not model test-time neural adaptation.

---
*Reproduce:* `./venv/bin/python run_adaptability.py --seeds 0 1 2`
