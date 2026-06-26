# Builder World (Experiment J) — building real apps from a one-line prompt

*Flagship write-up. Lead with the run output, then the method, then the numbers,
then the failures.* No pretrained model is used anywhere. Agents emit **real
JavaScript**, that JavaScript is **executed in Node** against hidden behavioural
tests, and an app counts as "built" only when every requirement passes for real.

---

## 0. The thing it actually produces

The operator's steer was blunt: push this toward what larger models do — *take a
vague task ("build a website that does X") and actually build a working app.* So
the deliverable is not a number, it is **five real, openable apps** the
civilization built from one-line prompts, sitting in `output_apps/`:

```
output_apps/
  counter/        index.html + app.js   "Build a counter app."        (3 features)
  tip_calculator/ index.html + app.js   "Build a tip calculator."     (3 features)
  todo/           index.html + app.js   "Build a to-do list app."     (3 features)
  shopping_cart/  index.html + app.js   "Build a shopping cart."      (4 features)
  notes/          index.html + app.js   "Build a notes app."          (6 features)
```

Open any `index.html` in a browser; the buttons drive a live reducer the agents
assembled. Here is the **frontier app** — `notes`, six features — *exactly as the
strongest cultured agent emitted it* (`output_apps/notes/app.js`, verbatim):

```js
function createApp() {
  const state = { items: [], count: 0, bill: 0, rate: 0, result: 0,
                  total: 0, input: '', filter: 'all' };
  function dispatch(action, payload) {
    switch (action) {
      case "ADD":    { state.items.push({text: payload, done: false}); break; }
      case "REMOVE": { state.items.splice(payload, 1); break; }
      case "EDIT":   { if (state.items[payload.i]) state.items[payload.i].text = payload.text; break; }
      case "TOGGLE": { if (state.items[payload]) state.items[payload].done = !state.items[payload].done; break; }
      case "FILTER": { state.filter = payload; break; }
      case "CLEAR":  { state.items = []; break; }
      default: break;
    }
  }
  return { dispatch, getState: () => state };
}
```

No template filled this in. Each `case` body is a **component** an agent had to
either discover by running candidate code in Node, or recall from inherited
culture, and the whole file only exists because every one of the six behavioural
tests passed when this exact code ran.

---

## 1. Hypothesis

**H_J.** The accumulation asymmetry this whole project demonstrates — *expensive to
discover, cheap to inherit* — applies to **building applications**. Under a fixed
per-agent build budget, the **frontier of buildable apps** (the most complex app a
generation can actually produce and run) **rises across generations only when a
component library accumulates culturally.** Without inheritance the frontier stays
flat; with it, generation N builds apps generation 1 could not.

Two mechanisms carry it, and they are the operator's two hints made literal:

- **Decomposition ("the sub-task thing").** A vague spec is decomposed into **one
  sub-task per user action** the app must support. Each sub-task is "find the
  handler whose tests pass for THIS behaviour." Decomposition turns a
  **multiplicative** joint search (every combination of every candidate handler,
  `|handlers|^features`) into an **additive** one (each handler found
  independently). The same budget that cannot touch the joint space comfortably
  covers the sum of parts.
- **Culture.** Each solved handler is a reusable, named **component** keyed by the
  action it implements. Solved components are contributed to a shared library,
  inherited by later agents, and **tried first**. A cultured agent recognises "this
  app needs an `ADD` behaviour and I inherited an `add_text` component" and plugs it
  in for ~1 trial; a fresh agent must blind-search a grid of candidate handlers per
  sub-task and, across a multi-feature app under a tight budget, runs out before
  assembling them all.

---

## 2. Method (honest, executable)

- **Component library.** 22 components, each a real `switch`-case body in JS:
  **14 genuine** handlers the specs need (`add_text`, `remove_at`, `toggle_at`,
  `clear_items`, `edit_at`, `set_filter`, `inc_count`, `dec_count`, `reset_count`,
  `set_bill`, `set_rate`, `compute_tip`, `add_priced`, `sum_prices`) and **8
  decoys** — plausible-but-wrong implementations (`push_raw` omits the `{text,done}`
  shape; `splice_first` ignores the index; `compute_flat` forgets the bill;
  `set_len` reads the wrong source; etc.). The decoys are what make the search
  *real*: each sub-task has a **unique** correct component, and a blind agent must
  rule the decoys out by running them.
- **Specs.** Five one-line prompts → 5 specs of rising feature count:
  `counter`(3), `tip_calculator`(3), `todo`(3), `shopping_cart`(4), `notes`(6).
  A spec is a list of sub-tasks; each sub-task carries hidden behavioural tests.
- **Grading is execution, not pattern-matching.** `grade_app` writes the assembled
  reducer to a temp file and **runs it in Node**, dispatching the test actions and
  asserting on resulting state. A sub-task is "solved" only if its component's real
  output matches. (A per-subtask grade cache keyed by `(action, component)` makes
  the multi-seed sweep tractable — **3,325 real Node executions** this run.)
- **Two builders.**
  - `build_monolithic(spec, budget)` — control. Searches the **joint** space: draws
    whole candidate assignments (one component per action at once) and runs the
    full app. Space size `≈ |candidates|^features`.
  - `build_decomposed(spec, budget, action_map, known_set)` — solves **each sub-task
    independently**. With culture (`action_map`: action→component, `known_set`:
    components seen to work) the inherited component is tried **first**; on a miss it
    falls back to blind per-sub-task search. Returns which actions were solved via
    culture vs. **newly discovered** (the discoveries are what get contributed back).
- **Three conditions × seeds 0/1/2, POP=5, GEN=8, BUDGET=40:**
  - **A — monolithic** (no decomposition, no culture).
  - **B — decomposed, no culture** (decomposition only; each generation starts
    from an empty library).
  - **C — decomposed + culture** (decomposition + an inherited, accumulating
    component library).

Re-run: `./venv/bin/python run_builder.py --seeds 0 1 2` (a few minutes).

---

## 3. Results (canonical — verified this run)

### 3.1 Frontier over generations (max features in a fully-built, test-passing app)

| Gen | A monolithic | B decomposed, no culture | C decomposed + culture |
|----:|:---:|:---:|:---:|
| 1 | 0 | 4.7 | 4.7 |
| 2 | 0 | 4.7 | **6.0** |
| 3 | 0 | 4.0 | 6.0 |
| 4 | 0 | 4.0 | 6.0 |
| 5 | 0 | 5.3 | 6.0 |
| 6 | 0 | 4.0 | 6.0 |
| 7 | 0 | 4.7 | 6.0 |
| 8 | 0 | 4.7 | 6.0 |

(means over seeds 0/1/2; the 6-feature `notes` app is the ceiling.)

- **A monolithic = 0 at every generation.** The smallest spec is 3 features; the
  joint space (≈22³ ≫ 40-trial budget) means the monolithic searcher essentially
  never lands a complete, test-passing assignment. *Without decomposition, nothing
  gets built.*
- **B decomposed-but-cultureless = a flat noisy band around 4**, no upward trend.
  Per-seed: seed0 `[4,4,4,4,4,4,6,4]`, seed1 `[4,6,4,4,6,4,4,4]`, seed2
  `[6,4,4,4,6,4,4,6]`. A lucky agent occasionally flukes a 6 — and it **never
  compounds**, because the next generation restarts from an empty library and must
  re-earn every component from scratch.
- **C decomposed + culture climbs to the ceiling and HOLDS.** Per-seed frontier:
  seed0 `[4,6,6,6,6,6,6,6]`, seed1 `[4,6,6,6,6,6,6,6]`, seed2 `[6,6,6,6,6,6,6,6]`.
  The component library grows from ~13–14 to **16 components** by generation 2 and
  stays saturated. *Once `notes` is built once, it stays built — forever, cheaply.*

![Builder frontier over generations](figures/25_builder_frontier.png)
![Culture size over generations](figures/27_builder_culture_growth.png)

### 3.2 Per-spec build rate, fresh agent vs. cultured agent (budget 40, n=27 each)

| App | Features | Fresh (no library) | Cultured (inherited library) |
|---|:---:|:---:|:---:|
| counter | 3 | 0.81 | **1.00** |
| tip_calculator | 3 | 0.52 | **1.00** |
| todo | 3 | 0.85 | **1.00** |
| shopping_cart | 4 | 0.33 | **1.00** |
| **notes** | **6** | **0.07** | **1.00** |

![Fresh vs cultured build rate](figures/26_builder_fresh_vs_cultured.png)

The trend is the whole point: **the harder the app, the more decisive culture is.**
The 3-feature apps are sometimes buildable fresh; the 6-feature `notes` app is
**essentially unbuildable fresh (0.07)** and **always buildable with an inherited
library (1.00)**. Difficulty and the cultural advantage scale together.

### 3.3 The accumulation mechanism, concretely

`notes` (`ADD/REMOVE/EDIT/TOGGLE/FILTER/CLEAR`) shares the **action names**
`ADD/REMOVE/TOGGLE` with `todo`. Once `todo` is built, those three components are in
the culture, so a later `notes` build **recalls them in ~1 trial each** — freeing
budget for the three `notes`-specific behaviours (`edit_at`, `set_filter`,
`clear_items`) that still have to be discovered once by a lucky agent. Those
discoveries then enter the culture too, and **every subsequent `notes` build is
one-shot.** That is the expensive-to-discover / cheap-to-inherit asymmetry — the
project's thesis — operating on whole applications instead of string transforms.

---

## 4. Failures & honest limits

- **The component library is fixed and finite.** Agents compose and discriminate
  among 22 pre-written handlers; they do **not** author arbitrary novel JavaScript
  from a blank file. What is demonstrated is the *mechanism* (decomposition +
  cultural reuse moving the frontier of buildable apps), not free-form codegen.
  (§6.5–6.6 elsewhere in this project show genuine grammar-based code *synthesis*;
  Builder World is about composition and inheritance at app scale.)
- **The budget is a tuned knob.** Push `BUDGET` past ~60 and even fresh starts can
  cover the grid (the cultural margin shrinks); drop it below ~20 and the cultured
  margin disappears too. 40 is the regime where decomposition matters and culture
  decides; this is stated plainly rather than hidden.
- **Monolithic A is a deliberately stiff control.** It uses no decomposition *and*
  no culture, so its flat zero conflates two disadvantages. B isolates
  decomposition (it builds, just doesn't improve); the A↔B↔C ladder separates the
  two mechanisms cleanly: A→B is "decomposition lets you build at all," B→C is
  "culture lets the frontier rise."
- **UI is scaffolding, logic is earned.** The emitted `index.html` (buttons, live
  state view) is a fixed wrapper; the *reducer* — the part graded by real execution
  — is what the agents actually assemble. We do not claim the agents designed the
  visual layout.
- **Apps are small and behavioural.** Real products have routing, persistence,
  styling, and ambiguous requirements. These are deterministic reducer apps with
  crisp hidden tests. The result is a clean demonstration of the accumulation
  law at app scale, not a production app generator.

---

## 5. Conclusion

Given five one-line prompts and no pretrained model, the civilization **builds five
real, runnable apps**, and the *frontier of what it can build rises across
generations* — but **only** under decomposition + culture. Decomposition is what
makes building tractable at all (monolithic search builds nothing); culture is what
makes the frontier *climb and hold* (cultureless decomposition plateaus and any
fluke evaporates next generation). The 6-feature `notes` app is unbuildable fresh
(0.07) and always buildable once the library is inherited (1.00). This is the same
sentence as the rest of the project, one level of abstraction higher: *the limiting
resource for hard construction is not per-agent compute but accumulated culture.*
Generation N ships an app generation 1 could not — because the components survived.
