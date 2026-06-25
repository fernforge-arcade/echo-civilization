# Reaching the Locked Computer-Use Levels — Brainstorm → Build

*Companion to the Computer-Use Benchmark (`REPORT.md` §6.4). Code:
`echo_civilization/frontier.py`, `echo_civilization/codegen.py`. Run:
`./venv/bin/python run_frontier.py`. Figure 19, `results/frontier.json`.*

---

## 1. The question the operator asked

The Computer-Use Benchmark marched the civilization's end-product agent up a graded
ladder of **real** computer projects and reported, honestly, where it stopped. It
stopped at two walls:

| Tier | Rungs | Why it was locked |
|------|-------|-------------------|
| **6** | `find_and_replace`, `word_frequency`, `sum_numbers` | *Runnable* on a real shell, but **out of the operation vocabulary.** The synthesiser searched over op **order** only; these need a richer primitive **and** an argument value (the literal to substitute) the agent was never told. An oracle search over the whole op vocabulary up to depth 4 could not hit them → proven unreachable. |
| **7** | `write_python_script`, `build_todo_webapp`, `refactor_repo` | **Not representable at all.** A single-file op-pipeline cannot express a general program; this needs open-ended **code generation**. |

The operator's ask: *brainstorm ways to make it so the agents could actually hit
those remaining levels* — and then build them. The hard constraint stays the same
as the whole project: **no pretrained LLM**, simple agents, and the result must
still honour the thesis (*a skill that is expensive to discover is cheap to
inherit, so culture decides who reaches the top*).

---

## 2. The brainstorm — six candidate mechanisms

Below is the full option space we considered, with the verdict on each. Two were
built (✅); the rest are recorded as the honest menu, with why they were deferred.

### For Tier 6 (richer single-file tool use)

**(a) ✅ Parametric operations.**
Stop treating an operation as a fixed function and give it *holes* for arguments:
`replace(<find>,<repl>)`, `prefix_lines(<text>)`, `split_on(<sep>)`… A program
becomes a tuple of `(op, arg)` cells instead of bare op names. This is the minimal
change that lets one op family cover an unbounded set of concrete behaviours
(redact *any* word, prefix *any* marker). *Cost:* the search space gains an
argument dimension — which motivates (b).

**(b) ✅ Argument synthesis by example (programming-by-example).**
The agent is handed a few **input→output examples** and must *infer the
hole-fillers from them* — never told the literal. Candidate arguments are **mined
from the examples themselves**: for a substitution, the `find` literal is a token
that *disappeared* (present in input, absent in output) and the `repl` literal one
that *appeared*. The program must satisfy **every** example, which rejects args
overfit to one instance. This is exactly the idea behind FlashFill / spreadsheet
auto-fill — a classic, fully non-LLM synthesis technique. It is the literal
realisation of the roadmap's "learned command arguments" item.

**(c) ✅ Reduction primitives.** Two Tier-6 tasks need no literal but a *fold* the
old vocabulary lacked: `word_freq` (tokenise → count → format) and `sum_numbers`
(parse ints → add). Added as ordinary primitives.

**(d) ⏸ Numeric / typed register.** A fuller version of (c): give the machine a
typed register so it can do general arithmetic and comparisons, not just two
hard-coded folds. Deferred — (c) covers the benchmark's Tier-6 rungs and a typed
register is a much larger surface to design and grade fairly.

### For Tier 7 (open-ended software)

**(e) ✅ Grammar-guided code synthesis.** Give the agent a **second action
space**: instead of an op-pipeline, it emits a program in a tiny **typed grammar
that compiles to real Python**, and we **run it in a subprocess against hidden
tests**, keeping the first program that passes them all. This is genuine
synthesis of executable code from examples (the DreamCoder / enumerate-and-test
lineage), with no pretrained model. It takes one Tier-7 rung — *"write a Python
script that reads a CSV and prints column averages"* — from **not representable**
to **reachable and really executed**.

**(f) ⏸ Learned grammar weights / neural-guided search.** Bias the grammar
enumeration with weights the population evolves (which productions tend to pay
off), so search gets cheaper over generations — the natural next step to make the
*Flask-app* and *repo-refactor* rungs tractable. Deferred: those rungs need a
multi-file action space and a test harness an order of magnitude larger; honestly
out of scope for this pass. **We do not claim them** — the ceiling moved up one
rung, it did not vanish.

---

## 3. What was built

### 3.1 Tier 6 — `frontier.py`

* **Parametric op table** (`PARAM_OPS`): `read_input`, `write_output`, `upper`,
  `grep`, `sort`, plus parametric `replace(find,repl)` / `prefix_lines(text)` and
  reductions `word_freq` / `sum_numbers`. Ops stay total (never raise), so blind
  search is safe.
* **Argument inference** (`infer_replace_args`, `mine_literals`): mines literal
  candidates from the example diffs.
* **Templated-macro synthesiser** (`synthesize_param`): a *skill* is now an
  op-**name** sequence *with holes* — a reusable **template**. Solving a task =
  recall an inherited template and **fill its holes from the examples**, else
  discover a template from scratch. Holes are refilled per task, so one template
  generalises across instances. **This is the key design move:** the inherited
  unit is a *parametrised program*, and "learning the argument" is filling the
  hole at solve time.

### 3.2 Tier 7 — `codegen.py`

* A small grammar (`render`) over `{skeleton, reducer, header, delimiter,
  format}` that emits a runnable Python script reading `argv[1]` as a CSV.
* `run_script` writes the source + a CSV to a throwaway temp dir and **executes
  it for real** (subprocess, 5 s timeout, minimal env), returning stdout.
* `synthesize_code` enumerates the grammar shallow-first, runs each candidate
  against the hidden tests, and keeps the first that passes all of them. An
  inherited **skeleton** is tried first (the cultural shortcut).

---

## 4. Results (seed 0, 10 trials — `results/frontier.json`, figure 19)

We report **two budget regimes**, which together keep the claim honest:
*generous* answers "did the ceiling move — can anyone reach it now?"; *tight*
answers "does **culture** still decide who reaches it?"

### Tier 6 — was 0/5 reachable (oracle-proven out of vocabulary)

| rung | generous: fresh | generous: cultured | tight: fresh | tight: cultured |
|------|:---:|:---:|:---:|:---:|
| find_and_replace | 1.00 | 1.00 | 0.10 | 1.00 |
| swap_words       | 0.90 | 0.90 | 0.10 | 0.90 |
| word_frequency   | 1.00 | 1.00 | 0.00 | 1.00 |
| sum_numbers      | 1.00 | 1.00 | 0.00 | 1.00 |
| redact_then_sort | 1.00 | 1.00 | 0.00 | 1.00 |

A cultured agent recalls the template and fills the holes in **~1 evaluation**; a
fresh agent needs **16–62** to discover it — fine under a generous budget,
**impossible under a tight one** (0.00–0.10). The ceiling **moved**, and culture
**still decides**. `swap_words` caps at 0.90 because when two arbitrary words are
swapped the example diff is occasionally ambiguous (several added/removed tokens) —
an honest limit of pure programming-by-example, not a bug.

### Tier 7 — was NOT REPRESENTABLE

| | generous: fresh | generous: cultured | tight: fresh | tight: cultured |
|--|:---:|:---:|:---:|:---:|
| csv → column averages | 1.00 | 1.00 | 0.00 | 1.00 |

The agent now synthesises and **runs real Python**. Cultured recalls the skeleton
in **3 real executions**; fresh must grind the grammar (**~147 executions**) — it
gets there with a generous budget but **fails under a tight one**. A synthesised,
verified-by-execution script (verbatim from the run):

```python
import sys, csv
path = sys.argv[1]
with open(path, newline='') as fh:
    rows = list(csv.reader(fh, delimiter=','))
rows = rows[1:] if rows else rows
if rows:
    ncol = len(rows[0])
    out = []
    for c in range(ncol):
        v = []
        for row in rows:
            try: v.append(float(row[c]))
            except (ValueError, IndexError): pass
        if v:
            r = sum(v) / len(v)
            out.append(str(r))
    print(' '.join(out))
```

---

## 5. The honest, *moved* ceiling

This pass raised the wall, it did not remove walls forever:

* **Tier 6 is now genuinely reachable** — and still gated by culture under budget.
* **One Tier-7 rung is reachable** via real code synthesis.
* **The other Tier-7 rungs (Flask app, repo refactor) remain out of reach.** They
  need a multi-file action space, a service to run, and a much larger test
  harness — mechanism (f) above is the route, and it is honestly future work. We
  do not claim them.

The shape of the result is the same as everywhere else in this project: the
mechanism that lets an agent reach a new rung is **expensive to discover and cheap
to inherit**, so **a population with accumulated culture clears rungs a fresh agent
cannot** — now demonstrated two full tiers higher than before.

Reproduce: `./venv/bin/python run_frontier.py --trials 10` (~2.5 min; `--quick`
for ~1 min).

---

## 6. Tier 8 — one rung higher (group-by aggregation)

The Tier-7 rung reached above is a *flat* per-column reduction. The next push
(see [`TIER8_FRONTIER_FINDINGS.md`](TIER8_FRONTIER_FINDINGS.md) and `REPORT.md`
§6.6) climbs to a structurally harder program — **group-by aggregation**: read a
CSV, group rows by a key column, aggregate a value column per group, print sorted
`key:value` pairs. It needs a **dict accumulator** and a two-pass shape, and the
agent must recover the key column, value column and reducer (none given). It is
synthesised as real Python and **really run** against hidden tests
(`echo_civilization/codegen2.py`, `run_tier8.py`).

Result (seeds 0 and 1 identical): generous budget → fresh and cultured both 1.00
(76 vs 16 real executions to solve); tight budget (45) → **fresh 0.00, cultured
1.00**. The frontier moved up another rung by mechanism, and culture still decides.

Reproduce: `./venv/bin/python run_tier8.py --trials 10` (~2 min).
