# Echo Civilization

An **artificial-civilization laboratory** that tests one question:

> *Can a population of simple learning agents accumulate knowledge and become more
> capable over generations through a civilization-like process?*

No pretrained models are used. Agents start with **minimal** capabilities and
acquire everything — copying, reversing, counting, navigating, communicating —
from scratch through interaction with environments and with each other.

The interesting result is **not** "can one agent solve a task?" but **"does
generation 30 have capabilities generation 1 could not achieve, because knowledge
accumulated culturally?"** — and the answer here is *yes*.

## The headline result

Four conditions are run with an **identical fixed problem-solving budget per
agent**, varying only the civilization machinery:

| Condition | gen 0 | final | accumulates? |
|---|---|---|---|
| **A** single agent, no memory/culture | ~0.5 | ~0.5 | no |
| **B** population, no sharing | ~0.5 | ~0.5 | no |
| **C** population + skill sharing/inheritance | ~0.45 | **~0.96** | **yes** |
| **D** full civilization (culture + teaching + reputation + inheritance) | ~0.45 | **~0.97** | **yes** |

Capability = fraction of a held-out suite of **hard** (composite/deep) tasks an
agent can solve using *accumulated knowledge only* (recall + recombination, no
fresh search). Because every agent has the same budget, the gap between C/D and
A/B is attributable to **accumulated culture, not compute**.

![capability](figures/01_average_intelligence.png)

## How accumulation works

Tasks are *"produce an output string from an input string"*. A **skill** is a
program built from primitive transforms (`copy`, `reverse`, caesar `inc/dec`,
`count`, `first/last`, `double`, `dedup`) and skills can be **composed**.

- Discovering a depth-*L* composite from scratch costs ~|primitives|^L
  evaluations → quickly exceeds the budget.
- An agent that **inherited** the constituent skills reaches the same composite in
  a handful of recombination checks.

So accumulated culture turns intractable searches into trivial ones. Later
generations stand on the shoulders of earlier ones.

## System components

| Module | Role |
|---|---|
| `agent.py` | Agent: identity, memory, skills, preferences, social profile, the task solver |
| `learning.py` | swappable learners: tabular **Q-learning**, **evolved MLP** (ES), random |
| `neural.py` | tiny numpy MLP controller (evolvable weights) |
| `memory.py` | short-term buffer + long-term store with salience-decay forgetting |
| `skills.py` | composable skill programs + primitives |
| `culture.py` | shared cultural memory (reputation, adoption, propagation, decay) |
| `teaching.py` | horizontal skill transfer (teacher → student) |
| `evolution.py` | generations: selection, mutation, vertical + cultural inheritance |
| `environments/` | Echo, Transformation, Memory, Grid (NN policy), Social (emergent language) |
| `evaluation.py` | the four baseline experiments + held-out capability metric |
| `visualization.py` | the five required graphs + supporting plots |
| `report.py` | generates `research_report.md` |
| `database.py` | SQLite logging of agents, skills, rewards, generations, propagation |

## The worlds

0. **Echo World** — reproduce a target string (learn to copy). Tabular Q-learning.
1. **Transformation World** — echo / reverse / count / shift and their compositions.
2. **Memory World** — remember a fact, recall after a delay (retention, forgetting, transfer).
3. **Grid World** — move, collect resources, avoid hazards, survive. Evolved NN policy.
4. **Social World** — agents agree on meanings of initially meaningless symbols (emergent communication).
5. **Computer World** *(Exp. E)* — operate a *simulated* VM (virtual filesystem +
   register) via shell-like ops; an **auto-curriculum** raises difficulty as the
   population improves. The full civilization climbs from "copy a file" to deep
   multi-step pipelines (level 1→5) and sustains it; a no-sharing control collapses.
6. **Real Computer World** *(Exp. F)* — the *same* learned macros execute as
   **real sandboxed `bash` commands** (`cat`/`grep`/`sort`/`uniq`/`wc`/`tr`/`tac`/`cp`
   in a temp dir, whitelisted, no network). Cultured agents solve real tasks in a
   handful of real commands; fresh agents fail within their execution budget —
   genuine, if bounded, computer-use agents.
7. **Autonomous Operation World** *(Exp. G)* — the highest abstraction: a firm of
   specialised agents runs *forever*, decomposing a continuous stream of customer
   orders into sub-tasks, delegating to specialists, earning revenue and paying
   wages, with bounded-tenure workforce churn. With a shared knowledge base the
   firm's profit compounds and it sustains ever-harder orders; an identical firm
   without institutional memory runs at a loss — institutional knowledge is the
   difference between a viable and a failing autonomous operation.

## Roadmap (raising the level of abstraction)

Done: worlds 0–7 above (including a working prototype of the autonomous-operation
world). Next: a wider sandboxed shell with **learned command arguments** and
agent-proposed sub-tasks; and a deeper autonomous world where agents **propose
their own goals**, with a multi-firm economy (competition, trade) over truly
open-ended horizons. The civilization machinery (skills, culture, teaching,
reputation, inheritance, specialization) is the substrate; each rung adds
hierarchy and economy. This is **not** a claim of AGI — it is a study of
cumulative culture as the lever for unbounded capability growth, tested at each
rung from copying a string to operating a real computer to running a business.

## Running

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/python run_experiments.py          # full run (~75s): worlds 0–7, A–G
./venv/bin/python run_experiments.py --quick   # fast smoke run
./venv/bin/python run_generalization.py        # the memorization-vs-generalization test (~45s)
```

Outputs:
- **[`REPORT.md`](REPORT.md)** — the flagship research write-up: hypothesis,
  methods, all worlds with worked examples, results & statistics, every figure
  embedded, conclusions, and limitations. **Start here.**
- **[`GENERALIZATION_FINDINGS.md`](GENERALIZATION_FINDINGS.md)** — the flagship
  write-up of the falsification test: train on a subset of composites, test on
  *disjoint, never-trained* composites stratified by depth. Full methodology, the
  train/test split, a worked example, multi-seed results, both figures, and the
  verdict. Confirms the headline is **real compositional generalization**
  (culture +0.49 on novel depth-3 tasks), not memorization. **Read this for the
  new result.**
- `GENERALIZATION_REPORT.md` — the terse machine-generated companion (auto-written
  by `run_generalization.py`).
- `research_report.md` — the machine-generated companion, auto-written each run.
- `figures/` — 17 PNGs (incl. computer-curriculum, real-OS, autonomous-firm, and
  the generalization-by-depth bars + accumulation curve).
- `results/echo_civilization.db` — all raw data (SQLite); `results/generalization.json`
  — the generalization summary.

## Design principle

The system is optimised for testing *"does intelligence accumulate?"*, not for
building one smart agent. Learning algorithms are swappable behind a single
interface so the experiment can be re-run with policy gradients, different ES
variants, etc., without touching the civilization machinery.
