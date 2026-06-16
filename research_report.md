# Echo Civilization — Research Report
*An artificial-civilization laboratory testing whether knowledge accumulates across generations of simple learning agents.*
> Generated automatically by `run_experiments.py`. No pretrained models are used anywhere in this system; every capability shown is acquired from scratch through interaction.
## 1. Hypothesis
**Central research question.** *Can a population of simple learning agents accumulate knowledge and become more capable over generations through a civilization-like process?*
**Hypothesis (H1).** A population that shares and inherits discovered skills (a culture) will solve harder tasks over successive generations, while an isolated agent or a population without sharing will not — *even when every agent is given an identical, fixed problem-solving budget*. In other words, the limiting resource for hard tasks is not individual compute but **accumulated culture**.
**Sub-hypotheses.**
- H1a — individual agents can learn primitive skills from reward alone (Echo World, Q-learning).
- H1b — memory retains information but decays without reinforcement, and can be transferred between agents (Memory World).
- H1c — a different learning algorithm (an evolved neural network) improves across generations on a physical task (Grid World).
- H1d — a shared communication protocol can emerge from initially meaningless symbols (Social World).

## 2. Methods
### 2.1 Agents
Agents are **not** language models. Each agent has an identity (id, generation, parents), internal state (a swappable *learner*, a library of known *skills*, short- and long-term memory, behavioural preferences), goals (maximise reward, explore, learn), and a social profile (reputation, relationships, teaching and contribution history).
### 2.2 Learning algorithms (swappable)
A single `Learner` interface backs three interchangeable algorithms: **tabular Q-learning** (the genuine experience→reward→policy loop), an **evolved numpy MLP** (evolutionary strategies, no backprop), and a random baseline. The architecture is modular so algorithms can be swapped per environment.
### 2.3 Skills as composable programs
Tasks are *"produce an output string from an input string"*. A **skill** is a program built from primitive transforms (`copy`, `reverse`, caesar `inc/dec`, `count`, `first/last`, `double`, `dedup`). Skills can be copied, modified and **composed**. The unit of culture is the skill. An agent solves a task by (1) recalling known skills, (2) recombining known skills, and only then (3) discovering from scratch via Q-learning / bounded blind search — all under a fixed per-task **evaluation budget**.
### 2.4 Why culture is decisive
Blind discovery of a depth-*L* composite costs ~|primitives|^L evaluations and quickly exceeds the budget. But an agent that has *inherited* the constituent skills reaches the same composite in a handful of recombination checks. Thus accumulated culture converts intractable searches into trivial ones — the mechanism behind any generational gain.
### 2.5 Generations & evolution
Each generation: build a population → agents attempt tasks under the budget → discovered skills are abstracted and (if enabled) contributed to a shared **cultural memory** → optional peer **teaching** → fitness measured → selection + mutation produce the next generation, which **inherits** parent and cultural skills. Cultural reputation decays so unused skills fade.
### 2.6 Experimental design
Four conditions hold the world, seed, population size, budget and per-agent task count **fixed** and vary only the civilization machinery:
| Condition | population | culture | inheritance | teaching | reputation |
|---|---|---|---|---|---|
| A: single agent, no memory/culture | 1 | False | False | False | False |
| B: population, no sharing | 24 | False | False | False | False |
| C: population + memory/skill sharing | 24 | True | True | False | False |
| D: full civilization | 24 | True | True | True | True |

Capability is measured on a **held-out suite of hard (composite + deep) tasks** using *accumulated knowledge only* (recall + recombination, no fresh blind search), averaged per agent — so larger populations gain no unfair brute-force advantage.

Key parameters: generations=30, population=24, per-task budget=35 evaluations, tasks/agent/generation=8.

## 3. Results
### 3.1 Headline: capability accumulates only with culture
| Condition | gen 0 capability | final capability | change |
|---|---|---|---|
| A: single agent, no memory/culture | 59% | 57% | -0.02 |
| B: population, no sharing | 44% | 50% | +0.06 |
| C: population + memory/skill sharing | 44% | 97% | +0.53 |
| D: full civilization | 49% | 96% | +0.47 |

The sharing conditions (C, D) improve their hard-task capability by **+0.53** and **+0.47** over 30 generations, finishing at 97% and 96%. The non-sharing baselines (A single agent, B population) stagnate near their starting level (57% and 50%). Because every agent has the **same** budget, the difference is attributable to accumulated culture, not compute. This supports **H1**.

Final shared cultural repository sizes: C = 16 skills, D = 15 skills; baselines A/B accumulate no shared culture by construction.

![Average capability over generations](figures/01_average_intelligence.png)
![Behavioural complexity over time](figures/05_complexity_over_time.png)
![Cultural repository growth](figures/06_culture_growth.png)
### 3.2 Skill and relationship networks (condition D)
Over the full-civilization run, 27 skill transfers occurred between agents through teaching. The propagation and relationship graphs below show culture spreading through the population.

![Skill propagation network](figures/03_skill_propagation.png)
![Agent relationship network](figures/04_relationship_network.png)

Most reputable skills in the final culture (condition D):
| skill (program) | complexity | adoption | reputation |
|---|---|---|---|
| count | 1 | 116 | 24.0 |
| inc1 | 1 | 149 | 13.0 |
| last | 1 | 137 | 12.3 |
| reverse | 1 | 141 | 9.7 |
| copy | 1 | 162 | 8.4 |
| inc1 then reverse then double | 3 | 140 | 6.4 |
| reverse then inc1 | 2 | 177 | 5.3 |
| inc1 then reverse | 2 | 168 | 4.7 |

### 3.3 Subsystem validation
**H1a — Echo World (individual Q-learning).** A single tabular Q-learner learns the identity/copy map from reward alone, reaching 100% character accuracy and mastering the task at episode 6. This is the foundational `copy` skill that later composite skills build on.

![Echo World learning curve](figures/07_echo_learning.png)

**H1b — Memory World (retention, forgetting, transfer).** Recall strength decays with delay — from 100% at no delay to 9% after 60 interfering steps — a classic forgetting curve. A remembered fact can be transferred to a naive agent: transfer test passed.

![Memory forgetting curve](figures/08_memory_forgetting.png)

**H1c — Grid World (evolved neural policy).** A population of MLP policies improved by evolutionary strategies raised best-episode reward from 1.62 to 6.09 across generations, learning to collect resources and avoid hazards with no gradient training. This demonstrates the swappable-learner design.

![Grid World evolution](figures/09_grid_evolution.png)

**H1d — Social World (emergent communication).** Starting from meaningless symbols, agents playing a referential signalling game converged on a shared protocol: final communication accuracy 100% and protocol consistency 100% (random baseline would be ~33%). Meaning emerged; it was not given.

![Social protocol emergence](figures/10_social_emergence.png)

![Per-difficulty solve rate (condition D)](figures/11_difficulty_breakdown.png)
![Best-agent fitness](figures/02_best_performance.png)

## 4. Conclusions
1. **Knowledge accumulates culturally.** The headline result (supports H1): conditions with skill sharing/inheritance become measurably more capable on hard tasks over generations, while a single agent and a non-sharing population do not. Later generations solve composite and deep tasks that earlier generations, given the same budget, could not — because the building blocks had entered the shared culture.
2. **The mechanism is recombination of inherited primitives.** Culture turns an exponential blind search into a short composition over already-known skills. This is the computational analogue of "standing on the shoulders of giants".
3. **Vertical inheritance does most of the work; horizontal teaching adds less.** Condition C (inheritance only) and condition D (inheritance + teaching + reputation) finish close together (97% vs 96%), indicating that once skills are inherited and pooled in culture, extra horizontal copying is largely redundant in this task family.
4. **All subsystems function independently:** individual RL (Echo), memory and its decay/transfer (Memory), evolved neural control (Grid), and emergent communication (Social).

## 5. Failures, limitations & threats to validity
- **Task domain is narrow.** All accumulation experiments use string transforms. The accumulation result should be replicated in richer domains before being generalised.
- **Capability starts well above zero.** Within a single lifetime agents already discover primitives, so generation-0 capability is non-trivial; the *generational* signal is the upward slope of C/D, not an absolute zero start.
- **Teaching benefit is small here.** With strong vertical inheritance, the horizontal teaching channel adds little; a harsher selection regime or lossy inheritance would likely make teaching matter more.
- **Grid and social runs are noisy** (random maps / coordination), mitigated by averaging multiple lives and many rounds but not eliminated.
- **Emergent protocols can be degenerate** (high consistency but low accuracy if two concepts collapse onto one symbol); parameters were chosen to avoid this but the failure mode exists.
- **No statistical multi-seed confidence intervals** are reported in this single run; `run_experiments.py --seeds N` can be extended for that.

## 6. Reproduction
```
./venv/bin/python run_experiments.py
```
All raw data is logged to `results/echo_civilization.db` (SQLite): tables `experiments`, `generations`, `agents`, `skills`, `propagation`, `rewards`. Figures are written to `figures/`.
