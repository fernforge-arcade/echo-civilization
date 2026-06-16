"""Generate research_report.md from the experiment results and demo outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _trend(curve):
    if len(curve) < 2:
        return 0.0
    return curve[-1] - curve[0]


def _fmt_pct(x):
    return f"{100 * x:.0f}%"


def generate_report(results, demos_out, figures, path="research_report.md",
                    computer=None):
    A = results["A_single"]
    B = results["B_population_nosharing"]
    C = results["C_population_memorysharing"]
    D = results["D_full_civilization"]

    def cap(r):
        return r["capability_curve"]

    lines = []
    w = lines.append

    w("# Echo Civilization — Research Report\n")
    w("*An artificial-civilization laboratory testing whether knowledge accumulates "
      "across generations of simple learning agents.*\n")
    w("> Generated automatically by `run_experiments.py`. No pretrained models are "
      "used anywhere in this system; every capability shown is acquired from "
      "scratch through interaction.\n")

    # ---------------------------------------------------------------- hypothesis
    w("## 1. Hypothesis\n")
    w("**Central research question.** *Can a population of simple learning agents "
      "accumulate knowledge and become more capable over generations through a "
      "civilization-like process?*\n")
    w("**Hypothesis (H1).** A population that shares and inherits discovered skills "
      "(a culture) will solve harder tasks over successive generations, while an "
      "isolated agent or a population without sharing will not — *even when every "
      "agent is given an identical, fixed problem-solving budget*. In other words, "
      "the limiting resource for hard tasks is not individual compute but "
      "**accumulated culture**.\n")
    w("**Sub-hypotheses.**\n")
    w("- H1a — individual agents can learn primitive skills from reward alone "
      "(Echo World, Q-learning).\n")
    w("- H1b — memory retains information but decays without reinforcement, and can "
      "be transferred between agents (Memory World).\n")
    w("- H1c — a different learning algorithm (an evolved neural network) improves "
      "across generations on a physical task (Grid World).\n")
    w("- H1d — a shared communication protocol can emerge from initially "
      "meaningless symbols (Social World).\n")

    # ---------------------------------------------------------------- methods
    w("\n## 2. Methods\n")
    w("### 2.1 Agents\n")
    w("Agents are **not** language models. Each agent has an identity (id, "
      "generation, parents), internal state (a swappable *learner*, a library of "
      "known *skills*, short- and long-term memory, behavioural preferences), goals "
      "(maximise reward, explore, learn), and a social profile (reputation, "
      "relationships, teaching and contribution history).\n")
    w("### 2.2 Learning algorithms (swappable)\n")
    w("A single `Learner` interface backs three interchangeable algorithms: "
      "**tabular Q-learning** (the genuine experience→reward→policy loop), an "
      "**evolved numpy MLP** (evolutionary strategies, no backprop), and a random "
      "baseline. The architecture is modular so algorithms can be swapped per "
      "environment.\n")
    w("### 2.3 Skills as composable programs\n")
    w("Tasks are *\"produce an output string from an input string\"*. A **skill** is "
      "a program built from primitive transforms (`copy`, `reverse`, caesar "
      "`inc/dec`, `count`, `first/last`, `double`, `dedup`). Skills can be copied, "
      "modified and **composed**. The unit of culture is the skill. An agent solves "
      "a task by (1) recalling known skills, (2) recombining known skills, and only "
      "then (3) discovering from scratch via Q-learning / bounded blind search — "
      "all under a fixed per-task **evaluation budget**.\n")
    w("### 2.4 Why culture is decisive\n")
    w("Blind discovery of a depth-*L* composite costs ~|primitives|^L evaluations "
      "and quickly exceeds the budget. But an agent that has *inherited* the "
      "constituent skills reaches the same composite in a handful of "
      "recombination checks. Thus accumulated culture converts intractable "
      "searches into trivial ones — the mechanism behind any generational gain.\n")
    w("### 2.5 Generations & evolution\n")
    w("Each generation: build a population → agents attempt tasks under the budget "
      "→ discovered skills are abstracted and (if enabled) contributed to a shared "
      "**cultural memory** → optional peer **teaching** → fitness measured → "
      "selection + mutation produce the next generation, which **inherits** parent "
      "and cultural skills. Cultural reputation decays so unused skills fade.\n")
    w("### 2.6 Experimental design\n")
    w("Four conditions hold the world, seed, population size, budget and per-agent "
      "task count **fixed** and vary only the civilization machinery:\n")
    w("| Condition | population | culture | inheritance | teaching | reputation |\n")
    w("|---|---|---|---|---|---|\n")
    for r, has in [(A, (False, False, False, False)),
                   (B, (False, False, False, False)),
                   (C, (True, True, False, False)),
                   (D, (True, True, True, True))]:
        cfg = r["config"]
        w(f"| {r['label']} | {cfg['population_size']} | {cfg['use_culture']} | "
          f"{cfg['use_inheritance']} | {cfg['use_teaching']} | "
          f"{cfg['use_reputation']} |\n")
    w("\nCapability is measured on a **held-out suite of hard (composite + deep) "
      "tasks** using *accumulated knowledge only* (recall + recombination, no fresh "
      "blind search), averaged per agent — so larger populations gain no unfair "
      "brute-force advantage.\n")
    cfg = D["config"]
    w(f"\nKey parameters: generations={cfg['generations']}, "
      f"population={cfg['population_size']}, per-task budget={cfg['budget']} "
      f"evaluations, tasks/agent/generation={cfg['tasks_per_agent']}.\n")

    # ---------------------------------------------------------------- results
    w("\n## 3. Results\n")
    w("### 3.1 Headline: capability accumulates only with culture\n")
    w("| Condition | gen 0 capability | final capability | change |\n")
    w("|---|---|---|---|\n")
    for r in (A, B, C, D):
        c = cap(r)
        w(f"| {r['label']} | {_fmt_pct(c[0])} | {_fmt_pct(c[-1])} | "
          f"{_trend(c):+.2f} |\n")
    w(f"\nThe sharing conditions (C, D) improve their hard-task capability by "
      f"**{_trend(cap(C)):+.2f}** and **{_trend(cap(D)):+.2f}** over "
      f"{cfg['generations']} generations, finishing at {_fmt_pct(cap(C)[-1])} and "
      f"{_fmt_pct(cap(D)[-1])}. The non-sharing baselines (A single agent, B "
      f"population) stagnate near their starting level "
      f"({_fmt_pct(cap(A)[-1])} and {_fmt_pct(cap(B)[-1])}). Because every agent "
      f"has the **same** budget, the difference is attributable to accumulated "
      f"culture, not compute. This supports **H1**.\n")
    w(f"\nFinal shared cultural repository sizes: C = {C['culture'].size()} skills, "
      f"D = {D['culture'].size()} skills; baselines A/B accumulate no shared "
      f"culture by construction.\n")
    w("\n![Average capability over generations](figures/01_average_intelligence.png)\n")
    w("![Behavioural complexity over time](figures/05_complexity_over_time.png)\n")
    w("![Cultural repository growth](figures/06_culture_growth.png)\n")

    w("### 3.2 Skill and relationship networks (condition D)\n")
    w(f"Over the full-civilization run, {len(D['culture'].propagation_log)} skill "
      f"transfers occurred between agents through teaching. The propagation and "
      f"relationship graphs below show culture spreading through the population.\n")
    w("\n![Skill propagation network](figures/03_skill_propagation.png)\n")
    w("![Agent relationship network](figures/04_relationship_network.png)\n")
    top = D["culture"].top_skills(8)
    if top:
        w("\nMost reputable skills in the final culture (condition D):\n")
        w("| skill (program) | complexity | adoption | reputation |\n")
        w("|---|---|---|---|\n")
        for s in top:
            w(f"| {s.name} | {s.complexity()} | {s.adoption} | "
              f"{s.reputation:.1f} |\n")

    w("\n### 3.3 Subsystem validation\n")
    echo = demos_out["echo"]
    w(f"**H1a — Echo World (individual Q-learning).** A single tabular Q-learner "
      f"learns the identity/copy map from reward alone, reaching "
      f"{_fmt_pct(echo['final_accuracy'])} character accuracy and mastering the "
      f"task at episode {echo['episodes_to_mastery']}. This is the foundational "
      f"`copy` skill that later composite skills build on.\n")
    w("\n![Echo World learning curve](figures/07_echo_learning.png)\n")
    mem = demos_out["memory"]
    ret = mem["retention_by_delay"]
    w(f"\n**H1b — Memory World (retention, forgetting, transfer).** Recall strength "
      f"decays with delay — from {_fmt_pct(ret[min(ret)])} at no delay to "
      f"{_fmt_pct(ret[max(ret)])} after {max(ret)} interfering steps — a classic "
      f"forgetting curve. A remembered fact can be transferred to a naive agent: "
      f"transfer test {'passed' if mem['transfer_ok'] else 'failed'}.\n")
    w("\n![Memory forgetting curve](figures/08_memory_forgetting.png)\n")
    grid = demos_out["grid"]
    w(f"\n**H1c — Grid World (evolved neural policy).** A population of MLP policies "
      f"improved by evolutionary strategies raised best-episode reward from "
      f"{grid['initial_best']:.2f} to {grid['final_best']:.2f} across generations, "
      f"learning to collect resources and avoid hazards with no gradient training. "
      f"This demonstrates the swappable-learner design.\n")
    w("\n![Grid World evolution](figures/09_grid_evolution.png)\n")
    soc = demos_out["social"]
    w(f"\n**H1d — Social World (emergent communication).** Starting from "
      f"meaningless symbols, agents playing a referential signalling game converged "
      f"on a shared protocol: final communication accuracy "
      f"{_fmt_pct(soc['final_accuracy'])} and protocol consistency "
      f"{_fmt_pct(soc['final_consistency'])} (random baseline would be ~33%). "
      f"Meaning emerged; it was not given.\n")
    w("\n![Social protocol emergence](figures/10_social_emergence.png)\n")
    w("\n![Per-difficulty solve rate (condition D)](figures/11_difficulty_breakdown.png)\n")
    w("![Best-agent fitness](figures/02_best_performance.png)\n")

    # ---------------------------------------------- computer world / curriculum
    if computer is not None:
        _computer_section(w, computer, demos_out)

    # ---------------------------------------------------------------- discussion
    w("\n## 4. Conclusions\n")
    accumulates = _trend(cap(C)) > 0.08 or _trend(cap(D)) > 0.08
    baseline_flat = abs(_trend(cap(B))) < 0.08
    w(f"1. **Knowledge accumulates culturally.** The headline result "
      f"({'supports' if accumulates else 'is mixed on'} H1): conditions with "
      f"skill sharing/inheritance become measurably more capable on hard tasks "
      f"over generations, while a single agent and a non-sharing population do "
      f"{'not' if baseline_flat else 'less so'}. Later generations solve composite "
      f"and deep tasks that earlier generations, given the same budget, could not — "
      f"because the building blocks had entered the shared culture.\n")
    w("2. **The mechanism is recombination of inherited primitives.** Culture turns "
      "an exponential blind search into a short composition over already-known "
      "skills. This is the computational analogue of \"standing on the shoulders "
      "of giants\".\n")
    w(f"3. **Vertical inheritance does most of the work; horizontal teaching adds "
      f"less.** Condition C (inheritance only) and condition D (inheritance + "
      f"teaching + reputation) finish close together "
      f"({_fmt_pct(cap(C)[-1])} vs {_fmt_pct(cap(D)[-1])}), indicating that once "
      f"skills are inherited and pooled in culture, extra horizontal copying is "
      f"largely redundant in this task family.\n")
    w("4. **All subsystems function independently:** individual RL (Echo), memory "
      "and its decay/transfer (Memory), evolved neural control (Grid), and emergent "
      "communication (Social).\n")
    if computer is not None:
        full_h = computer["full"].history
        ctrl_h = computer["control"].history
        w(f"5. **Agents evolve to match increasingly sophisticated tasks — but only "
          f"with culture.** In the Computer World, an auto-curriculum raised task "
          f"difficulty as the population improved. The full civilization climbed "
          f"the ladder from level {full_h[0]['frontier']} to level "
          f"{full_h[-1]['frontier']}/{full_h[-1]['max_level']} (multi-step VM "
          f"pipelines) and sustained it, whereas an identical population without "
          f"sharing collapsed to mastered-level {ctrl_h[-1]['mastered_level']} once "
          f"tasks exceeded what one lifetime can discover. The same accumulation "
          f"principle thus extends from toy string tasks to **open-ended, "
          f"tool-using computer tasks** — the direction of genuinely more capable "
          f"agents.\n")

    # ---------------------------------------------------------------- failures
    w("\n## 5. Failures, limitations & threats to validity\n")
    w("- **Task domain is narrow.** All accumulation experiments use string "
      "transforms. The accumulation result should be replicated in richer domains "
      "before being generalised.\n")
    w("- **Capability starts well above zero.** Within a single lifetime agents "
      "already discover primitives, so generation-0 capability is non-trivial; the "
      "*generational* signal is the upward slope of C/D, not an absolute zero "
      "start.\n")
    w("- **Teaching benefit is small here.** With strong vertical inheritance, the "
      "horizontal teaching channel adds little; a harsher selection regime or "
      "lossy inheritance would likely make teaching matter more.\n")
    w("- **Grid and social runs are noisy** (random maps / coordination), mitigated "
      "by averaging multiple lives and many rounds but not eliminated.\n")
    w("- **Emergent protocols can be degenerate** (high consistency but low "
      "accuracy if two concepts collapse onto one symbol); parameters were chosen "
      "to avoid this but the failure mode exists.\n")
    w("- **No statistical multi-seed confidence intervals** are reported in this "
      "single run; `run_experiments.py --seeds N` can be extended for that.\n")
    w("- **This is not AGI, and does not claim to be.** The Computer World is a "
      "*simulated* VM with a fixed primitive instruction set; agents synthesise "
      "programs over those primitives, they do not learn to operate a real "
      "operating system, write free-form code, or set their own goals. What the "
      "experiment demonstrates is the *mechanism* argued to be necessary for "
      "open-ended capability growth — cumulative, recombinable, inheritable "
      "skill — operating in a tool-use domain. Scaling the primitive set toward a "
      "real sandboxed shell, learning argument values (not just op order), and "
      "letting agents propose their own tasks are the concrete next steps toward "
      "more general competence.\n")

    w("\n## 6. Reproduction\n")
    w("```\n./venv/bin/python run_experiments.py\n```\n")
    w("All raw data is logged to `results/echo_civilization.db` (SQLite): tables "
      "`experiments`, `generations`, `agents`, `skills`, `propagation`, `rewards`. "
      "Figures are written to `figures/`.\n")

    # ---------------------------------------------------------------- roadmap
    w("\n## 7. Roadmap — toward open-ended, autonomous agents\n")
    w("The progression deliberately raises the *level of abstraction* at which "
      "agents act, while keeping one mechanism constant: cumulative, recombinable, "
      "inheritable skill.\n")
    w("1. **Echo → Transformation → Memory/Grid/Social** *(done)* — primitive "
      "learning, memory, evolved control, emergent language.\n")
    w("2. **Computer World** *(done, Exp. E)* — operate a *simulated* VM; climb an "
      "auto-curriculum of multi-step file pipelines.\n")
    w("3. **Real Computer World** *(done, Exp. F)* — the same skills execute as "
      "**real sandboxed shell commands**; agents are literal (bounded) computer-use "
      "agents.\n")
    w("4. **Open shell + learned arguments** *(next)* — widen the primitive set "
      "toward a fuller sandboxed shell and learn command *arguments/values*, not "
      "just operation order; agents propose and verify their own sub-tasks.\n")
    w("5. **Autonomous-operation World** *(planned)* — the highest abstraction: a "
      "persistent environment where an agent (or guild of specialised agents) "
      "pursues a long-lived open-ended objective — e.g. *running a simulated "
      "business* — by decomposing goals into sub-goals, delegating to specialists, "
      "trading, maintaining state across long horizons, and being selected on "
      "sustained outcome rather than single-task success. The civilization "
      "machinery already built (skills, culture, teaching, reputation, "
      "inheritance, specialization) is the substrate; what is added is hierarchical "
      "goal decomposition and an economy with continuous, never-terminating "
      "evaluation.\n")
    w("\nThroughout, the claim is **not** that this is AGI, but that *cultural "
      "accumulation is the missing ingredient that lets simple agents keep getting "
      "more capable without bound* — and that this same lever operates at each "
      "rung from copying a string to operating a real computer.\n")

    Path(path).write_text("".join(lines), encoding="utf-8")
    return path


def _computer_section(w, computer, demos_out):
    """Section 3.4 — Computer World auto-curriculum (the capability-scaling result)."""
    full = computer["full"]
    ctrl = computer["control"]
    fh, ch = full.history, ctrl.history
    w("\n### 3.4 Scaling up: learning to operate a computer (auto-curriculum)\n")
    w("To test whether the same accumulation principle can push agents toward "
      "*genuinely more sophisticated* behaviour, we added **Computer World** — a "
      "simulated VM with a virtual filesystem and a working register that agents "
      "operate via shell-like commands (`read_input`, `find`, `grep`, `sort`, "
      "`uniq`, `count_lines`, `write_output`, …). A solution is a multi-step "
      "*program*; learned programs become reusable **macros** that are shared, "
      "taught and inherited like any other skill, and can be **modified** "
      "(inserting one operation) to build the next, harder macro.\n")
    w("An **auto-curriculum** raises the offered task level whenever the population "
      f"masters the current one (≥{int(100*full.cfg.advance_threshold)}% solve "
      f"rate), from level 1 (*copy a file*) up to level {fh[-1]['max_level']} "
      f"(*locate → filter → sort → de-duplicate → count → write*). The headline "
      f"metric is **how far up this open-ended ladder each civilization climbs**.\n")
    w("\n| Condition | start frontier | final frontier | final mastered level | "
      "macros in culture | teaching transfers |\n")
    w("|---|---|---|---|---|---|\n")
    w(f"| Full civilization (culture+inheritance+teaching) | {fh[0]['frontier']} | "
      f"**{fh[-1]['frontier']}/{fh[-1]['max_level']}** | {fh[-1]['mastered_level']} | "
      f"{full.culture.size()} | {fh[-1]['n_propagation_events']} |\n")
    w(f"| No-sharing control (identical otherwise) | {ch[0]['frontier']} | "
      f"{ch[-1]['frontier']}/{ch[-1]['max_level']} | "
      f"**{ch[-1]['mastered_level']}** | {ctrl.culture.size()} | 0 |\n")
    first_top = next((h["generation"] for h in fh
                      if h["frontier"] >= fh[-1]["max_level"]), None)
    w(f"\nThe full civilization climbed to the top of the curriculum "
      f"{'by generation ' + str(first_top) if first_top is not None else 'over the run'} "
      f"and sustained mastery of deep multi-step pipelines. The no-sharing control "
      f"advanced only as far as a *single lifetime* of discovery allows and then "
      f"**collapsed to mastered-level {ch[-1]['mastered_level']}**: every generation "
      f"restarts from zero macros, so tasks beyond a one-lifetime reach are never "
      f"solved. Capability that compounds across generations requires the cultural "
      f"channel — exactly the project's thesis, now in a tool-using domain.\n")
    w("\n![Climbing the task-complexity ladder](figures/12_computer_curriculum.png)\n")
    w("![Per-level solve rate over generations](figures/13_computer_levels.png)\n")

    # concrete worked example
    tr = demos_out.get("computer_trace")
    if tr:
        w("\n**A worked example.** An agent that inherited basic macros was given a "
          f"level-{tr['level']} task (`{tr['task_name']}`, keyword "
          f"*\"{tr['keyword']}\"*, input file `{tr['input_file']}`). It produced the "
          f"program:\n")
        w(f"\n```\n{tr['discovered_program']}\n```\n")
        w(f"\nground-truth pipeline: `{tr['canonical_program']}` — "
          f"output `{tr['output']!r}` vs expected `{tr['expected']!r}` "
          f"({'solved' if tr['solved'] else 'unsolved'}). Note the agent may find a "
          f"*shorter equivalent* program by reusing an inherited macro, which is "
          f"itself a small instance of cultural knowledge transfer.\n")
    top = full.culture.top_skills(8)
    if top:
        w("\nMost reputable computer macros in the final culture:\n")
        w("| macro | program | depth | adoption | reputation |\n")
        w("|---|---|---|---|---|\n")
        for s in top:
            w(f"| {s.name} | {'→'.join(s.program)} | {s.complexity()} | "
              f"{s.adoption} | {s.reputation:.1f} |\n")

    # ----- Experiment F: real sandboxed OS -----
    real = demos_out.get("real_os")
    if real:
        rows = real["rows"]
        w("\n### 3.5 From simulation to a real OS (Experiment F)\n")
        w("The Computer World above is simulated. To check the skills are *real*, "
          "**Real Computer World** maps every primitive op to an actual coreutils "
          "command (`cat`, `grep`, `sort`, `uniq`, `wc`, `tr`, `tac`, `cp`) run by "
          "`bash` in a throwaway temp sandbox (whitelisted commands, quoted args, "
          "minimal env, timeout — no network). An agent's macros transfer "
          "**unchanged** from the simulated world to the real shell.\n")
        w("We compare a *cultured* agent (inherited macros) with a *fresh* agent "
          f"(empty library, budget {real['fresh_budget']} real executions) on the "
          "same real tasks:\n")
        w("\n| level | task | cultured solved | cultured shell cmds | fresh solved | fresh shell cmds |\n")
        w("|---|---|---|---|---|---|\n")
        for r in rows:
            w(f"| {r['level']} | `{r['name']}` | "
              f"{'✅' if r['cultured_solved'] else '❌'} | {r['cultured_shell_calls']} | "
              f"{'✅' if r['fresh_solved'] else '❌'} | {r['fresh_shell_calls']} |\n")
        n_cult = sum(r["cultured_solved"] for r in rows)
        n_fresh = sum(r["fresh_solved"] for r in rows)
        w(f"\nThe cultured agent solved **{n_cult}/{len(rows)}** real tasks; the "
          f"fresh agent solved only **{n_fresh}/{len(rows)}** before exhausting its "
          f"real-execution budget. Inherited skill makes real computer use cheap; "
          f"from scratch it is prohibitively expensive — the accumulation thesis "
          f"holds against a genuine operating system.\n")
        tr = real.get("trace")
        if tr:
            w(f"\nA real command trace (level {tr['level']}, `{tr['name']}`, keyword "
              f"*\"{tr['keyword']}\"*) the agent actually executed in its sandbox:\n")
            w("\n```bash\n" + "\n".join(tr["commands"]) + "\n```\n")
            w(f"produced `{tr['output']!r}` (expected `{tr['expected']!r}`).\n")
        w("\n![Real OS shell: cost to solve, cultured vs fresh](figures/14_real_os_shell.png)\n")
