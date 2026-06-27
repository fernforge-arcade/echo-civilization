"""The required visualizations.

Produces five figures:
  1. average intelligence (capability) over generations,
  2. best-agent performance over generations,
  3. skill-propagation network,
  4. agent relationship network,
  5. complexity of behaviours over time,
plus a couple of supporting plots (per-difficulty solve rate; culture growth;
Echo-World learning curve; Social-World protocol emergence).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_average_intelligence(results, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, r in results.items():
        cc = r["capability_curve"]
        ax.plot(range(len(cc)), cc, marker="o", ms=3, label=r["label"])
    ax.set_title("1. Average capability on held-out hard tasks over generations")
    ax.set_xlabel("generation")
    ax.set_ylabel("avg fraction of hard tasks solvable\n(accumulated knowledge only)")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_best_performance(results, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, r in results.items():
        best = [h["best_fitness"] for h in r["history"]]
        ax.plot(range(len(best)), best, marker="s", ms=3, label=r["label"])
    ax.set_title("2. Best-agent fitness over generations")
    ax.set_xlabel("generation")
    ax.set_ylabel("best lifetime fitness")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_skill_propagation(culture, path, max_edges=150):
    """Directed network of teaching events: teacher -> student, per skill."""
    G = nx.DiGraph()
    for prog_name, frm, to, gen in culture.propagation_log[:max_edges]:
        G.add_edge(frm, to)
    fig, ax = plt.subplots(figsize=(8, 7))
    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "no propagation events\n(sharing disabled in this condition)",
                ha="center", va="center")
    else:
        pos = nx.spring_layout(G, seed=1, k=0.5)
        deg = dict(G.degree())
        sizes = [80 + 40 * deg[n] for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color="#3b7dd8",
                               alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.25, arrows=True,
                               arrowsize=7, ax=ax, edge_color="#888")
        ax.set_title(f"3. Skill-propagation network "
                     f"({G.number_of_nodes()} agents, {G.number_of_edges()} transfers)")
    ax.axis("off")
    return _save(fig, path)


def plot_relationship_network(population, path):
    """Undirected affinity network from agent relationships."""
    G = nx.Graph()
    for a in population:
        G.add_node(a.id, rep=a.reputation)
        for other, aff in a.relationships.items():
            if aff > 0.1:
                G.add_edge(a.id, other, weight=aff)
    fig, ax = plt.subplots(figsize=(8, 7))
    if G.number_of_edges() == 0:
        ax.text(0.5, 0.5, "no relationships formed\n(no social interaction in this condition)",
                ha="center", va="center")
    else:
        pos = nx.spring_layout(G, seed=2, k=0.6)
        reps = [max(0.0, population_rep(population, n)) for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=[60 + 120 * r for r in reps],
                               node_color=reps, cmap="viridis", ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
        ax.set_title(f"4. Agent relationship network "
                     f"({G.number_of_nodes()} agents, {G.number_of_edges()} ties)")
    ax.axis("off")
    return _save(fig, path)


def population_rep(population, agent_id):
    for a in population:
        if a.id == agent_id:
            return a.reputation
    return 0.0


def plot_complexity_over_time(results, path):
    """Complexity of behaviours: avg/max difficulty of solved tasks and avg skill
    program length, over generations, for the full-civilization condition."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, r in results.items():
        hist = r["history"]
        avg_diff = [h["avg_difficulty_solved"] for h in hist]
        ax.plot(range(len(avg_diff)), avg_diff, marker="o", ms=3,
                label=r["label"])
    ax.set_title("5. Behavioural complexity over time\n(avg composition depth of solved tasks)")
    ax.set_xlabel("generation")
    ax.set_ylabel("avg difficulty (program length) solved")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_culture_growth(results, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, r in results.items():
        cs = [h["culture_size"] for h in r["history"]]
        ax.plot(range(len(cs)), cs, marker="o", ms=3, label=r["label"])
    ax.set_title("Supporting: cultural repository size over generations")
    ax.set_xlabel("generation")
    ax.set_ylabel("# distinct skills in shared culture")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_difficulty_breakdown(result, path):
    """Per-difficulty solve rate over generations for one condition."""
    hist = result["history"]
    diffs = sorted({d for h in hist for d in h["solve_rate_by_difficulty"]})
    fig, ax = plt.subplots(figsize=(8, 5))
    for d in diffs:
        ys = [h["solve_rate_by_difficulty"].get(d, np.nan) for h in hist]
        ax.plot(range(len(ys)), ys, marker="o", ms=3, label=f"difficulty {d}")
    ax.set_title(f"Supporting: solve rate by task difficulty ({result['label']})")
    ax.set_xlabel("generation")
    ax.set_ylabel("solve rate")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_echo_learning(curve, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(curve)), curve, color="#d8543b")
    ax.set_title("Echo World: tabular Q-learning accuracy while learning to copy")
    ax.set_xlabel("training episode")
    ax.set_ylabel("character-copy accuracy")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_social_emergence(result, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(result["accuracy_curve"], label="communication accuracy")
    ax.plot(result["consistency_curve"], label="protocol consistency")
    ax.set_title("Social World: emergence of a shared communication protocol")
    ax.set_xlabel("interaction round")
    ax.set_ylabel("fraction")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, path)


def _smooth(xs, k=3):
    out = []
    for i in range(len(xs)):
        lo = max(0, i - k + 1)
        out.append(float(np.mean(xs[lo:i + 1])))
    return out


def plot_computer_curriculum(full_hist, control_hist, path):
    """Headline extension figure: how high up the open-ended task-complexity
    ladder each civilization climbs over generations (auto-curriculum)."""
    fig, ax = plt.subplots(figsize=(8.5, 5))
    gens_f = range(len(full_hist))
    gens_c = range(len(control_hist))
    ax.plot(gens_f, [h["frontier"] for h in full_hist], color="#c0392b",
            lw=2, label="full civilization — curriculum frontier unlocked")
    ax.plot(gens_f, _smooth([h["mastered_level"] for h in full_hist]),
            color="#e67e22", ls="--", label="full civilization — level mastered (smoothed)")
    ax.plot(gens_c, _smooth([h["mastered_level"] for h in control_hist]),
            color="#2980b9", ls="--", label="no-sharing control — level mastered (smoothed)")
    ax.set_title("Computer World: climbing the task-complexity ladder\n"
                 "(agents evolving to match increasingly sophisticated tasks)")
    ax.set_xlabel("generation")
    ax.set_ylabel("curriculum level (1=copy file … 5=deep pipeline)")
    ax.set_yticks(range(0, full_hist[0]["max_level"] + 1))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_computer_levels(full_hist, path):
    """Per-level solve rate over generations for the full computer civilization."""
    max_level = full_hist[0]["max_level"]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for lvl in range(1, max_level + 1):
        ys = [h["solve_rate_by_level"].get(lvl) for h in full_hist]
        gens = [i for i, y in enumerate(ys) if y is not None]
        vals = [y for y in ys if y is not None]
        if gens:
            ax.plot(gens, vals, marker="o", ms=2, label=f"level {lvl}")
    ax.set_title("Computer World: solve rate per task level over generations\n"
                 "(deeper levels only become solvable after macros accumulate)")
    ax.set_xlabel("generation")
    ax.set_ylabel("solve rate (when that level was offered)")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_enterprise(full_hist, control_hist, path):
    """Experiment G: autonomous firm — cumulative profit and rising order
    sophistication over a long, never-terminating run, with vs without a shared
    knowledge base."""
    fig, ax = plt.subplots(figsize=(8.5, 5))
    days_f = [h["day"] for h in full_hist]
    days_c = [h["day"] for h in control_hist]
    ax.plot(days_f, [h["cum_profit"] for h in full_hist], color="#16a085", lw=2,
            label="firm WITH shared knowledge base")
    ax.plot(days_c, [h["cum_profit"] for h in control_hist], color="#c0392b", lw=2,
            label="firm WITHOUT knowledge base (control)")
    ax.set_xlabel("business day (continuous operation)")
    ax.set_ylabel("cumulative profit")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.step(days_f, [h["max_order_level"] for h in full_hist], color="#16a085",
             ls=":", alpha=0.7, where="post",
             label="firm ambition: hardest order level sold")
    ax2.set_ylabel("order sophistication level (1–5)")
    ax2.set_ylim(0, full_hist[0].get("max_level_done", 5) and 5.5)
    ax.set_title("Experiment G — autonomous firm running forever\n"
                 "(institutional knowledge compounds into profit & sophistication)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
    return _save(fig, path)


def plot_real_os(demo, path):
    """Experiment F: real sandboxed-shell execution — cost (real commands run) to
    solve each level, cultured vs fresh, with solved/failed marked."""
    rows = demo["rows"]
    levels = [r["level"] for r in rows]
    cult = [r["cultured_shell_calls"] for r in rows]
    fresh = [r["fresh_shell_calls"] for r in rows]
    x = np.arange(len(levels))
    fig, ax = plt.subplots(figsize=(8.5, 5))
    b1 = ax.bar(x - 0.2, cult, 0.38, label="cultured agent (inherited macros)",
                color="#27ae60")
    b2 = ax.bar(x + 0.2, fresh, 0.38, label="fresh agent (no macros)",
                color="#c0392b")
    for r, rect in zip(rows, b1):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.5,
                "✓" if r["cultured_solved"] else "✗", ha="center", fontsize=9)
    for r, rect in zip(rows, b2):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.5,
                "✓" if r["fresh_solved"] else "✗", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}\n{r['name']}" for l, r in zip(levels, rows)],
                       fontsize=7)
    ax.set_title("Experiment F — real OS shell: cost to solve a task\n"
                 "(✓ solved, ✗ failed within budget; bar height = real commands run)")
    ax.set_ylabel("real shell commands executed")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_grid_evolution(curve, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([c["avg"] for c in curve], marker="o", ms=3, label="population mean")
    ax.plot([c["best"] for c in curve], marker="s", ms=3, label="best")
    ax.set_title("Grid World: evolved neural-network policy fitness over generations")
    ax.set_xlabel("generation")
    ax.set_ylabel("episode reward")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, path)


# ----------------------------------------------------- generalization experiment
SHORT_LABELS = {
    "A_single": "A single",
    "B_population_nosharing": "B no-share",
    "C_population_memorysharing": "C sharing",
    "D_full_civilization": "D full civ",
}
_COND_COLORS = {
    "A_single": "#7f8c8d",
    "B_population_nosharing": "#e67e22",
    "C_population_memorysharing": "#27ae60",
    "D_full_civilization": "#c0392b",
}


def plot_generalization_bars(summary, path):
    """Grouped bars: per-condition solve rate on each suite, with seed std error
    bars. The story is the depth-3 group: novel composites needing an inherited
    intermediate abstraction."""
    suites = [("train2_newinputs", "trained depth-2\n(new inputs, in-dist)"),
              ("held2", "NOVEL depth-2\n(needs primitives)"),
              ("held3", "NOVEL depth-3\n(needs depth-2 abstraction)")]
    conds = list(summary)
    x = np.arange(len(suites))
    w = 0.2
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, name in enumerate(conds):
        means = [summary[name][s][0] for s, _ in suites]
        stds = [summary[name][s][1] for s, _ in suites]
        ax.bar(x + (i - 1.5) * w, means, w, yerr=stds, capsize=3,
               label=SHORT_LABELS.get(name, name),
               color=_COND_COLORS.get(name, None))
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in suites])
    ax.set_ylabel("frozen solve rate (no discovery, no test-time learning)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Compositional generalization: memorization vs. recombination\n"
                 "(held-out programs never seen in training; mean ± SD over seeds)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(title="condition", fontsize=9)
    return _save(fig, path)


_ADAPT_SHORT = dict(SHORT_LABELS, FRESH="FRESH (gen-0)")
_ADAPT_COLORS = dict(_COND_COLORS, FRESH="#2c3e50")


def plot_adaptability_bars(summary, tight_budget, generous_budget, oracle_rate, path):
    """Grouped bars: per-condition solve rate on the NOVEL combinator family at the
    TIGHT budget (where culture decides) with a faint generous-budget overlay (the
    reachable ceiling). Conditions ordered A,B,C,D,FRESH; oracle drawn as a line."""
    order = ["A_single", "B_population_nosharing", "C_population_memorysharing",
             "D_full_civilization", "FRESH"]
    conds = [c for c in order if c in summary]
    x = np.arange(len(conds))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    tight = [summary[c]["tight"][0] for c in conds]
    tight_sd = [summary[c]["tight"][1] for c in conds]
    gen = [summary[c]["generous"][0] for c in conds]
    ax.bar(x, gen, 0.6, color="#d0d3d4", label=f"generous budget ({generous_budget})",
           zorder=1)
    ax.bar(x, tight, 0.42, yerr=tight_sd, capsize=4, zorder=2,
           color=[_ADAPT_COLORS.get(c) for c in conds],
           label=f"TIGHT budget ({tight_budget})")
    ax.axhline(oracle_rate, ls="--", color="#8e44ad", lw=1.5,
               label=f"oracle (knows inner f's): {oracle_rate:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels([_ADAPT_SHORT.get(c, c) for c in conds])
    ax.set_ylabel("frozen solve rate on NOVEL combinator family")
    ax.set_ylim(0, 1.05)
    ax.set_title("Adaptability to a structurally NOVEL task family\n"
                 "(higher-order combinators nobody trained on; mean ± SD over seeds)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9)
    return _save(fig, path)


def plot_adaptability_curve(curves, path):
    """Solve rate vs. search budget for the cultured civ (D) vs a FRESH gen-0 agent
    on the novel family. The gap between the curves IS the inherited-library edge:
    D climbs to the ceiling far earlier; FRESH needs a far larger budget."""
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for name, curve in curves.items():
        if not curve:
            continue
        bs = [b for b, _ in curve]
        vs = [v for _, v in curve]
        ax.plot(bs, vs, marker="o", ms=4, label=_ADAPT_SHORT.get(name, name),
                color=_ADAPT_COLORS.get(name))
    ax.set_xscale("log")
    ax.set_xlabel("search budget (consistency checks allowed per task)")
    ax.set_ylabel("frozen solve rate on novel family")
    ax.set_ylim(0, 1.05)
    ax.set_title("Adaptation curve: inherited abstractions move the budget frontier\n"
                 "(novel combinator family, seed 0)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(title="condition", fontsize=9)
    return _save(fig, path)


def plot_parametric_bars(summary, tight_budget, generous_budget, oracle_rate, path):
    """Grouped bars: per-condition frozen solve rate on the NOVEL high-argument suite
    at the TIGHT budget (where inherited schemas decide) over a faint generous-budget
    overlay (the ceiling both can reach). Conditions A,B,C,D,FRESH; oracle as a line.
    Mirrors plot_adaptability_bars but the lever here is the inherited parametric
    SCHEMA (a family + its argument inverter), not a concrete inner program."""
    order = ["A_single", "B_population_nosharing", "C_population_memorysharing",
             "D_full_civilization", "FRESH"]
    conds = [c for c in order if c in summary]
    x = np.arange(len(conds))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    tight = [summary[c]["tight"][0] for c in conds]
    tight_sd = [summary[c]["tight"][1] for c in conds]
    gen = [summary[c]["generous"][0] for c in conds]
    ax.bar(x, gen, 0.6, color="#d0d3d4", label=f"generous budget ({generous_budget})",
           zorder=1)
    ax.bar(x, tight, 0.42, yerr=tight_sd, capsize=4, zorder=2,
           color=[_ADAPT_COLORS.get(c) for c in conds],
           label=f"TIGHT budget ({tight_budget})")
    ax.axhline(oracle_rate, ls="--", color="#8e44ad", lw=1.5,
               label=f"oracle (holds every schema): {oracle_rate:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels([_ADAPT_SHORT.get(c, c) for c in conds])
    ax.set_ylabel("frozen solve rate on NOVEL high-argument suite")
    ax.set_ylim(0, 1.05)
    ax.set_title("Parametric abstraction: binding a NOVEL argument to an inherited schema\n"
                 "(args 3/4/5 unseen during accumulation; mean ± SD over seeds)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9)
    return _save(fig, path)


def plot_parametric_curve(curves, path):
    """Solve rate vs. search budget for the cultured civ (D) vs a FRESH gen-0 agent
    on the novel high-argument suite. D inverts the argument per inherited family in a
    handful of checks; FRESH must blind-sweep the {14 families × 7 args × 2 inners}
    grid, so it needs an order of magnitude more budget to reach the same ceiling."""
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for name, curve in curves.items():
        if not curve:
            continue
        bs = [b for b, _ in curve]
        vs = [v for _, v in curve]
        ax.plot(bs, vs, marker="o", ms=4, label=_ADAPT_SHORT.get(name, name),
                color=_ADAPT_COLORS.get(name))
    ax.set_xscale("log")
    ax.set_xlabel("search budget (consistency checks allowed per task)")
    ax.set_ylabel("frozen solve rate on novel high-argument suite")
    ax.set_ylim(0, 1.05)
    ax.set_title("Argument-binding frontier: inherited schemas move the budget wall\n"
                 "(novel high-argument suite, seed 0)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(title="condition", fontsize=9)
    return _save(fig, path)


def plot_generalization_curve(curves, path, suite_label="novel depth-3"):
    """Per-generation frozen solve rate on the held-out suite (seed 0)."""
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for name, curve in curves.items():
        if not curve:
            continue
        gens = [g for g, _ in curve]
        vals = [v for _, v in curve]
        ax.plot(gens, vals, marker="o", ms=3, label=SHORT_LABELS.get(name, name),
                color=_COND_COLORS.get(name, None))
    ax.set_xlabel("generation")
    ax.set_ylabel(f"frozen solve rate on {suite_label}")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Accumulation of generalization over generations\n"
                 f"({suite_label}, never-trained composites, seed 0)")
    ax.grid(alpha=0.3)
    ax.legend(title="condition", fontsize=9)
    return _save(fig, path)


# ======================================================================
# Experiment J — Builder World
# ======================================================================

def plot_builder_frontier(payload, path):
    """Frontier of buildable apps (max feature_count fully built) vs generation,
    averaged over seeds, for the three conditions."""
    curves = payload["curves"]
    labels = {
        "A_monolithic": "A  monolithic (no decompose, no culture)",
        "B_decomposed_no_culture": "B  decomposed, culture wiped each gen",
        "C_decomposed_culture": "C  decomposed + accumulating culture",
    }
    colors = {"A_monolithic": "#bbbbbb",
              "B_decomposed_no_culture": "#1f77b4",
              "C_decomposed_culture": "#d62728"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for cond in ["A_monolithic", "B_decomposed_no_culture", "C_decomposed_culture"]:
        fr = curves[cond]["frontier"]
        ax.plot(range(len(fr)), fr, marker="o", ms=5, lw=2,
                color=colors[cond], label=labels[cond])
    maxfc = max(sp["feature_count"] for sp in payload["specs"])
    ax.axhline(maxfc, ls="--", color="#999", lw=1)
    ax.text(0.1, maxfc + 0.05, f"hardest app in catalogue ({maxfc} features)",
            fontsize=8, color="#666")
    ax.set_title("Builder World: frontier of buildable apps over generations\n"
                 "(most-complex app a generation can actually build & run; "
                 "build budget fixed)")
    ax.set_xlabel("generation")
    ax.set_ylabel("frontier = max features in a fully-built, test-passing app")
    ax.set_ylim(-0.2, maxfc + 0.6)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="center right")
    return _save(fig, path)


def plot_builder_fresh_vs_cultured(payload, path):
    """Per-spec build success: fresh agent vs cultured (full inherited library)."""
    fvc = payload["fresh_vs_cultured"]
    names = list(fvc.keys())
    feats = [fvc[n]["feature_count"] for n in names]
    fresh = [fvc[n]["fresh"] for n in names]
    cult = [fvc[n]["cultured"] for n in names]
    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, fresh, w, label="fresh (no culture)", color="#1f77b4")
    ax.bar(x + w / 2, cult, w, label="cultured (inherited library)", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n({f} feat)" for n, f in zip(names, feats)], fontsize=8)
    ax.set_ylabel(f"build success rate (single agent, budget "
                  f"{payload['config']['build_budget']})")
    ax.set_ylim(0, 1.05)
    ax.set_title("Builder World: a single agent's build rate, fresh vs cultured\n"
                 "(culture lifts the buildable frontier from ~3 features to all 6)")
    for xi, v in zip(x - w / 2, fresh):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
    for xi, v in zip(x + w / 2, cult):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9)
    return _save(fig, path)


def plot_builder_culture_growth(payload, path):
    """Size of the inherited component library over generations (condition C),
    with the frontier overlaid to show culture-size driving capability."""
    curves = payload["curves"]
    cs = curves["C_decomposed_culture"]["culture_size"]
    fr = curves["C_decomposed_culture"]["frontier"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(cs)), cs, marker="o", ms=5, lw=2, color="#2ca02c",
            label="shared component library size")
    ax.set_xlabel("generation")
    ax.set_ylabel("# components in shared culture", color="#2ca02c")
    ax.tick_params(axis="y", labelcolor="#2ca02c")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(range(len(fr)), fr, marker="s", ms=5, lw=2, ls="--",
             color="#d62728", label="frontier (features)")
    ax2.set_ylabel("frontier = max features built", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax.set_title("Builder World: accumulating culture lifts the build frontier\n"
                 "(condition C — library grows, harder apps become buildable)")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="lower right")
    return _save(fig, path)


# ----------------------------------------------------------------------
# Experiment K — Stack World (full-stack, resilient builder)
# ----------------------------------------------------------------------

_STACK_ORDER = ["BRITTLE", "RESILIENT", "BRITTLE+CULTURE", "RESILIENT+CULTURE"]
_STACK_LABELS = {
    "BRITTLE": "brittle (blind search, no culture)",
    "RESILIENT": "resilient (repair, no culture)",
    "BRITTLE+CULTURE": "brittle + culture",
    "RESILIENT+CULTURE": "resilient + culture",
}
_STACK_COLORS = {
    "BRITTLE": "#bbbbbb",
    "RESILIENT": "#1f77b4",
    "BRITTLE+CULTURE": "#ff7f0e",
    "RESILIENT+CULTURE": "#d62728",
}


def plot_stack_frontier(payload, path):
    """Frontier (largest full-stack app a generation can fully build & boot, in
    endpoints) over generations, for all four conditions."""
    conds = payload["conditions"]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for name in _STACK_ORDER:
        rows = conds[name]
        fr = [r["frontier"] for r in rows]
        ax.plot(range(len(fr)), fr, marker="o", ms=5, lw=2,
                color=_STACK_COLORS[name], label=_STACK_LABELS[name])
    maxep = max(sp["endpoints"] for sp in payload["meta"]["specs"])
    ax.axhline(maxep, ls="--", color="#999", lw=1)
    ax.text(0.1, maxep + 0.3, f"largest spec ({maxep}-endpoint platform)",
            fontsize=8, color="#666")
    ax.set_title("Stack World: frontier of buildable full-stack apps over "
                 "generations\n(largest app a generation builds so every REST "
                 "endpoint passes its Node tests)")
    ax.set_xlabel("generation")
    ax.set_ylabel("frontier = endpoints in a fully-built, test-passing app")
    ax.set_ylim(-0.5, maxep + 1.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="center right")
    return _save(fig, path)


def plot_stack_reliability(payload, path):
    """Per-condition reliability: endpoint pass-rate and the fraction of passing
    endpoints that needed a repair (recovery), at the final generation."""
    conds = payload["conditions"]
    names = _STACK_ORDER
    endpoint_rate = [conds[n][-1]["endpoint_rate"] for n in names]
    recovery = [conds[n][-1]["recovery_rate"] for n in names]
    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, endpoint_rate, w, label="endpoint pass rate",
           color="#1f77b4")
    ax.bar(x + w / 2, recovery, w,
           label="share of passes recovered by repair", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels([_STACK_LABELS[n] for n in names], fontsize=7.5,
                       rotation=12, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("rate (final generation, averaged over seeds)")
    ax.set_title("Stack World: per-endpoint reliability and repair-driven "
                 "recovery\n(resilience lifts the pass rate and recovers "
                 "near-misses; culture pushes pass rate to 1.0)")
    for xi, v in zip(x - w / 2, endpoint_rate):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
    for xi, v in zip(x + w / 2, recovery):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="upper left")
    return _save(fig, path)


def plot_stack_culture_growth(payload, path):
    """Endpoint-type vocabulary in shared culture vs the frontier it unlocks
    (RESILIENT+CULTURE condition)."""
    rows = payload["conditions"]["RESILIENT+CULTURE"]
    cs = [r["culture_size"] for r in rows]
    fr = [r["frontier"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(cs)), cs, marker="o", ms=5, lw=2, color="#2ca02c",
            label="endpoint types in shared culture")
    ax.set_xlabel("generation")
    ax.set_ylabel("# proven endpoint-type configs", color="#2ca02c")
    ax.tick_params(axis="y", labelcolor="#2ca02c")
    ax.set_ylim(-0.3, 6)
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(range(len(fr)), fr, marker="s", ms=5, lw=2, ls="--",
             color="#d62728", label="frontier (endpoints)")
    ax2.set_ylabel("frontier = max endpoints built", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax.set_title("Stack World: a 5-type endpoint vocabulary unlocks unbounded "
                 "resources\n(once create/list/read/update/delete are proven, "
                 "every new resource is near-free)")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8,
              loc="center right")
    return _save(fig, path)
