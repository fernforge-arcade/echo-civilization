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
