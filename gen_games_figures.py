#!/usr/bin/env python
"""Figures for Experiment L (Game World ladder).

Reads results/games.json (produced by run_games.py) and writes four PNGs to figures/:
  games_learning_curves.png   — 2x2 per-rung SOLO/POP/CIV mean-score curves with std bands
  games_techtree.png          — EchoCraft achievement dependency DAG (depth-layered)
  games_echocraft_depth.png   — EchoCraft max tech-depth vs generation (CIV rises, POP flat)
  games_culture_advantage.png — headline: culture advantage (CIV-POP) per rung
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from echo_civilization.games.craft import ACHIEVEMENTS, ACH_DEPTH, MAX_DEPTH

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results", "games.json")
FIGDIR = os.path.join(HERE, "figures")

COND_COLOR = {"SOLO": "#888888", "POP": "#d1495b", "CIV": "#2e86ab"}
COND_LABEL = {"SOLO": "SOLO (lone lifelong learner)",
              "POP": "POP (population, no sharing)",
              "CIV": "CIV (population + culture + generations)"}
RUNG_TITLE = {
    "tictactoe": "Tic-Tac-Toe  (cx≈5, solved)",
    "connect4": "Connect Four 6×5  (cx≈21)",
    "minichess": "Los Alamos minichess 6×6  (cx≈60)",
    "echocraft": "EchoCraft  (open-ended, cx≈100)",
}
RUNG_ORDER = ["tictactoe", "connect4", "minichess", "echocraft"]


def load():
    with open(RESULTS) as f:
        return json.load(f)


def fig_learning_curves(d):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for ax, rn in zip(axes.flat, RUNG_ORDER):
        r = d["rungs"][rn]
        for cond in ("SOLO", "POP", "CIV"):
            cd = r["conditions"][cond]
            y = np.array(cd["mean_curve"])
            std = np.array(cd.get("mean_curve_std", np.zeros_like(y)))
            x = np.arange(1, len(y) + 1)
            c = COND_COLOR[cond]
            ax.plot(x, y, "-o", color=c, lw=2, ms=4, label=cond)
            ax.fill_between(x, y - std, y + std, color=c, alpha=0.13)
        ax.set_title(RUNG_TITLE[rn], fontsize=11)
        ax.set_xlabel("generation")
        ax.set_ylabel("capability (mean pop. score)")
        ax.grid(alpha=0.25)
        ax.set_ylim(0, 1.02)
    axes.flat[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Game World ladder — capability over generations, by condition\n"
                 "(matched budget; POP & CIV have full generational turnover)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(FIGDIR, "games_learning_curves.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print("wrote", out)


def fig_techtree(d):
    # Build the dependency DAG from the ACHIEVEMENTS chain (edges = "unlocks next depth").
    G = nx.DiGraph()
    for name, depth, recipe in ACHIEVEMENTS:
        G.add_node(name, depth=depth, recipe=recipe)
    # Connect each node to the shallowest deeper nodes it plausibly gates: link consecutive
    # depths through the crafting spine so the figure reads as a tree.
    spine = [
        ("collect_wood", "place_table"),
        ("place_table", "make_wood_pickaxe"),
        ("make_wood_pickaxe", "collect_stone"),
        ("collect_stone", "place_furnace"),
        ("collect_stone", "make_stone_pickaxe"),
        ("make_stone_pickaxe", "collect_coal"),
        ("make_stone_pickaxe", "collect_iron"),
        ("collect_coal", "smelt_iron"),
        ("collect_iron", "smelt_iron"),
        ("place_furnace", "smelt_iron"),
        ("smelt_iron", "make_iron_pickaxe"),
        ("make_iron_pickaxe", "collect_diamond"),
    ]
    G.add_edges_from(spine)

    # Layered layout: x = depth, y = spread within a depth layer.
    by_depth = {}
    for n in G.nodes:
        by_depth.setdefault(ACH_DEPTH[n], []).append(n)
    pos = {}
    for depth, nodes in by_depth.items():
        nodes = sorted(nodes)
        for i, n in enumerate(nodes):
            y = (i - (len(nodes) - 1) / 2) * 1.3
            pos[n] = (depth, y)

    fig, ax = plt.subplots(figsize=(13, 6))
    recipe_nodes = [n for n in G.nodes if G.nodes[n]["recipe"]]
    innate_nodes = [n for n in G.nodes if not G.nodes[n]["recipe"]]
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#bbbbbb",
                           arrows=True, arrowsize=12, width=1.3,
                           node_size=1500, connectionstyle="arc3,rad=0.03")
    nx.draw_networkx_nodes(G, pos, nodelist=innate_nodes, ax=ax,
                           node_color="#cfe8f3", edgecolors="#2e86ab",
                           node_size=1500, linewidths=1.5)
    nx.draw_networkx_nodes(G, pos, nodelist=recipe_nodes, ax=ax,
                           node_color="#ffd8a8", edgecolors="#e8590c",
                           node_shape="s", node_size=1700, linewidths=1.8)
    nx.draw_networkx_labels(G, pos, ax=ax,
                            labels={n: n.replace("_", "\n") for n in G.nodes},
                            font_size=7)
    for depth in range(MAX_DEPTH + 1):
        ax.text(depth, min(y for _, y in pos.values()) - 1.2, f"depth {depth}",
                ha="center", va="top", fontsize=8, color="#666")
    ax.set_title("EchoCraft tech tree — 13 achievements, depth 0→8\n"
                 "orange squares = craftable recipes (the inheritable cultural unit); "
                 "blue = gatherable", fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    out = os.path.join(FIGDIR, "games_techtree.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print("wrote", out)


def fig_echocraft_depth(d):
    ec = d["rungs"]["echocraft"]["conditions"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for cond in ("SOLO", "POP", "CIV"):
        ex = ec[cond]["extras"]
        x = np.arange(1, len(ex["best_max_depth"]) + 1)
        ax1.plot(x, ex["best_max_depth"], "-o", color=COND_COLOR[cond],
                 lw=2, ms=4, label=cond)
        ax2.plot(x, ex["best_crafter"], "-o", color=COND_COLOR[cond],
                 lw=2, ms=4, label=cond)
    ax1.axhline(MAX_DEPTH, ls="--", color="#444", lw=1, alpha=0.6)
    ax1.text(1, MAX_DEPTH + 0.05, "bottom of tech tree (depth 8)",
             fontsize=8, color="#444")
    ax1.set_ylabel("max tech depth reached (best agent)")
    ax1.set_title("EchoCraft — how deep the tech tree gets mined")
    ax2.set_ylabel("Crafter score (geo-mean of achievement rates)")
    ax2.set_title("EchoCraft — Crafter score")
    for ax in (ax1, ax2):
        ax.set_xlabel("generation")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    fig.suptitle("Only culture keeps climbing: CIV reaches the bottom of the tree "
                 "over generations; POP stays shallow", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(FIGDIR, "games_echocraft_depth.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print("wrote", out)


def fig_culture_advantage(d):
    labels, advs, cxs = [], [], []
    for rn in RUNG_ORDER:
        r = d["rungs"][rn]
        civ = r["conditions"]["CIV"]["final_mean"]
        pop = r["conditions"]["POP"]["final_mean"]
        advs.append(civ - pop)
        cxs.append(r["complexity"])
        labels.append(RUNG_TITLE[rn].split("  ")[0])
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2e86ab" if a >= 0 else "#d1495b" for a in advs]
    bars = ax.bar(range(len(advs)), advs, color=colors, width=0.6)
    ax.axhline(0, color="#333", lw=1)
    ax.set_xticks(range(len(advs)))
    ax.set_xticklabels([f"{l}\n(cx≈{c})" for l, c in zip(labels, cxs)], fontsize=9)
    ax.set_ylabel("culture advantage  (CIV − POP final capability)")
    ax.set_title("Culture's payoff is NOT monotone in game-tree complexity\n"
                 "it is largest where skills can't be re-found in one lifetime AND "
                 "culture is stored losslessly (EchoCraft)", fontsize=11)
    for b, a in zip(bars, advs):
        ax.text(b.get_x() + b.get_width() / 2,
                a + (0.008 if a >= 0 else -0.008),
                f"{a:+.3f}", ha="center",
                va="bottom" if a >= 0 else "top", fontsize=10, fontweight="bold")
    ax.annotate("connect4: a lone agent masters it in one lifetime\n"
                "and averaged weights are a lossy culture → culture hurts",
                xy=(1.3, advs[1] * 0.55), xytext=(1.7, -0.055),
                fontsize=8, color="#d1495b", ha="left",
                arrowprops=dict(arrowstyle="->", color="#d1495b"))
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    out = os.path.join(FIGDIR, "games_culture_advantage.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print("wrote", out)


def main():
    d = load()
    fig_learning_curves(d)
    fig_techtree(d)
    fig_echocraft_depth(d)
    fig_culture_advantage(d)


if __name__ == "__main__":
    main()
