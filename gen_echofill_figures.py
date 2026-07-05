"""Figures for the wrangling experiment: the naive-vs-cultured capability gap and
the cost comparison against an LLM baseline."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = json.loads(Path("results/echofill_bench.json").read_text())
FIG = Path("figures")
FIG.mkdir(exist_ok=True)

INK = "#1a1a2e"
NAIVE_C = "#c44"
CULT_C = "#2a9d8f"
LLM_C = "#e9c46a"


def fig_arms():
    tasks = [r["task"] for r in RES["cultured"]]
    naive = [r["accuracy"] * 100 for r in RES["naive"]]
    cult = [r["accuracy"] * 100 for r in RES["cultured"]]
    kinds = [r["kind"] for r in RES["cultured"]]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    y = range(len(tasks))
    h = 0.38
    ax.barh([i + h / 2 for i in y], naive, height=h, color=NAIVE_C,
            label="naive agent (no culture)")
    ax.barh([i - h / 2 for i in y], cult, height=h, color=CULT_C,
            label="cultured agent (inherited skill library)")
    ax.set_yticks(list(y))
    labels = [f"{t}\n({k})" for t, k in zip(tasks, kinds)]
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("held-out row accuracy (%)")
    ax.set_xlim(0, 108)
    ax.set_title("Same held-out wrangling suite: culture solves the composite\n"
                 "tasks a from-scratch agent cannot compose",
                 color=INK, fontsize=12)
    for i, (nv, cv) in enumerate(zip(naive, cult)):
        ax.text(nv + 1, i + h / 2, f"{nv:.0f}%", va="center", fontsize=8, color=NAIVE_C)
        ax.text(cv + 1, i - h / 2, f"{cv:.0f}%", va="center", fontsize=8, color=CULT_C)
    ax.legend(loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "28_echofill_arms.png", dpi=130)
    print("wrote figures/28_echofill_arms.png")


def fig_cost():
    # cost to process a column of N rows: LLM (paid per row) vs cultured ($0)
    sizes = [1_000, 10_000, 100_000, 1_000_000]
    tpr_in, tpr_out = RES["summary"]["llm_tokens_per_row"]
    llm = [n * (tpr_in / 1e6 * 1.00 + tpr_out / 1e6 * 5.00) for n in sizes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    x = range(len(sizes))
    ax1.bar([i - 0.2 for i in x], llm, width=0.4, color=LLM_C,
            label="LLM per row (Haiku 4.5)")
    ax1.bar([i + 0.2 for i in x], [0] * len(sizes), width=0.4, color=CULT_C,
            label="cultured agents ($0)")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([f"{n:,}" for n in sizes], rotation=20, fontsize=8)
    ax1.set_ylabel("cost to process one column (USD)")
    ax1.set_xlabel("rows")
    ax1.set_title("Cost of the same wrangling job", fontsize=11, color=INK)
    for i, c in enumerate(llm):
        ax1.text(i - 0.2, c, f"${c:,.0f}" if c >= 1 else f"${c:.2f}",
                 ha="center", va="bottom", fontsize=8)
    ax1.legend(fontsize=9)
    ax1.spines[["top", "right"]].set_visible(False)

    # accuracy vs cost scatter (per 100k rows)
    acc = {"naive": RES["summary"]["naive_row_accuracy"] * 100,
           "cultured": RES["summary"]["cultured_row_accuracy"] * 100,
           "LLM": 100.0}
    cost = {"naive": 0, "cultured": 0, "LLM": RES["summary"]["llm_cost_100k"]}
    cols = {"naive": NAIVE_C, "cultured": CULT_C, "LLM": LLM_C}
    for k in acc:
        ax2.scatter(cost[k], acc[k], s=180, color=cols[k], zorder=3,
                    edgecolor="white")
        ax2.annotate(k, (cost[k], acc[k]), textcoords="offset points",
                     xytext=(8, 6), fontsize=10)
    ax2.set_xlabel("cost per 100k rows (USD)")
    ax2.set_ylabel("held-out accuracy (%)")
    ax2.set_title("Accuracy vs cost (100k rows)", fontsize=11, color=INK)
    ax2.set_ylim(30, 108)
    ax2.set_xlim(-1.5, RES["summary"]["llm_cost_100k"] + 3)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG / "29_echofill_cost.png", dpi=130)
    print("wrote figures/29_echofill_cost.png")


if __name__ == "__main__":
    fig_arms()
    fig_cost()
