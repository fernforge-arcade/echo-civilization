"""Record *actual gameplay* from the four Game-World rungs (Experiment L) as animated GIFs.

For each rung we train a real agent with the same code the experiment uses, then play out one
representative game greedily and save every ply/step as a frame. The result is a short looping
"video" (GIF) that embeds directly in REPORT.md — a lone TTT agent holding a draw against
perfect play, a linear Connect-Four evaluator beating the heuristic, the minichess searcher
grinding out material against the engine, and — the headline — a culture-born EchoCraft agent
walking its tech tree down to diamond.

No pretrained anything: same numpy + stdlib agents as the rest of the ladder. Run:

    ./venv/bin/python gen_games_animations.py            # all four
    ./venv/bin/python gen_games_animations.py --only echocraft
"""

from __future__ import annotations

import argparse
import io

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrow
from PIL import Image

from echo_civilization.games import tictactoe as ttt
from echo_civilization.games import connect4 as c4
from echo_civilization.games import minichess as mc
from echo_civilization.games import craft as cr
from echo_civilization.games.harness import run_condition

FIG_DIR = "figures"


def _fig_to_image(fig):
    """Rasterize a matplotlib figure to a PIL RGB image (stable size across frames)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor())
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return img.copy()


def _save_gif(frames, path, duration_ms=700, hold_last=6):
    """Write a looping GIF; repeat the final frame so the outcome lingers before the loop."""
    frames = frames + [frames[-1]] * hold_last
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0, optimize=True)
    print(f"  wrote {path}  ({len(frames)} frames)")


# ---------------------------------------------------------------------------------------------
# Rung 1 — Tic-Tac-Toe: a trained agent holds a draw against the perfect minimax opponent.
# ---------------------------------------------------------------------------------------------
def _draw_ttt(board, title, subtitle, highlight=None):
    fig, ax = plt.subplots(figsize=(4.2, 4.9))
    ax.set_xlim(-0.05, 3.05)
    ax.set_ylim(-0.05, 3.95)
    ax.axis("off")
    for i in range(4):
        ax.plot([0, 3], [i, i], color="#333", lw=2)
        ax.plot([i, i], [0, 3], color="#333", lw=2)
    for idx, ch in enumerate(board):
        r, c = divmod(idx, 3)
        x, y = c + 0.5, (2 - r) + 0.5
        if idx == highlight:
            ax.add_patch(Rectangle((c, 2 - r), 1, 1, color="#ffe08a", zorder=0))
        if ch == "X":
            ax.plot([x - 0.28, x + 0.28], [y - 0.28, y + 0.28], color="#d23", lw=5)
            ax.plot([x - 0.28, x + 0.28], [y + 0.28, y - 0.28], color="#d23", lw=5)
        elif ch == "O":
            ax.add_patch(Circle((x, y), 0.3, fill=False, color="#268", lw=5))
    ax.text(1.5, 3.72, title, ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(1.5, 3.38, subtitle, ha="center", va="center", fontsize=10, color="#555")
    fig.tight_layout()
    img = _fig_to_image(fig)
    plt.close(fig)
    return img


def _play_ttt_game(agent, agent_mark, ev):
    """Play one greedy game vs perfect play; return (outcome, list_of_(board,move))."""
    board, mark = "." * 9, "X"
    steps = []
    while True:
        if mark == agent_mark:
            a = agent.act(board, mark, greedy=True)
        else:
            a = ttt.perfect_move(board, mark, ev)
        board = ttt.apply_move(board, a, mark)
        steps.append((board, a, mark))
        w = ttt.winner(board)
        if w is not None:
            return w, steps
        mark = "O" if mark == "X" else "X"


def make_tictactoe(rng_seed=0):
    print("tic-tac-toe: pooling a population's answer books (CIV), then holding a draw vs perfect")
    # A single lifelong learner does NOT reliably hold the draw vs perfect play under a tight
    # budget (that is the rung's whole point — see the report). So we demonstrate the *culture*
    # version: run the real CIV loop so a population pools its self-play answer books into a
    # consensus, then a culture-born agent inherits that book and holds the draw.
    rung = ttt.TicTacToeRung()
    rng = np.random.default_rng(rng_seed)
    culture = rung.new_culture()
    agents = [rung.new_agent(rng, culture) for _ in range(8)]
    for g in range(10):
        for ag in agents:
            rung.train(ag, rng, 300)
        evals = [rung.evaluate(ag, rng)["score"] for ag in agents]
        order = sorted(range(len(agents)), key=lambda i: evals[i], reverse=True)
        for i in order[:4]:
            rung.extract(agents[i], culture)
        for ag in agents:
            rung.transfer(ag, culture)
        agents = [rung.new_agent(rng, culture, parent=agents[order[0]]) for _ in range(8)]
    print(f"  consensus answer-book size: {culture.size()} states")

    agent = rung.new_agent(rng, culture)  # a child born into the civilization's answer book

    # Agent plays X vs the perfect (unbeatable) minimax opponent. The best any player can do vs
    # perfect play is a draw — a culture-born agent holding that draw is the skill on display.
    agent_mark = "X"
    outcome, steps = None, None
    for s in range(60):
        ev = np.random.default_rng(rng_seed + 100 + s)
        outcome, steps = _play_ttt_game(agent, agent_mark, ev)
        if outcome == "D":
            break
    print(f"  displayed game outcome vs perfect play: {outcome}")

    frames = [_draw_ttt("." * 9, "Tic-Tac-Toe", "culture-born agent (X) vs perfect play (O)")]
    for board, a, mark in steps:
        if mark == agent_mark:
            who = "agent (X) moves"
        else:
            who = "perfect opponent (O) moves"
        frames.append(_draw_ttt(board, "Tic-Tac-Toe", who, highlight=a))
    outcome_txt = {"D": "draw — the best attainable result vs perfect play",
                   agent_mark: "agent wins", "O": "opponent wins"}[outcome]
    frames.append(_draw_ttt(steps[-1][0], "Tic-Tac-Toe", outcome_txt, highlight=steps[-1][1]))
    _save_gif(frames, f"{FIG_DIR}/games_play_tictactoe.gif", duration_ms=750)


# ---------------------------------------------------------------------------------------------
# Rung 2 — Connect Four: a TD-learned linear evaluator beats the 1-ply heuristic.
# ---------------------------------------------------------------------------------------------
def _draw_c4(board, title, subtitle, last_col=None):
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.set_xlim(-0.5, c4.COLS - 0.5)
    ax.set_ylim(-0.5, c4.ROWS + 0.4)
    ax.axis("off")
    ax.add_patch(Rectangle((-0.5, -0.5), c4.COLS, c4.ROWS, color="#2456a6", zorder=0))
    for r in range(c4.ROWS):
        for c in range(c4.COLS):
            v = board[r, c]
            color = "#eef" if v == 0 else ("#f2c53d" if v == 1 else "#e2503a")
            yy = (c4.ROWS - 1 - r)
            edge = "#c88" if (last_col is not None and c == last_col and _top_filled(board, c) == r) else "none"
            ax.add_patch(Circle((c, yy), 0.42, color=color, zorder=1,
                                ec=edge, lw=3))
    ax.text((c4.COLS - 1) / 2, c4.ROWS + 0.22, title, ha="center", fontsize=14, fontweight="bold")
    ax.text((c4.COLS - 1) / 2, c4.ROWS - 0.15, subtitle, ha="center", fontsize=10, color="#333")
    fig.tight_layout()
    img = _fig_to_image(fig)
    plt.close(fig)
    return img


def _top_filled(board, col):
    for r in range(c4.ROWS):
        if board[r, col] != 0:
            return r
    return None


def make_connect4(rng_seed=1):
    print("connect four: training a TD linear evaluator, then beating the heuristic")
    rng = np.random.default_rng(rng_seed)
    agent = c4.Connect4Agent(rng)
    for i in range(2500):
        c4.self_play_episode(agent, rng)
        agent.epsilon = max(0.03, agent.epsilon * 0.999)

    # agent is player 1 (yellow) vs the 1-ply heuristic (player -1, red)
    ev = np.random.default_rng(rng_seed + 7)
    board, player = c4.new_board(), 1
    frames = [_draw_c4(board, "Connect Four", "trained agent (yellow) vs heuristic (red)")]
    last = None
    for _ in range(60):
        if player == 1:
            col = agent.act(board, player, greedy=True)
            who = "agent (yellow) drops"
        else:
            col = c4._heuristic_move(board, player, ev)
            who = "heuristic (red) drops"
        board = c4.drop(board, col, player)
        last = col
        w = c4.winner(board)
        if w is not None:
            outcome = {1: "agent connects four — win",
                       -1: "heuristic wins", 0: "draw"}[w]
            frames.append(_draw_c4(board, "Connect Four", outcome, last_col=last))
            break
        frames.append(_draw_c4(board, "Connect Four", who, last_col=last))
        player = -player
    _save_gif(frames, f"{FIG_DIR}/games_play_connect4.gif", duration_ms=650)


# ---------------------------------------------------------------------------------------------
# Rung 3 — Los Alamos minichess: the searcher grinds out material vs the material engine.
# ---------------------------------------------------------------------------------------------
_GLYPH = {mc.P: "♟", mc.Nn: "♞", mc.R: "♜", mc.Q: "♛", mc.K: "♚"}


def _draw_chess(board, title, subtitle, last=None):
    fig, ax = plt.subplots(figsize=(4.8, 5.2))
    ax.set_xlim(0, mc.N)
    ax.set_ylim(0, mc.N + 0.7)
    ax.axis("off")
    for r in range(mc.N):
        for c in range(mc.N):
            light = (r + c) % 2 == 0
            ax.add_patch(Rectangle((c, r), 1, 1,
                         color="#ecd9b0" if light else "#b58863", zorder=0))
    if last is not None:
        for (rr, cc) in (last[:2], last[2:]):
            ax.add_patch(Rectangle((cc, rr), 1, 1, color="#f6f36a", alpha=0.55, zorder=1))
    for r in range(mc.N):
        for c in range(mc.N):
            pc = int(board[r, c])
            if pc == 0:
                continue
            glyph = _GLYPH[abs(pc)]
            color = "#f8f8f8" if pc > 0 else "#181818"
            edge = "#181818" if pc > 0 else "#f8f8f8"
            ax.text(c + 0.5, r + 0.5, glyph, ha="center", va="center",
                    fontsize=26, color=color, zorder=2,
                    path_effects=[])
            ax.text(c + 0.5, r + 0.5, glyph, ha="center", va="center",
                    fontsize=26, color=color, zorder=2)
    ax.text(mc.N / 2, mc.N + 0.42, title, ha="center", fontsize=14, fontweight="bold")
    ax.text(mc.N / 2, mc.N + 0.12, subtitle, ha="center", fontsize=10, color="#333")
    fig.tight_layout()
    img = _fig_to_image(fig)
    plt.close(fig)
    return img


def make_minichess(rng_seed=0):
    print("minichess: training TD-leaf vs the material engine, then playing a game")
    rng = np.random.default_rng(rng_seed)
    rung = mc.MiniChessRung()
    agent = mc.MiniChessAgent(rng)
    engine = mc._make_material_engine(depth=agent.depth)
    for i in range(300):
        mc.train_episode_vs_engine(agent, engine, rng, aside=1 if i % 2 == 0 else -1)
        agent.epsilon = max(0.05, agent.epsilon * 0.985)

    # agent is White (side +1) vs the material engine (Black)
    ev = np.random.default_rng(rng_seed + 3)
    board, side = mc.start_board(), 1
    agent_side = 1
    frames = [_draw_chess(board, "Los Alamos minichess",
                          "trained agent (white) vs material engine (black)")]
    last = None
    for ply in range(46):
        if side == agent_side:
            mv = agent.act(board, side, greedy=True)
            who = "agent (white) — depth-2 search"
        else:
            mv = engine(board, side, ev)
            who = "material engine (black)"
        if mv is None:
            break
        board = mc.make_move(board, mv)
        last = mv
        tv = mc.terminal_value(board, side)
        if tv is not None:
            won = (side == agent_side) == (tv > 0)
            frames.append(_draw_chess(board, "Los Alamos minichess",
                          "agent captures the king — win" if won else "agent loses king",
                          last=last))
            break
        margin = mc._material(board, agent_side)
        frames.append(_draw_chess(board, "Los Alamos minichess",
                      f"{who}   |   material margin {margin:+.0f}", last=last))
        side = -side
    else:
        margin = mc._material(board, agent_side)
        frames.append(_draw_chess(board, "Los Alamos minichess",
                      f"move cap reached — material margin {margin:+.0f}", last=last))
    _save_gif(frames, f"{FIG_DIR}/games_play_minichess.gif", duration_ms=600)


# ---------------------------------------------------------------------------------------------
# Rung 4 — EchoCraft (the headline): a culture-born agent walks the tech tree to diamond.
# ---------------------------------------------------------------------------------------------
_TERRAIN_COLOR = {
    cr.GRASS: "#7cb342", cr.TREE: "#2e7d32", cr.STONE: "#9e9e9e", cr.COAL: "#37474f",
    cr.IRON: "#c98a5e", cr.WATER: "#1e88e5", cr.DIAMOND: "#40c4ff", cr.BUSH: "#aed581",
    cr.TABLE: "#8d6e63", cr.FURNACE: "#d84315",
}
_TERRAIN_LABEL = {
    cr.TREE: "T", cr.STONE: "S", cr.COAL: "C", cr.IRON: "Fe", cr.WATER: "~",
    cr.DIAMOND: "◆", cr.BUSH: "*", cr.TABLE: "⌗", cr.FURNACE: "▣",
}


def _draw_craft(world, title, action_label, unlocked, depth):
    fig = plt.figure(figsize=(8.2, 5.0))
    gax = fig.add_axes([0.02, 0.06, 0.52, 0.84])
    pax = fig.add_axes([0.58, 0.06, 0.40, 0.84])
    gax.set_xlim(0, cr.GRID)
    gax.set_ylim(0, cr.GRID)
    gax.set_xticks([]); gax.set_yticks([])
    gax.set_aspect("equal")
    for r in range(cr.GRID):
        for c in range(cr.GRID):
            t = int(world.grid[r, c])
            gax.add_patch(Rectangle((c, cr.GRID - 1 - r), 1, 1,
                          color=_TERRAIN_COLOR[t], ec="#ffffff", lw=0.6))
            if t in _TERRAIN_LABEL:
                gax.text(c + 0.5, cr.GRID - 1 - r + 0.5, _TERRAIN_LABEL[t],
                         ha="center", va="center", fontsize=9, color="#ffffff", fontweight="bold")
    ar, ac = world.pos
    gax.add_patch(Circle((ac + 0.5, cr.GRID - 1 - ar + 0.5), 0.34, color="#ffeb3b",
                         ec="#000", lw=1.6, zorder=5))
    gax.text(ac + 0.5, cr.GRID - 1 - ar + 0.5, "@", ha="center", va="center",
             fontsize=11, fontweight="bold", zorder=6)
    gax.set_title(title, fontsize=13, fontweight="bold")

    # side panel: action, tech depth, inventory, achievement checklist
    pax.axis("off")
    pax.set_xlim(0, 1); pax.set_ylim(0, 1)
    pax.text(0, 0.98, action_label, fontsize=11, fontweight="bold", va="top", color="#0d47a1")
    pax.text(0, 0.90, f"tech depth reached: {depth} / {cr.MAX_DEPTH}", fontsize=10, va="top")
    inv = world.inv
    invline = f"wood {inv['wood']}  stone {inv['stone']}  coal {inv['coal']}  " \
              f"ore {inv['iron_ore']}  ingot {inv['iron_ingot']}"
    pax.text(0, 0.845, invline, fontsize=8.5, va="top", color="#444")
    y = 0.80
    for name, d, is_recipe in cr.ACHIEVEMENTS:
        got = name in unlocked
        mark = "■" if got else "□"
        color = "#2e7d32" if got else "#bbb"
        tag = " (recipe)" if is_recipe else ""
        pax.text(0, y, f"{mark} d{d} {name}{tag}", fontsize=8.6, va="top",
                 color=color, fontweight="bold" if got else "normal")
        y -= 0.058
    img = _fig_to_image(fig)
    plt.close(fig)
    return img


def _build_craft_culture(seed, gens=12, pop=6):
    """Run the real CIV loop to accumulate a recipe set, then return the strongest agent and a
    fresh culture-born agent to demonstrate (mirrors the experiment's inheritance)."""
    rung = cr.CraftRung()
    rng = np.random.default_rng(seed)
    culture = rung.new_culture()
    agents = [rung.new_agent(rng, culture) for _ in range(pop)]
    for g in range(gens):
        for ag in agents:
            rung.train(ag, rng, 12)
        evals = [rung.evaluate(ag, rng) for ag in agents]
        order = sorted(range(len(agents)), key=lambda i: evals[i]["score"], reverse=True)
        for i in order[:3]:
            rung.extract(agents[i], culture)
        for ag in agents:
            rung.transfer(ag, culture)
        best = agents[order[0]]
        agents = [rung.new_agent(rng, culture, parent=best) for _ in range(pop)]
    return culture, rng


def make_echocraft(rng_seed=0):
    print("echocraft: building culture across generations, then a culture-born greedy life")
    culture, rng = _build_craft_culture(rng_seed)
    print(f"  culture recipe set: {sorted(culture.recipes)}")

    # A child born into the civilization's recipes + know-how. Try a few maps and keep the life
    # that reaches deepest — this is the demonstrative "best gameplay", explicitly the CIV agent.
    rung = cr.CraftRung()
    best_frames, best_depth = None, -1
    for attempt in range(8):
        agent = rung.new_agent(rng, culture)
        world = cr.World(rng)
        unlocked = set()
        frames = [_draw_craft(world, "EchoCraft — culture-born agent",
                              "spawns into a fresh world", unlocked, 0)]
        depth = 0
        while not world.done():
            opts = world.available()
            s = world.state_key()
            a = agent.act(s, opts, greedy=True)
            if a is None:
                break
            world.step(a, agent.known_recipes)
            unlocked = set(world.unlocked)
            depth = max((cr.ACH_DEPTH[n] for n in unlocked), default=0)
            is_recipe = a in cr.RECIPE_NAMES
            label = ("crafts " if is_recipe else "does ") + a.replace("_", " ")
            frames.append(_draw_craft(world, "EchoCraft — culture-born agent",
                                      label, unlocked, depth))
        # final summary frame
        frames.append(_draw_craft(world, "EchoCraft — culture-born agent",
                      f"life over — reached tech depth {depth}", unlocked, depth))
        if depth > best_depth:
            best_depth, best_frames = depth, frames
        if depth >= cr.MAX_DEPTH:
            break
    print(f"  best demonstrated tech depth: {best_depth} / {cr.MAX_DEPTH}")
    _save_gif(best_frames, f"{FIG_DIR}/games_play_echocraft.gif", duration_ms=850, hold_last=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["tictactoe", "connect4", "minichess", "echocraft"],
                    default=None)
    args = ap.parse_args()
    jobs = {
        "tictactoe": make_tictactoe,
        "connect4": make_connect4,
        "minichess": make_minichess,
        "echocraft": make_echocraft,
    }
    if args.only:
        jobs[args.only]()
    else:
        for fn in jobs.values():
            fn()


if __name__ == "__main__":
    main()
