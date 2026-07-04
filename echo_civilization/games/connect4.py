"""Rung 2 — Connect Four (reduced 6x5, win = 4 in a row).

Deterministic, adversarial, ~10^21 game-tree states — far too large for a full table, so the
agent learns a *linear evaluation* of positions and plays a 1-ply lookahead with it. Weights
are learned by TD(0) on afterstate values through self-play (terminal reward +/-1).

The cultural unit is the learned weight vector itself: strong agents contribute their weights
to a running civilization consensus, and new agents are born with those weights as a
warm-start. Inheriting a *tuned evaluation function* is worth much more here than the TTT
answer book was, because the position space is far too large to chart move-by-move.
"""

from __future__ import annotations

import numpy as np

ROWS, COLS, CONNECT = 5, 6, 4

# feature layout (from the side-to-move's perspective, sign-symmetric):
#  0 just_won            1 opp_can_win_reply
#  2 my_open3            3 opp_open3
#  4 my_open2            5 opp_open2
#  6 center_control      7 my_pieces-opp_pieces   8 bias
N_FEATURES = 9


def new_board():
    return np.zeros((ROWS, COLS), dtype=np.int8)


def legal_cols(board):
    return [c for c in range(COLS) if board[0, c] == 0]


def drop(board, col, player):
    """Return a new board with `player` dropped in `col` (assumes legal)."""
    b = board.copy()
    for r in range(ROWS - 1, -1, -1):
        if b[r, c := col] == 0:
            b[r, c] = player
            break
    return b


def _build_line_idx():
    """Flat cell indices for every length-CONNECT window: shape (n_lines, CONNECT)."""
    lines = []
    for r in range(ROWS):
        for c in range(COLS):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                cells = [(r + dr * k, c + dc * k) for k in range(CONNECT)]
                if all(0 <= rr < ROWS and 0 <= cc < COLS for rr, cc in cells):
                    lines.append([rr * COLS + cc for rr, cc in cells])
    return np.array(lines, dtype=np.intp)


LINE_IDX = _build_line_idx()
CENTER = COLS // 2


def _line_sums(board):
    return board.ravel()[LINE_IDX].sum(axis=1)


def winner(board):
    """Return 1, -1, 0 (draw, board full) or None (unfinished)."""
    s = _line_sums(board)
    if (s == CONNECT).any():
        return 1
    if (s == -CONNECT).any():
        return -1
    if (board[0] != 0).all():
        return 0
    return None


def features(board, player):
    """Feature vector of `board` from `player`'s perspective (after `player` just moved)."""
    opp = -player
    vals = board.ravel()[LINE_IDX]                 # (n_lines, CONNECT)
    line_sum = vals.sum(axis=1)
    just_won = 1.0 if (line_sum == CONNECT * player).any() else 0.0
    empties = (vals == 0).sum(axis=1)
    mine = (vals == player).sum(axis=1)
    theirs = (vals == opp).sum(axis=1)
    my3 = float(((mine == 3) & (empties == 1)).sum())
    op3 = float(((theirs == 3) & (empties == 1)).sum())
    my2 = float(((mine == 2) & (empties == 2)).sum())
    op2 = float(((theirs == 2) & (empties == 2)).sum())
    # opponent immediate-win reply (only when the game is still live)
    opp_can_win = 0.0
    if just_won == 0.0 and op3 > 0 and (board[0] == 0).any():
        for c in np.nonzero(board[0] == 0)[0]:
            if winner(drop(board, int(c), opp)) == opp:
                opp_can_win = 1.0
                break
    col = board[:, CENTER]
    center_ctrl = float((col == player).sum() - (col == opp).sum())
    material = float((board == player).sum() - (board == opp).sum())
    return np.array([just_won, opp_can_win, my3, op3, my2, op2,
                     center_ctrl, material, 1.0], dtype=float)


class Connect4Agent:
    """Linear afterstate evaluator learned by TD(0) self-play."""

    def __init__(self, rng, alpha=0.02, epsilon=0.2):
        self.rng = rng
        self.alpha, self.epsilon = alpha, epsilon
        self.w = rng.normal(0, 0.01, N_FEATURES)
        # sensible priors so early self-play isn't pure noise
        self.w[0] += 1.0    # winning is good
        self.w[1] -= 1.0    # letting opp win is bad
        self.w[2] += 0.3    # my threats good
        self.w[3] -= 0.3    # opp threats bad

    def value_after(self, board, player):
        return float(self.w @ features(board, player))

    def act(self, board, player, greedy=False):
        cols = legal_cols(board)
        if (not greedy) and self.rng.random() < self.epsilon:
            return int(self.rng.choice(cols))
        vals = [self.value_after(drop(board, c, player), player) for c in cols]
        best = max(vals)
        top = [cols[i] for i, v in enumerate(vals) if v >= best - 1e-9]
        return int(self.rng.choice(top))

    def td_update(self, feat, target):
        pred = float(self.w @ feat)
        self.w += self.alpha * (target - pred) * feat
        # keep weights bounded so a bad self-play run can't blow up
        np.clip(self.w, -5, 5, out=self.w)


def _heuristic_move(board, player, rng):
    """1-ply greedy opponent: win if possible, else block, else prefer center."""
    cols = legal_cols(board)
    for c in cols:
        if winner(drop(board, c, player)) == player:
            return c
    for c in cols:
        if winner(drop(board, c, -player)) == -player:
            return c
    order = sorted(cols, key=lambda c: abs(c - COLS // 2))
    return order[0]


def _random_move(board, player, rng):
    return int(rng.choice(legal_cols(board)))


def self_play_episode(agent, rng):
    """One self-play game with TD(0) afterstate backups per side. Terminal reward +/-1/0."""
    board, player = new_board(), 1
    # per side: list of (afterstate_feature) for that side's own moves
    afters = {1: [], -1: []}
    while True:
        col = agent.act(board, player)
        board = drop(board, col, player)
        afters[player].append(features(board, player))
        w = winner(board)
        if w is not None:
            for p in (1, -1):
                r = 1.0 if w == p else (-1.0 if w == -p else 0.0)
                feats = afters[p]
                for i, f in enumerate(feats):
                    if i + 1 < len(feats):
                        target = 0.9 * float(agent.w @ feats[i + 1])
                    else:
                        target = r
                    agent.td_update(f, target)
            return w
        player = -player
    # unreachable


def evaluate(agent, rng, n_games=60):
    """Win/draw rate vs random and vs the 1-ply heuristic, alternating who starts."""
    def play(opponent, agent_player):
        board, player = new_board(), 1
        while True:
            if player == agent_player:
                c = agent.act(board, player, greedy=True)
            else:
                c = opponent(board, player, rng)
            board = drop(board, c, player)
            w = winner(board)
            if w is not None:
                return w
            player = -player

    wr = wh = dr = dh = 0
    half = n_games // 2
    for g in range(n_games):
        ap = 1 if g < half else -1
        w = play(_random_move, ap)
        if w == ap:
            wr += 1
        elif w == 0:
            dr += 1
        w = play(_heuristic_move, ap)
        if w == ap:
            wh += 1
        elif w == 0:
            dh += 1
    win_random = wr / n_games
    win_heur = wh / n_games
    score = 0.4 * win_random + 0.6 * win_heur + 0.1 * (dh / n_games)
    return {"score": score, "win_vs_random": win_random,
            "win_vs_heuristic": win_heur, "draw_vs_heuristic": dh / n_games}


class Connect4Culture:
    """Consensus weight vector: a reputation-weighted average of contributed evaluations."""

    def __init__(self):
        self.sum = np.zeros(N_FEATURES)
        self.count = 0.0

    def contribute(self, w, reputation=1.0):
        self.sum += reputation * np.asarray(w)
        self.count += reputation

    def consensus(self):
        if self.count == 0:
            return None
        return self.sum / self.count

    def size(self):
        return int(self.count)


class Connect4Rung:
    name = "connect4"
    complexity = 21

    def new_culture(self):
        return Connect4Culture()

    def new_agent(self, rng, culture=None, parent=None):
        ag = Connect4Agent(rng)
        if culture is not None:
            c = culture.consensus()
            if c is not None:
                ag.w = 0.5 * ag.w + 0.5 * c   # born blending own priors with culture
        return ag

    def train(self, agent, rng, episodes):
        for _ in range(episodes):
            self_play_episode(agent, rng)
            agent.epsilon = max(0.05, agent.epsilon * 0.999)

    def evaluate(self, agent, rng):
        return evaluate(agent, rng, n_games=60)

    def extract(self, agent, culture):
        rep = max(0.1, self.evaluate(agent, rng=agent.rng)["score"])
        culture.contribute(agent.w, reputation=rep)

    def transfer(self, agent, culture):
        c = culture.consensus()
        if c is not None:
            agent.w = 0.7 * agent.w + 0.3 * c  # nudge toward consensus

    def culture_size(self, culture):
        return culture.size() if culture else 0
