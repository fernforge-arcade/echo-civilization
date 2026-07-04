"""Rung 1 — Tic-Tac-Toe.

Deterministic, adversarial, ~10^5 game-tree states, *solved*. An agent learns by tabular
Q-learning through self-play. The cultural unit is a shared answer book: a mapping
board-state -> move that strong agents vote into, and that new agents warm-start from.

Capability is graded externally against a *perfect* minimax opponent (optimum = never lose,
i.e. draw) and against a random opponent. This is the easy rung: a lone agent can master it
given enough games, so culture's job here is only to make that faster — the control case for
the "culture matters more as environments get harder" thesis.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache

import numpy as np

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # cols
    (0, 4, 8), (2, 4, 6),              # diagonals
]


def winner(board):
    """Return 'X', 'O', 'D' (draw) or None (unfinished) for a 9-char board."""
    for a, b, c in WIN_LINES:
        if board[a] != "." and board[a] == board[b] == board[c]:
            return board[a]
    if "." not in board:
        return "D"
    return None


def legal_moves(board):
    return [i for i, c in enumerate(board) if c == "."]


def apply_move(board, i, mark):
    return board[:i] + mark + board[i + 1:]


@lru_cache(maxsize=None)
def minimax(board, mark):
    """Perfect play. Returns (value, best_move) from `mark`'s perspective.

    value: +1 win, 0 draw, -1 loss. Used both as the unbeatable evaluation opponent and,
    at reduced strength, as a curriculum sparring partner.
    """
    w = winner(board)
    if w == mark:
        return 1, None
    if w == "D":
        return 0, None
    if w is not None:
        return -1, None
    other = "O" if mark == "X" else "X"
    best_val, best_move = -2, None
    for i in legal_moves(board):
        v, _ = minimax(apply_move(board, i, mark), other)
        v = -v
        if v > best_val:
            best_val, best_move = v, i
    return best_val, best_move


def perfect_move(board, mark, rng, epsilon=0.0):
    """Minimax move, optionally epsilon-random (a weaker sparring opponent)."""
    if epsilon and rng.random() < epsilon:
        return int(rng.choice(legal_moves(board)))
    _, mv = minimax(board, mark)
    return mv


def random_move(board, mark, rng):
    return int(rng.choice(legal_moves(board)))


class TicTacToeAgent:
    """Tabular Q-learner over board strings, learning by self-play.

    Inherits an answer book (state->move) as a Q warm-start: inherited moves get an optimistic
    initial Q so the agent trusts culture first, then refines from its own experience.
    """

    def __init__(self, rng, alpha=0.3, gamma=0.95, epsilon=0.25):
        self.rng = rng
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.q = defaultdict(lambda: np.zeros(9))
        self.inherited = set()

    def act(self, board, mark, greedy=False):
        moves = legal_moves(board)
        key = (board, mark)
        if (not greedy) and self.rng.random() < self.epsilon:
            return int(self.rng.choice(moves))
        qv = self.q[key]
        masked = np.full(9, -np.inf)
        for m in moves:
            masked[m] = qv[m]
        best = float(np.max(masked))
        # break ties randomly for exploration diversity
        top = [m for m in moves if qv[m] >= best - 1e-9]
        return int(self.rng.choice(top))

    def update(self, key, action, reward, next_key, next_moves, done):
        if done or not next_moves:
            best_next = 0.0
        else:
            nq = self.q[next_key]
            best_next = max(nq[m] for m in next_moves)
        target = reward + self.gamma * best_next
        self.q[key][action] += self.alpha * (target - self.q[key][action])

    def answer_book(self, min_visits=None):
        """Export greedy policy for states the agent has an opinion on (nonzero Q)."""
        book = {}
        for key, qv in self.q.items():
            board, mark = key
            moves = legal_moves(board)
            if not moves or float(np.max(np.abs(qv))) < 1e-6:
                continue
            masked = np.full(9, -np.inf)
            for m in moves:
                masked[m] = qv[m]
            book[key] = int(np.argmax(masked))
        return book

    def inherit(self, book, optimistic=0.5):
        for key, move in book.items():
            self.q[key][move] = max(self.q[key][move], optimistic)
            self.inherited.add(key)


def self_play_episode(agent, rng, teacher_prob=0.35):
    """One training game. With probability `teacher_prob` the agent plays one side against an
    epsilon-corrupted minimax teacher (graded signal that anchors it to real play); otherwise
    it plays both sides against itself (explores the state space). Learning is Monte-Carlo
    control per side: each of a side's own (state,action) pairs is nudged toward the discounted
    terminal return. Terminal reward is +1 win / -1 loss / +0.5 draw — a draw is a *good*
    outcome, since a draw is the best attainable result against perfect play."""
    board, mark = "." * 9, "X"
    history = {"X": [], "O": []}
    use_teacher = rng.random() < teacher_prob
    teacher_mark = rng.choice(["X", "O"]) if use_teacher else None
    while True:
        if use_teacher and mark == teacher_mark:
            action = perfect_move(board, mark, rng, epsilon=0.4)
        else:
            action = agent.act(board, mark)
            history[mark].append(((board, mark), action))
        board = apply_move(board, action, mark)
        w = winner(board)
        if w is not None:
            for m in ("X", "O"):
                r = 0.5 if w == "D" else (1.0 if w == m else -1.0)
                g = r
                for key, act in reversed(history[m]):
                    cur = agent.q[key][act]
                    agent.q[key][act] = cur + agent.alpha * (g - cur)
                    g *= agent.gamma
            return w
        mark = "O" if mark == "X" else "X"


def evaluate(agent, rng, n_games=100):
    """Grade capability against perfect and random opponents.

    Against perfect play the best possible is a draw, so we report a *safety* score:
      loss_rate_vs_perfect (lower is better) and win_rate_vs_random (higher is better).
    Combined `score` in [0,1]: 0.5*(1-loss_vs_perfect) + 0.5*win_vs_random.
    """
    def play(opponent, agent_mark):
        board, mark = "." * 9, "X"
        while True:
            if mark == agent_mark:
                a = agent.act(board, mark, greedy=True)
            else:
                a = opponent(board, mark)
            board = apply_move(board, a, mark)
            w = winner(board)
            if w is not None:
                return w
            mark = "O" if mark == "X" else "X"

    losses_p = draws_p = 0
    wins_r = 0
    half = n_games // 2
    for g in range(n_games):
        am = "X" if g < half else "O"
        wp = play(lambda b, m: perfect_move(b, m, rng), am)
        if wp == "D":
            draws_p += 1
        elif wp != am:
            losses_p += 1
        wr = play(lambda b, m: random_move(b, m, rng), am)
        if wr == am:
            wins_r += 1
    loss_vs_perfect = losses_p / n_games
    win_vs_random = wins_r / n_games
    score = 0.5 * (1 - loss_vs_perfect) + 0.5 * win_vs_random
    return {
        "score": score,
        "loss_vs_perfect": loss_vs_perfect,
        "draw_vs_perfect": draws_p / n_games,
        "win_vs_random": win_vs_random,
    }


class TTTCulture:
    """Shared answer book. Agents vote (state,mark) -> move; the plurality vote becomes the
    civilization's recommended reply, weighted by how many agents agree."""

    def __init__(self):
        self.votes = {}  # key -> {move: weight}

    def contribute(self, book):
        for key, move in book.items():
            self.votes.setdefault(key, {})
            self.votes[key][move] = self.votes[key].get(move, 0.0) + 1.0

    def consensus(self):
        book = {}
        for key, moves in self.votes.items():
            book[key] = max(moves.items(), key=lambda kv: kv[1])[0]
        return book

    def size(self):
        return len(self.votes)


class TicTacToeRung:
    """Adapter binding Tic-Tac-Toe to the generational harness."""

    name = "tictactoe"
    complexity = 5  # log10 game-tree complexity (literature)

    def new_culture(self):
        return TTTCulture()

    def new_agent(self, rng, culture=None, parent=None):
        ag = TicTacToeAgent(rng)
        if culture is not None:
            ag.inherit(culture.consensus())
        return ag

    def train(self, agent, rng, episodes):
        for _ in range(episodes):
            self_play_episode(agent, rng)
            agent.epsilon = max(0.05, agent.epsilon * 0.9995)

    def evaluate(self, agent, rng):
        return evaluate(agent, rng, n_games=120)

    def extract(self, agent, culture):
        culture.contribute(agent.answer_book())

    def transfer(self, agent, culture):
        agent.inherit(culture.consensus())

    def culture_size(self, culture):
        return culture.size() if culture else 0
