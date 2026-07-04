"""Rung 3 — Los Alamos minichess (6x6, no bishops).

This is the historical first chess-like program a computer ever played (MANIAC I, 1956): a 6x6
board with king, queen, two rooks, two knights and six pawns per side, no bishops, no castling,
no two-square pawn advance, no en-passant. Game-tree complexity is astronomically smaller than
full chess but still far beyond tabular methods, which makes it the right "chess" rung to test
whether an *inherited evaluation function* accumulates strength.

The agent plays a depth-1 alpha-beta search using a linear evaluation (material + piece-square
+ mobility + king safety), and learns the evaluation weights by TD-leaf self-play. The cultural
unit is the weight vector — a shared, evolving sense of what a good position looks like.

No pretrained anything; the eval starts near-random (only a rough material prior) and self-play
plus culture do the rest.
"""

from __future__ import annotations

import numpy as np

N = 6
EMPTY = 0
# piece codes: positive = white, negative = black
P, Nn, R, Q, K = 1, 2, 3, 4, 5  # pawn, knight, rook, queen, king
PIECE_VALUE = {P: 1.0, Nn: 3.0, R: 5.0, Q: 9.0, K: 0.0}
NAMES = {P: "P", Nn: "N", R: "R", Q: "Q", K: "K"}


def start_board():
    b = np.zeros((N, N), dtype=np.int8)
    back = [R, Nn, Q, K, Nn, R]
    for c in range(N):
        b[0, c] = back[c]          # white back rank (row 0)
        b[1, c] = P                # white pawns
        b[N - 2, c] = -P           # black pawns
        b[N - 1, c] = -back[c]     # black back rank
    return b


def in_bounds(r, c):
    return 0 <= r < N and 0 <= c < N


KNIGHT_D = [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]
ROOK_D = [(1, 0), (-1, 0), (0, 1), (0, -1)]
QUEEN_D = ROOK_D + [(1, 1), (1, -1), (-1, 1), (-1, -1)]


def gen_moves(board, side):
    """All pseudo-legal moves for `side` (+1 white / -1 black). A move is (r0,c0,r1,c1).

    Pawns move toward the far rank (white up in row index, black down), capture diagonally,
    and promote to queen on the last rank. No castling / en-passant / double-step (Los Alamos
    rules). King-capture legality is handled by search (a side that loses its king has lost)."""
    moves = []
    for r in range(N):
        for c in range(N):
            pc = board[r, c]
            if pc == 0 or np.sign(pc) != side:
                continue
            t = abs(pc)
            if t == P:
                fwd = side
                # forward push
                if in_bounds(r + fwd, c) and board[r + fwd, c] == 0:
                    moves.append((r, c, r + fwd, c))
                # captures
                for dc in (-1, 1):
                    rr, cc = r + fwd, c + dc
                    if in_bounds(rr, cc) and board[rr, cc] != 0 and np.sign(board[rr, cc]) != side:
                        moves.append((r, c, rr, cc))
            elif t == Nn:
                for dr, dc in KNIGHT_D:
                    rr, cc = r + dr, c + dc
                    if in_bounds(rr, cc) and (board[rr, cc] == 0 or np.sign(board[rr, cc]) != side):
                        moves.append((r, c, rr, cc))
            elif t == K:
                for dr, dc in QUEEN_D:
                    rr, cc = r + dr, c + dc
                    if in_bounds(rr, cc) and (board[rr, cc] == 0 or np.sign(board[rr, cc]) != side):
                        moves.append((r, c, rr, cc))
            else:  # sliding: rook or queen
                dirs = ROOK_D if t == R else QUEEN_D
                for dr, dc in dirs:
                    rr, cc = r + dr, c + dc
                    while in_bounds(rr, cc):
                        if board[rr, cc] == 0:
                            moves.append((r, c, rr, cc))
                        else:
                            if np.sign(board[rr, cc]) != side:
                                moves.append((r, c, rr, cc))
                            break
                        rr, cc = rr + dr, cc + dc
    return moves


def make_move(board, mv):
    r0, c0, r1, c1 = mv
    b = board.copy()
    pc = b[r0, c0]
    b[r0, c0] = 0
    # promotion
    if abs(pc) == P and (r1 == N - 1 or r1 == 0):
        pc = Q * np.sign(pc)
    b[r1, c1] = pc
    return b


def king_alive(board, side):
    return (board == K * side).any()


def terminal_value(board, side):
    """Return +/-1 if a king is missing (from `side`'s perspective), else None."""
    if not king_alive(board, -side):
        return 1.0
    if not king_alive(board, side):
        return -1.0
    return None


# --- linear evaluation -------------------------------------------------------
# feature vector (from `side`'s perspective, sign-symmetric):
#   material diff for P,N,R,Q  (4) + mobility diff (1) + center-control diff (1)
#   + king-advancement/back-rank safety (1) + pawn-advance diff (1) + bias (1)
N_FEATURES = 9
_CENTER = np.zeros((N, N))
for _r in range(N):
    for _c in range(N):
        _CENTER[_r, _c] = 1.0 - (abs(_r - 2.5) + abs(_c - 2.5)) / 6.0  # peak in the middle


def features(board, side):
    f = np.zeros(N_FEATURES)
    for i, t in enumerate((P, Nn, R, Q)):
        f[i] = ((board == t * side).sum() - (board == -t * side).sum())
    # piece development: minor/major pieces off their starting back rank (cheap, no move-gen).
    # White back rank is row 0, black back rank is row N-1.
    my_back, th_back = (0, N - 1) if side > 0 else (N - 1, 0)
    heavy = np.isin(np.abs(board), [Nn, R, Q])
    dev_me = int((heavy & (board * side > 0) & (np.arange(N)[:, None] != my_back)).sum())
    dev_th = int((heavy & (board * side < 0) & (np.arange(N)[:, None] != th_back)).sum())
    f[4] = 0.2 * (dev_me - dev_th)
    # center control by piece occupancy
    f[5] = float((_CENTER * (board * side > 0)).sum() - (_CENTER * (board * side < 0)).sum())
    # king safety: prefer own king on its back two ranks
    kmine = np.argwhere(board == K * side)
    ktheir = np.argwhere(board == K * -side)
    safe_me = 1.0 if len(kmine) and (kmine[0][0] <= 1 if side > 0 else kmine[0][0] >= N - 2) else 0.0
    safe_th = 1.0 if len(ktheir) and (ktheir[0][0] <= 1 if -side > 0 else ktheir[0][0] >= N - 2) else 0.0
    f[6] = safe_me - safe_th
    # pawn advancement (rows advanced toward promotion)
    my_p = np.argwhere(board == P * side)
    th_p = np.argwhere(board == P * -side)
    adv_me = sum((rp[0] if side > 0 else N - 1 - rp[0]) for rp in my_p)
    adv_th = sum((rp[0] if -side > 0 else N - 1 - rp[0]) for rp in th_p)
    f[7] = 0.1 * (adv_me - adv_th)
    f[8] = 1.0
    return f


# material weights are FROZEN at this prior; only the positional features (indices 4..8) learn.
# Self-play material diff averages ~0, so TD would otherwise erode [1,3,5,9] toward noise and the
# agent would start hanging pieces. Freezing material lets culture accumulate *positional* skill,
# which is the thing worth inheriting here.
MATERIAL_PRIOR = np.array([1.0, 3.0, 5.0, 9.0])
_LEARN_MASK = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=float)


class MiniChessAgent:
    def __init__(self, rng, alpha=0.02, epsilon=0.2, depth=1):
        self.rng = rng
        self.alpha, self.epsilon, self.depth = alpha, epsilon, depth
        self.w = rng.normal(0, 0.05, N_FEATURES)
        self.w[0:4] = MATERIAL_PRIOR
        self.w[8] = 0.0

    def evaluate(self, board, side):
        return float(self.w @ features(board, side))

    def search(self, board, side, depth, alpha, beta):
        """Negamax alpha-beta returning (value, best_move) from `side`'s view."""
        tv = terminal_value(board, side)
        if tv is not None:
            return tv * 100.0, None
        if depth == 0:
            return self.evaluate(board, side), None
        moves = gen_moves(board, side)
        if not moves:
            return self.evaluate(board, side), None
        # capture-first ordering sharpens alpha-beta pruning (big speedup at depth>=2)
        moves.sort(key=lambda m: -PIECE_VALUE.get(abs(int(board[m[2], m[3]])), 0.0))
        best_val, best_mv = -1e9, None
        for mv in moves:
            child = make_move(board, mv)
            val, _ = self.search(child, -side, depth - 1, -beta, -alpha)
            val = -val
            if val > best_val:
                best_val, best_mv = val, mv
            alpha = max(alpha, val)
            if alpha >= beta:
                break
        return best_val, best_mv

    def act(self, board, side, greedy=False):
        moves = gen_moves(board, side)
        if not moves:
            return None
        if (not greedy) and self.rng.random() < self.epsilon:
            return moves[int(self.rng.integers(len(moves)))]
        _, mv = self.search(board, side, self.depth, -1e9, 1e9)
        return mv if mv is not None else moves[0]

    def td_update(self, feat, target):
        pred = float(self.w @ feat)
        self.w += self.alpha * (target - pred) * feat * _LEARN_MASK  # material frozen
        self.w[0:4] = MATERIAL_PRIOR
        np.clip(self.w, -20, 20, out=self.w)


def _material_move(board, side, rng):
    """Baseline opponent: greedy 1-ply on pure material (captures the most valuable piece)."""
    best, bv = None, -1e9
    for mv in gen_moves(board, side):
        tgt = board[mv[2], mv[3]]
        v = PIECE_VALUE.get(abs(int(tgt)), 0.0) if tgt != 0 else 0.0
        # tiny center bonus to break ties toward development
        v += 0.01 * _CENTER[mv[2], mv[3]]
        if not king_alive(make_move(board, mv), -side):
            v += 1000  # take the king if you can
        if v > bv:
            bv, best = v, mv
    return best


def _random_move(board, side, rng):
    moves = gen_moves(board, side)
    return moves[int(rng.integers(len(moves)))] if moves else None


def _material(board, side):
    """Material sum from `side`'s view (own pieces minus opponent's, standard values)."""
    tot = 0.0
    for t, v in PIECE_VALUE.items():
        tot += v * ((board == t * side).sum() - (board == -t * side).sum())
    return tot


def _reward(board, side):
    """Continuous outcome in [-1,1]: king capture is decisive, else the material margin.

    A win/draw/loss signal alone is too sparse here — at shallow search two sound players just
    trade down to a draw. Material margin gives a dense gradient, so learned positional weights
    are rewarded for the small edges that eventually win material."""
    if not king_alive(board, -side):
        return 1.0
    if not king_alive(board, side):
        return -1.0
    return float(np.tanh(_material(board, side) / 3.0))


def self_play_episode(agent, rng, max_plies=48):
    board, side = start_board(), 1
    afters = {1: [], -1: []}
    for _ply in range(max_plies):
        mv = agent.act(board, side)
        if mv is None:
            break
        board = make_move(board, mv)
        afters[side].append(features(board, side))
        tv = terminal_value(board, side)
        if tv is not None:
            _backup(agent, afters, {side: 1.0, -side: -1.0})
            return side if tv > 0 else -side
        side = -side
    # ran to the ply cap: reward each side by its final material margin
    _backup(agent, afters, {1: _reward(board, 1), -1: _reward(board, -1)})
    return 0


def _backup(agent, afters, final_reward):
    for p in (1, -1):
        feats = afters[p]
        for i, f in enumerate(feats):
            target = 0.9 * float(agent.w @ feats[i + 1]) if i + 1 < len(feats) else final_reward[p]
            agent.td_update(f, target)


def _make_material_engine(depth=2):
    """A fixed opponent with real search but a purely material eval (no positional knowledge).

    Because it searches to the same depth as the learner, it never hangs pieces — so beating it
    requires genuine positional understanding, which is exactly what self-play + culture supply.
    This gives the rung real headroom above the material prior."""
    eng = MiniChessAgent(np.random.default_rng(12345), epsilon=0.0, depth=depth)
    eng.w = np.zeros(N_FEATURES)
    eng.w[0:4] = MATERIAL_PRIOR

    def move(board, side, rng):
        return eng.act(board, side, greedy=True)
    return move


def train_episode_vs_engine(agent, engine_move, rng, aside, max_plies=40):
    """One training game: the learner (playing `aside`) faces the fixed material engine, which is
    a stationary, tactically-sound target — so TD-leaf learns *positional* weights that convert
    into material, rather than chasing a moving self-play target."""
    board, side = start_board(), 1
    afters = []
    for _ in range(max_plies):
        if side == aside:
            moves = gen_moves(board, side)
            if not moves:
                break
            if rng.random() < agent.epsilon:
                mv = moves[int(rng.integers(len(moves)))]
            else:
                _, mv = agent.search(board, side, agent.depth, -1e9, 1e9)
                mv = mv if mv is not None else moves[0]
            board = make_move(board, mv)
            afters.append(features(board, aside))
        else:
            mv = engine_move(board, side, rng)
            if mv is None:
                break
            board = make_move(board, mv)
        tv = terminal_value(board, side)
        if tv is not None:
            r = 1.0 if (side == aside) == (tv > 0) else -1.0
            _backup_side(agent, afters, r)
            return r
        side = -side
    r = _reward(board, aside)
    _backup_side(agent, afters, r)
    return r


def _backup_side(agent, afters, final_reward):
    for i, f in enumerate(afters):
        target = 0.9 * float(agent.w @ afters[i + 1]) if i + 1 < len(afters) else final_reward
        agent.td_update(f, target)


def evaluate(agent, rng, n_games=16, max_plies=48):
    """Primary score = average outcome vs a depth-matched *material engine*, where outcome is the
    continuous material margin (king capture = decisive). The material prior alone draws the
    engine (~0.5); learned positional weights are what push the margin positive, so this is where
    culture shows up. `win_vs_engine` is the fraction of games with a positive final margin."""
    engine = _make_material_engine(depth=agent.depth)

    def play(opponent, agent_side):
        board, side = start_board(), 1
        for _ in range(max_plies):
            mv = agent.act(board, side, greedy=True) if side == agent_side else opponent(board, side, rng)
            if mv is None:
                break
            board = make_move(board, mv)
            tv = terminal_value(board, side)
            if tv is not None:
                return 1.0 if (side == agent_side) == (tv > 0) else -1.0
            side = -side
        return _reward(board, agent_side)

    margins, wins = [], 0
    wr = wm = 0
    half = n_games // 2
    n_sanity = max(4, n_games // 2)
    for g in range(n_games):
        aside = 1 if g < half else -1
        r = play(engine, aside)
        margins.append(r)
        wins += 1 if r > 0.02 else 0
        if g < n_sanity:
            if play(_random_move, aside) > 0.02:
                wr += 1
            if play(_material_move, aside) > 0.02:
                wm += 1
    avg = float(np.mean(margins))
    return {
        "score": 0.5 + 0.5 * avg,          # in [0,1], 0.5 = even with the engine
        "win_vs_engine": wins / n_games,
        "margin_vs_engine": avg,
        "win_vs_random": wr / n_sanity,
        "win_vs_material": wm / n_sanity,
    }


class MiniChessCulture:
    def __init__(self):
        self.sum = np.zeros(N_FEATURES)
        self.count = 0.0

    def contribute(self, w, reputation=1.0):
        self.sum += reputation * np.asarray(w)
        self.count += reputation

    def consensus(self):
        return None if self.count == 0 else self.sum / self.count

    def size(self):
        return int(self.count)


class MiniChessRung:
    name = "minichess"
    complexity = 60  # between Connect Four (21) and full chess (123) on the log10 scale

    def new_culture(self):
        return MiniChessCulture()

    def new_agent(self, rng, culture=None, parent=None):
        ag = MiniChessAgent(rng)
        if culture is not None:
            c = culture.consensus()
            if c is not None:
                ag.w = 0.5 * ag.w + 0.5 * c
        return ag

    def train(self, agent, rng, episodes):
        engine = _make_material_engine(depth=agent.depth)
        for i in range(episodes):
            train_episode_vs_engine(agent, engine, rng, aside=1 if i % 2 == 0 else -1)
            agent.epsilon = max(0.05, agent.epsilon * 0.985)

    def evaluate(self, agent, rng):
        return evaluate(agent, rng, n_games=10)

    def extract(self, agent, culture):
        rep = max(0.1, self.evaluate(agent, agent.rng)["score"])
        culture.contribute(agent.w, reputation=rep)

    def transfer(self, agent, culture):
        c = culture.consensus()
        if c is not None:
            agent.w = 0.7 * agent.w + 0.3 * c

    def culture_size(self, culture):
        return culture.size() if culture else 0
