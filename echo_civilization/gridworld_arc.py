"""Echo-ARC: a symbolic grid world and its perception layer.

This is the substrate for abstraction *invention* (see `abstraction.py`). Where the
string world (skills.py) worked over `str -> str`, this world works over small
integer grids (`numpy` arrays, colours 0-9, 0 = background) the way ARC does.

Two things live here:

  1. Perception. `perceive()` turns a raw grid into a symbolic world model:
     connected components (objects), their bounding boxes, sizes, colours and
     spatial relations. This is the "pixels -> objects -> relations" bridge the
     brief asks for -- not deep-learning vision, just a useful symbolic parse.

  2. A composable grid DSL. Every op is a total `grid -> grid` function (never
     raises, always returns a grid), so a *program* is a tuple of op names exactly
     like in the string world. That uniformity is deliberate: the same fragment
     mining, MDL scoring and staged search machinery works unchanged, and invented
     abstractions slot in as new ops.

Keeping ops non-parametric (no free integer arguments) keeps enumeration a clean
discrete search; expressive power instead comes from *composition* and, crucially,
from the abstractions the system invents on top of these primitives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_DIM = 20  # hard cap so tiling can't blow grids up unboundedly


# --------------------------------------------------------------------------- #
# Perception: pixels -> objects -> relations
# --------------------------------------------------------------------------- #
@dataclass
class GridObject:
    cells: list            # list[(r, c)]
    color: int
    def __post_init__(self):
        rs = [r for r, _ in self.cells]
        cs = [c for _, c in self.cells]
        self.top, self.left, self.bottom, self.right = min(rs), min(cs), max(rs), max(cs)
        self.size = len(self.cells)
        self.height = self.bottom - self.top + 1
        self.width = self.right - self.left + 1

    def touches_border(self, H, W) -> bool:
        return self.top == 0 or self.left == 0 or self.bottom == H - 1 or self.right == W - 1


def connected_components(grid: np.ndarray, same_color: bool = False):
    """4-connected components of non-background cells.

    same_color=False groups any adjacent non-zero cells (shape-level objects);
    same_color=True groups only adjacent cells of equal colour.
    """
    H, W = grid.shape
    seen = np.zeros((H, W), dtype=bool)
    objs = []
    for r in range(H):
        for c in range(W):
            if grid[r, c] == 0 or seen[r, c]:
                continue
            col = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and grid[ny, nx] != 0:
                        if same_color and int(grid[ny, nx]) != col:
                            continue
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            # dominant colour of the blob
            vals = [int(grid[y, x]) for y, x in cells]
            color = max(set(vals), key=vals.count)
            objs.append(GridObject(cells, color))
    return objs


def perceive(grid: np.ndarray) -> dict:
    """Full symbolic world model of a grid."""
    H, W = grid.shape
    objs = connected_components(grid)
    colors = sorted(set(int(v) for v in grid.flatten() if v != 0))
    return {
        "shape": (H, W),
        "objects": objs,
        "n_objects": len(objs),
        "colors": colors,
        "n_colors": len(colors),
        "symmetric_h": bool(np.array_equal(grid, np.fliplr(grid))),
        "symmetric_v": bool(np.array_equal(grid, np.flipud(grid))),
        "n_nonzero": int(np.count_nonzero(grid)),
    }


# --------------------------------------------------------------------------- #
# DSL: total grid -> grid primitives
# --------------------------------------------------------------------------- #
def _cap(g: np.ndarray) -> np.ndarray:
    return g if g.shape[0] <= MAX_DIM and g.shape[1] <= MAX_DIM else None


def _crop(g):
    ys, xs = np.nonzero(g)
    if len(ys) == 0:
        return g
    return g[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()


def _compress(g):
    """Delete all-zero rows and columns."""
    if g.size == 0:
        return g
    rows = np.any(g != 0, axis=1)
    cols = np.any(g != 0, axis=0)
    if not rows.any() or not cols.any():
        return g
    return g[np.ix_(rows, cols)].copy()


def _gravity_down(g):
    out = np.zeros_like(g)
    H = g.shape[0]
    for c in range(g.shape[1]):
        col = g[:, c]
        vals = col[col != 0]
        if len(vals):
            out[H - len(vals):, c] = vals
    return out


def _gravity_right(g):
    out = np.zeros_like(g)
    W = g.shape[1]
    for r in range(g.shape[0]):
        row = g[r, :]
        vals = row[row != 0]
        if len(vals):
            out[r, W - len(vals):] = vals
    return out


def _keep_component(g, largest: bool):
    objs = connected_components(g)
    if not objs:
        return g.copy()
    tgt = max(objs, key=lambda o: o.size) if largest else min(objs, key=lambda o: o.size)
    out = np.zeros_like(g)
    for y, x in tgt.cells:
        out[y, x] = g[y, x]
    return out


def _tile_h(g):
    out = np.hstack([g, g])
    return out if _cap(out) is not None else g


def _tile_v(g):
    out = np.vstack([g, g])
    return out if _cap(out) is not None else g


def _sym_h(g):
    """Overlay the left-right mirror (non-zero wins) -> horizontally symmetric."""
    m = np.fliplr(g)
    return np.where(g != 0, g, m)


def _sym_v(g):
    m = np.flipud(g)
    return np.where(g != 0, g, m)


def _color_cycle(g):
    """Cycle every non-zero colour 1..9 -> 2..9,1."""
    out = g.copy()
    nz = out != 0
    out[nz] = (out[nz] % 9) + 1
    return out


BASE_OPS = {
    "flip_h": lambda g: np.fliplr(g).copy(),
    "flip_v": lambda g: np.flipud(g).copy(),
    "rot90": lambda g: np.rot90(g, 1).copy(),
    "rot180": lambda g: np.rot90(g, 2).copy(),
    "rot270": lambda g: np.rot90(g, 3).copy(),
    "transpose": lambda g: g.T.copy(),
    "crop": _crop,
    "compress": _compress,
    "tile_h": _tile_h,
    "tile_v": _tile_v,
    "gravity_d": _gravity_down,
    "gravity_r": _gravity_right,
    "keep_largest": lambda g: _keep_component(g, True),
    "keep_smallest": lambda g: _keep_component(g, False),
    "sym_h": _sym_h,
    "sym_v": _sym_v,
    "color_cycle": _color_cycle,
}

BASE_OP_NAMES = list(BASE_OPS.keys())


def apply_op(op, g, library=None):
    """Apply one op by name. If it's an invented abstraction, look it up in library
    and run its expansion (a tuple of base/lower ops)."""
    if op in BASE_OPS:
        return BASE_OPS[op](g)
    if library is not None and op in library.ops:
        return run_program(library.ops[op].expansion, g, library)
    raise KeyError(f"unknown op {op!r}")


def run_program(program, g: np.ndarray, library=None) -> np.ndarray:
    out = g
    for op in program:
        out = apply_op(op, out, library)
        if out is None or out.size == 0:
            return out if out is not None else np.zeros((1, 1), dtype=g.dtype)
    return out


def grids_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.array_equal(a, b)
