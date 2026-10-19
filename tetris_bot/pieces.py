"""Tetromino definitions shared by the replica, the fast simulator and the web controller.

Coordinates: x grows to the right, y grows DOWNWARD (row 0 is the top of the matrix), which
matches screen coordinates.  Each piece lives in an SRS bounding box; a piece position is the
(x, y) of the top-left corner of that box plus a rotation state 0 / 1 (R) / 2 / 3 (L).
"""
import numpy as np

W = 10            # matrix width
VISIBLE = 20      # visible rows
HIDDEN = 4        # buffer rows above the visible matrix
ROWS = VISIBLE + HIDDEN
SPAWN_Y = HIDDEN - 2   # box y at spawn: spawn-orientation minos sit in the two rows above the skyline

PIECES = "IOTSZJL"
I, O, T, S, Z, J, L = range(7)
NONE = 7          # "no piece" marker (empty hold slot / unknown next piece)

_BOX = {I: 4, O: 2, T: 3, S: 3, Z: 3, J: 3, L: 3}
_SPAWN_CELLS = {
    I: [(0, 1), (1, 1), (2, 1), (3, 1)],
    O: [(0, 0), (1, 0), (0, 1), (1, 1)],
    T: [(1, 0), (0, 1), (1, 1), (2, 1)],
    S: [(1, 0), (2, 0), (0, 1), (1, 1)],
    Z: [(0, 0), (1, 0), (1, 1), (2, 1)],
    J: [(0, 0), (0, 1), (1, 1), (2, 1)],
    L: [(2, 0), (0, 1), (1, 1), (2, 1)],
}
SPAWN_X = {I: 3, O: 4, T: 3, S: 3, Z: 3, J: 3, L: 3}


def _rot_cw(cells, n):
    return sorted((n - 1 - y, x) for x, y in cells)


# SHAPES[piece][rot] -> list of 4 (dx, dy) offsets inside the bounding box
SHAPES = []
for _p in range(7):
    cells, states = sorted(_SPAWN_CELLS[_p]), []
    for _ in range(4):
        states.append(tuple(cells))
        cells = _rot_cw(cells, _BOX[_p])
    SHAPES.append(tuple(states))

# SRS wall kicks, written in the usual "y up" convention and converted to y-down below.
_KICKS_JLSTZ = {
    (0, 1): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
    (1, 0): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
    (1, 2): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
    (2, 1): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
    (2, 3): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
    (3, 2): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
    (3, 0): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
    (0, 3): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
}
_KICKS_I = {
    (0, 1): [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
    (1, 0): [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],
    (1, 2): [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)],
    (2, 1): [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],
    (2, 3): [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],
    (3, 2): [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
    (3, 0): [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],
    (0, 3): [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)],
}


def kicks(piece, r_from, r_to):
    """Kick offsets (dx, dy) in y-down coordinates to try, in order."""
    if piece == O:
        return [(0, 0)]
    table = _KICKS_I if piece == I else _KICKS_JLSTZ
    return [(dx, -dy) for dx, dy in table[(r_from, r_to)]]


# Rotation states worth considering for a straight drop (the others give identical results).
DISTINCT_ROTS = {I: (0, 1), O: (0,), T: (0, 1, 2, 3), S: (0, 1), Z: (0, 1), J: (0, 1, 2, 3), L: (0, 1, 2, 3)}

# Guideline colours, used by the mock renderer and as reference hues by the screen reader.
COLORS = {
    I: (0, 200, 230),    # cyan
    O: (230, 210, 0),    # yellow
    T: (190, 0, 235),    # purple
    S: (0, 200, 40),     # green
    Z: (225, 10, 20),    # red
    J: (0, 95, 215),     # blue
    L: (240, 140, 0),    # orange
}

# Score table from play.tetris.com (multiplied by the level in the real game).
LINE_SCORE = (0, 100, 300, 500, 800)
TSPIN_SCORE = (400, 800, 1200, 1600)


class PlacementTable:
    """All straight-drop placements (rotation, box x) of one piece, as index arrays."""

    def __init__(self, piece):
        rot, bx, cols, dys = [], [], [], []
        for r in DISTINCT_ROTS[piece]:
            cells = SHAPES[piece][r]
            xs = [c[0] for c in cells]
            for x in range(-min(xs), W - max(xs)):
                rot.append(r)
                bx.append(x)
                cols.append([x + c[0] for c in cells])
                dys.append([c[1] for c in cells])
        self.rot = np.array(rot)
        self.bx = np.array(bx)
        self.cols = np.array(cols)
        self.dys = np.array(dys)
        self.n = len(rot)
        self.arange = np.arange(self.n)[:, None]
        # index of the placement that corresponds to "rotated in place at the spawn column"
        self.spawn_idx = {r: int(np.nonzero((self.rot == r) & (self.bx == SPAWN_X[piece]))[0][0])
                          for r in DISTINCT_ROTS[piece]}


TABLES = [PlacementTable(p) for p in range(7)]
