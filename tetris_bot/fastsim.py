"""Fast placement-level Tetris simulator used for reinforcement learning.

Instead of single key presses, one "action" here is a whole placement: optionally press hold,
rotate, shift sideways, hard drop.  `expand()` returns every such placement for a position in
one vectorised numpy pass, together with the resulting boards, the reward and a feature vector
describing each resulting position ("afterstate").  That is what the value network scores.
"""
import numpy as np
from .pieces import (W, ROWS, HIDDEN, SPAWN_X, SPAWN_Y, SHAPES, TABLES, DISTINCT_ROTS, NONE,
                     LINE_SCORE)
from .engine import Bag, PREVIEW

REWARD_SCALE = 1.0 / 1000.0       # 1 reward unit = 1000 points at level 1
N_BOARD_FEATS = 22
N_FEATS = N_BOARD_FEATS + 8 + 7   # + hold one-hot (7 pieces + empty) + upcoming piece one-hot

_NORM = np.array([20.0] * 10 + [10, 20, 20, 20, 40, 40, 20, 20, 40, 10, 10, 1], np.float32)
_ROWIDX = np.arange(ROWS)[None, :]
_LINE_REWARD = np.array(LINE_SCORE, np.float64) * REWARD_SCALE
_SPAWN_CELLS = [(np.array([SPAWN_Y + dy for dx, dy in SHAPES[p][0]]),
                 np.array([SPAWN_X[p] + dx for dx, dy in SHAPES[p][0]])) for p in range(7)]


class State:
    """Everything the player can see: matrix, falling piece, hold slot, preview queue, B2B flag."""
    __slots__ = ("board", "cur", "hold", "queue", "b2b")

    def __init__(self, board, cur, hold=NONE, queue=(), b2b=False):
        self.board = board          # (ROWS, W) bool, True = filled, row 0 = top
        self.cur = cur
        self.hold = hold
        self.queue = tuple(queue)
        self.b2b = b2b


def column_tops(board):
    """y of the highest filled cell in each column (ROWS for an empty column)."""
    return np.where(board.any(0), board.argmax(0), ROWS)


def board_features(boards, b2b, height_bias=0.0):
    """Hand-picked descriptors of a batch of boards; `boards` is (N, ROWS, W) bool."""
    n = boards.shape[0]
    f = np.empty((n, N_BOARD_FEATS), np.float32)
    h = np.where(boards.any(1), ROWS - boards.argmax(1), 0)                 # column heights
    f[:, 0:10] = h
    f[:, 10] = (h - boards.sum(1)).sum(1)                                   # holes
    f[:, 11] = np.abs(np.diff(h, axis=1)).sum(1)                            # bumpiness
    f[:, 12] = h.max(1)
    f[:, 13] = h.mean(1)
    rowsum = boards.sum(2)
    f[:, 14] = ((boards[:, :, 1:] != boards[:, :, :-1]).sum((1, 2))         # row transitions
                + (~boards[:, :, 0]).sum(1) + (~boards[:, :, -1]).sum(1) - 2 * (rowsum == 0).sum(1))
    f[:, 15] = ((boards[:, 1:, :] != boards[:, :-1, :]).sum((1, 2))         # column transitions
                + (~boards[:, -1, :]).sum(1))
    left = np.concatenate([h[:, 1:2], h[:, :-1]], axis=1)
    right = np.concatenate([h[:, 1:], h[:, -2:-1]], axis=1)
    well = np.maximum(0, np.minimum(left, right) - h)
    f[:, 16] = well.sum(1)                                                  # total well depth
    f[:, 17] = well.max(1)                                                  # deepest well
    above = np.cumsum(boards, axis=1)
    holemask = (~boards) & (above > 0)
    f[:, 18] = (above * holemask).sum((1, 2))                               # blocks stacked on holes
    f[:, 19] = holemask.any(2).sum(1)                                       # rows containing holes
    f[:, 20] = (rowsum == W - 1).sum(1)                                     # rows one cell short
    f[:, 21] = b2b
    if height_bias:                 # pretend the stack is taller: makes the agent play it safe
        f[:, 0:10] += height_bias
        f[:, 12:14] += height_bias
    f /= _NORM
    return f


def _reachable(tb, piece, by, ceil_y, start_x):
    """Placements whose rotate-then-slide path at height `ceil_y` is not blocked by the stack."""
    ok = by >= ceil_y
    valid = np.zeros(tb.n, bool)
    for r in DISTINCT_ROTS[piece]:
        hit = np.nonzero((tb.rot == r) & (tb.bx == start_x))[0]
        if len(hit) == 0 or not ok[hit[0]]:
            continue
        if r == 2:        # reached with two clockwise turns, so the in-between state must fit too
            mid = np.nonzero((tb.rot == 1) & (tb.bx == start_x))[0]
            if len(mid) == 0 or not ok[mid[0]]:
                continue
        s = int(hit[0])
        i = s
        while i >= 0 and tb.rot[i] == r and ok[i]:
            valid[i] = True
            i -= 1
        i = s + 1
        while i < tb.n and tb.rot[i] == r and ok[i]:
            valid[i] = True
            i += 1
    return np.nonzero(valid)[0]


def spawn_row(tops, piece):
    """Box y of a freshly spawned piece: it appears above the skyline and drops one row at once
    if there is room (same rule as the engine)."""
    cells = SHAPES[piece][0]
    rest = min(tops[SPAWN_X[piece] + dx] - dy for dx, dy in cells) - 1
    if rest < SPAWN_Y:
        return None                      # blocked: spawning this piece would end the game
    return SPAWN_Y + 1 if rest >= SPAWN_Y + 1 else SPAWN_Y


def place_all(board, tops, piece, ceil_y, start_x=None):
    """Drop `piece` in every reachable (rotation, column). Returns rot, bx, by, boards, lines, lockout."""
    tb = TABLES[piece]
    by = (tops[tb.cols] - tb.dys).min(1) - 1                 # box y where the piece comes to rest
    cols, dys, rot, bx = tb.cols, tb.dys, tb.rot, tb.bx
    if start_x is not None or tops.min() < ceil_y + 4:
        idx = _reachable(tb, piece, by, ceil_y, SPAWN_X[piece] if start_x is None else start_x)
        by, cols, dys, rot, bx = by[idx], cols[idx], dys[idx], rot[idx], bx[idx]
    n = len(by)
    ys = by[:, None] + dys
    boards = np.broadcast_to(board, (n, ROWS, W)).copy()
    boards[np.arange(n)[:, None], ys, cols] = True
    full = boards.all(2)
    lines = full.sum(1)
    hit = np.nonzero(lines)[0]
    if len(hit):
        order = np.argsort(~full[hit], axis=1, kind="stable")      # full rows float to the top ...
        sub = np.take_along_axis(boards[hit], order[:, :, None], axis=1)
        sub[_ROWIDX < lines[hit, None]] = False                    # ... and are emptied
        boards[hit] = sub
    lockout = (ys < HIDDEN).all(1)
    return rot, bx, by, boards, lines, lockout


class Candidates:
    """All placements available in a State, as parallel arrays (one entry per placement)."""
    __slots__ = ("n", "use_hold", "piece", "rot", "bx", "by", "drop", "boards", "lines", "reward",
                 "dead", "b2b", "X", "_state", "_opt", "_children")

    def child(self, i):
        """The State that results from playing placement i."""
        cur, hold, queue = self._children[self._opt[i]]
        return State(self.boards[i], cur, hold, queue, bool(self.b2b[i]))


def expand(st, ceil_y=None, start_x=None, allow_hold=True, hold_margin=0, height_bias=0.0):
    """Enumerate every placement of the current piece and of the piece reachable through hold.

    ceil_y / start_x describe where the falling piece is right now (default: where it spawns);
    sideways moves and rotations are assumed to happen at that height.  `hold_margin` lowers the
    assumed height for the piece that comes out of hold (used in live play, where pieces keep
    falling while keys are pressed).
    """
    tops = column_tops(st.board)
    q = st.queue
    if ceil_y is None:
        ceil_y = spawn_row(tops, st.cur)
        if ceil_y is None:
            ceil_y = SPAWN_Y
    options = [(False, st.cur, ceil_y, start_x, (q[0] if q else NONE, st.hold, q[1:]))]
    if allow_hold:
        alt, child = NONE, None
        if st.hold != NONE:
            if st.hold != st.cur:
                alt, child = st.hold, (q[0] if q else NONE, st.cur, q[1:])
        elif q and q[0] != st.cur:
            alt, child = q[0], (q[1] if len(q) > 1 else NONE, st.cur, q[2:])
        if alt != NONE and spawn_row(tops, alt) is not None:
            options.append((True, alt, spawn_row(tops, alt) + hold_margin, None, child))
    parts = []
    for k, (use_hold, piece, cy, sx, child) in enumerate(options):
        rot, bx, by, boards, lines, lockout = place_all(st.board, tops, piece, cy, sx)
        n = len(rot)
        if n == 0:
            continue
        dead = lockout.copy()
        nxt = child[0]
        if nxt != NONE:                                           # block out: next piece cannot spawn
            sy, sxs = _SPAWN_CELLS[nxt]
            dead |= boards[:, sy, sxs].any(1)
        tail = np.zeros((n, 15), np.float32)
        tail[:, child[1]] = 1.0                                   # hold slot after the move
        if nxt != NONE:
            tail[:, 8 + nxt] = 1.0                                # piece that falls next
        parts.append((np.full(n, use_hold), np.full(n, piece), rot, bx, by, by - cy, boards, lines, dead,
                      np.full(n, k), tail))
    c = Candidates()
    c._state, c._children = st, [o[4] for o in options]
    if not parts:
        c.n = 0
        return c
    cat = [np.concatenate(x) for x in zip(*parts)]
    c.use_hold, c.piece, c.rot, c.bx, c.by, c.drop, c.boards, c.lines, c.dead, c._opt, tail = cat
    c.n = len(c.rot)
    c.b2b = np.where(c.lines == 4, True, np.where(c.lines > 0, False, st.b2b))
    c.reward = _LINE_REWARD[c.lines] * np.where((c.lines == 4) & st.b2b, 1.5, 1.0)
    c.X = np.concatenate([board_features(c.boards, c.b2b, height_bias), tail], axis=1)
    return c


class TetrisSim:
    """A full game on top of `expand`: keeps the piece queue, score, level and statistics."""

    def __init__(self, seed=None):
        self.bag = Bag(seed)
        queue = [self.bag.next() for _ in range(PREVIEW + 1)]
        self.state = State(np.zeros((ROWS, W), bool), queue[0], NONE, queue[1:], False)
        self.score = 0
        self.lines = 0
        self.level = 1
        self.pieces = 0
        self.clear_count = [0] * 5
        self.over = False

    def candidates(self):
        return expand(self.state)

    def step(self, cands, i):
        """Play placement i of `cands` (which must come from self.candidates())."""
        n = int(cands.lines[i])
        pts = LINE_SCORE[n] * (3 if (n == 4 and self.state.b2b) else 2) // 2
        self.score += pts * self.level + 2 * int(cands.drop[i])
        self.lines += n
        self.clear_count[n] += 1
        self.level = 1 + self.lines // 10
        self.pieces += 1
        st = cands.child(i)
        queue = list(st.queue)
        if st.cur == NONE:
            st.cur = self.bag.next()
        while len(queue) < PREVIEW:
            queue.append(self.bag.next())
        st.queue = tuple(queue)
        self.state = st
        if cands.dead[i]:
            self.over = True
