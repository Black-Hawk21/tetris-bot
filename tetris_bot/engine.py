"""Key-level replica of the play.tetris.com marathon game.

This is the "real" game: it is driven by the same keys as the website (left / right / rotate /
soft drop / hard drop / hold), has gravity, lock delay, SRS wall kicks, a 7-bag randomiser,
a 3-piece preview, hold, T-spin detection, back-to-back and the score table from the site.

The RL agent does not train on this class directly (it is too slow to enumerate moves key by
key); it trains on `fastsim`, and `tests/test_consistency.py` checks that both agree.
"""
import random
import numpy as np
from .pieces import (W, ROWS, HIDDEN, SPAWN_X, SPAWN_Y, SHAPES, NONE, T, kicks,
                     LINE_SCORE, TSPIN_SCORE)

PREVIEW = 3
LOCK_DELAY = 0.5


class Bag:
    """7-bag randomiser: every 7 pieces contain each tetromino exactly once."""

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.buf = []

    def next(self):
        if not self.buf:
            self.buf = list(range(7))
            self.rng.shuffle(self.buf)
        return self.buf.pop()


def gravity_seconds(level):
    """Seconds per row at a given level (Tetris guideline curve)."""
    level = min(level, 20)
    return (0.8 - (level - 1) * 0.007) ** (level - 1)


class Engine:
    def __init__(self, seed=None, spawn_y=None):
        self.spawn_y = spawn_y                       # None = guideline spawn above the skyline
        self.bag = Bag(seed)
        self.board = np.zeros((ROWS, W), np.int8)      # 0 = empty, piece id + 1 otherwise
        self.queue = [self.bag.next() for _ in range(PREVIEW)]
        self.hold = NONE
        self.can_hold = True
        self.score = 0
        self.lines = 0
        self.level = 1
        self.pieces = 0
        self.b2b = False
        self.over = False
        self.last_clear = ""           # text description of the last scoring event
        self.clear_count = [0] * 5     # how many singles / doubles / triples / tetrises
        self.tspins = 0
        self.last_lock = None
        self._fall_t = 0.0
        self._lock_t = 0.0
        self._spawn()

    # ------------------------------------------------------------------ helpers
    def cells(self, rot=None, x=None, y=None):
        rot = self.rot if rot is None else rot
        x = self.x if x is None else x
        y = self.y if y is None else y
        return [(x + dx, y + dy) for dx, dy in SHAPES[self.piece][rot]]

    def collides(self, rot, x, y):
        for dx, dy in SHAPES[self.piece][rot]:
            cx, cy = x + dx, y + dy
            if cx < 0 or cx >= W or cy >= ROWS or (cy >= 0 and self.board[cy, cx]):
                return True
        return False

    def ghost_y(self):
        y = self.y
        while not self.collides(self.rot, self.x, y + 1):
            y += 1
        return y

    def _spawn(self, piece=None):
        if piece is None:
            piece = self.queue.pop(0)
            self.queue.append(self.bag.next())
        self.piece, self.rot = piece, 0
        self.x, self.y = SPAWN_X[piece], SPAWN_Y if self.spawn_y is None else self.spawn_y
        self._last_rot = False
        self._fall_t = self._lock_t = 0.0
        if self.collides(self.rot, self.x, self.y):
            self.over = True                      # block out
        elif self.spawn_y is None and not self.collides(self.rot, self.x, self.y + 1):
            self.y += 1                           # guideline: a new piece drops one row at once

    # ------------------------------------------------------------------ player actions
    def move(self, dx):
        if self.over or self.collides(self.rot, self.x + dx, self.y):
            return False
        self.x += dx
        self._last_rot = False
        self._lock_t = 0.0
        return True

    def rotate(self, d):
        """d = +1 clockwise ("rotate right"), -1 counter-clockwise ("rotate left")."""
        if self.over:
            return False
        new = (self.rot + d) % 4
        for kx, ky in kicks(self.piece, self.rot, new):
            if not self.collides(new, self.x + kx, self.y + ky):
                self.rot, self.x, self.y = new, self.x + kx, self.y + ky
                self._last_rot = True
                self._lock_t = 0.0
                return True
        return False

    def soft_drop(self):
        if self.over or self.collides(self.rot, self.x, self.y + 1):
            return False
        self.y += 1
        self.score += 1
        self._last_rot = False
        return True

    def hard_drop(self):
        if self.over:
            return
        gy = self.ghost_y()
        if gy != self.y:
            self._last_rot = False
        self.score += 2 * (gy - self.y)
        self.y = gy
        self._lock()

    def hold_piece(self):
        if self.over or not self.can_hold:
            return False
        old = self.hold
        self.hold = self.piece
        self._spawn(None if old == NONE else old)
        self.can_hold = False
        return True

    def press(self, key):
        """Apply one key tap, named as on the website's control screen."""
        if key == "left":
            return self.move(-1)
        if key == "right":
            return self.move(1)
        if key == "up":
            return self.rotate(1)
        if key == "z":
            return self.rotate(-1)
        if key == "down":
            return self.soft_drop()
        if key == "space":
            return self.hard_drop()
        if key == "c":
            return self.hold_piece()
        raise ValueError(key)

    # ------------------------------------------------------------------ time
    def tick(self, dt):
        """Advance real time by dt seconds (gravity and lock delay)."""
        if self.over:
            return
        if self.collides(self.rot, self.x, self.y + 1):
            self._lock_t += dt
            if self._lock_t >= LOCK_DELAY:
                self._lock()
            return
        self._fall_t += dt
        g = gravity_seconds(self.level)
        while self._fall_t >= g and not self.collides(self.rot, self.x, self.y + 1):
            self._fall_t -= g
            self.y += 1
            self._last_rot = False

    # ------------------------------------------------------------------ locking / scoring
    def _is_tspin(self):
        if self.piece != T or not self._last_rot:
            return False
        n = 0
        for cx, cy in ((self.x, self.y), (self.x + 2, self.y), (self.x, self.y + 2), (self.x + 2, self.y + 2)):
            if cx < 0 or cx >= W or cy >= ROWS or self.board[cy, cx]:
                n += 1
        return n >= 3

    def _lock(self):
        cells = self.cells()
        tspin = self._is_tspin()
        for cx, cy in cells:
            if cy >= 0:
                self.board[cy, cx] = self.piece + 1
        full = self.board.all(axis=1)
        n = int(full.sum())
        # kept so a front end can play a line-clear animation
        self.last_lock = {"board": self.board.copy(), "rows": np.nonzero(full)[0].tolist()}
        if n:
            keep = self.board[~full]
            self.board = np.vstack([np.zeros((n, W), np.int8), keep])
        base = TSPIN_SCORE[n] if tspin else LINE_SCORE[n]
        difficult = n > 0 and (tspin or n == 4)
        if difficult and self.b2b:
            base = base * 3 // 2
        self.score += base * self.level
        name = ("T-Spin " if tspin else "") + ["", "Single", "Double", "Triple", "Tetris"][n]
        self.last_clear = ("Back-to-Back " if difficult and self.b2b else "") + name.strip()
        if n:
            self.b2b = difficult
            self.clear_count[n] += 1
        self.tspins += tspin
        self.lines += n
        self.level = 1 + self.lines // 10
        self.pieces += 1
        self.can_hold = True
        if all(cy < HIDDEN for _, cy in cells):
            self.over = True                      # lock out
            return
        self._spawn()

    # ------------------------------------------------------------------ display
    def render_text(self):
        ghost = set() if self.over else set((x, y) for x, y in self.cells(y=self.ghost_y()))
        active = set() if self.over else set(self.cells())
        out = []
        for y in range(HIDDEN, ROWS):
            row = ""
            for x in range(W):
                if (x, y) in active:
                    row += "[]"
                elif self.board[y, x]:
                    row += "##"
                elif (x, y) in ghost:
                    row += "::"
                else:
                    row += " ."
            out.append("|" + row + "|")
        out.append("+" + "-" * (2 * W) + "+")
        return "\n".join(out)
