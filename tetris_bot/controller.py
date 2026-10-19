"""Turns a chosen placement into the key taps shown on the website's control screen."""


def key_plan(use_hold, rot_from, x_from, rot_to, x_to):
    """Keys to press, in order: hold (C), rotate (Up = right, Z = left), shift, hard drop (Space)."""
    keys = ["c"] if use_hold else []
    turn = (rot_to - rot_from) % 4
    keys += {0: [], 1: ["up"], 2: ["up", "up"], 3: ["z"]}[turn]
    dx = x_to - x_from
    keys += ["right" if dx > 0 else "left"] * abs(dx)
    keys.append("space")
    return keys


# --------------------------------------------------------------------------------------------
# Live play: look at the screen, decide, press keys.
# --------------------------------------------------------------------------------------------
import math
import time
import numpy as np
from .pieces import W, ROWS, HIDDEN, VISIBLE, SPAWN_X, SHAPES, NONE, PIECES
from .engine import gravity_seconds
from .fastsim import State
from .agent import rank
from .pieces import kicks
from . import vision


def simulate_keys(board, piece, rot, x, y, keys, gravity, tap_time, head_start=0.0):
    """Predict where a key sequence really puts the piece, with the piece falling `gravity` seconds
    per row while the keys are being pressed.  Returns the 4 final cells [(x, y), ...]."""
    def hits(r, px, py):
        for dx, dy in SHAPES[piece][r]:
            cx, cy = px + dx, py + dy
            if cx < 0 or cx >= W or cy >= ROWS or (cy >= 0 and board[cy, cx]):
                return True
        return False

    clock = head_start
    for k in keys:
        while clock >= gravity and not hits(rot, x, y + 1):       # rows fallen since the last key
            clock -= gravity
            y += 1
        if hits(rot, x, y + 1):
            clock = 0.0
        if k in ("up", "z"):
            new = (rot + (1 if k == "up" else -1)) % 4
            for kx, ky in kicks(piece, rot, new):
                if not hits(new, x + kx, y + ky):
                    rot, x, y = new, x + kx, y + ky
                    break
        elif k in ("left", "right"):
            dx = -1 if k == "left" else 1
            if not hits(rot, x + dx, y):
                x += dx
        elif k == "space":
            while not hits(rot, x, y + 1):
                y += 1
        clock += tap_time
    return sorted((x + dx, y + dy) for dx, dy in SHAPES[piece][rot])


class ScreenIO:
    """Real screen + real keyboard (mss for capture, pynput for key presses)."""

    def __init__(self, monitor=1):
        import mss
        from pynput.keyboard import Controller, Key
        self.sct = mss.mss()
        self.mon = self.sct.monitors[monitor]
        self.kb = Controller()
        self.keys = {"left": Key.left, "right": Key.right, "up": Key.up, "down": Key.down,
                     "space": Key.space, "z": "z", "c": "c"}
        self.region = dict(self.mon)

    def _shot(self, region):
        raw = self.sct.grab(region)
        return np.frombuffer(raw.bgra, np.uint8).reshape(raw.height, raw.width, 4)[:, :, 2::-1]

    def grab_full(self):
        return self._shot(self.mon)

    def focus_on(self, bounds, full_shape):
        """Capture only the box `bounds` = (left, top, right, bottom) in full-screenshot pixels."""
        scale = full_shape[1] / self.mon["width"]            # 2.0 on HiDPI / Retina displays
        l, t, r, b = bounds
        l, t = max(0, int(l / scale)), max(0, int(t / scale))
        r = min(self.mon["width"], int(math.ceil(r / scale)))
        b = min(self.mon["height"], int(math.ceil(b / scale)))
        self.region = {"left": self.mon["left"] + l, "top": self.mon["top"] + t, "width": r - l, "height": b - t}
        return l * scale, t * scale                          # offset of the new capture, in pixels

    def grab(self):
        return self._shot(self.region)

    def tap(self, key, hold):
        k = self.keys[key]
        self.kb.press(k)
        time.sleep(hold)
        self.kb.release(k)

    def sleep(self, dt):
        time.sleep(dt)

    def now(self):
        return time.perf_counter()


class Bot:
    """Plays one game through an IO object (the real screen, or the replica for testing).

    Every placement is closed-loop: press the rotate / shift keys, look at the screen to check the
    piece really is where it should be, press more keys if it is not, and only then hard drop.
    Key presses that the game swallowed therefore cost a little time instead of a misplaced piece.
    """

    def __init__(self, net, io, layout, depth=2, tap_hold=0.04, tap_gap=0.04, hold_wait=0.07,
                 verify=True, adapt=True, settle=0.9, grace=0.15, spawn_wait=0.35, latency=0.04,
                 caution=1.0, max_caution=8.0, verbose=True, stop=lambda: False, log_path=None):
        self.net, self.io, self.lay = net, io, layout
        self.depth, self.tap_hold, self.tap_gap, self.hold_wait = depth, tap_hold, tap_gap, hold_wait
        self.verify, self.adapt = verify, adapt
        self.settle, self.grace, self.spawn_wait = settle, grace, spawn_wait
        self.latency, self.caution, self.max_caution = latency, caution, max_caution
        self.verbose, self.stop = verbose, stop
        self.lines = self.pieces = self.resyncs = self.blind = 0
        self.corrected = self.hold_retries = self.redrops = 0   # extra keys / repeated holds / repeated drops
        self.b2b = False
        self.hold = NONE
        self._recent, self._recent_first, self._window = [], [], []   # recent pieces: 1 = needed a fix
        self._floor = min(0.02, tap_hold)
        self.lead_in, self._lead_floor = 0.0, 0.0        # pause between seeing a piece and the first key
        self._logf = open(log_path, "w") if log_path else None

    # ------------------------------------------------------------------ small helpers
    def _log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    def _note(self, text):
        """Diagnostics go to the log file (bot_log.txt), not to the screen."""
        if self._logf:
            self._logf.write(f"[piece {self.pieces + 1:4d}] {text}\n")
            self._logf.flush()

    def _send(self, keys):
        for k in keys:
            self.io.tap(k, self.tap_hold)
            self.io.sleep(self.tap_gap)

    def _see_piece(self, stack):
        """Cells [(x, y), ...] of the falling piece on screen right now, or None if unclear."""
        filled, _ = vision.read_cells(self.io.grab(), self.lay)
        if (stack & ~filled).any():
            return None
        ys, xs = np.nonzero(filled & ~stack)
        if len(ys) != 4:
            return None
        return sorted(zip(xs.tolist(), ys.tolist()))

    @staticmethod
    def _locate(piece, cells, prefer):
        """(rot, box x) of `piece` given its cells; `prefer` orders look-alike rotation states."""
        for r in prefer:
            shape = SHAPES[piece][r]
            x = cells[0][0] - min(d[0] for d in shape)
            y = min(c[1] for c in cells) - min(d[1] for d in shape)
            if sorted((x + dx, y + dy) for dx, dy in shape) == cells:
                return r, x
        return None

    @staticmethod
    def _shape(cells):
        top = min(c[1] for c in cells)
        return sorted((x, y - top) for x, y in cells)

    def _tune(self, needed_fix, first_key_lost):
        """Self-adjusting key timing.  If the first key of a piece keeps getting lost, the game is
        not listening yet right after a piece appears: wait longer before the first key.  If
        presses get lost in general, hold keys longer.  When all is well for a while, speed up."""
        if not self.adapt:
            return
        self._recent = (self._recent + [int(needed_fix)])[-12:]
        self._recent_first = (self._recent_first + [int(first_key_lost)])[-12:]
        self._window.append(int(needed_fix))
        if first_key_lost and sum(self._recent_first) >= 2 and self.lead_in < 0.25:
            self._lead_floor = min(0.2, max(self._lead_floor, self.lead_in + 0.01))
            self.lead_in = min(0.25, self.lead_in + 0.03)
            self._recent, self._recent_first, self._window = [], [], []
            self._note(f"the first key of a piece keeps getting lost: waiting {self.lead_in * 1000:.0f} ms before it from now on")
        elif needed_fix and sum(self._recent) >= 3 and self.tap_hold < 0.07:
            self._floor = min(0.05, max(self._floor, self.tap_hold + 0.003))
            self.tap_hold = self.tap_gap = min(0.07, self.tap_hold * 1.2 + 0.002)
            self._recent, self._recent_first, self._window = [], [], []
            self._note(f"key presses are being lost: slowing taps to {self.tap_hold * 1000:.0f} ms")
        elif len(self._window) >= 40:
            if sum(self._window) <= 2:                       # going well: try a little faster
                if self.tap_hold > self._floor:
                    self.tap_hold = self.tap_gap = max(self._floor, self.tap_hold - 0.004)
                    self._note(f"40 good pieces: speeding taps up to {self.tap_hold * 1000:.0f} ms")
                self.lead_in = max(self._lead_floor, self.lead_in - 0.01)
            self._window = []

    # ------------------------------------------------------------------ seeing the next piece
    def _wait_for_piece(self, expected, expected_cur, t_drop, expect_clear=False, dropped=None):
        """Poll the screen until a falling piece can be identified. Returns an Observation or None.
        `dropped` = (stack before the drop, cells where the piece should have landed)."""
        idle_since, told, redrops = self.io.now(), False, 0
        while not self.stop():
            img = self.io.grab()
            if expected is not None:
                obs = vision.observe(img, self.lay, expected)
                if obs.active is not None:
                    return obs
                waited = self.io.now() - t_drop
                if dropped is not None and redrops < 3 and waited > 0.06:
                    # Is the piece we "dropped" still hanging in the air above its landing spot?
                    # Then the hard-drop key was swallowed: press it again.
                    old, landing = dropped
                    ys, xs = np.nonzero(obs.filled & ~old)
                    cells = sorted(zip(xs.tolist(), ys.tolist()))
                    if (len(cells) == 4 and not (old & ~obs.filled).any() and cells != landing
                            and self._shape(cells) == self._shape(landing)):
                        self._note("hard drop not registered: pressing Space again")
                        self.io.tap("space", self.tap_hold)
                        self.io.sleep(self.tap_gap)
                        self.redrops += 1
                        redrops += 1
                        t_drop = self.io.now()
                        continue
                extra = obs.filled & ~expected
                in_sync = not (expected & ~obs.filled).any()
                if in_sync and expected_cur != NONE and extra.sum() < 4 and not extra[2:].any():
                    # The new piece is still (partly) above the visible rows.  We know which piece
                    # it is from the preview, so the cells that do show are enough to place it.
                    by_vis = self._partial_spawn(expected_cur, extra)
                    if by_vis is None and waited > self.spawn_wait and not extra.any():
                        by_vis = -2                       # nothing visible at all yet: act blind
                    if by_vis is not None:
                        obs.active = (expected_cur, 0, SPAWN_X[expected_cur], by_vis)
                        obs.stack = expected.copy()
                        self.blind += by_vis == -2 and not extra.any()
                        return obs
                # The screen disagrees with what we expect.  After a line clear that is normal for
                # a moment (animation); otherwise a piece went astray, so re-read the screen from scratch.
                if waited > (self.settle if expect_clear else self.grace):
                    fresh = vision.observe(img, self.lay, None)
                    if fresh.active is not None or waited > self.settle:
                        self.resyncs += 1
                        rows = ["".join("#" if e else "." for e in er) + "   " + "".join("#" if f else "." for f in fr)
                                for er, fr in zip(expected, obs.filled)]
                        self._note("MISPLACED PIECE - board is not what was planned.  expected | on screen\n    "
                                   + "\n    ".join(rows))
                        expected = None
                        if fresh.active is not None:
                            return fresh
            else:
                obs = vision.observe(img, self.lay, None)
                if obs.active is not None:
                    return obs
                if not told and self.io.now() - idle_since > 4.0:
                    self._log("No falling piece visible (paused, game over, or window hidden). Waiting...")
                    told = True
            self.io.sleep(0.012)
        return None

    @staticmethod
    def _partial_spawn(piece, extra):
        """If the visible cells are exactly the lower part of `piece` at its spawn column, return
        its box y in visible-row coordinates."""
        ys, xs = np.nonzero(extra)
        seen = set(zip(xs.tolist(), ys.tolist()))
        if not seen:
            return None
        for by_vis in (-1, -2):
            cells = {(SPAWN_X[piece] + dx, by_vis + dy) for dx, dy in SHAPES[piece][0]}
            if seen == {c for c in cells if c[1] >= 0}:
                return by_vis
        return None

    # ------------------------------------------------------------------ pressing keys
    def _execute(self, stack, piece, rot, x, r_to, x_to, use_hold, t_seen):
        """Bring the piece to rotation r_to / column x_to and hard drop it, checking the screen on
        the way.  Returns (done, needed_fix, first_key_lost); done=False means "the hold did not
        happen, plan again"."""
        wait = self.lead_in - (self.io.now() - t_seen)
        if wait > 0:
            self.io.sleep(wait)
        if use_hold:
            self.io.tap("c", self.tap_hold)
            self.io.sleep(max(self.hold_wait, self.lead_in))
            rot, x = 0, SPAWN_X[piece]
            if self.verify:
                for _ in range(4):
                    cells = self._see_piece(stack)
                    seen = vision.match_piece(cells) if cells else None
                    if seen:
                        break
                    self.io.sleep(0.03)
                if seen and seen[0] != piece:            # the swap did not happen
                    self.hold_retries += 1
                    self._note(f"hold key not registered (still see {PIECES[seen[0]]}, wanted {PIECES[piece]})")
                    return False, True, True
                if seen:
                    rot, x = self._locate(piece, cells, (0, 1, 3, 2))
        self._send(key_plan(False, rot, x, r_to, x_to)[:-1])
        needed_fix = first_key_lost = False
        if self.verify:
            want = sorted((x_to + dx, dy - min(d[1] for d in SHAPES[piece][r_to])) for dx, dy in SHAPES[piece][r_to])
            last, fixes = None, 0
            for _ in range(10):
                self.io.sleep(max(0.03, self.tap_gap))            # let the screen catch up
                cells = self._see_piece(stack)
                if cells is None:
                    break                                         # not visible: trust the plan
                shape = self._shape(cells)
                if shape == want:
                    break
                if shape != last:                                 # might be a stale frame: look twice
                    last = shape
                    continue
                where = self._locate(piece, cells, (r_to, rot, 0, 1, 3, 2))
                if where is None or fixes >= 4:
                    break
                if fixes == 0:      # was it the very first key that went missing?
                    first_key_lost = (where[0] == rot) if r_to != rot else (where[1] == x)
                fix = key_plan(False, where[0], where[1], r_to, x_to)[:-1]
                if fixes % 2 == 1:                                # second try: turn the other way round
                    turn = (where[0] - r_to) % 4
                    fix = ["z"] * turn + [k for k in fix if k in ("left", "right")]
                self._note(f"{PIECES[piece]} is at rot {where[0]} col {where[1]}, wanted rot {r_to} col {x_to}: pressing {fix}")
                self._send(fix)
                fixes, needed_fix, last = fixes + 1, True, None
        self.io.tap("space", self.tap_hold)
        self.io.sleep(self.tap_gap)
        return True, needed_fix, first_key_lost

    # ------------------------------------------------------------------ main loop
    def run(self, max_pieces=None):
        expected, expected_cur, t_drop, expect_clear, dropped = None, NONE, self.io.now(), False, None
        allow_hold = True
        while not self.stop() and (max_pieces is None or self.pieces < max_pieces):
            obs = self._wait_for_piece(expected, expected_cur, t_drop, expect_clear, dropped)
            if obs is None:
                break
            t_seen = self.io.now()
            piece, rot, bx, by_vis = obs.active
            board = np.zeros((ROWS, W), bool)
            board[HIDDEN:] = obs.stack
            hold = obs.hold if obs.hold != NONE else self.hold
            queue = []
            for p in obs.next:
                if p == NONE:
                    break
                queue.append(p)
            st = State(board, piece, hold, queue, self.b2b)

            # The piece keeps falling while keys are pressed.  Walk down the agent's ranking and
            # take the first placement whose key sequence still gets there under gravity.
            level = 1 + self.lines // 10
            g = gravity_seconds(level)
            per_tap = self.tap_hold + self.tap_gap
            start = by_vis + HIDDEN
            bias = min(self.max_caution, self.caution * 6 * per_tap / g)
            c, order = rank(self.net, st, depth=self.depth, ceil_y=start, start_x=bx, height_bias=bias,
                            allow_hold=allow_hold)
            i = None
            for j in order[:15]:
                p, r_to, x_to = int(c.piece[j]), int(c.rot[j]), int(c.bx[j])
                want = sorted((x_to + dx, int(c.by[j]) + dy) for dx, dy in SHAPES[p][r_to])
                if c.use_hold[j]:
                    args = (p, 0, SPAWN_X[p], HIDDEN - 1, key_plan(False, 0, SPAWN_X[p], r_to, x_to), g, per_tap)
                    heads = (self.hold_wait, self.hold_wait + g)
                else:
                    args = (piece, rot, bx, start, key_plan(False, rot, bx, r_to, x_to), g, per_tap)
                    heads = (self.latency, self.latency + g)
                if i is None:
                    i = int(j)                            # fallback: the agent's first choice
                if all(simulate_keys(board, *args, head_start=h) == want for h in heads):
                    i = int(j)
                    break
            if i is None:                                 # nothing fits any more: just drop it
                self.io.tap("space", self.tap_hold)
                self.io.sleep(self.tap_gap)
                t_drop, expected, expected_cur, dropped = self.io.now(), None, NONE, None
                self.pieces += 1
                continue
            use_hold = bool(c.use_hold[i])
            done, needed_fix, first_lost = self._execute(obs.stack, int(c.piece[i]), rot, bx, int(c.rot[i]),
                                                         int(c.bx[i]), use_hold, t_seen)
            self._tune(needed_fix, first_lost)
            if not done:                                  # hold was not taken: look again and re-plan
                expected, expected_cur, t_drop, expect_clear, dropped = obs.stack, piece, self.io.now(), False, None
                allow_hold = self.hold_retries % 2 == 1   # every other retry: place the piece itself
                continue
            allow_hold = True
            self.corrected += needed_fix
            t_drop = self.io.now()
            self.pieces += 1
            n = int(c.lines[i])
            self.lines += n
            expect_clear = n > 0
            self.b2b = bool(c.b2b[i])
            child = c.child(i)
            self.hold = child.hold
            expected, expected_cur = c.boards[i][HIDDEN:].copy(), child.cur
            landing = sorted((int(c.bx[i]) + dx, int(c.by[i]) + dy - HIDDEN)
                             for dx, dy in SHAPES[int(c.piece[i])][int(c.rot[i])])
            dropped = (obs.stack, landing)
            if self.verbose and (n or self.pieces % 25 == 0):
                what = ["", "single", "double", "triple", "TETRIS"][n]
                self._log(f"piece {self.pieces:5d} | lines {self.lines:4d} | level {level:2d} | {what}")
        return self.pieces

    def summary(self):
        return (f"{self.pieces} pieces, {self.lines} lines | pieces that needed extra key presses: {self.corrected} | "
                f"hold retries: {self.hold_retries} | repeated hard drops: {self.redrops} | misplaced pieces: {self.resyncs} | "
                f"key timing now {self.tap_hold * 1000:.0f} ms, wait before first key {self.lead_in * 1000:.0f} ms")
