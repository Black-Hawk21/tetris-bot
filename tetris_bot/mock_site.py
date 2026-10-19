"""A stand-in for the website, for testing the controller without a browser.

It runs the replica engine on a simulated clock, shows it through the same renderer as the
playable window, ignores key presses during the line-clear animation, lets pieces fall with
gravity while the bot is "pressing keys", and can drop a key now and then to test recovery.
"""
import random
from .engine import Engine
from .render import render


class MockSiteIO:
    def __init__(self, seed=0, cell=30, clear_anim=0.4, entry_delay=0.05, grab_cost=0.02,
                 drop_key_prob=0.0, level=None, poll_fps=None, dead_time=0.0, visible_spawn=False):
        """poll_fps: the game only looks at the keyboard that many times per second, so a short
        tap can fall between two looks, and two quick taps can blur into one.
        dead_time: seconds after a piece appears during which input is ignored.
        visible_spawn: pieces appear inside the visible rows (as on play.tetris.com)."""
        from .pieces import HIDDEN
        self.eng = Engine(seed, spawn_y=HIDDEN if visible_spawn else None)
        self.poll_fps, self.dead_time, self.spawn_t, self._last_poll = poll_fps, dead_time, 0.0, {}
        self.cell, self.clear_anim, self.entry_delay, self.grab_cost = cell, clear_anim, entry_delay, grab_cost
        self.t = 0.0
        self.anim_until, self.flash = 0.0, None
        self.rng = random.Random(seed)
        self.drop_key_prob = drop_key_prob
        self.fixed_level = level
        self.keys_sent = self.keys_ignored = 0

    def _after_lock(self, pieces_before):
        if self.eng.pieces != pieces_before:
            rows = self.eng.last_lock["rows"]
            self.anim_until = self.t + (self.clear_anim if rows else self.entry_delay)
            self.spawn_t = self.anim_until
            self.flash = (self.eng.last_lock["board"], rows) if rows else None

    def _advance(self, dt):
        end = self.t + dt
        while self.t < end - 1e-9:
            step = min(0.005, end - self.t)
            self.t += step
            if self.t >= self.anim_until:
                self.flash = None
                if self.fixed_level:
                    self.eng.level = self.fixed_level
                n = self.eng.pieces
                self.eng.tick(step)
                self._after_lock(n)

    def grab(self):
        self._advance(self.grab_cost)
        busy = self.t < self.anim_until
        return render(self.eng, self.cell, flash=self.flash if busy else None, show_piece=not busy)

    def _registers(self, key, hold):
        if self.t < self.anim_until or self.t < self.spawn_t + self.dead_time:
            return False
        if self.rng.random() < self.drop_key_prob:
            return False
        if self.poll_fps:
            first = -int(-(self.t * self.poll_fps - 1e-9) // 1)          # first look at or after key-down
            last = int(((self.t + hold) * self.poll_fps + 1e-9) // 1)    # last look before key-up
            if first > last:
                return False                                             # tap fell between two looks
            merged = self._last_poll.get(key) == first - 1               # no key-up seen since last tap
            self._last_poll[key] = last
            return not merged
        return True

    def tap(self, key, hold):
        self.keys_sent += 1
        if self._registers(key, hold):
            n = self.eng.pieces
            self.eng.press(key)
            self._after_lock(n)
        else:
            self.keys_ignored += 1
        self._advance(hold)

    def sleep(self, dt):
        self._advance(dt)

    def now(self):
        return self.t
