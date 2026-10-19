"""The learning part: a small value network and the move chooser that uses it.

V(afterstate) estimates the discounted future reward of a position *after* a piece has been
placed.  To pick a move the agent scores every reachable placement with
    immediate reward + gamma * V(resulting position)
and plays the best one.  With depth=2 it also looks one more piece ahead using the preview.
"""
import numpy as np
from .fastsim import N_FEATS, expand

GAMMA = 0.95            # default discount; each saved network remembers the one it was trained with
DEATH_PENALTY = 1.0     # in reward units (= 1000 points), on top of losing all future reward


class ValueNet:
    """Plain numpy multilayer perceptron with ReLU units, trained with Adam on a Huber loss."""

    def __init__(self, sizes=(N_FEATS, 64, 64, 1), seed=0):
        rng = np.random.default_rng(seed)
        self.W = [(rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32)
                  for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b, np.float32) for b in sizes[1:]]
        self.W[-1] *= 0.1
        self._m = [np.zeros_like(p) for p in self.W + self.b]
        self._v = [np.zeros_like(p) for p in self.W + self.b]
        self._t = 0
        self.gamma = GAMMA

    def predict(self, X):
        a = X
        for W, b in zip(self.W[:-1], self.b[:-1]):
            a = np.maximum(a @ W + b, 0.0)
        return (a @ self.W[-1] + self.b[-1])[:, 0]

    def train_batch(self, X, y, lr=1e-3):
        acts = [X]
        for W, b in zip(self.W[:-1], self.b[:-1]):
            acts.append(np.maximum(acts[-1] @ W + b, 0.0))
        err = (acts[-1] @ self.W[-1] + self.b[-1])[:, 0] - y
        loss = float(np.mean(np.where(np.abs(err) < 1, 0.5 * err * err, np.abs(err) - 0.5)))
        delta = (np.clip(err, -1, 1) / len(y))[:, None].astype(np.float32)
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        for k in range(len(self.W) - 1, -1, -1):
            gW[k] = acts[k].T @ delta
            gb[k] = delta.sum(0)
            if k:
                delta = (delta @ self.W[k].T) * (acts[k] > 0)
        self._t += 1
        params, grads = self.W + self.b, gW + gb
        corr = lr * np.sqrt(1 - 0.999 ** self._t) / (1 - 0.9 ** self._t)
        for p, g, m, v in zip(params, grads, self._m, self._v):
            m *= 0.9
            m += 0.1 * g
            v *= 0.999
            v += 0.001 * g * g
            p -= corr * m / (np.sqrt(v) + 1e-8)
        return loss

    def copy_weights_from(self, other):
        for a, b in zip(self.W + self.b, other.W + other.b):
            a[...] = b

    def save(self, path):
        np.savez(path, gamma=np.float32(self.gamma), **{f"W{i}": w for i, w in enumerate(self.W)},
                 **{f"b{i}": b for i, b in enumerate(self.b)})

    @classmethod
    def load(cls, path):
        d = np.load(path)
        n = sum(1 for k in d.files if k.startswith("W"))
        net = cls(sizes=[d["W0"].shape[0]] + [d[f"W{i}"].shape[1] for i in range(n)])
        net.W = [d[f"W{i}"] for i in range(n)]
        net.b = [d[f"b{i}"] for i in range(n)]
        net.gamma = float(d["gamma"]) if "gamma" in d.files else 0.99
        return net


def score_candidates(net, c):
    """reward + gamma * V for every candidate; game-ending placements get the death penalty."""
    q = c.reward + net.gamma * net.predict(c.X)
    q[c.dead] = c.reward[c.dead] - DEATH_PENALTY
    return q


def rank(net, st, depth=1, beam=8, **expand_kw):
    """All placements for State `st`, best first. Returns (candidates, list of indices)."""
    c = expand(st, **expand_kw)
    if c.n == 0:
        return c, []
    q = score_candidates(net, c)
    order = np.argsort(-q)
    if depth < 2:
        return c, order.tolist()
    top, deep = order[:beam], []
    for i in top:                                  # look one more piece ahead for the best few
        qi = q[i]
        if not c.dead[i]:
            child = c.child(int(i))
            if child.cur != 7:
                c2 = expand(child)
                qi = c.reward[i] + net.gamma * (score_candidates(net, c2).max() if c2.n else -DEATH_PENALTY)
        deep.append(qi)
    top = top[np.argsort(-np.array(deep), kind="stable")]
    return c, top.tolist() + order[beam:].tolist()


def choose(net, st, depth=1, beam=8, **expand_kw):
    """Best placement for State `st`. Returns (candidates, index) or (candidates, None) if stuck."""
    c, order = rank(net, st, depth, beam, **expand_kw)
    return c, (order[0] if order else None)


def engine_keys(net, eng, depth=2):
    """Key taps the agent wants to press for the current piece of a key-level `Engine`."""
    from .fastsim import State
    from .pieces import SPAWN_X
    from .controller import key_plan
    st = State(eng.board > 0, eng.piece, eng.hold, eng.queue, eng.b2b)
    c, i = choose(net, st, depth=depth, ceil_y=eng.y, start_x=eng.x, allow_hold=eng.can_hold)
    if i is None:
        return ["space"]
    if c.use_hold[i]:
        return key_plan(True, 0, SPAWN_X[int(c.piece[i])], int(c.rot[i]), int(c.bx[i]))
    return key_plan(False, eng.rot, eng.x, int(c.rot[i]), int(c.bx[i]))
