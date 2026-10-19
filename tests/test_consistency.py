"""The fast simulator must agree with the key-level engine: same boards, same score, same queue."""
import os, sys, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tetris_bot.engine import Engine
from tetris_bot.fastsim import TetrisSim
from tetris_bot.pieces import SPAWN_X
from tetris_bot.controller import key_plan


def run(seed, max_pieces=400, clean=False):
    rng = random.Random(seed)
    eng, sim = Engine(seed), TetrisSim(seed)
    n = 0
    while not sim.over and n < max_pieces:
        assert eng.piece == sim.state.cur and tuple(eng.queue) == sim.state.queue and eng.hold == sim.state.hold
        c = sim.candidates()
        assert c.n > 0
        if clean:      # prefer placements that clear lines / make few holes so games last long
            score = c.lines * 10 - c.X[:, 10] * 10 - c.X[:, 12] * 20 + np.array([rng.random() for _ in range(c.n)])
            i = int(np.argmax(score))
        else:
            i = rng.randrange(c.n)
        for key in key_plan(bool(c.use_hold[i]), 0, SPAWN_X[int(c.piece[i])], int(c.rot[i]), int(c.bx[i])):
            eng.press(key)
        sim.step(c, i)
        assert np.array_equal(eng.board > 0, sim.state.board), f"board mismatch seed={seed} piece={n}"
        assert eng.score == sim.score, (eng.score, sim.score)
        assert eng.lines == sim.lines and eng.b2b == sim.state.b2b
        assert eng.over == sim.over, (eng.over, sim.over, n)
        n += 1
    return n, sim.lines


if __name__ == "__main__":
    tot = lines = 0
    for seed in range(300):
        a, b = run(seed, clean=seed % 2 == 0)
        tot += a; lines += b
    print(f"consistency OK: {tot} placements, {lines} line clears compared against the engine")
