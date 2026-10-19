"""Self-play reinforcement learning in the replica.

The agent starts with random weights and no knowledge of Tetris strategy.  It plays the fast
simulator, and after every placement the value network is nudged towards
    n-step reward + gamma^n * V(position n placements later)        (temporal-difference learning)
using a replay buffer of its own recent experience.

    python train.py --steps 600000 --out models/value_net.npz
"""
import argparse, time
from collections import deque
import numpy as np
from tetris_bot.agent import ValueNet, score_candidates, choose, GAMMA, DEATH_PENALTY
from tetris_bot.fastsim import TetrisSim, N_FEATS


def messy_start(sim, rng, moves):
    """Play `moves` random placements so the game starts from a messed-up board."""
    for _ in range(moves):
        c = sim.candidates()
        alive = np.nonzero(~c.dead)[0] if c.n else []
        if len(alive) == 0:
            break
        sim.step(c, int(rng.choice(alive)))
    sim.score = sim.lines = sim.pieces = 0
    sim.level, sim.clear_count = 1, [0] * 5


def evaluate(net, games=10, max_pieces=1000, depth=1, seed0=10_000, messy=0):
    """Greedy play. Returns dict of averages (points are level-1 points, i.e. without level multiplier).
    With messy > 0 every game starts after that many random placements (tests recovery)."""
    pieces = lines = tetrises = died = clears = 0
    pts = 0.0
    for g in range(games):
        sim = TetrisSim(seed0 + g)
        if messy:
            messy_start(sim, np.random.default_rng(seed0 + g), messy)
        while not sim.over and sim.pieces < max_pieces:
            c, i = choose(net, sim.state, depth=depth)
            if i is None:
                sim.over = True
                break
            pts += c.reward[i] * 1000
            sim.step(c, i)
        pieces += sim.pieces; lines += sim.lines; died += sim.over
        tetrises += sim.clear_count[4]; clears += sum(sim.clear_count[1:])
    return dict(pieces=pieces / games, lines=lines / games, died=died / games,
                pts_per_piece=pts / max(pieces, 1), tetris_share=4 * tetrises / max(lines, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600_000, help="total placements to play")
    ap.add_argument("--out", default="models/value_net.npz")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--nstep", type=int, default=3)
    ap.add_argument("--gamma", type=float, default=GAMMA, help="discount per placement")
    ap.add_argument("--buffer", type=int, default=200_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eps-end", type=float, default=0.01)
    ap.add_argument("--eps-steps", type=int, default=60_000)
    ap.add_argument("--cap", type=int, default=2000, help="max pieces per training game")
    ap.add_argument("--target-sync", type=int, default=2000)
    ap.add_argument("--eval-every", type=int, default=25_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None, help="continue from saved weights")
    ap.add_argument("--messy", type=float, default=0.5,
                    help="share of training games that start from a randomly messed-up board")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    net = ValueNet.load(a.init) if a.init else ValueNet((N_FEATS, a.hidden, a.hidden, 1), a.seed)
    if not a.init:
        net.gamma = a.gamma
    gamma = net.gamma
    target = ValueNet((N_FEATS, a.hidden, a.hidden, 1))
    target.copy_weights_from(net)

    B = a.buffer
    buf_x = np.zeros((B, N_FEATS), np.float32)      # afterstate
    buf_x2 = np.zeros((B, N_FEATS), np.float32)     # afterstate n placements later
    buf_r = np.zeros(B, np.float32)                 # discounted reward collected in between
    buf_g = np.zeros(B, np.float32)                 # gamma^n, or 0 if the game ended
    size = pos = 0

    def push(x, r, x2, g):
        nonlocal size, pos
        buf_x[pos], buf_r[pos], buf_x2[pos], buf_g[pos] = x, r, x2, g
        pos = (pos + 1) % B
        size = min(size + 1, B)

    def flush(window, final_x, alive):
        """Emit n-step transitions for everything still in the window (end of a game)."""
        while window:
            ret, g = 0.0, 1.0
            for _, r in window:
                ret += g * r
                g *= gamma
            push(window[0][0], ret, final_x, g if alive else 0.0)
            window.popleft()

    t0, best, ep, losses = time.time(), -1e9, 0, deque(maxlen=2000)
    ep_len, ep_lines = deque(maxlen=50), deque(maxlen=50)
    def new_game():
        sim = TetrisSim(int(rng.integers(1 << 30)))
        if rng.random() < a.messy:
            messy_start(sim, rng, int(rng.integers(3, 16)))
        return sim

    sim, window, prev_x = new_game(), deque(), None
    for step in range(1, a.steps + 1):
        eps = max(a.eps_end, 1.0 - (1.0 - a.eps_end) * step / a.eps_steps)
        c = sim.candidates()
        alive = np.nonzero(~c.dead)[0] if c.n else []
        if len(alive) == 0:                              # no survivable placement: game over
            if prev_x is not None:
                window.append((prev_x, -DEATH_PENALTY))
                flush(window, prev_x, alive=False)
            ended = True
        else:
            if rng.random() < eps:
                i = int(rng.choice(alive))
            else:
                i = int(np.argmax(score_candidates(net, c)))
            x, r = c.X[i], float(c.reward[i])
            if prev_x is not None:
                window.append((prev_x, r))               # reward earned after leaving prev_x
                if len(window) == a.nstep:
                    ret = sum(gamma ** k * w[1] for k, w in enumerate(window))
                    push(window[0][0], ret, x, gamma ** a.nstep)
                    window.popleft()
            sim.step(c, i)
            prev_x = x
            ended = sim.pieces >= a.cap
            if ended:
                flush(window, x, alive=True)             # capped, not dead: keep bootstrapping
        if ended:
            ep += 1
            ep_len.append(sim.pieces); ep_lines.append(sim.lines)
            sim, window, prev_x = new_game(), deque(), None

        if size >= 2000:
            idx = rng.integers(0, size, a.batch)
            y = buf_r[idx] + buf_g[idx] * target.predict(buf_x2[idx])
            frac = step / a.steps
            lr = a.lr * (1.0 if frac < 0.5 else 0.3 if frac < 0.8 else 0.1)
            losses.append(net.train_batch(buf_x[idx], y, lr))
            if step % a.target_sync == 0:
                target.copy_weights_from(net)

        if step % a.eval_every == 0:
            ev = evaluate(net, games=6, max_pieces=1000)
            ev2 = evaluate(net, games=6, max_pieces=1000, messy=12)
            fitness = ev["pts_per_piece"] * ev["pieces"] + ev2["pts_per_piece"] * ev2["pieces"]
            tag = ""
            if fitness > best:
                best, tag = fitness, "  <- saved"
                net.save(a.out)
            print(f"step {step:7d} | games {ep:5d} | eps {eps:.3f} | train len {np.mean(ep_len or [0]):6.0f} "
                  f"| loss {np.mean(losses or [0]):.4f} | eval: pieces {ev['pieces']:6.0f} died {ev['died']:.2f} "
                  f"pts/piece {ev['pts_per_piece']:6.1f} tetris share {ev['tetris_share']:.2f} "
                  f"| messy start: pieces {ev2['pieces']:6.0f} died {ev2['died']:.2f} pts/piece {ev2['pts_per_piece']:6.1f} "
                  f"| {time.time() - t0:5.0f}s{tag}", flush=True)
    print("done; best weights in", a.out)


if __name__ == "__main__":
    main()
