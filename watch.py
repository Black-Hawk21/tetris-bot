"""Watch the trained agent play the replica in the terminal (it presses the same keys as on the site).

    python watch.py                  # animated, about 8 pieces per second
    python watch.py --fast --pieces 3000 --games 5      # no animation, just results
"""
import argparse, time
from tetris_bot.agent import ValueNet, engine_keys
from tetris_bot.engine import Engine
from tetris_bot.pieces import PIECES, NONE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/value_net.npz")
    ap.add_argument("--pieces", type=int, default=500)
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--depth", type=int, default=2, help="1 = current piece only, 2 = also look at the next piece")
    ap.add_argument("--fast", action="store_true", help="no animation")
    ap.add_argument("--fps", type=float, default=8)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    net = ValueNet.load(a.model)
    for g in range(a.games):
        eng = Engine(None if a.seed is None else a.seed + g)
        while not eng.over and eng.pieces < a.pieces:
            for k in engine_keys(net, eng, a.depth):
                eng.press(k)
            if not a.fast:
                hold = "-" if eng.hold == NONE else PIECES[eng.hold]
                print("\033[H\033[J" + eng.render_text())
                print(f"score {eng.score}  level {eng.level}  lines {eng.lines}  pieces {eng.pieces}")
                print(f"hold {hold}  next {' '.join(PIECES[p] for p in eng.queue)}  {eng.last_clear}")
                time.sleep(1.0 / a.fps)
        c = eng.clear_count
        print(f"game {g + 1}: {'TOPPED OUT' if eng.over else 'still alive'} after {eng.pieces} pieces | score {eng.score} | "
              f"level {eng.level} | lines {eng.lines} (singles {c[1]}, doubles {c[2]}, triples {c[3]}, tetrises {c[4]})")


if __name__ == "__main__":
    main()
