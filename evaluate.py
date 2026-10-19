"""Measure a trained network in the fast simulator.

    python evaluate.py --games 20 --pieces 2000
"""
import argparse
from tetris_bot.agent import ValueNet
from train import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/value_net.npz")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--pieces", type=int, default=2000)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--messy", type=int, default=0, help="random placements before each game starts")
    a = ap.parse_args()
    ev = evaluate(ValueNet.load(a.model), a.games, a.pieces, a.depth, seed0=50_000, messy=a.messy)
    print(f"{a.games} games of up to {a.pieces} pieces, search depth {a.depth}, messy start {a.messy}:")
    print(f"  average pieces survived : {ev['pieces']:.0f}")
    print(f"  games ending in top-out : {ev['died'] * 100:.0f}%")
    print(f"  points per piece (lvl 1): {ev['pts_per_piece']:.1f}")
    print(f"  lines per game          : {ev['lines']:.0f}")
    print(f"  lines cleared as Tetris : {ev['tetris_share'] * 100:.0f}%")


if __name__ == "__main__":
    main()
