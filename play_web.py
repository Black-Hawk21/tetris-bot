"""Let the trained agent play on https://play.tetris.com/ with real key presses.

    1. Open the site in your browser, start a game, and leave the game visible on screen.
    2. python play_web.py            # you get a few seconds to click on the game window
    3. Press Esc to stop the bot (Esc also pauses the game).

    python play_web.py --dry-run     # only show what the bot sees; press nothing

The bot reads the screen (board, falling piece, HOLD, NEXT), picks a placement with the trained
value network, and taps: C (hold), Up / Z (rotate), Left / Right, Space (hard drop).
Before each hard drop it looks at the screen again to make sure the piece is where it should be.
"""
import argparse, sys, time
import numpy as np
from tetris_bot import vision
from tetris_bot.agent import ValueNet
from tetris_bot.controller import Bot, ScreenIO


def locate(io, manual):
    """Find the playfield on screen, then narrow screen capture down to the game area."""
    full = io.grab_full()
    if manual:
        x0, y0, cell = (float(v) for v in manual.split(","))
        lay = vision.Layout(x0, y0, cell)
    else:
        lay = vision.find_board(full)
    if lay is None:
        from PIL import Image
        Image.fromarray(full).save("screen_debug.png")
        sys.exit("Could not find the black 10x20 playfield on screen. Make sure the game is fully visible and "
                 "a game is running (best right after it starts). Saved what I saw to screen_debug.png.\n"
                 "You can also pass the playfield position by hand: --board LEFT,TOP,CELLSIZE (in pixels).")
    off = io.focus_on(lay.bounds(), full.shape)
    img = io.grab()
    lay = vision.find_board(img) or vision.Layout(lay.x0 - off[0], lay.y0 - off[1], lay.cell)
    return lay, img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/value_net.npz")
    ap.add_argument("--delay", type=float, default=5, help="seconds to wait before starting")
    ap.add_argument("--depth", type=int, default=2, help="1 = current piece only, 2 = also plan the next piece")
    ap.add_argument("--tap-ms", type=float, default=40,
                    help="starting key timing: each key is held this long, then released this long")
    ap.add_argument("--no-adapt", action="store_true", help="keep the key timing fixed instead of self-tuning")
    ap.add_argument("--no-verify", action="store_true",
                    help="do not check the piece on screen before dropping (faster, but lost keys misplace pieces)")
    ap.add_argument("--caution", type=float, default=1.0,
                    help="how much lower to keep the stack as the game speeds up (0 = never adapt)")
    ap.add_argument("--start-level", type=int, default=1,
                    help="level shown on the site when the bot starts (it counts lines itself after that)")
    ap.add_argument("--monitor", type=int, default=1, help="which screen (1 = primary)")
    ap.add_argument("--board", default=None, help="manual playfield position: LEFT,TOP,CELLSIZE in pixels")
    ap.add_argument("--max-pieces", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="show what the bot sees, press no keys")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    net = ValueNet.load(a.model)
    io = ScreenIO(a.monitor)
    print(f"Click on the game now. Starting in {a.delay:.0f} s ...", flush=True)
    time.sleep(a.delay)
    lay, img = locate(io, a.board)
    obs = vision.observe(img, lay)
    vision.draw_overlay(img, lay, obs).save("bot_view.png")
    print(f"Playfield found: one block = {lay.cell:.1f} px. What I see -> {obs.describe()}")
    print("Saved bot_view.png (squares = stack, circles = falling piece, letters = hold / next).")
    if a.dry_run:
        return

    stopped = []
    from pynput import keyboard
    listener = keyboard.Listener(on_press=lambda k: stopped.append(1) if k == keyboard.Key.esc else None)
    listener.start()
    print("Playing. Press Esc to stop.", flush=True)
    bot = Bot(net, io, lay, depth=a.depth, tap_hold=a.tap_ms / 1000, tap_gap=a.tap_ms / 1000,
              verify=not a.no_verify, adapt=not a.no_adapt, caution=a.caution, verbose=not a.quiet,
              stop=lambda: bool(stopped), log_path="bot_log.txt")
    bot.lines = 10 * (a.start_level - 1)
    try:
        bot.run(a.max_pieces)
    except KeyboardInterrupt:
        pass
    print("Stopped. " + bot.summary())
    print("Details of every corrected or misplaced piece are in bot_log.txt.")


if __name__ == "__main__":
    main()
