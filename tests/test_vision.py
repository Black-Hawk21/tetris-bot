"""Screen reader checks: the real play.tetris.com screenshot, and rendered replica frames."""
import os, sys, random
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tetris_bot import vision
from tetris_bot.engine import Engine
from tetris_bot.render import render
from tetris_bot.pieces import SHAPES, PIECES, T, O, Z, J, S, I, NONE, HIDDEN

HERE = os.path.dirname(os.path.abspath(__file__))


def test_real_screenshot():
    path = os.path.join(HERE, "tetris_com_screenshot.png")
    img = np.array(Image.open(path).convert("RGB"))
    truth = ["..........", "..........", "..oo......", "..o.......", "..o......."] + [".........."] * 10 + \
            ["...#......", "...#####..", "#.######..", "#.######.#", "#.########"]
    for scale in (1.0, 0.8, 0.66, 0.5, 1.25):
        im = img if scale == 1.0 else np.array(Image.fromarray(img).resize(
            (int(img.shape[1] * scale), int(img.shape[0] * scale)), Image.BILINEAR))
        lay = vision.find_board(im)
        assert lay is not None
        o = vision.observe(im, lay)
        got = ["".join("#" if o.stack[r, q] else "o" if o.filled[r, q] else "." for q in range(10)) for r in range(20)]
        assert got == truth, "\n".join(got)
        assert o.active == (J, 1, 1, 2) and o.hold == T and o.next == [O, Z, Z], o.describe()
    print("real screenshot: board, falling piece, hold and next read correctly at 5 zoom levels")


def test_fullscreen_screenshot():
    """Whole 1920x1080 desktop with the browser: grid lines are lighter here, and the cropped
    capture the bot uses while playing must give the same reading."""
    img = np.array(Image.open(os.path.join(HERE, "tetris_com_fullscreen.png")).convert("RGB"))
    lay = vision.find_board(img)
    assert lay is not None and abs(lay.cell - 40.6) < 0.5, lay and lay.as_dict()
    l, t, r, b = (int(v) for v in lay.bounds())
    crop = img[max(0, t):b, max(0, l):r]
    for im, la in ((img, lay), (crop, vision.find_board(crop))):
        o = vision.observe(im, la)
        assert o.active == (T, 0, 7, 10) and o.hold == NONE and o.next == [S, I, J], o.describe()
        assert sorted(zip(*np.nonzero(o.stack))) == [(18, 5), (19, 3), (19, 4), (19, 5)]
    print("full-screen screenshot: playfield found and read correctly (full capture and cropped capture)")


def test_rendered_frames():
    frames = 0
    for cell in (22, 30, 48):
        lay = vision.find_board(render(Engine(0), cell))
        for seed in range(60):
            rng, e = random.Random(seed), Engine(seed)
            for _ in range(rng.randrange(1, 14)):
                for _ in range(rng.randrange(4)):
                    e.press("up")
                for _ in range(rng.randrange(6)):
                    e.press(rng.choice(["left", "right"]))
                if rng.random() < 0.3:
                    e.press("c")
                e.press("space")
            e.tick(2.3)
            for _ in range(rng.randrange(4)):
                e.press("up")
            if e.over or min(y for _, y in e.cells()) < HIDDEN or any((x, y + 1) in [(q, r) for r, q in zip(*np.nonzero(e.board))] for x, y in e.cells()):
                continue                                   # piece off-screen or resting on the stack: skip
            o = vision.observe(render(e, cell), lay)
            assert np.array_equal(o.stack, e.board[HIDDEN:] > 0), (seed, cell)
            assert o.hold == e.hold and o.next == e.queue, (seed, cell, o.describe())
            p, r, x, y = o.active
            assert p == e.piece and sorted((x + dx, y + dy) for dx, dy in SHAPES[p][r]) == \
                sorted((cx, cy - HIDDEN) for cx, cy in e.cells()), (seed, cell)
            frames += 1
    print(f"rendered replica: {frames} frames read correctly")


if __name__ == "__main__":
    test_real_screenshot()
    test_fullscreen_screenshot()
    test_rendered_frames()
