"""Draws the replica with the same layout and colours as play.tetris.com.

Used by the playable window (`replica_gui.py`) and by the offline test of the screen reader, so
the bot can be tried against something that looks like the website before touching the website.
"""
import numpy as np
from PIL import Image, ImageDraw
from .pieces import W, VISIBLE, HIDDEN, SHAPES, COLORS, NONE
from .vision import HOLD_CENTRE, NEXT_CENTRES

BOARD_X, BOARD_Y = 9.0, 1.0          # playfield origin in cells
CANVAS_W, CANVAS_H = 28.0, 22.0      # canvas size in cells
PANEL, BLACK, GRID, BG = (36, 36, 36), (0, 0, 0), (27, 26, 26), (232, 232, 232)


def _block(d, x, y, s, col):
    d.rectangle([x + 1, y + 1, x + s - 2, y + s - 2], fill=col)
    hi = tuple(min(255, int(v * 1.25 + 40)) for v in col)
    d.rectangle([x + s * 0.15, y + s * 0.08, x + s * 0.85, y + s * 0.2], fill=hi)


def _preview(d, piece, centre, cell):
    if piece == NONE:
        return
    s = cell * 0.96
    cells = SHAPES[piece][0]
    xs, ys = [c[0] for c in cells], [c[1] for c in cells]
    w, h = max(xs) - min(xs) + 1, max(ys) - min(ys) + 1
    ox = (BOARD_X + centre[0]) * cell - w * s / 2
    oy = (BOARD_Y + centre[1]) * cell - h * s / 2
    for cx, cy in cells:
        _block(d, ox + (cx - min(xs)) * s, oy + (cy - min(ys)) * s, s, COLORS[piece])


def render(eng, cell=30, flash=None, show_piece=True):
    """RGB frame (numpy uint8) of an Engine. `flash` = (board, rows) draws a line-clear flash."""
    img = Image.new("RGB", (int(CANVAS_W * cell), int(CANVAS_H * cell)), BG)
    d = ImageDraw.Draw(img)
    bx, by = BOARD_X * cell, BOARD_Y * cell
    rim = max(3, cell // 5)
    d.rectangle([bx - 2 * rim, by - 2 * rim, bx + W * cell + 2 * rim - 1, by + VISIBLE * cell + 2 * rim - 1], fill=(120, 120, 120))
    d.rectangle([bx - rim, by - rim, bx + W * cell + rim - 1, by + VISIBLE * cell + rim - 1], fill=PANEL)
    d.rectangle([bx, by, bx + W * cell - 1, by + VISIBLE * cell - 1], fill=BLACK)
    for q in range(1, W):
        d.line([bx + q * cell, by, bx + q * cell, by + VISIBLE * cell - 1], fill=GRID)
    for r in range(1, VISIBLE):
        d.line([bx, by + r * cell, bx + W * cell - 1, by + r * cell], fill=GRID)
    board = eng.board if flash is None else flash[0]
    for r in range(HIDDEN, HIDDEN + VISIBLE):
        for q in range(W):
            v = board[r, q]
            if v:
                col = (255, 255, 255) if (flash and r in flash[1]) else COLORS[v - 1]
                _block(d, bx + q * cell, by + (r - HIDDEN) * cell, cell, col)
    if show_piece and flash is None and not eng.over:
        col = COLORS[eng.piece]
        dim = tuple(v // 3 for v in col)
        for cx, cy in eng.cells(y=eng.ghost_y()):
            if cy >= HIDDEN:
                x, y = bx + cx * cell, by + (cy - HIDDEN) * cell
                d.rectangle([x + 1, y + 1, x + cell - 2, y + cell - 2], outline=dim)
        for cx, cy in eng.cells():
            if cy >= HIDDEN:
                _block(d, bx + cx * cell, by + (cy - HIDDEN) * cell, cell, col)
    # HOLD and NEXT panels
    for cx, y0, y1, title in ((HOLD_CENTRE[0], 1.2, 5.6, "HOLD"), (NEXT_CENTRES[0][0], 1.2, 10.6, "NEXT")):
        d.rectangle([(BOARD_X + cx - 2.5) * cell, (BOARD_Y + y0) * cell, (BOARD_X + cx + 2.5) * cell, (BOARD_Y + y1) * cell], fill=PANEL)
        d.text(((BOARD_X + cx - 0.6) * cell, (BOARD_Y + y0 + 0.2) * cell), title, fill=(255, 255, 255))
    _preview(d, eng.hold, HOLD_CENTRE, cell)
    for p, c in zip(eng.queue, NEXT_CENTRES):
        _preview(d, p, c, cell)
    # score panel
    px, py = (BOARD_X - 7.4) * cell, (BOARD_Y + 12) * cell
    d.rectangle([px, py, px + 5 * cell, py + 7 * cell], fill=PANEL)
    for k, (name, val) in enumerate((("SCORE", eng.score), ("LEVEL", eng.level), ("LINES", eng.lines))):
        d.text((px + 0.5 * cell, py + (0.4 + 2.1 * k) * cell), name, fill=(255, 255, 255))
        d.text((px + 0.5 * cell, py + (1.2 + 2.1 * k) * cell), str(val), fill=(255, 255, 255))
    if eng.over:
        d.text((bx + 3 * cell, by - 0.9 * cell), "GAME OVER", fill=(200, 0, 0))
    return np.asarray(img)
