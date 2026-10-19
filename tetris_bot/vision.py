"""Reads the game from pixels: matrix, falling piece, hold slot and the 3-piece preview.

Everything is located relative to the black 10x20 playfield, so the browser zoom level does not
matter.  The offsets of the HOLD and NEXT panels were measured on a play.tetris.com screenshot.
"""
import numpy as np
from .pieces import W, VISIBLE, HIDDEN, SHAPES, NONE, I, O, T, S, Z, J, L, PIECES

# Panel positions in units of one playfield cell, relative to the playfield's top-left corner.
HOLD_CENTRE = (-4.885, 3.74)
NEXT_CENTRES = ((14.90, 3.74), (14.90, 6.356), (14.90, 8.97))
SLOT_HALF = (2.15, 1.2)

REF_HUE = {I: 186, O: 54, T: 289, S: 135, Z: 359, J: 205, L: 39}     # degrees, measured on the site
_HUES = np.array([REF_HUE[p] for p in range(7)], np.float32)
_PATTERNS = {            # occupancy of a 3x2 preview box, top row then bottom row
    (0, 1, 0, 1, 1, 1): T, (1, 0, 0, 1, 1, 1): J, (0, 0, 1, 1, 1, 1): L,
    (0, 1, 1, 1, 1, 0): S, (1, 1, 0, 0, 1, 1): Z,
}
DARK = 32               # the playfield background (incl. its grid lines) is darker than this


class Layout:
    """Pixel geometry of the playfield inside a captured image."""

    def __init__(self, x0, y0, cell):
        self.x0, self.y0, self.cell = float(x0), float(y0), float(cell)
        frac = np.linspace(0.32, 0.68, 5)                 # sample points inside each cell
        self._ys = np.rint(self.y0 + (np.arange(VISIBLE)[:, None] + frac[None, :]) * self.cell).astype(int)
        self._xs = np.rint(self.x0 + (np.arange(W)[:, None] + frac[None, :]) * self.cell).astype(int)

    def as_dict(self):
        return {"x0": self.x0, "y0": self.y0, "cell": self.cell}

    def bounds(self):
        """Smallest box (left, top, right, bottom) containing playfield, HOLD and NEXT panels."""
        c = self.cell
        return (self.x0 + (HOLD_CENTRE[0] - SLOT_HALF[0] - 0.3) * c, self.y0 - 0.3 * c,
                self.x0 + (NEXT_CENTRES[0][0] + SLOT_HALF[0] + 0.3) * c, self.y0 + 20.3 * c)

    def fits(self, shape):
        l, t, r, b = self.bounds()
        return l >= 0 and t >= 0 and r < shape[1] and b < shape[0]


def _hue_deg(rgb):
    """Hue in degrees for an (..., 3) float array."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.max(-1), rgb.min(-1)
    d = np.where(mx - mn == 0, 1, mx - mn)
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4))
    return h * 60.0


def _nearest_piece(hue, allowed=None):
    diff = np.abs(hue - _HUES)
    diff = np.minimum(diff, 360 - diff)
    if allowed is not None:
        diff = np.where(np.isin(np.arange(7), allowed), diff, 1e9)
    return int(np.argmin(diff))


def _dark_boxes(img, thresh):
    """Bounding boxes (area, x0, y0, x1, y1) of dark regions shaped like the playfield (1 : 2)."""
    Himg, Wimg = img.shape[:2]
    k = max(1, max(Himg, Wimg) // 420)
    mask = img[::k, ::k].max(2) < thresh
    h, w = mask.shape
    seen = np.zeros_like(mask)
    found = []
    for sy, sx in zip(*np.nonzero(mask[::4, ::4])):       # seed points for flood fill
        sy, sx = int(sy) * 4, int(sx) * 4
        if seen[sy, sx]:
            continue
        stack, area = [(sy, sx)], 0
        seen[sy, sx] = True
        y0 = y1 = sy
        x0 = x1 = sx
        while stack:
            y, x = stack.pop()
            area += 1
            y0, y1, x0, x1 = min(y0, y), max(y1, y), min(x0, x), max(x1, x)
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        if bw < 20 or not 1.6 <= bh / bw <= 2.4 or area < 0.4 * bw * bh:
            continue
        found.append((area, x0 * k, y0 * k, min(Wimg - 1, x1 * k + k - 1), min(Himg - 1, y1 * k + k - 1)))
    return sorted(found, reverse=True)


def _comb(profile, start, lo, cell, n_lines):
    """How well a comb of grid lines (first cell edge at `start`, spacing `cell`) fits a brightness
    profile: brightness on the lines minus brightness in the cell centres."""
    k = np.arange(1, n_lines + 1)
    lines = np.rint(start + k * cell - lo).astype(int)
    mids = np.rint(start + (np.arange(n_lines + 1) + 0.5) * cell - lo).astype(int)
    if lines.min() < 0 or mids.min() < 0 or max(lines.max(), mids.max()) >= len(profile):
        return -1e9
    return float(profile[lines].mean() - profile[mids].mean())


def _fit_grid(img, box):
    """Pin down the playfield inside a rough dark box by lining up its faint grid lines.
    Returns (contrast, Layout) or None.  The contrast tells a real playfield from any other
    dark rectangle on screen (a terminal window, say)."""
    Himg, Wimg = img.shape[:2]
    _, bx0, by0, bx1, by1 = box
    bw, bh = bx1 - bx0 + 1, by1 - by0 + 1
    guess = bw / W
    ya, yb = by0 + int(0.06 * bh), by0 + int(0.6 * bh)       # upper part: mostly empty cells
    lo = max(0, bx0 - 6)
    prof = np.median(img[ya:yb, lo:min(Wimg, bx1 + 7)].max(2), axis=0).astype(np.float32)
    if len(prof) < 30:
        return None
    wide = np.maximum(prof, np.maximum(np.roll(prof, 1), np.roll(prof, -1)))    # widen thin lines
    best = (-1e9, 0.0, guess)
    for cell in np.arange(0.84 * guess, 1.03 * guess, 0.1):
        for off in np.arange(-2.0, 0.5 * guess, 0.5):
            if bx0 + off + W * cell > bx1 + 3:
                break
            sc = _comb(wide, bx0 + off, lo, cell, W - 1)
            if sc > best[0]:
                best = (sc, off, cell)
    cx, off0, cell0 = best
    best = (-1e9, off0, cell0)                       # second pass on the sharp profile: exact fit
    for cell in np.arange(cell0 - 0.6, cell0 + 0.6, 0.05):
        for off in np.arange(off0 - 3.0, off0 + 3.0, 0.5):
            sc = _comb(prof, bx0 + off, lo, cell, W - 1)
            if sc > best[0]:
                best = (sc, off, cell)
    _, off, cell = best
    x0 = bx0 + off
    ylo = max(0, by0 - 6)
    xa, xb = int(x0 + 0.3 * cell), int(x0 + 9.7 * cell)
    rprof = np.median(img[ylo:yb, xa:xb].max(2), axis=1).astype(np.float32)
    rprof = np.maximum(rprof, np.maximum(np.roll(rprof, 1), np.roll(rprof, -1)))
    n = int((yb - by0) / cell) - 2
    if n < 3:
        return None
    best = (-1e9, 0.0)
    for off in np.arange(-2.0, 0.5 * cell, 0.5):
        sc = _comb(rprof, by0 + off, ylo, cell, n)
        if sc > best[0]:
            best = (sc, off)
    cy, off = best
    return min(cx, cy), Layout(x0, by0 + off, cell)


def find_board(img):
    """Locate the playfield: a dark rectangle twice as tall as wide with a 10 x 20 grid in it."""
    fits = [f for f in (_fit_grid(img, b) for b in _dark_boxes(img, 48)[:6]) if f and f[0] >= 6]
    if fits:
        return max(fits, key=lambda f: f[0])[1]
    return _find_board_plain(img)


def _find_board_plain(img):
    """Fallback for a playfield without visible grid lines: edges of the pure-black area."""
    Himg, Wimg = img.shape[:2]
    boxes = _dark_boxes(img, DARK)
    if not boxes:
        return None
    _, x0, y0, x1, y1 = boxes[0]
    pad = 3 * max(1, max(Himg, Wimg) // 420) + 3
    ya, yb = y0 + (y1 - y0) // 20, y0 + int((y1 - y0) * 0.4)           # upper part: mostly empty
    xa, xb = max(0, x0 - pad), min(Wimg, x1 + pad + 1)
    run = _longest_run((img[ya:yb, xa:xb].max(2) < DARK).mean(0) > 0.35)
    if run is None:
        return None
    left, right = xa + run[0], xa + run[1]
    yt = max(0, y0 - pad)
    run = _longest_run((img[yt:yb, left:right + 1].max(2) < DARK).mean(1) > 0.35)
    if run is None:
        return None
    return Layout(left, yt + run[0], (right - left + 1) / W)


def _longest_run(flags):
    """(first, last) index of the longest stretch of True values.  The playfield has a thin dark
    rim just outside its border; taking the longest run skips it."""
    best, start = None, None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            if best is None or i - start > best[1] - best[0] + 1:
                best = (start, i - 1)
            start = None
    return best


def read_cells(img, lay):
    """Returns (filled, colour): 20x10 bool and 20x10 piece-colour id (-1 where empty)."""
    patch = img[lay._ys[:, None, :, None], lay._xs[None, :, None, :]].astype(np.float32)   # 20,10,5,5,3
    rgb = patch.reshape(VISIBLE, W, -1, 3).mean(2)
    mx, mn = rgb.max(-1), rgb.min(-1)
    filled = mx > 90                      # the ghost piece is only an outline, its centre stays black
    hue = _hue_deg(rgb)
    diff = np.abs(hue[..., None] - _HUES)
    colour = np.argmin(np.minimum(diff, 360 - diff), axis=-1).astype(np.int8)
    colour[(mx - mn) < 40] = 7            # greyish / white cell (flash animation, garbage)
    colour[~filled] = -1
    return filled, colour


def read_slot(img, lay, centre):
    """Which tetromino is drawn in a HOLD / NEXT slot (NONE if the slot is empty)."""
    c = lay.cell
    xa, xb = int(lay.x0 + (centre[0] - SLOT_HALF[0]) * c), int(lay.x0 + (centre[0] + SLOT_HALF[0]) * c)
    ya, yb = int(lay.y0 + (centre[1] - SLOT_HALF[1]) * c), int(lay.y0 + (centre[1] + SLOT_HALF[1]) * c)
    if xa < 0 or ya < 0 or xb > img.shape[1] or yb > img.shape[0]:
        return NONE
    step = max(1, int(c) // 16)
    reg = img[ya:yb:step, xa:xb:step].astype(np.float32)
    mx, mn = reg.max(2), reg.min(2)
    mask = (mx > 110) & (mx - mn > 70)
    if mask.mean() < 0.06:
        return NONE
    xs = np.nonzero(mask.mean(0) > 0.08)[0]
    ys = np.nonzero(mask.mean(1) > 0.04)[0]
    if len(xs) == 0 or len(ys) == 0:
        return NONE
    bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    hue = float(np.median(_hue_deg(reg[mask])))
    if bw / bh > 2.8:
        return I
    if bw / bh < 1.25:
        return O
    box = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    occ = tuple(int(box[r * bh // 2:(r + 1) * bh // 2, q * bw // 3:(q + 1) * bw // 3].mean() > 0.5)
                for r in range(2) for q in range(3))
    if occ in _PATTERNS:
        return _PATTERNS[occ]
    return _nearest_piece(hue, allowed=[T, S, Z, J, L])


_SHAPE_LOOKUP = {}
for _p in range(7):
    for _r in (0, 1, 3, 2):
        cells = SHAPES[_p][_r]
        mx_, my_ = min(c[0] for c in cells), min(c[1] for c in cells)
        key = frozenset((c[0] - mx_, c[1] - my_) for c in cells)
        _SHAPE_LOOKUP.setdefault(key, (_p, _r, mx_, my_))


def match_piece(cells):
    """cells: iterable of (x, y). Returns (piece, rot, box_x, box_y) if they form a tetromino."""
    cells = list(cells)
    if len(cells) != 4:
        return None
    mx_, my_ = min(c[0] for c in cells), min(c[1] for c in cells)
    hit = _SHAPE_LOOKUP.get(frozenset((x - mx_, y - my_) for x, y in cells))
    if hit is None:
        return None
    p, r, ox, oy = hit
    return p, r, mx_ - ox, my_ - oy


def find_active(filled, colour, expected_stack=None):
    """Find the falling piece. Returns ((piece, rot, box_x, box_y_visible), cells) or None.

    With `expected_stack` (what the matrix should look like after our last move) the falling
    piece is simply "whatever is on screen that is not part of the stack".  Without it, the piece
    is the 4 same-coloured connected cells at the very top that still have room to fall.
    """
    if expected_stack is not None:
        if (expected_stack & ~filled).any():
            return None
        ys, xs = np.nonzero(filled & ~expected_stack)
        cells = list(zip(xs.tolist(), ys.tolist()))
        m = match_piece(cells)
        return (m, cells) if m else None
    ys, xs = np.nonzero(filled)
    if len(ys) == 0:
        return None
    sy, sx = int(ys[0]), int(xs[0])                       # top-most, then left-most filled cell
    col = colour[sy, sx]
    comp, stack = {(sx, sy)}, [(sx, sy)]
    while stack and len(comp) <= 4:
        x, y = stack.pop()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < W and 0 <= ny < VISIBLE and (nx, ny) not in comp and colour[ny, nx] == col:
                comp.add((nx, ny))
                stack.append((nx, ny))
    m = match_piece(comp)
    if m is None:
        return None
    for x, y in comp:                                     # a piece lying on the stack is not falling
        if y + 1 >= VISIBLE or (filled[y + 1, x] and (x, y + 1) not in comp):
            return None
    return m, list(comp)


class Observation:
    __slots__ = ("stack", "active", "hold", "next", "filled")

    def describe(self):
        a = "none" if self.active is None else f"{PIECES[self.active[0]]} rot{self.active[1]} x{self.active[2]} y{self.active[3]}"
        names = lambda ps: " ".join("-" if p == NONE else PIECES[p] for p in ps)
        return f"falling: {a} | hold: {names([self.hold])} | next: {names(self.next)} | stack cells: {int(self.stack.sum())}"


def observe(img, lay, expected_stack=None):
    """Full reading of one frame. `active` is None if no falling piece can be identified."""
    o = Observation()
    filled, colour = read_cells(img, lay)
    o.filled = filled
    found = find_active(filled, colour, expected_stack)
    o.stack = filled.copy()
    o.active = None
    if found:
        o.active = found[0]
        for x, y in found[1]:
            o.stack[y, x] = False
    o.hold = read_slot(img, lay, HOLD_CENTRE)
    o.next = [read_slot(img, lay, c) for c in NEXT_CENTRES]
    return o


def draw_overlay(img, lay, obs):
    """Debug picture: what the bot thinks it sees, drawn over the capture."""
    from PIL import Image, ImageDraw
    pic = Image.fromarray(img.astype(np.uint8)).convert("RGB")
    d = ImageDraw.Draw(pic)
    c = lay.cell
    for r in range(VISIBLE):
        for q in range(W):
            box = [lay.x0 + q * c + c * 0.3, lay.y0 + r * c + c * 0.3, lay.x0 + q * c + c * 0.7, lay.y0 + r * c + c * 0.7]
            if obs.stack[r, q]:
                d.rectangle(box, outline=(255, 255, 255), width=3)
            elif obs.filled[r, q]:
                d.ellipse(box, outline=(255, 255, 255), width=3)
    d.rectangle([lay.x0, lay.y0, lay.x0 + 10 * c, lay.y0 + 20 * c], outline=(255, 0, 255), width=2)
    for centre, p in [(HOLD_CENTRE, obs.hold)] + list(zip(NEXT_CENTRES, obs.next)):
        box = [lay.x0 + (centre[0] - SLOT_HALF[0]) * c, lay.y0 + (centre[1] - SLOT_HALF[1]) * c,
               lay.x0 + (centre[0] + SLOT_HALF[0]) * c, lay.y0 + (centre[1] + SLOT_HALF[1]) * c]
        d.rectangle(box, outline=(255, 0, 255), width=2)
        d.text((box[0] + 4, box[1] + 2), "-" if p == NONE else PIECES[p], fill=(255, 255, 255))
    return pic
