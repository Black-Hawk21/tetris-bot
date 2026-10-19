"""Playable replica window that looks like play.tetris.com (same layout, colours and keys).

    python replica_gui.py            # play it yourself
    python replica_gui.py --bot      # watch the trained agent play it

It is also a safe practice target for the screen bot: start this window, then run
`python play_web.py` and the bot will read this window and press keys exactly as on the website.

Keys: Left/Right move, Up rotate right, Z rotate left, Down soft drop, Space hard drop, C hold,
Esc pause, R restart.
"""
import argparse, json, time
import pygame
from tetris_bot.engine import Engine
from tetris_bot.pieces import HIDDEN
from tetris_bot.render import render, CANVAS_W, CANVAS_H

KEYS = {pygame.K_LEFT: "left", pygame.K_RIGHT: "right", pygame.K_UP: "up", pygame.K_DOWN: "down",
        pygame.K_SPACE: "space", pygame.K_z: "z", pygame.K_c: "c"}
CLEAR_ANIM = 0.35          # seconds a cleared line flashes; input is ignored meanwhile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=int, default=30, help="size of one block in pixels")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--level", type=int, default=None, help="pin the level (gravity speed)")
    ap.add_argument("--bot", action="store_true", help="let the trained agent play")
    ap.add_argument("--model", default="models/value_net.npz")
    ap.add_argument("--guideline-spawn", action="store_true",
                    help="pieces appear above the visible rows (default: at the top row, as on the site)")
    ap.add_argument("--stats-file", default=None, help="write score/lines here as JSON (used by tests)")
    a = ap.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((int(CANVAS_W * a.cell), int(CANVAS_H * a.cell)))
    pygame.display.set_caption("Tetris replica")
    clock = pygame.time.Clock()
    spawn_y = None if a.guideline_spawn else HIDDEN
    eng = Engine(a.seed, spawn_y)
    net = None
    if a.bot:
        from tetris_bot.agent import ValueNet, engine_keys
        net = ValueNet.load(a.model)
    paused, anim_until, flash, plan, last_stats = False, 0.0, None, [], 0.0
    while True:
        dt = clock.tick(60) / 1000.0
        now = time.perf_counter()
        busy = now < anim_until
        presses = []
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    paused = not paused
                elif ev.key == pygame.K_r:
                    eng, plan = Engine(a.seed, spawn_y), []
                elif ev.key in KEYS and not a.bot:
                    presses.append(KEYS[ev.key])
        if a.bot and not paused and not busy and not eng.over:
            if not plan:
                plan = engine_keys(net, eng)
            presses.append(plan.pop(0))
        if not paused and not busy:
            n = eng.pieces
            if a.level:
                eng.level = a.level
            for k in presses:
                eng.press(k)
            eng.tick(dt)
            if eng.pieces != n:
                plan = []
                if eng.last_lock["rows"]:
                    anim_until, flash = now + CLEAR_ANIM, (eng.last_lock["board"], eng.last_lock["rows"])
        busy = now < anim_until
        frame = render(eng, a.cell, flash=flash if busy else None, show_piece=not busy)
        screen.blit(pygame.image.frombuffer(frame.tobytes(), frame.shape[1::-1], "RGB"), (0, 0))
        pygame.display.flip()
        if a.stats_file and now - last_stats > 0.5:
            last_stats = now
            with open(a.stats_file, "w") as f:
                json.dump({"score": eng.score, "lines": eng.lines, "level": eng.level, "pieces": eng.pieces,
                           "tetrises": eng.clear_count[4], "over": eng.over}, f)


if __name__ == "__main__":
    main()
