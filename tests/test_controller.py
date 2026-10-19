"""End-to-end test of the website controller against the mock site (pixels in, key taps out)."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tetris_bot.agent import ValueNet
from tetris_bot.controller import Bot
from tetris_bot.mock_site import MockSiteIO
from tetris_bot.render import render
from tetris_bot.engine import Engine
from tetris_bot import vision

MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "value_net.npz")


def run(name, pieces=300, bot_kw=None, **mock_kw):
    net = ValueNet.load(os.environ.get("MODEL", MODEL))
    io = MockSiteIO(**mock_kw)
    lay = vision.find_board(render(Engine(0), io.cell))
    bot = Bot(net, io, lay, depth=2, verbose=False, stop=lambda: io.eng.over, **(bot_kw or {}))
    t = time.time()
    bot.run(max_pieces=pieces)
    e = io.eng
    holes = int(((e.board > 0).cumsum(0) > 0).sum() - (e.board > 0).sum())
    print(f"{name:44s} pieces {e.pieces:4d} lines {e.lines:4d} tetrises {e.clear_count[4]:3d} level {e.level:2d} "
          f"score {e.score:7d} | over {str(e.over):5s} | misplaced {bot.resyncs:3d} fixed-before-drop {bot.corrected:3d} "
          f"| keys lost {io.keys_ignored:4d}/{io.keys_sent:4d} | tap {bot.tap_hold * 1000:2.0f} ms, lead-in {bot.lead_in * 1000:3.0f} ms "
          f"| {io.t / max(e.pieces, 1):.2f} s/piece | holes at end {holes}", flush=True)
    return e, bot


OLD = dict(verify=False, adapt=False, tap_hold=0.025, tap_gap=0.025)      # how the first version behaved

if __name__ == "__main__":
    site = dict(visible_spawn=True)
    e, bot = run("ideal input", seed=1, **site)
    assert not e.over and bot.resyncs == 0
    e, bot = run("ideal input, pieces spawn off-screen", seed=1)
    assert not e.over and bot.resyncs == 0
    e, bot = run("small window (cell = 20 px)", seed=5, cell=20, **site)
    assert not e.over and bot.resyncs == 0
    print("-- game samples the keyboard 30x per second (short taps get lost) --")
    run("  first version (blind 25 ms taps)", seed=2, bot_kw=OLD, poll_fps=30, **site)
    e, bot = run("  now (check the screen, adapt timing)", seed=2, poll_fps=30, **site)
    assert not e.over and bot.resyncs <= 3
    print("-- game samples the keyboard 20x per second --")
    run("  first version", seed=6, bot_kw=OLD, poll_fps=20, **site)
    e, bot = run("  now", pieces=200, seed=6, poll_fps=20, **site)
    assert not e.over and bot.resyncs <= 3
    print("-- game ignores input for 120 ms after each new piece --")
    run("  first version", seed=3, bot_kw=OLD, dead_time=0.12, **site)
    e, bot = run("  now", seed=3, dead_time=0.12, **site)
    assert not e.over and bot.resyncs <= 3
    print("-- 3% of all key presses vanish at random --")
    run("  first version", seed=4, bot_kw=OLD, drop_key_prob=0.03, **site)
    e, bot = run("  now", pieces=200, seed=4, drop_key_prob=0.03, **site)
    assert not e.over and bot.resyncs <= 5
    print("controller OK")
