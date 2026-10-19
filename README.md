# Tetris RL bot for play.tetris.com

A self-learning Tetris agent. It learned to play inside a software replica of the game
(no human strategy coded in), and a screen-reading controller lets it play the real site
with keyboard presses.

```
tetris-rl-bot/
  tetris_bot/
    pieces.py       tetromino shapes, SRS wall kicks, colours, score table
    engine.py       the replica: key-level game (gravity, lock delay, hold, 7-bag, T-spins, B2B, levels)
    fastsim.py      fast "one placement per step" simulator used for training
    agent.py        value network (numpy) + move chooser
    vision.py       reads board / falling piece / HOLD / NEXT from screen pixels
    controller.py   turns a decision into key taps; live loop; screen + keyboard access
    render.py       draws the replica with the website's layout
    mock_site.py    simulated website used by the tests
  train.py          reinforcement learning (self-play)
  evaluate.py       measure a trained network
  watch.py          watch the agent play in the terminal
  replica_gui.py    playable replica window (you or the agent)
  play_web.py       play on https://play.tetris.com/
  models/value_net.npz   trained weights
  tests/            consistency, vision and controller tests
```

## Install

```
pip install -r requirements.txt
```

Python 3.9+. `pygame` is only needed for `replica_gui.py`.
On macOS, give your terminal *Screen Recording* and *Accessibility* permission
(System Settings > Privacy & Security), otherwise it cannot see the screen or press keys.

## Play on the website

1. Open https://play.tetris.com/, get past the ads and start a game.
2. Keep the whole game visible (playfield, HOLD and NEXT), then run

   ```
   python play_web.py --dry-run     # first time: check what the bot sees, presses nothing
   python play_web.py               # play
   ```

3. You get 5 seconds to click on the game so it has keyboard focus.
4. **Esc stops the bot** (and pauses the game). Don't type elsewhere while it runs: it sends real key presses.

`bot_view.png` is written at start-up and shows what the bot recognised: squares on stack
cells, circles on the falling piece, letters on HOLD / NEXT. If that picture is wrong, the bot
will play wrong, so look at it once.

Useful options:

| option | meaning |
| --- | --- |
| `--tap-ms 40` | starting key timing (hold and pause per key). The bot tunes this itself while playing |
| `--no-adapt` | keep the key timing fixed |
| `--no-verify` | skip the on-screen check before each hard drop (faster; lost keys then misplace pieces) |
| `--depth 1` | plan only the current piece (faster, weaker). Default 2 also plans the next piece |
| `--caution 1.0` | how strongly it keeps the stack low as the game speeds up (0 = off) |
| `--start-level N` | if you start the bot in the middle of a game |
| `--board LEFT,TOP,CELL` | playfield position in pixels, if automatic detection fails |
| `--monitor 2` | game is on the second screen |

**Checked placements.** Before every hard drop the bot looks at the screen again and compares
the falling piece with where it should be. If a key press was swallowed by the game it presses
the missing keys (and repeats a swallowed hold or hard drop). When that happens often it slows
its key taps or waits longer after a piece appears; when things go well it speeds up again.
When it stops it prints how many pieces needed extra keys and how many were misplaced, and
`bot_log.txt` lists each case.

Practice run without the website: start `python replica_gui.py` and then `python play_web.py`.
The bot reads that window and presses keys exactly the same way.

## Watch / measure it in the replica

```
python watch.py                         # terminal animation
python watch.py --fast --pieces 3000    # just the result
python replica_gui.py --bot             # window that looks like the site
python evaluate.py                      # statistics over 20 games
```

## How it learns

* **State and action.** One action = one whole placement: optionally hold, rotate, shift, hard
  drop. For every reachable placement the simulator produces the resulting board.
* **Value network.** A small neural network (37 inputs, two hidden layers of 64) estimates how
  much score is still to come from a board *after* a placement. Inputs: the 10 column heights,
  holes, bumpiness, wells, row/column transitions, back-to-back flag, the piece in HOLD and the
  piece that falls next.
* **Choosing.** Score every placement by `reward now + 0.95 * V(resulting board)` and take the
  best. When playing, it also tries the best 8 placements together with every placement of the
  next piece (depth 2).
* **Learning.** Temporal-difference learning from self-play with a replay buffer and a target
  network (n-step, n = 3). Reward is the line-clear score from the site's table
  (100 / 300 / 500 / 800, x1.5 for back-to-back Tetris), in units of 1000 points; topping out
  costs 1. Exploration: random placements, 100% at the start falling to 1%.
* The network starts from random weights. Nothing tells it that holes are bad or that Tetrises
  are good; that comes out of the score.

* **Recovery practice.** In the second half of training, half of the games start from a board
  messed up by a few random placements, so it also learns to dig out after a misplaced piece.

The shipped weights came from 650,000 placements of self-play (about 20 minutes on one CPU core):

```
python train.py --steps 200000 --messy 0 --out models/stage1.npz
python train.py --steps 450000 --init models/stage1.npz --eps-steps 1 --messy 0.5 --out models/value_net.npz
```

(Stage 1 of the shipped weights was a snapshot at 200,000 steps of a longer run, so a rerun
will not give identical numbers.)

## What was measured

Everything below was run in the replica. **It has not been run on the live website** - the
machine it was built on cannot reach play.tetris.com.

| test | result |
| --- | --- |
| Fast simulator, 10 games x 2000 pieces, depth 2 | no top-outs; 116.9 points per piece at level 1; 97% of lines cleared as Tetrises |
| Same, depth 1, 20 games x 3000 pieces | 15% of games topped out; 105.8 points per piece; 85% Tetris lines |
| Start after 6 random placements, depth 2, 10 x 1000 pieces | no top-outs; 95.2 points per piece |
| Key-level replica (`watch.py --fast`), 3 x 1000 pieces | all alive; 92-98 Tetrises per game |
| Simulated website (pixels in, key taps out, game speeding up), first version of the controller, 6 games played to the end | topped out at level 13-18 after 129-175 lines; 240,000-346,000 points |
| Simulated website that only samples the keyboard 30 times a second, 300 pieces | first version: topped out after 22 pieces, 18 misplaced. Now: 0 misplaced, 29 Tetrises |
| Simulated website that ignores input for 120 ms after each new piece, 300 pieces | first version: topped out after 34 pieces. Now: 0 misplaced, 28 Tetrises |
| Simulated website dropping 3% of key presses at random, 200 pieces | first version: 24 misplaced. Now: 0 misplaced |
| Real screen capture + real key presses on the replica window | 120 pieces, 46 lines, 11 Tetrises, none misplaced |
| Reading your two play.tetris.com screenshots (game-only and full 1920x1080 desktop) | board, falling piece, HOLD and NEXT all correct, at several zoom levels |
| Fast simulator vs key-level replica | identical boards and scores over 54,160 placements |

Run the tests yourself: `python tests/test_consistency.py`, `python tests/test_vision.py`,
`python tests/test_controller.py`.

## Limits

* Placements are "rotate, slide, hard drop" only. It does not do T-spins or tucks under
  overhangs (the replica scores T-spins, the agent just never sets them up).
* The reward ignores soft/hard-drop points and the level multiplier; those do not change which
  placement is best.
* Speed. Each key tap takes 40-80 ms plus one look at the screen per piece, so at high levels pieces fall several rows while the
  bot is still moving them. The bot simulates that and keeps the stack lower as the level
  rises, but past roughly level 12 gravity wins.
* The screen reader was built from two screenshots of play.tetris.com (classic skin, the layout
  with HOLD left and NEXT right). A different skin or layout needs new numbers at the top of
  `tetris_bot/vision.py`.
* The bot assumes the game starts at level 1 and gains a level every 10 lines (use `--start-level` otherwise).
