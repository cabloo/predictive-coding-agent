"""Run one agent on one game and report when it mastered the game and how long it held it.

    python -m pcagent.run --game cue --delay 2 --seed 1
    python -m pcagent.run --game two-need --delay 2 --seed 3 --out mine.json

The command line runs the agent on the CPU in one thread. A run that masters continues for 100,000 ticks (or 20
times its time to mastery, if longer) so the hold can be judged; ``--ticks`` caps the run either way. A file written
with ``--out`` has the format of ``results/acceptance.json``, so ``pcagent.report`` and ``tools/make_figure.py``
read it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import torch

from . import scoring
from .agent import Agent
from .worlds import GAMES, anchors


def run(game_name="cue", delay=0, seed=1, ticks=200_000, bodies=32, curve_every=1000, anchor_ticks=3000,
        hold_x=20.0, hold_min=100_000, hindsight=True, log=print):
    """Run one agent on one game. Returns the anchors, the bar, the curve, the tick of mastery (or None) and the hold.

    The curve has one point per ``curve_every`` ticks: the mean score over those ticks. Ticks after the last full
    window are run but not scored. The number of threads is the caller's choice (the command line sets one).
    """
    if game_name not in GAMES:
        raise ValueError("game must be one of %s, got %r" % (sorted(GAMES), game_name))
    if ticks < curve_every:
        raise ValueError("ticks must be at least one window of %d ticks, got %d" % (curve_every, ticks))
    make = lambda B, s: GAMES[game_name](B, seed=s, delay=delay)                    # noqa: E731
    anch = anchors(make, bodies, anchor_ticks)
    bar = scoring.solved_bar(anch)
    log("%s game, delay %d, seed %d: oracle %.4f, floor %.4f, bar %.4f"
        % (game_name, delay, seed, anch["oracle"], anch["floor"], bar))
    game = make(bodies, seed)
    agent = Agent(game.n_extero, game.n_proprio, game.n_intero, seed=seed, hindsight=hindsight)
    curve, hits, n, t0 = [], 0.0, 0, time.monotonic()
    with torch.inference_mode():
        for t in range(ticks):
            cmd = agent.step(game.extero(), game.proprio(), game.intero())
            game.step(cmd)
            hits += game.score()
            n += 1
            if agent.diverged:
                log("stopped at tick %d: the agent's weights are no longer finite" % (t + 1))
                break
            if (t + 1) % curve_every == 0:
                curve.append({"tick": t + 1, "score": hits / n})
                hits, n = 0.0, 0
                log("tick %7d  score %.3f  %s  (%.0f ticks/s)"
                    % (t + 1, curve[-1]["score"], "at or above the bar" if curve[-1]["score"] >= bar else "below",
                       (t + 1) / (time.monotonic() - t0)))
                if scoring.run_until(curve, bar, t + 1, hold_x, hold_min):
                    break
    hold = scoring.hold_ratio(curve, bar)
    if hold is None:
        log("never mastered in %d ticks (bar %.4f)" % (curve[-1]["tick"] if curve else 0, bar))
    else:
        log("mastered at tick %d; held %s%d ticks, R %s%.1f; held: %s"
            % (hold["ttm"], ">= " if hold["censored"] else "", hold["hold"], ">= " if hold["censored"] else "",
               hold["R"], {True: "yes", False: "no", None: "undetermined (the run ended first)"}[hold["held"]]))
    return {"game": game_name, "delay": delay, "seed": seed, "hindsight": hindsight, "anchors": anch, "bar": bar,
            "curve": curve, "ticks_to_mastery": None if hold is None else hold["ttm"], "hold": hold,
            "diverged": agent.diverged}


def as_cell(res):
    """One run in the format of a results file's cell."""
    return {"game": res["game"], "delay": res["delay"], "seed": res["seed"], "hindsight": res["hindsight"],
            "bar": res["bar"], "oracle": res["anchors"]["oracle"], "floor": res["anchors"]["floor"],
            "diverged": res["diverged"], "curve": [[c["tick"], c["score"]] for c in res["curve"]]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--game", default="cue", choices=sorted(GAMES))
    ap.add_argument("--delay", type=int, default=0, help="ticks between the answer and its outcome")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ticks", type=int, default=200_000, help="the most ticks to run (at least 1,000)")
    ap.add_argument("--bodies", type=int, default=32)
    ap.add_argument("--no-hindsight", dest="hindsight", action="store_false",
                    help="the control: memory corrects no weight")
    ap.add_argument("--out", default=None, help="write the run here as a results file (JSON)")
    a = ap.parse_args(argv)
    if a.ticks < 1000:
        ap.error("--ticks must be at least 1000 (one scored window)")
    if a.delay < 0:
        ap.error("--delay must be 0 or more")
    torch.set_num_threads(1)
    res = run(a.game, a.delay, a.seed, a.ticks, a.bodies, hindsight=a.hindsight)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"cells": [as_cell(res)]}, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
