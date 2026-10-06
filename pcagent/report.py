"""Summarize a results file as the two tables in the README.

    python -m pcagent.report results/acceptance.json

A results file is ``{"cells": [{"game", "delay", "seed", "hindsight", "bar", "oracle", "floor", "diverged",
"curve": [[tick, score], ...]}, ...]}``.
"""
from __future__ import annotations

import json
import sys

from . import scoring


def _curve(cell):
    return [{"tick": t, "score": s} for t, s in cell["curve"]]


def main_table(cells):
    """One row per game and delay, over the seeds, for the runs with hindsight on."""
    rows = ["| Game | Delay (ticks) | Pass bar | Mastered at tick, by seed | Hold ratio R, lowest seed "
            "| Windows below the bar after mastery | Lowest window after mastery |",
            "|---|---|---|---|---|---|---|"]
    keys = []
    for c in cells:
        if c["hindsight"] and (c["game"], c["delay"]) not in keys:
            keys.append((c["game"], c["delay"]))
    for game, delay in keys:
        group = sorted((c for c in cells if c["hindsight"] and (c["game"], c["delay"]) == (game, delay)),
                       key=lambda c: c["seed"])
        holds = [scoring.hold_ratio(_curve(c), c["bar"]) for c in group]
        if any(h is None for h in holds):
            rows.append("| %s | %d | %.3f | not every seed mastered | | | |" % (game, delay, group[0]["bar"]))
            continue
        after = [[s for t, s in c["curve"] if t >= h["ttm"]] for c, h in zip(group, holds)]
        worst = min(holds, key=lambda h: h["R"])
        rows.append("| %s | %d | %.3f | %s | %s%.1f | %d of %d | %.3f |" % (
            game, delay, group[0]["bar"], ", ".join("{:,}".format(h["ttm"]) for h in holds),
            "≥ " if worst["censored"] else "", worst["R"],
            sum(x < c["bar"] for c, a in zip(group, after) for x in a), sum(len(a) for a in after),
            min(min(a) for a in after)))
    return "\n".join(rows)


def control_table(cells):
    """One row per control run (hindsight off)."""
    rows = ["| Game | Delay (ticks) | Pass bar | Best window | Mean score | Ran for (ticks) | Outcome |",
            "|---|---|---|---|---|---|---|"]
    for c in cells:
        if c["hindsight"]:
            continue
        scores = [s for _, s in c["curve"]]
        if not scores:
            rows.append("| %s | %d | %.3f | | | 0 | no full window was scored |" % (c["game"], c["delay"], c["bar"]))
            continue
        mastered = scoring.ticks_to_mastery(_curve(c), c["bar"]) is not None
        rows.append("| %s | %d | %.3f | %.3f | %.3f | %s | %s |" % (
            c["game"], c["delay"], c["bar"], max(scores), sum(scores) / len(scores),
            "{:,}".format(c["curve"][-1][0]),
            "mastered" if mastered else ("never mastered; weights became non-finite" if c["diverged"]
                                         else "never mastered")))
    return "\n".join(rows)


def summary(cells):
    """Counts over the runs with hindsight on."""
    on = [c for c in cells if c["hindsight"]]
    holds = [scoring.hold_ratio(_curve(c), c["bar"]) for c in on]
    ttm = sorted(h["ttm"] for h in holds if h is not None)
    return {"runs": len(on), "mastered": len(ttm), "held": sum(bool(h and h["held"]) for h in holds),
            "sustained_losses": sum(bool(h and not h["censored"]) for h in holds),
            "fastest": ttm[0] if ttm else None, "slowest": ttm[-1] if ttm else None,
            "median": ttm[len(ttm) // 2] if ttm else None}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__)
        return 2
    with open(argv[0]) as fh:
        cells = json.load(fh)["cells"]
    print(main_table(cells))
    print()
    print(control_table(cells))
    print()
    print(summary(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
