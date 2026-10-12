"""Summarize a results file as the tables in docs/results.md.

    python -m pcagent.report results/acceptance.json

``summary_table`` gives the README's short table.

A results file is ``{"cells": [{"game", "delay", "seed", "hindsight", "information", "bar", "oracle", "floor",
"diverged", "curve": [[tick, score], ...]}, ...]}``. A run is one of three kinds: the agent as published
(``hindsight`` and ``information`` both true), the comparison (``information`` false: the information term casts no
vote) and the control (``hindsight`` false: memory corrects no weight).
"""
from __future__ import annotations

import json
import sys

from . import scoring


def _curve(cell):
    return [{"tick": t, "score": s} for t, s in cell["curve"]]


def _runs(cells, information=True):
    """The runs with hindsight on, with or without the information term."""
    return [c for c in cells if c["hindsight"] and c.get("information", True) == information]


def _groups(cells):
    """The runs grouped by game and delay, in file order, each group sorted by seed."""
    keys = []
    for c in cells:
        if (c["game"], c["delay"]) not in keys:
            keys.append((c["game"], c["delay"]))
    return [(k, sorted((c for c in cells if (c["game"], c["delay"]) == k), key=lambda c: c["seed"])) for k in keys]


def _after_mastery(cell, hold):
    return [s for t, s in cell["curve"] if t >= hold["ttm"]]


def outside_before_mastery(cell, hold):
    """Ticks a body spent outside the band up to mastery: one minus the score, summed over the windows up to the
    one mastery is dated to. It is finer than the time to mastery, which is counted in whole windows."""
    out, last = 0.0, 0
    for t, s in cell["curve"]:
        if t > hold["ttm"]:
            break
        out += (1.0 - s) * (t - last)
        last = t
    return out


def main_table(cells, information=True):
    """One row per game and delay, over the seeds, for the runs with hindsight on."""
    rows = ["| Game | Delay (ticks) | Pass bar | Mastered at tick, by seed | Hold ratio R, lowest seed "
            "| Windows below the bar after mastery | Lowest window after mastery |",
            "|---|---|---|---|---|---|---|"]
    for (game, delay), group in _groups(_runs(cells, information)):
        holds = [scoring.hold_ratio(_curve(c), c["bar"]) for c in group]
        if any(h is None for h in holds):
            rows.append("| %s | %d | %.3f | not every seed mastered | | | |" % (game, delay, group[0]["bar"]))
            continue
        after = [_after_mastery(c, h) for c, h in zip(group, holds)]
        worst = min(holds, key=lambda h: h["R"])
        rows.append("| %s | %d | %.3f | %s | %s%.1f | %d of %d | %.3f |" % (
            game, delay, group[0]["bar"], ", ".join("{:,}".format(h["ttm"]) for h in holds),
            "≥ " if worst["censored"] else "", worst["R"],
            sum(x < c["bar"] for c, a in zip(group, after) for x in a), sum(len(a) for a in after),
            min(min(a) for a in after)))
    return "\n".join(rows)


def summary_table(cells):
    """The README's short table: one row per game and delay, for the agent as published."""
    rows = ["| Game | Delay (ticks) | Runs that mastered it and held it | Ticks to mastery, in each run |",
            "|---|---|---|---|"]
    for (game, delay), group in _groups(_runs(cells)):
        holds = [scoring.hold_ratio(_curve(c), c["bar"]) for c in group]
        rows.append("| %s | %d | %d of %d | %s |" % (
            game, delay, sum(bool(h and h["held"]) for h in holds), len(group),
            ", ".join("never" if h is None else "{:,}".format(h["ttm"]) for h in holds)))
    return "\n".join(rows)


def comparison_table(cells):
    """One row per game and delay: the agent as published beside the same agent without the information term."""
    rows = ["| Game | Delay (ticks) | Held, with the term | Held, without it "
            "| Ticks outside the band before mastery, mean of the seeds, with | without "
            "| Ticks to mastery over the seeds, with | without | Windows below the bar after mastery, with | without |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    without = dict(_groups(_runs(cells, False)))
    for key, group in _groups(_runs(cells)):
        cols = []
        for g in (group, without.get(key, [])):
            holds = [scoring.hold_ratio(_curve(c), c["bar"]) for c in g]
            got = [(c, h) for c, h in zip(g, holds) if h is not None]
            every = len(got) == len(g) and bool(g)
            cols.append(("%d of %d" % (sum(bool(h and h["held"]) for h in holds), len(g)),
                         "{:,}".format(sum(h["ttm"] for _, h in got)) if every else "",
                         "%d" % sum(x < c["bar"] for c, h in got for x in _after_mastery(c, h)),
                         "{:,.0f}".format(sum(outside_before_mastery(c, h) for c, h in got) / len(g)) if every else ""))
        rows.append("| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            key[0], key[1], cols[0][0], cols[1][0], cols[0][3], cols[1][3], cols[0][1], cols[1][1], cols[0][2],
            cols[1][2]))
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


def summary(cells, information=True):
    """Counts over the runs with hindsight on, with or without the information term."""
    on = _runs(cells, information)
    holds = [scoring.hold_ratio(_curve(c), c["bar"]) for c in on]
    got = [(c, h) for c, h in zip(on, holds) if h is not None]
    ttm = sorted(h["ttm"] for _, h in got)
    return {"runs": len(on), "mastered": len(ttm), "held": sum(bool(h and h["held"]) for h in holds),
            "sustained_losses": sum(bool(h and not h["censored"]) for h in holds),
            "fastest": ttm[0] if ttm else None, "slowest": ttm[-1] if ttm else None,
            "median": ttm[len(ttm) // 2] if ttm else None, "ticks_to_mastery_summed": sum(ttm),
            "mastered_in_first_window": sum(h["ttm"] == c["curve"][0][0] for c, h in got),
            "ticks_outside_band_before_mastery_summed": round(sum(outside_before_mastery(c, h) for c, h in got)),
            "windows_below_bar_after_mastery": sum(x < c["bar"] for c, h in got for x in _after_mastery(c, h)),
            "runs_with_a_window_below_bar": sum(any(x < c["bar"] for x in _after_mastery(c, h)) for c, h in got)}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__)
        return 2
    with open(argv[0]) as fh:
        cells = json.load(fh)["cells"]
    print(main_table(cells))
    print()
    print(comparison_table(cells))
    print()
    if any(not c["hindsight"] for c in cells):
        print(control_table(cells))
        print()
    print(summary(cells))
    print(summary(cells, information=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
