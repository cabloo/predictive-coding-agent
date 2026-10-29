"""Summarize a results file as the tables in docs/results.md.

    python -m pcagent.report results/acceptance.json

``summary_table`` gives the README's short table.

A results file is ``{"cells": [{"game", "delay", "seed", "hindsight", "bar", "oracle", "floor", "diverged",
"curve": [[tick, score], ...]}, ...]}``. A run is one of two kinds: the agent as published (``hindsight`` true) and
the control (``hindsight`` false: memory corrects no weight). The file's ``"events"`` counts, over the first
30,000 ticks of every run of the agent as published, the cue ticks and the other ticks of a body, and how many of
each were eventful. A run continued past the cap also has
``"continued_at"``, the tick where its two parts were joined. A run of the agent as published also has
``"precision": {"channels": [name, ...], "readings": [[tick, [one precision per channel], memory's], ...]}``, read
every 5,000 ticks from the running averages its precisions come from. The file's ``"learning_precision"`` holds each
of those runs run again from tick 0 to its mastery (at least 5,000 ticks), with every precision recorded:
``{"window", "ceiling", "runs": [{"game", "delay", "seed", "same_as_published", "largest_score_difference",
"mastered_at", "ticks", "channels", "windows": [[end, ticks, [[mean, uses at the ceiling, lowest] per channel],
[closings, mean, closings at the ceiling, lowest] for memory's, score], ...]}, ...]}``.
"""
from __future__ import annotations

import json
import statistics
import sys

from . import scoring


def _curve(cell):
    return [{"tick": t, "score": s} for t, s in cell["curve"]]


def _runs(cells):
    """The runs of the agent as published (hindsight on)."""
    return [c for c in cells if c["hindsight"]]


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


def main_table(cells):
    """One row per game and delay, over the seeds, for the runs with hindsight on."""
    rows = ["| Game | Delay (ticks) | Pass bar | Mastered at tick, by seed | Hold ratio R, lowest seed "
            "| Windows below the bar after mastery | Lowest window after mastery |",
            "|---|---|---|---|---|---|---|"]
    for (game, delay), group in _groups(_runs(cells)):
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


CEILING = 10.0


def _sight_median(cell):
    """The median, over a run's readings, of the mean precision of its sight channels."""
    names, readings = cell["precision"]["channels"], cell["precision"]["readings"]
    idx = [i for i, n in enumerate(names) if n == "sight"]
    return statistics.median(sum(p[i] for i in idx) / len(idx) for _, p, _ in readings)


def _at_ceiling(cell, sight):
    """How many of a run's precision readings are at the ceiling, and how many there are: sight's channels if
    ``sight``, else the fullness's, its change's and memory's. (The movement's precision weights nothing: the
    movement is copied, not forecast, so its error is zero.)"""
    names, readings = cell["precision"]["channels"], cell["precision"]["readings"]
    keep = ("sight",) if sight else ("fullness", "change")
    vals = [x for _, p, m in readings for x in ([p[i] for i, n in enumerate(names) if n in keep]
                                                + ([] if sight else [m]))]
    return sum(x >= CEILING - 1e-9 for x in vals), len(vals)


def precision_table(cells):
    """One row per game and delay: where the precisions of the agent as published sat at its saved readings."""
    rows = ["| Game | Delay (ticks) | Readings | Sight's precision, median, by seed | Sight's readings at the ceiling "
            "| Fullness, change and memory: readings at the ceiling |",
            "|---|---|---|---|---|---|"]
    for (game, delay), group in _groups(_runs(cells)):
        sight = [_at_ceiling(c, True) for c in group]
        other = [_at_ceiling(c, False) for c in group]
        rows.append("| %s | %d | %d | %s | %d%% | %d%% |" % (
            game, delay, sum(len(c["precision"]["readings"]) for c in group),
            ", ".join("%.1f" % _sight_median(c) for c in group),
            round(100.0 * sum(a for a, _ in sight) / sum(n for _, n in sight)),
            round(100.0 * sum(a for a, _ in other) / sum(n for _, n in other))))
    return "\n".join(rows)


def _learning_span(run):
    """The windows of a re-run up to its own mastery (all of them if it did not master in what was recorded)."""
    t = run["mastered_at"]
    return [w for w in run["windows"] if t is None or w[0] <= t]


def learning_uses(run, kind):
    """Over a re-run's ticks up to its mastery: the precisions used for ``kind`` (a channel name, or "memory" for
    memory's at its closings), as (sum of the precisions, uses at the ceiling, uses)."""
    span = _learning_span(run)
    if kind == "memory":
        return (sum((m[1] or 0.0) * m[0] for _, _, _, m, _ in span), sum(m[2] for _, _, _, m, _ in span),
                sum(m[0] for _, _, _, m, _ in span))
    idx = [i for i, n in enumerate(run["channels"]) if n == kind]
    return (sum(ch[i][0] * n for _, n, ch, _, _ in span for i in idx), sum(ch[i][1] for _, _, ch, _, _ in span
                                                                            for i in idx),
            sum(n for _, n, _, _, _ in span) * len(idx))


def _share(part, whole):
    """A share in whole percent, rounded; one that rounds to 100 without being all of it reads '>99%'."""
    pct = round(100.0 * part / whole)
    return ">99%" if pct == 100 and part < whole else "%d%%" % pct


def learning_table(doc):
    """One row per game and delay: the precisions used while the agent learned, from the re-runs in the results
    file's ``learning_precision`` (each run of the agent as published run again from tick 0, every precision
    recorded). Each cell is the mean precision used up to mastery, over the seeds, and in brackets the share of
    uses at the ceiling."""
    rows = ["| Game | Delay (ticks) | Re-runs equal to the published run | Re-run mastered at tick, by seed "
            "| Sight | Fullness | Change | Memory's |",
            "|---|---|---|---|---|---|---|---|"]
    keys = []
    for r in doc["runs"]:
        if (r["game"], r["delay"]) not in keys:
            keys.append((r["game"], r["delay"]))
    for game, delay in keys:
        group = sorted((r for r in doc["runs"] if (r["game"], r["delay"]) == (game, delay)), key=lambda r: r["seed"])
        cols = []
        for kind in ("sight", "fullness", "change", "memory"):
            s, at, n = (sum(x) for x in zip(*(learning_uses(r, kind) for r in group)))
            cols.append("%.1f (%s)" % (s / n, _share(at, n)))
        rows.append("| %s | %d | %d of %d | %s | %s |" % (
            game, delay, sum(r["same_as_published"] for r in group), len(group),
            ", ".join("not by {:,}".format(r["ticks"]) if r["mastered_at"] is None else "{:,}".format(r["mastered_at"])
                      for r in group), " | ".join(cols)))
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
    on = _runs(cells)
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
        doc = json.load(fh)
    cells = doc["cells"]
    print(main_table(cells))
    print()
    if "learning_precision" in doc:
        print(learning_table(doc["learning_precision"]))
        print()
    if all("precision" in c for c in _runs(cells)):
        print(precision_table(cells))
        print()
    if any(not c["hindsight"] for c in cells):
        print(control_table(cells))
        print()
    print(summary(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
