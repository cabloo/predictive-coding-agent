"""How a run is judged: the pass bar, mastery, and how long mastery was held.

A curve is a list of ``{"tick": ..., "score": ...}`` points, each the mean score over the ticks since the last point.
"""
from __future__ import annotations

SOLVED_SHARE = 0.9      # the bar sits 90% of the way from the floor to the oracle
MASTERY_POINTS = 3      # consecutive points at or above the bar that count as mastery; mirrored for a loss
HELD_X = 10.0           # mastery counts as held when it lasts at least 10 times as long as it took to reach


def solved_bar(anchors, share=SOLVED_SHARE):
    """The score that counts as solved: ``floor + share * (oracle - floor)``."""
    return anchors["floor"] + share * (anchors["oracle"] - anchors["floor"])


def ticks_to_mastery(curve, bar, k=MASTERY_POINTS):
    """The tick of the first of ``k`` consecutive points at or above the bar, or None. One point is a spike."""
    s = [c["score"] for c in curve]
    for i in range(len(s) - k + 1):
        if all(x >= bar for x in s[i:i + k]):
            return curve[i]["tick"]
    return None


def _loss_index(scores, start, bar, k):
    for j in range(start, len(scores) - k + 1):
        if all(x < bar for x in scores[j:j + k]):
            return j
    return None


def hold_ratio(curve, bar, k=MASTERY_POINTS):
    """How long mastery held, in units of how long it took to reach. None if the curve never mastered.

    ``hold`` runs from mastery to the first sustained loss (``k`` consecutive points below the bar). With no loss
    before the curve ends, ``hold`` is the time to the last point and is censored: the run ended, the hold did not.
    ``R = hold / ticks_to_mastery``. ``held`` is True when ``R >= 10``, False when a loss came sooner, and None when
    the run stopped before it could show either. The ``_first`` keys use a single point below the bar instead, so a
    brief dip that the sustained rule forgives stays visible.
    """
    if k < 1:
        raise ValueError("k must be >= 1, got %r" % (k,))
    s = [float(c["score"]) for c in curve]
    t = [int(c["tick"]) for c in curve]
    mi = next((i for i in range(len(s) - k + 1) if all(x >= bar for x in s[i:i + k])), None)
    if mi is None:
        return None
    ttm = t[mi]
    out = {"ttm": ttm}
    for suffix, kk in (("", k), ("_first", 1)):
        j = _loss_index(s, mi + 1, bar, kk)
        hold = (t[-1] if j is None else t[j]) - ttm
        out["hold" + suffix] = hold
        out["R" + suffix] = hold / ttm if ttm > 0 else float("inf")
        out["censored" + suffix] = j is None
    out["held"] = True if out["R"] >= HELD_X else (None if out["censored"] else False)
    return out


def run_until(curve, bar, tick, hold_x=20.0, hold_min=100_000, k=MASTERY_POINTS):
    """Whether a run that has mastered has now run long enough to judge the hold.

    A mastered run continues to ``mastery + max(hold_x * ticks_to_mastery, hold_min)`` ticks whatever happens in
    between, so that a loss and a recovery are both seen.
    """
    m = ticks_to_mastery(curve, bar, k)
    return m is not None and tick >= m + max(hold_x * m, hold_min)
