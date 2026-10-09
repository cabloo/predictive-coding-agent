"""Helpers for the agent's memory (Idea 10): when a moment is worth remembering, and the line that reads an outcome
from the fullness.

Pure functions of tensors. Nothing here knows about trials, delays or blank inputs.
"""
from __future__ import annotations

import torch

WRITE_BAR = 0.5     # a body remembers a tick whose news is at least half the usual
FIT_RATE = 0.05     # rate of the line that forecasts an outcome from the fullness
STAT_RATE = 0.01    # rate of the running statistics


def news(surprise, deviation, q):
    """How newsworthy this tick is, per body, on the sight and movement channels.

    News is what neither the forecast nor the channel's running mean predicted: ``d2 = min(surprise^2, deviation^2)``.
    ``q`` is the running mean of ``d2`` per channel (``None`` before the first tick) and is updated before the
    division. Returns ``(sum_c d2 / (sum_c q + 1e-12), q)``.
    """
    d2 = torch.minimum(surprise ** 2, deviation ** 2).double()
    m = d2.mean(0)
    q = m.clone() if q is None else (1.0 - STAT_RATE) * q + STAT_RATE * m
    return d2.sum(-1) / (q.sum() + 1e-12), q


def fit_step(fit, residual, level):
    """Move the line ``fit`` (``[slope, offset]`` per need) a fraction of the way toward the least-squares line,
    across the bodies, of the outcomes on ``[L, 1]``.

    ``residual`` is the outcome beyond what the line says now, so the step is toward the fit of the outcomes
    themselves. The slope is fitted on centered values, so it does not depend on the offset. The line reads the
    fullness only.
    """
    B = residual.shape[0]
    lm, rm = level.mean(0), residual.mean(0)
    lc = level - lm
    slope = ((residual - rm) * lc).sum(0) / ((lc ** 2).sum(0) + B * 1e-4)
    return fit + FIT_RATE * torch.stack([slope, rm - slope * lm], dim=-1)
