"""Helpers for the agent's memory (Idea 10): when a moment is worth remembering, and how an outcome is measured.

Pure functions of tensors. Nothing here knows about trials, delays or blank inputs.
"""
from __future__ import annotations

import torch

WRITE_BAR = 0.5     # a body remembers a tick whose news is at least half the usual
FIT_RATE = 0.05     # rate of the fit of the usual change in fullness
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


def beyond_usual(level, level_prev, fit):
    """The change in each need beyond what its level predicts: ``(L_t - L_{t-1}) - (a L_{t-1} + a0)``.

    ``fit`` holds ``[a, a0]`` per need.
    """
    return (level - level_prev) - (fit[:, 0] * level_prev + fit[:, 1])


def fit_step(fit, residual, level_prev):
    """Move ``fit`` a fraction of the way toward the least-squares fit, across the bodies, of this tick's change on
    ``[L, 1]``.

    ``residual`` is the change beyond the current fit, so the step is toward the fit of the change itself. The
    slope is fitted on centered values, so it does not depend on the offset. The fit reads the level only. Run on
    every body it tracks the usual change at a level, feeding included; run on the bodies that wrote no moment it
    tracks the bare leak.
    """
    B = residual.shape[0]
    lm, rm = level_prev.mean(0), residual.mean(0)
    lc = level_prev - lm
    slope = ((residual - rm) * lc).sum(0) / ((lc ** 2).sum(0) + B * 1e-4)
    return fit + FIT_RATE * torch.stack([slope, rm - slope * lm], dim=-1)
