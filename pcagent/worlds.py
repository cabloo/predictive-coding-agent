"""The two games the agent is tested on, and the reference policies that anchor a score.

Both games run ``B`` bodies in parallel. A trial is ``delay + 1`` ticks, back to back:

    tick 0           the cue is shown; the movement made on this tick is the answer
    ticks 1..delay   nothing is shown, and movements change nothing
    tick delay       the outcome: each need gains ``BITE * credit`` (with no delay, on the cue tick itself)

The score is the share of bodies whose needs are all inside the comfort band. The agent is never told the score, the
trial structure, or the delay: it senses the cue, its own last movement and its fullness, and nothing else.
"""
from __future__ import annotations

import torch

BAND_LO, BAND_HI = 0.5, 1.0
_DIRS = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))      # east, north, west, south


def grade(cmd, cue, eps=1e-8):
    """Credit for a movement against a cue direction: ``max(0, 1 - angle / 90 degrees)``. No movement earns 0."""
    n = torch.linalg.vector_norm(cmd, dim=-1)
    cos = ((cmd / n.clamp_min(eps).unsqueeze(-1)) * cue).sum(-1).clamp(-1.0, 1.0)
    credit = (1.0 - torch.rad2deg(torch.arccos(cos)) / 90.0).clamp_min(0.0)
    return torch.where(n > eps, credit, torch.zeros_like(credit))


def _unit(cmd):
    n = torch.linalg.vector_norm(cmd, dim=-1, keepdim=True)
    return torch.where(n > 1e-8, cmd / n.clamp_min(1e-8), torch.zeros_like(cmd)).detach()


class CueGame:
    """One need. The cue is one of four directions; moving that way on the cue tick earns the bite ``delay`` ticks
    later.

    The leak is set per trial, so the timing is the only thing that changes with ``delay``: a body that always answers
    correctly sits at 0.75 just after each bite and falls to 0.60 before the next, inside the band at every delay;
    a body that guesses sits near 0.19.
    """

    n_extero, n_proprio, n_intero = 2, 2, 1
    LEAK_PER_TRIAL, BITE = 0.20, 0.15

    def __init__(self, B, seed=0, delay=0):
        if int(delay) < 0:
            raise ValueError("delay must be >= 0, got %r" % (delay,))
        self.B = B
        self.g = torch.Generator().manual_seed(seed)
        self.delay = int(delay)
        self.T = self.delay + 1
        self.leak = 1.0 - (1.0 - self.LEAK_PER_TRIAL) ** (1.0 / self.T)      # per tick
        self.dirs = torch.tensor(_DIRS)
        self.cue_idx = torch.randint(0, 4, (B,), generator=self.g)
        self.full = torch.full((B, 1), 0.75)
        self.vel = torch.zeros(B, 2)
        self.credit = torch.zeros(B)
        self.t = 0

    @property
    def phase(self):
        return self.t % self.T

    def cue(self):
        return self.dirs[self.cue_idx]

    def extero(self):
        """Sight: the cue on the cue tick, blank otherwise."""
        return self.cue() if self.phase == 0 else torch.zeros(self.B, 2)

    def proprio(self):
        """The body's last movement, as a unit direction."""
        return self.vel

    def intero(self):
        """Fullness."""
        return self.full

    def oracle_cmd(self):
        return self.cue()

    def step(self, cmd):
        self.vel = _unit(cmd)
        if self.phase == 0:
            self.credit = grade(cmd.detach(), self.cue())
        self.full = (1.0 - self.leak) * self.full
        if self.phase == self.delay:
            self.full = self.full + self.BITE * self.credit.unsqueeze(1)
            self.cue_idx = torch.randint(0, 4, (self.B,), generator=self.g)
        self.t += 1
        return self.credit

    def score(self):
        f = self.full.squeeze(-1)
        return float(((f >= BAND_LO) & (f <= BAND_HI)).float().mean())


class TwoNeedGame:
    """Two needs (food and water), both offered every trial in two different directions, and room to overshoot.

    The one answer is graded against each direction: pointing at one takes it, pointing between two that are
    90 degrees apart takes half of each, pointing elsewhere takes nothing. Declining is possible and is needed,
    because a bite can push a level above the band.

    The two leaks sum to a constant, but the split between them is redrawn every ``HOLD`` trials and held in
    between. That is what makes the body necessary: with equal leaks a fixed schedule that never reads the levels
    scores well, and with the held split the best such schedule does not (see :meth:`blind_policies`).
    """

    n_extero, n_proprio, n_intero = 4, 2, 2
    LEAK_PER_TRIAL, BITE, JITTER, HOLD, LEVEL_MAX = 0.10, 0.30, 0.08, 20, 1.5

    def __init__(self, B, seed=0, delay=0):
        if int(delay) < 0:
            raise ValueError("delay must be >= 0, got %r" % (delay,))
        self.B = B
        self.g = torch.Generator().manual_seed(seed)
        self.delay = int(delay)
        self.T = self.delay + 1
        self.dirs = torch.tensor(_DIRS)
        self.lv = torch.full((B, 2), 0.75)                  # [food, water]
        self.vel = torch.zeros(B, 2)
        self.credit = torch.zeros(B, 2)
        self.trial = 0
        self.t = 0
        self._draw_cues()
        self._draw_jitter()

    def _draw_cues(self):
        f = torch.randint(0, 4, (self.B,), generator=self.g)
        off = torch.randint(1, 4, (self.B,), generator=self.g)      # never the same direction
        self.food_idx, self.water_idx = f, (f + off) % 4

    def _draw_jitter(self):
        self.jit = (torch.rand(self.B, generator=self.g) * 2.0 - 1.0) * self.JITTER

    def _leak_trial(self):
        """Per need, per trial: the split is jittered, the sum is fixed."""
        return torch.stack([self.LEAK_PER_TRIAL + self.jit, self.LEAK_PER_TRIAL - self.jit], dim=-1)

    @property
    def phase(self):
        return self.t % self.T

    def cues(self):
        return self.dirs[self.food_idx], self.dirs[self.water_idx]

    def extero(self):
        """Sight: ``[food direction, water direction]`` on the cue tick, blank otherwise."""
        if self.phase != 0:
            return torch.zeros(self.B, self.n_extero)
        return torch.cat(self.cues(), dim=-1)

    def proprio(self):
        return self.vel

    def intero(self):
        """The two levels."""
        return self.lv

    def oracle_cmd(self):
        """Serve the lower need; decline (no movement) when that bite, after this trial's leak, would overshoot."""
        f, w = self.cues()
        pick_food = self.lv[:, 0] <= self.lv[:, 1]
        lvl = torch.where(pick_food, self.lv[:, 0], self.lv[:, 1])
        L = torch.where(pick_food, self._leak_trial()[:, 0], self._leak_trial()[:, 1])
        over = lvl * (1.0 - L) + self.BITE > BAND_HI
        cmd = torch.where(pick_food.unsqueeze(-1), f, w)
        return torch.where(over.unsqueeze(-1), torch.zeros_like(cmd), cmd)

    @staticmethod
    def blind_policies():
        """Policies that use the cues but never the body. The best of them is part of the floor.

        The periodic schedules cycle over {food, water, skip} per trial; an agent that can carry time could learn
        one of those instead of reading its needs, so they are scored too.
        """
        def food_only(b):
            return b.cues()[0]

        def water_only(b):
            return b.cues()[1]

        def coin(b):
            f, w = b.cues()
            c = torch.rand(b.B, 1, generator=b.g) < 0.5
            return torch.where(c, f, w)

        def split_half(b):
            f, w = b.cues()
            take = (torch.rand(b.B, 1, generator=b.g) < 0.5).to(f.dtype)
            return (f + w) * take

        def _cycle(pattern):
            def pick(b, t):
                f, w = b.cues()
                step = pattern[(t // b.T) % len(pattern)]
                return f if step == "F" else (w if step == "W" else torch.zeros_like(f))
            return pick
        periodic = {"cycle_" + p: _cycle(p) for p in ("FW", "FWS", "FFW", "FWW", "FWSS", "FFWS", "FWWS", "FFWW")}
        return {"food_only": food_only, "water_only": water_only, "coin": coin, "split_half": split_half, **periodic}

    def step(self, cmd):
        self.vel = _unit(cmd)
        if self.phase == 0:
            f, w = self.cues()
            c = cmd.detach()
            self.credit = torch.stack([grade(c, f), grade(c, w)], dim=-1)
        per_tick = 1.0 - (1.0 - self._leak_trial()) ** (1.0 / self.T)
        self.lv = (1.0 - per_tick) * self.lv
        if self.phase == self.delay:
            self.lv = (self.lv + self.BITE * self.credit).clamp(0.0, self.LEVEL_MAX)
            self.trial += 1
            self._draw_cues()
            if self.trial % self.HOLD == 0:
                self._draw_jitter()
        self.t += 1
        return self.credit

    def score(self):
        return float(((self.lv >= BAND_LO) & (self.lv <= BAND_HI)).all(-1).float().mean())


GAMES = {"cue": CueGame, "two-need": TwoNeedGame}


def anchors(make_game, B=32, ticks=3000, seed=7):
    """Score the reference policies on a fresh game.

    ``make_game(B, seed)`` builds the game. Each policy is scored over the second half of ``ticks``. The oracle is
    the game's own hand-written policy (the ceiling). The floor is the best score among the policies that ignore
    what the game is about: a fixed movement and a random one (they ignore the cue), and any ``blind_policies``
    the game declares (in the two-need game they read the cues and ignore the body's levels).
    """
    def run(pick):
        game = make_game(B, seed)
        hits, n = 0.0, 0
        for t in range(ticks):
            game.step(pick(game, t) if pick.__code__.co_argcount > 1 else pick(game))
            if t >= ticks // 2:
                hits += game.score()
                n += 1
        return hits / max(n, 1)

    fixed = run(lambda b: torch.tensor([1.0, 0.0]).expand(B, 2))
    rand = run(lambda b: torch.randn(B, 2, generator=b.g))
    oracle = run(lambda b: b.oracle_cmd())
    out = {"oracle": oracle, "blind_fixed": fixed, "blind_random": rand, "floor": max(fixed, rand)}
    blind = getattr(type(make_game(B, seed)), "blind_policies", lambda: {})() or {}
    for name, pick in blind.items():
        out["blind_" + name] = run(pick)
        out["floor"] = max(out["floor"], out["blind_" + name])
    return out
