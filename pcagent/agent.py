"""A predictive-coding agent that learns to act for a delayed outcome by hindsight anticipation.

The agent keeps a sparse internal state, predicts its next senses from it, and learns from its own prediction errors.
Its preferences concern only its fullness: a range (a comfort band) one tick ahead, and the middle of that range at
the next eventful moment. The movement it makes is found while the state settles, by following the slope of those
preferences. One extra forecast, of how much its fullness will change by the next eventful moment, lets them reach
across a delay; the part of it read from the state is the anticipation, and a memory of the last eventful moment
corrects it in hindsight.

The comments name the parts as docs/how-it-works.md does, "Idea 1" to "Idea 12":

    Idea 1   predict the next senses; the forecast learns by a normalized delta rule
    Idea 2   a sparse state of thresholded units with lateral inhibition
    Idea 3   settling: the state is found each tick by a relaxation, run twice
    Idea 4   the input weights learn from the forecast's error, sent back one step
    Idea 5   a second level predicts the first
    Idea 6   the comfort band, read under the forecast's uncertainty
    Idea 7   the movement is found during the settle, by following the slope of the preferences
    Idea 8   the weights that forecast the change in fullness read a centered state
    Idea 9   one anticipation, read from the state and fed back into it on the next tick
    Idea 10  memory sets the anticipation's target at one remembered moment: the change in fullness that followed
    Idea 11  the anticipation has a setpoint of its own: fullness at the next moment should sit at the band's middle
    Idea 12  regulators: attention, a learning-rate gate, a hold on ticks with little surprise, thresholds that
             weigh each tick by its surprise, and running averages that no single outcome can rewrite

Shapes: ``B`` bodies run in parallel and share every weight. What the agent senses on a tick is one vector,
``[sight | last movement | fullness | change in fullness]``. In the code ``level`` is the fullness. ``z1`` and ``z2``
are the states of the two levels of units and ``u1`` and ``u2`` their potentials before thresholding. Level 1 has two
states per tick: ``z1`` from the settle that includes the goal, and ``z1_up``, the goal-free state, from the settle
that does not.

In the comments, "surprise" means a signed prediction error (what was sensed minus what was forecast), not the
negative log probability of an observation. docs/equations.md gives every update below as an equation.
"""
from __future__ import annotations

import math

import torch

from . import memory

LOG_VAR_MIN, LOG_VAR_MAX = math.log(1e-8), 0.0
STAT_RATE = 0.01            # running statistics: forecast skill, anticipation power, outcomes, the gate's fast average
HOLD_EPS = 1e-3             # floor inside the hold's surprise ratio
_SQRT_2 = math.sqrt(2)
_LOG_SQRT_2PI = math.log(math.sqrt(2 * math.pi))


def _normal_cdf(x):
    return 0.5 * (1 + torch.erf(x / _SQRT_2))


def _normal_pdf(x):
    return torch.exp(-x ** 2 / 2.0 - _LOG_SQRT_2PI)


def _unit_rows(*mats):
    """Scale the matrices together so that each joint row has length 1."""
    n = torch.sqrt(sum((m ** 2).sum(-1) for m in mats)).clamp_min(1e-12).unsqueeze(-1)
    return [m / n for m in mats]


def _direction(m):
    """The unit direction of a movement, or zero for no movement."""
    n = torch.linalg.vector_norm(m, dim=-1, keepdim=True)
    return torch.where(n > 1e-12, m / n.clamp_min(1e-12), 0.0)


def _center(z):
    """Idea 8: the state centered over its active units (zero for a body with none active)."""
    f = (z > 0).to(z.dtype)
    n = f.sum(-1, keepdim=True)
    return z - z.sum(-1, keepdim=True) / n.clamp_min(1.0) * f


class Agent:
    """The agent. Call :meth:`step` once per tick with what the body senses; it returns the movement.

    Every default below is the value the published results were run with.

    Args:
        extero_dim: width of the sight input.
        proprio_dim: width of the movement (and of the "last movement" input).
        intero_dim: number of needs (fullness levels).
        widths: units at level 1 and level 2.
        seed: seeds the initial weights. Nothing else in the agent is random.
        band_lo, band_hi: the comfort band.
        forecast_lr: rate of the forecast weights (Idea 1) and, times ``hindsight_rate``, of the anticipation's.
        perception_lr: rate of the level-1 input weights (Idea 4).
        level2_lr, level2_input_lr: rates of level 2's prediction of level 1 and of its input weights (Idea 5).
        uncertainty_lr: rate of the weights that predict the variance of the change forecast (Idea 6).
        settle_steps, settle_rate: the relaxation (Idea 3).
        firing_rate, threshold_lr, threshold_floor: the share of ticks a unit should be active on, and the step and
            floor of the thresholds that steer toward it (Idea 2).
        level2_hold: share of level 2's potentials held from one tick to the next (Idea 5).
        turn_rate, turn_anneal: how fast the movement turns during the settle, and how that slows (Idea 7).
        hindsight_rate: memory's corrections run at this multiple of the ordinary rates (Idea 10).
        step_limit: the largest step of the anticipation's weights, as a share of their size plus one (Idea 10).
        anticipation_gain: the push has two terms, one through the weights that forecast the next change and one
            through the anticipation's weights; this multiplies the second (Idea 11).
        sample_cap: the most one outcome's squared surprise counts for, in units of the running average it is
            folded into (Idea 12).
        skill_offset: a constant added to the one-tick forecast's skill when the push weighs the first term
            against the second (Idea 11).
        attention_gain, attention_cap: slope and ceiling of the attention gain (Idea 12).
        gate_floor: the learning-rate gate's floor; its ceiling is ``min(10, 1 / largest level-1 rate)`` (Idea 12).
        hold_max, hold_steepness: the largest hold on a tick with little surprise, and how steeply it sets in
            (Idea 12).
        init_scale: scale of the forecast weights at birth.
        hindsight: False switches memory's corrections off, as a control. Memory still records moments, outcomes
            and its statistics, but corrects no weight, so the anticipation and the line stay at zero.
        device: where the tensors live. The results were run on the CPU.

    Attributes worth reading from outside:
        info: a few numbers from the last tick (the gate, the size of the anticipation's weights, the number of
            active units, and the last hindsight surprise).
        diverged: True once level 1's state, the movement or one of the main weights is no longer finite.
    """

    def __init__(self, extero_dim: int, proprio_dim: int = 2, intero_dim: int = 1, *, widths=(256, 64), seed: int = 0,
                 band_lo: float = 0.5, band_hi: float = 1.0,
                 forecast_lr: float = 0.1, perception_lr: float = 0.03, level2_lr: float = 0.02,
                 level2_input_lr: float = 0.03, uncertainty_lr: float = 0.02,
                 settle_steps: int = 40, settle_rate: float = 0.1,
                 firing_rate: float = 0.056, threshold_lr: float = 0.01, threshold_floor: float = 0.0,
                 level2_hold: float = 0.9, turn_rate: float = 10.0, turn_anneal: float = 10.0,
                 hindsight_rate: float = 3.0, step_limit: float = 0.005, anticipation_gain: float = 8.0,
                 skill_offset: float = 0.05, attention_gain: float = 3.0, attention_cap: float = 10.0,
                 gate_floor: float = 1.0, hold_max: float = 0.9, hold_steepness: float = 3.0,
                 sample_cap: float = 10.0, init_scale: float = 0.1, hindsight: bool = True, device: str = "cpu"):
        if int(extero_dim) < 1 or int(proprio_dim) < 1 or int(intero_dim) < 1:
            raise ValueError("extero_dim, proprio_dim and intero_dim must each be >= 1")
        widths = tuple(int(w) for w in widths)
        if len(widths) != 2 or min(widths) < 1:
            raise ValueError("widths must be two positive level sizes (level 1, level 2), got %r" % (widths,))
        if not float(band_hi) > float(band_lo):
            raise ValueError("band_hi must exceed band_lo")
        if int(settle_steps) < 1 or not 0.0 < float(settle_rate) <= 1.0:
            raise ValueError("settle_steps >= 1 and settle_rate in (0, 1] required")
        if max(float(forecast_lr), float(perception_lr)) > 1.0:
            raise ValueError("forecast_lr and perception_lr must be <= 1 (the gate's ceiling is 1 / the larger)")
        if not float(sample_cap) > 1.0:
            raise ValueError("sample_cap must exceed 1")
        self.extero_dim, self.proprio_dim, self.intero_dim = int(extero_dim), int(proprio_dim), int(intero_dim)
        self.widths, self.device = widths, device
        self.band_lo, self.band_hi = float(band_lo), float(band_hi)
        self.forecast_lr, self.perception_lr = float(forecast_lr), float(perception_lr)
        self.level2_lr, self.level2_input_lr = float(level2_lr), float(level2_input_lr)
        self.uncertainty_lr = float(uncertainty_lr)
        self.settle_steps, self.settle_rate = int(settle_steps), float(settle_rate)
        self.firing_rate, self.threshold_lr = float(firing_rate), float(threshold_lr)
        self.threshold_floor = float(threshold_floor)
        self.level2_hold = float(level2_hold)
        self.turn_rate, self.turn_anneal = float(turn_rate), float(turn_anneal)
        self.hindsight_rate, self.step_limit = float(hindsight_rate), float(step_limit)
        self.anticipation_gain = float(anticipation_gain)
        self.skill_offset = float(skill_offset)
        self.attention_gain, self.attention_cap = float(attention_gain), float(attention_cap)
        self.gate_floor = float(gate_floor)
        self.gate_max = min(10.0, 1.0 / max(max(self.forecast_lr, self.perception_lr), 1e-12))
        self.hold_max, self.hold_steepness = float(hold_max), float(hold_steepness)
        self.sample_cap = float(sample_cap)
        self.nlms_floor = 1e-3                       # floor on a unit's gain in the input-weight steps (both levels)
        self.hindsight = bool(hindsight)

        # what is sensed: [sight | last movement | fullness | change in fullness]
        e, p, i = self.extero_dim, self.proprio_dim, self.intero_dim
        self.i_e, self.i_p = slice(0, e), slice(e, e + p)
        self.i_l, self.i_d = slice(e + p, e + p + i), slice(e + p + i, e + p + 2 * i)
        self.n_s = e + p + 2 * i

        # weights. The random draws come from one generator, in this order: C, K, Km, K2.
        n1, n2 = widths
        g = torch.Generator().manual_seed(int(seed))
        rnd = lambda *s: torch.randn(*s, generator=g)                                    # noqa: E731
        self.C = (rnd(self.n_s, n1) * float(init_scale) / math.sqrt(n1)).to(device)      # forecast of the next senses
        K, Km, A = _unit_rows(rnd(n1, self.n_s), rnd(n1, p), torch.zeros(n1, n1))
        self.K, self.Km, self.A = K.to(device), Km.to(device), A.to(device)   # input: surprise, movement, last state
        self.W2 = torch.zeros(n1, n2, device=device)                          # level 2's prediction of level 1
        K2, A2 = _unit_rows(rnd(n2, n1), torch.zeros(n2, n2))
        self.K2, self.A2 = K2.to(device), A2.to(device)
        self.Ka = torch.zeros(n1, i, device=device)                                      # input from the anticipation
        self.Ca = torch.zeros(i, n1, device=device)                                      # the anticipation's weights
        self.from_level = torch.zeros(i, 2, device=device)   # the part of the outcome the fullness alone forecasts:
                                                             # [slope, offset] per need (Idea 10)
        self.Wv = torch.zeros(i, n1, device=device)                                      # variance of the change
        self.bv = torch.full((i,), math.log(0.001), device=device)
        self.c0 = torch.zeros(self.n_s, device=device)
        self.C[self.i_p] = 0.0                                                # the movement is known, not forecast
        self.theta = [torch.zeros(n1, device=device), torch.zeros(n2, device=device)]
        self.L = [torch.zeros(n1, n1, device=device), torch.zeros(n2, n2, device=device)]

        # running statistics and regulators
        self.skill_stats = [torch.zeros(self.n_s, device=device) for _ in range(3)]      # mean, variance, squared error
        self.skill = torch.zeros(self.n_s, device=device)                                # per sensed channel
        self.attention = torch.ones(self.n_s, device=device)
        self.var_ebar = None                    # running squared error of the change forecast
        self.hold_mbar = None                   # running mean |surprise| per channel
        self.hold = None                        # this tick's hold per body and unit
        self.news_weight = None                 # per body: 1 on a tick with the usual surprise, 0 on a held one
        self.level_lo, self.level_hi = None, None      # range of fullness seen so far
        self.ant_power = None                   # running mean power of the anticipation
        self.gate = 1.0                         # the learning-rate gate
        self.gate_fast, self.gate_slow = None, None
        self.news_q = None                      # running news per sight / movement channel
        self.outcome_stats = None               # the outcome's running mean, variance, and the forecast's error
        self.diverged = False
        self.info: dict = {}
        self._B = None
        self._t = 0

    # ------------------------------------------------------------------ per-body state
    def _init_state(self, B):
        n1, n2 = self.widths
        z = lambda *s: torch.zeros(*s, device=self.device)                               # noqa: E731
        self._B, self._t = B, 0
        self.z1, self.z1_prev, self.z1_up = z(B, n1), z(B, n1), z(B, n1)
        self.eps, self.m = z(B, self.n_s), z(B, self.proprio_dim)
        self.z2, self.z2_prev, self.eps2 = z(B, n2), z(B, n2), z(B, n1)
        self.u1, self.u2 = z(B, n1), z(B, n2)
        self.o_prev, self.l_prev = None, None
        self.ant = z(B, self.intero_dim)
        # memory: one remembered moment per body
        self.mem_z, self.mem_zp = z(B, n1), z(B, n1)
        self.mem_eps, self.mem_mov = z(B, self.n_s), z(B, self.proprio_dim)
        self.mem_ant = z(B, self.intero_dim)
        self.mem_level = z(B, self.intero_dim)         # the fullness at the remembered moment
        self.mem_has = torch.zeros(B, dtype=torch.bool, device=self.device)
        self.mem_G = z(B, self.intero_dim)             # the change in fullness since that moment

    # ------------------------------------------------------------------ Idea 6: the comfort band
    def _expected_slope(self, mean, var):
        """The band's slope averaged over a normal with this mean and variance.

        Outside the band the cost is ``R (d/R)^2 + d`` in the distance ``d`` past the nearer edge, ``R`` the band's
        width. The slope is negative below the band and positive above it.
        """
        R = self.band_hi - self.band_lo
        sd = var.clamp_min(0.0).sqrt().clamp_min(1e-12)
        d_hi, d_lo = mean - self.band_hi, self.band_lo - mean
        a_hi, a_lo = d_hi / sd, d_lo / sd
        cdf_hi, cdf_lo = _normal_cdf(a_hi), _normal_cdf(a_lo)
        step = cdf_hi - cdf_lo
        pdf_hi, pdf_lo = _normal_pdf(a_hi), _normal_pdf(a_lo)
        e_above = d_hi * cdf_hi + sd * pdf_hi
        e_below = d_lo * cdf_lo + sd * pdf_lo
        return step + 2.0 / R * (e_above - e_below)

    def _forecast_var(self, z1):
        return torch.exp((z1 @ self.Wv.t() + self.bv).clamp(LOG_VAR_MIN, LOG_VAR_MAX))

    # ------------------------------------------------------------------ Idea 11: the anticipation's own setpoint
    def _one_tick_weight(self):
        """The weight of the push's first term: the one-tick forecast's skill (plus ``skill_offset``) as a share of
        the two skills together."""
        s1 = float(self.skill[self.i_d].mean())
        tot = s1
        if self.outcome_stats is not None:
            _, v, e = self.outcome_stats
            tot = tot + float(((v - e).clamp_min(0.0) / (v + 1e-08)).mean())
        return (s1 + self.skill_offset) / (tot + self.skill_offset)

    def _push_context(self, level):
        """Everything the push needs that does not change while the state settles."""
        Cd = self.C[self.i_d]
        return {"Cd": Cd, "CdT": Cd.t(), "WvT": self.Wv.t(), "c0d": self.c0[self.i_d], "CaT": self.Ca.t(),
                "w1": self._one_tick_weight(),
                "from_level": level * self.from_level[:, 0] + self.from_level[:, 1],
                "middle": 0.5 * (self.band_lo + self.band_hi), "half_width": 0.5 * (self.band_hi - self.band_lo)}

    def _push(self, z1, level, ctx):
        """Which units, if more active, would bring the agent's two reads back toward their setpoints.

        The first term is the comfort band read on fullness + the forecast of the next change (Idea 6). The second
        is the anticipation's own setpoint (Idea 11): fullness at the next eventful moment, which is the fullness
        now plus the forecast change until then, should sit at the band's middle. Its error is counted in
        half-widths of the band and goes through the anticipation's weights.

        The result is used only to turn the movement (see ``_settle1``); it is not added to the units' input.
        """
        f = _center(z1) @ ctx["CdT"]
        f = f + ctx["c0d"]
        var = torch.exp((z1 @ ctx["WvT"] + self.bv).clamp(LOG_VAR_MIN, LOG_VAR_MAX))
        slope = self._expected_slope(level + f, var)
        push = -slope @ ctx["Cd"]
        push = ctx["w1"] * push
        fired = (z1 > 0).to(z1.dtype)                    # Idea 8: the first term loses its mean over active units
        push = push - (fired * push).sum(-1, keepdim=True) / fired.sum(-1, keepdim=True).clamp_min(1.0)
        ahead = z1 @ ctx["CaT"]                          # the forecast change until the next moment: the state's
        ahead = ahead + ctx["from_level"]                # part and the fullness's part
        error = ctx["middle"] - (level + ahead)
        error = error / ctx["half_width"]
        through_anticipation = error @ self.Ca
        through_anticipation = self.anticipation_gain * through_anticipation
        return push + through_anticipation

    # ------------------------------------------------------------------ Ideas 2, 3, 7: settling
    def _fire(self, u, layer):
        return torch.where(u > self.theta[layer], u, 0.0).clamp_min(0.0)

    def _regulate(self, layer, z):
        """Move the thresholds toward the target firing rate. Level 1 (layer 0) weighs each body's tick by its
        surprise, ``news_weight`` (Idea 12)."""
        a = (z > 0).to(z.dtype)
        if layer == 0:
            n = self.news_weight.to(a.dtype).unsqueeze(-1)
            self.theta[layer] = self.theta[layer] + self.threshold_lr * ((n * a).mean(0) - self.firing_rate * n.mean())
        else:
            self.theta[layer] = self.theta[layer] + self.threshold_lr * (a.mean(0) - self.firing_rate)
        self.theta[layer] = self.theta[layer].clamp_min(self.threshold_floor)

    def _inhibition(self):
        """Units with overlapping input weights inhibit each other. Recomputed from the weights every tick."""
        out = []
        for W in (torch.cat([self.K, self.Km], dim=1), self.K2):
            Wn = W / W.norm(dim=1, keepdim=True).clamp_min(1e-12)
            out.append((Wn @ Wn.t()).clamp_min(0.0).fill_diagonal_(0.0))
        self.L = out

    def _turn(self, m, g, k):
        """Idea 7: turn the unit movement ``m`` toward ``g``, the direction the push asks for, fast early in the
        settle and slowly late."""
        eta = self.settle_rate
        kk = self.turn_rate / (1.0 + k / self.turn_anneal)
        d = m + kk * eta * (g - (g * m).sum(-1, keepdim=True) * m)
        n = torch.linalg.vector_norm(d, dim=-1, keepdim=True)
        return torch.where(n > 1e-12, d / n.clamp_min(1e-12), 0.0)

    def _settle1(self, base, level, goal: bool, m_fixed, u_prev):
        """Relax level 1 from zero. With ``goal`` the push turns the movement; otherwise the movement is ``m_fixed``."""
        B, n1 = base.shape
        u = torch.zeros(B, n1, device=base.device)
        z = torch.zeros_like(u)
        lam = self.hold
        mix = (1.0 - lam, lam)
        m = m_fixed if m_fixed is not None else torch.zeros(B, self.proprio_dim, device=base.device)
        KmT, L0T = self.Km.t(), self.L[0].t()
        mix_u = mix[1] * u_prev
        b_fixed = None
        if m_fixed is not None:
            b_fixed = base + _direction(m_fixed) @ KmT
        zf = (z > 0).to(z.dtype)
        ctx = self._push_context(level) if goal else None
        for k in range(self.settle_steps):
            if b_fixed is not None:
                b = b_fixed
            else:
                b = base + _direction(m) @ KmT
            if goal:
                push = self._push(z, level, ctx)
                m = self._turn(m, zf * push @ self.Km, k)
            target = b - z @ L0T
            target = mix[0] * target + mix_u
            u = u + self.settle_rate * (target - u)
            z = self._fire(u, 0)
            zf = (z > 0).to(z.dtype)
        return z, m, u

    def _settle2(self, base, u_prev):
        """Relax level 2. It starts from last tick's potentials and always holds a share of them."""
        lam = self.level2_hold
        u, mix = u_prev.clone(), ((1.0 - lam, lam) if lam else None)
        z = self._fire(u, 1)
        L1T = self.L[1].t()
        mix_u = mix[1] * u_prev if mix is not None else None
        for _ in range(self.settle_steps):
            target = base - z @ L1T
            if mix is not None:
                target = mix[0] * target + mix_u
            u = u + self.settle_rate * (target - u)
            z = self._fire(u, 1)
        return z, u

    # ------------------------------------------------------------------ Idea 12: regulators
    def _update_skill(self, o, pred):
        """Per sensed channel: the share of its variance the one-tick forecast explains."""
        b = STAT_RATE
        mu, v, e = self.skill_stats
        mu = (1.0 - b) * mu + b * o.mean(0)
        v = (1.0 - b) * v + b * ((o - mu) ** 2).mean(0)
        e = (1.0 - b) * e + b * ((o - pred) ** 2).mean(0)
        self.skill_stats = [mu, v, e]
        self.skill = (v - e).clamp_min(0.0) / (v + 1e-08)

    def _attend(self, eps):
        """Amplify the surprise of channels that would move the agent's other good forecasts. Needs are held at 1."""
        Cs = self.C * self.skill.unsqueeze(1)
        full = (Cs.t() @ Cs @ self.K * self.K).sum(0)
        own = (Cs * self.K.t()).sum(1)
        lev = (full - own ** 2).clamp_min(0.0).sqrt()
        tot = lev.mean()
        lev = lev / tot.clamp_min(1e-12)
        self.attention = (1.0 + self.attention_gain * (lev - 1.0).clamp_min(0.0)).clamp(max=self.attention_cap)
        self.attention[self.i_d] = 1.0
        self.attention[self.i_l] = 1.0
        return eps * self.attention

    def _update_hold(self, eps):
        """A body with little surprise holds its potentials; one with the usual surprise or more is rewritten."""
        a = eps.abs().double()
        am = a.mean(0)
        self.hold_mbar = am.clone() if self.hold_mbar is None else 0.99 * self.hold_mbar + 0.01 * am
        sal = ((a + HOLD_EPS) / (self.hold_mbar.to(a.device) + HOLD_EPS)).mean(-1)
        h = (self.hold_steepness * (1.0 - sal).clamp_min(0.0)).clamp(max=1.0).to(torch.float32)
        lam = (self.hold_max * h).unsqueeze(-1).expand(-1, self.widths[0])
        self.hold = lam.contiguous()
        self.news_weight = 1.0 - h

    def _gate_value(self):
        """Learn faster when the anticipation's recent surprise is far from what has become usual."""
        if self.gate_fast is None:
            return 1.0
        u = (self.gate_fast + 1e-12) / (self.gate_slow + 1e-12)
        return min(self.gate_max, max(self.gate_floor, self.gate_max * abs(u - 1.0)))

    def _gate_fold(self, x):
        """Fold one outcome's squared surprise into the gate's two averages, each taking it at no more than
        ``sample_cap`` times its own value."""
        if self.gate_fast is None:
            self.gate_fast, self.gate_slow = x, x
            return
        r = STAT_RATE / 10.0
        cap = self.sample_cap
        x_fast = min(x, cap * self.gate_fast) if self.gate_fast > 0.0 else x
        x_slow = min(x, cap * self.gate_slow) if self.gate_slow > 0.0 else x
        self.gate_fast = (1.0 - STAT_RATE) * self.gate_fast + STAT_RATE * x_fast
        self.gate_slow = (1.0 - r) * self.gate_slow + r * x_slow

    # ------------------------------------------------------------------ Ideas 1, 4, 6, 9: learning from the last tick
    @staticmethod
    def _rescale_blocks(old, new, sight):
        """Idea 4: each unit's sight weights, and the rest of its input weights, keep the lengths they had before
        the step."""
        (K0, Km0, A0), (K1, Km1, A1) = old, new
        other = [c for c in range(K0.shape[1]) if not sight.start <= c < sight.stop]

        def norms(K, Km, A):
            rest = torch.sqrt((K[:, other] ** 2).sum(-1) + (Km ** 2).sum(-1) + (A ** 2).sum(-1))
            return K[:, sight].norm(dim=-1), rest
        e0, r0 = norms(K0, Km0, A0)
        e1, r1 = norms(K1, Km1, A1)
        s_e = (e0 / e1.clamp_min(1e-12)).unsqueeze(-1)
        s_r = (r0 / r1.clamp_min(1e-12)).unsqueeze(-1)
        K = K1 * s_r
        K[:, sight] = K1[:, sight] * s_e
        return K, Km1 * s_r, A1 * s_r

    def _apply_input_step(self, dW):
        mats = [self.K, self.Km, self.A]
        new, c = [], 0
        for W in mats:
            new.append(W + dW[:, c:c + W.shape[1]])
            c += W.shape[1]
        self.K, self.Km, self.A = self._rescale_blocks(mats, new, self.i_e)

    def _learn_level1(self, o, pred):
        B = o.shape[0]
        self.gate = self._gate_value()
        delta = o - pred
        # Idea 4: the error goes back one step through the forecast weights, to the units that were active
        fired = (self.z1 > 0).to(o.dtype)
        g = delta @ self.C * fired
        gd = delta[:, self.i_d] @ self.C[self.i_d] * fired
        g = g - fired * (gd.sum(-1, keepdim=True) / fired.sum(-1, keepdim=True).clamp_min(1.0))
        self._update_skill(o, pred)
        # Idea 1: the forecast weights, by a normalized delta rule (those for the change read the centered state)
        power = 1e-06 + float((self.z1 ** 2).sum(-1).mean())
        dC = self.forecast_lr * delta.t() @ self.z1 / B / power
        zc = _center(self.z1)
        dC[self.i_d] = self.forecast_lr * delta[:, self.i_d].t() @ zc / B / power
        dC = dC * self.gate
        self.C = (self.C + dC).clamp(-1.0, 1.0)
        self.C[self.i_p] = 0.0
        dc0 = self.forecast_lr * delta.mean(0)
        dc0 = dc0 * self.gate
        self.c0 = self.c0 + dc0
        self.c0[self.i_p] = 0.0
        gain = (self.C ** 2).sum(0)
        # Idea 4: the input weights (from surprise, movement and the last state), with the inputs that formed
        # the state whose forecast is being corrected
        gm = g * self.gate
        x = torch.cat([self.eps, _direction(self.m), self.z1_prev], dim=-1)
        gs = gm / (gain + self.nlms_floor)
        dW = self.perception_lr * gs.t() @ (x / ((x ** 2).sum(-1, keepdim=True) + 1e-06)) / B
        self._apply_input_step(dW)
        # Idea 9: the input weights from the anticipation, from the same error (this step is not gated)
        p = float((self.ant ** 2).sum(-1).mean())
        self.ant_power = p if self.ant_power is None else (1.0 - STAT_RATE) * self.ant_power + STAT_RATE * p
        gs = g / (gain + self.nlms_floor)
        pw = (self.ant ** 2).sum(-1, keepdim=True)
        dKa = self.perception_lr * gs.t() @ (self.ant / (pw + self.ant_power + 1e-12)) / B
        self.Ka = (self.Ka + dKa).clamp(-1.0, 1.0)
        # Idea 6: the variance of the change forecast (not gated)
        e = o[:, self.i_d] - pred[:, self.i_d]
        e2 = e ** 2
        em = e2.mean(0)
        self.var_ebar = em.clone() if self.var_ebar is None else 0.99 * self.var_ebar + 0.01 * em
        var = self._forecast_var(self.z1)
        r = ((e2 - var) / torch.maximum(var, self.var_ebar.clamp_min(1e-12))).clamp(-1.0, 1000.0)
        y = self.z1 @ self.Wv.t() + self.bv
        stuck = (y >= LOG_VAR_MAX) & (r > 0) | (y <= LOG_VAR_MIN) & (r < 0)
        r = torch.where(stuck, torch.zeros_like(r), r)
        self.Wv = self.Wv + self.uncertainty_lr * (r.t() @ self.z1) / B / power
        self.bv = self.bv + self.uncertainty_lr * r.mean(0)

    def _learn_level2(self, z1_up):
        """Idea 5: level 2 learns to predict level 1's goal-free state; its own input weights learn as in Idea 4."""
        B = z1_up.shape[0]
        delta2 = z1_up - self.z2 @ self.W2.t()
        g2 = delta2 @ self.W2 * (self.z2 > 0).to(z1_up.dtype)
        dW2 = self.level2_lr * delta2.t() @ self.z2 / B / (1e-06 + float((self.z2 ** 2).sum(-1).mean()))
        self.W2 = self.W2 + dW2
        gs = g2 / ((self.W2 ** 2).sum(0) + self.nlms_floor)
        x = torch.cat([self.eps2, self.z2_prev], dim=-1)
        dW = self.level2_input_lr * gs.t() @ (x / ((x ** 2).sum(-1, keepdim=True) + 1e-06)) / B
        n1 = self.eps2.shape[-1]
        self.K2, self.A2 = _unit_rows(self.K2 + dW[:, :n1], self.A2 + dW[:, n1:])

    # ------------------------------------------------------------------ Ideas 9, 10: anticipation and memory
    def _anticipate(self, z1):
        """The anticipation: the state's part of the forecast of how much the fullness will change by the next
        eventful moment (the other part is read from the fullness itself, ``from_level``)."""
        return z1 @ self.Ca.t()

    def _anticipation_input(self, level):
        """The value fed back into the state: the anticipation read from last tick's goal-free state, limited in
        size to the width of the range of fullness seen so far."""
        lo, hi = level.min(0).values, level.max(0).values
        self.level_lo = lo.clone() if self.level_lo is None else torch.minimum(self.level_lo, lo)
        self.level_hi = hi.clone() if self.level_hi is None else torch.maximum(self.level_hi, hi)
        a = self._anticipate(self.z1_up)
        R = self.level_hi - self.level_lo
        return torch.maximum(torch.minimum(a, R), -R)

    def _update_outcome_stats(self, target, err):
        """The outcome's running mean and variance, and the forecast's running squared error. The squared error
        of one closing enters at no more than ``sample_cap`` times the running value (Idea 12)."""
        b = STAT_RATE
        e2 = (err ** 2).mean(0)
        if self.outcome_stats is None:
            self.outcome_stats = [target.mean(0).clone(), ((target - target.mean(0)) ** 2).mean(0), e2.clone()]
        else:
            mu, v, e = self.outcome_stats
            mu = (1.0 - b) * mu + b * target.mean(0)
            capped = torch.where(e > 0, torch.minimum(e2, self.sample_cap * e), e2)
            self.outcome_stats = [mu, (1.0 - b) * v + b * ((target - mu) ** 2).mean(0), (1.0 - b) * e + b * capped]

    def _correct_anticipation(self, err, closing):
        """Idea 10: the anticipation's weights step by the surprise at the remembered moment (the outcome minus what
        that moment forecast), by the delta rule, at ``hindsight_rate`` times the forecast's rate and with a limit
        on the step."""
        B = err.shape[0]
        lr = self.forecast_lr * self.hindsight_rate
        lr = lr * self.gate
        zs = self.mem_z[closing]
        dC = lr * err[closing].t() @ zs / B / (1e-06 + float((zs ** 2).sum(-1).mean()))
        nC = torch.linalg.vector_norm(self.Ca, dim=-1)
        nd = torch.linalg.vector_norm(dC, dim=-1)
        sc = torch.where(nd > 0, (self.step_limit * (nC + 1.0) / nd.clamp_min(1e-12)).clamp(max=1.0),
                         torch.ones_like(nd))                            # the step limit, per need
        self.Ca = self.Ca + dC * sc.unsqueeze(-1)

    def _correct_perception(self, err, closing, Ca_old):
        """Idea 10: the same error, sent back through the anticipation's weights, trains the input weights that
        formed the remembered state, with the remembered inputs."""
        B = err.shape[0]
        rate = self.hindsight_rate
        rate = rate * self.gate
        gain = (Ca_old ** 2).sum(0) + self.nlms_floor
        back = err @ Ca_old
        zk = self.mem_z
        gs = back * (zk > 0).to(zk.dtype) * closing.to(zk.dtype).unsqueeze(-1) / gain
        x = torch.cat([self.mem_eps, self.mem_mov, self.mem_zp], dim=-1)
        dW = rate * self.perception_lr * gs.t() @ (x / ((x ** 2).sum(-1, keepdim=True) + 1e-06)) / B
        dKa = torch.zeros_like(self.Ka)
        if self.ant_power is not None:
            al = self.mem_ant
            pw = (al ** 2).sum(-1, keepdim=True)
            dKa = dKa + rate * self.perception_lr * gs.t() @ (al / (pw + self.ant_power + 1e-12)) / B
        self._apply_input_step(dW)
        self.Ka = (self.Ka + dKa).clamp(-1.0, 1.0)

    def _memory_tick(self, o, eps, level, z1_up, m, ant):
        """One tick of memory, after both settles.

        1. Every body adds this tick's change in fullness to its open outcome. The outcome of a moment is the whole
           change in fullness from that moment to the next one: the leak, and whatever was eaten.
        2. A body whose sight or movement carries news writes a moment.
        3. A writing body that already holds a moment closes it. What that moment forecast has two parts: one read
           from the fullness at the moment (a line, ``from_level``) and one read from the remembered state (the
           anticipation). The surprise is the outcome against their sum. It corrects the anticipation's weights
           and, sent back one step, the input weights that formed the remembered state; the line is refitted to
           the outcomes themselves, so the anticipation is left with what the fullness alone does not tell.
           Nothing else is corrected.
        4. The writing bodies store the new moment (the goal-free state, the inputs that formed it and the
           fullness) and start a new outcome.
        """
        if self.l_prev is not None:
            self.mem_G = self.mem_G + (level - self.l_prev)
        news_channels = list(range(self.n_s))[self.i_e] + list(range(self.n_s))[self.i_p]
        s, self.news_q = memory.news(eps[:, news_channels], (o - self.skill_stats[0])[:, news_channels], self.news_q)
        write = s >= memory.WRITE_BAR
        closing = write & self.mem_has
        if bool(closing.any()):
            by_level = self.mem_level * self.from_level[:, 0] + self.from_level[:, 1]
            forecast = self._anticipate(self.mem_z) + by_level
            target = self.mem_G.clone()
            err = torch.where(closing.unsqueeze(-1), target - forecast, torch.zeros_like(forecast))
            Ca_old = self.Ca
            self._update_outcome_stats(target[closing], err[closing])
            self._gate_fold(float((err[closing] ** 2).mean()))
            if self.hindsight:
                self._correct_anticipation(err, closing)
                self._correct_perception(err, closing, Ca_old)
                self.from_level = memory.fit_step(self.from_level, (target - by_level)[closing], self.mem_level[closing])
            self.info["surprise_rms"] = float((err[closing] ** 2).mean().sqrt())
        idx = write.nonzero(as_tuple=True)[0]
        if idx.numel():
            self.mem_z[idx] = z1_up[idx]
            self.mem_eps[idx] = eps[idx]
            self.mem_mov[idx] = _direction(m)[idx]
            self.mem_zp[idx] = self.z1[idx]
            self.mem_ant[idx] = ant[idx]
            self.mem_level[idx] = level[idx]
            self.mem_has[idx] = True
            self.mem_G[idx] = 0.0

    # ------------------------------------------------------------------ one tick
    @torch.no_grad()
    def step(self, extero, proprio, intero):
        """One tick. ``extero`` is sight, ``proprio`` the last movement, ``intero`` the fullness of each need.

        Each is ``(bodies, dim)``. Returns the movement, ``(bodies, proprio_dim)``: a unit direction, or zero.
        Learning from last tick's forecast happens first, so this tick's state is formed with updated weights.
        """
        if extero.dim() != 2 or proprio.dim() != 2 or intero.dim() != 2:
            raise ValueError("extero, proprio, intero must be (bodies, dim)")
        B = extero.shape[0]
        dims = (self.extero_dim, self.proprio_dim, self.intero_dim)
        if (extero.shape[1], proprio.shape[1], intero.shape[1]) != dims \
                or proprio.shape[0] != B or intero.shape[0] != B:
            raise ValueError("input shapes do not match (extero, proprio, intero) = (%d, %d, %d)" % dims)
        if self._B != B:
            self._init_state(B)
        extero, proprio, intero = (x.to(self.device, torch.float32) for x in (extero, proprio, intero))
        change = intero - self.l_prev if self.l_prev is not None else torch.zeros_like(intero)
        o = torch.cat([extero, proprio, intero, change], dim=-1)
        self._inhibition()
        # Idea 1: what last tick's state forecast for this tick. The movement is known, not forecast.
        pred = self.z1 @ self.C.t()
        pred[:, self.i_d] = _center(self.z1) @ self.C[self.i_d].t()
        pred[:, self.i_p] = _direction(self.m)
        eps = o - pred                                   # surprise: drives the state (no bias)
        pred = pred + self.c0                            # the full forecast: trains the weights
        if self._t > 0:
            self._learn_level1(o, pred)
        base = self.z1 @ self.A.t() + self._attend(eps) @ self.K.t()
        ant = self._anticipation_input(intero)           # Idea 9: the anticipation comes back in as an input
        base = base + ant @ self.Ka.t()
        top_down = self.z2 @ self.W2.t()
        self._update_hold(eps)
        u_prev = self.u1
        # Ideas 3 and 7: two settles. The first includes level 2's prediction and the goal, which turns the movement.
        # It gives the movement and the state the forecast reads.
        z1, m, u1 = self._settle1(base + top_down, intero, goal=True, m_fixed=None, u_prev=u_prev)
        # The second has neither, and the movement fixed: what the agent perceives, given what it did.
        z1_up, _, _ = self._settle1(base, intero, goal=False, m_fixed=m, u_prev=u_prev)
        self._regulate(0, z1)
        u2_prev = self.u2
        if self._t > 0:
            self._learn_level2(z1_up)
        eps2 = z1_up - self.z2 @ self.W2.t()
        z2, u2 = self._settle2(self.z2 @ self.A2.t() + eps2 @ self.K2.t(), u2_prev)
        self._regulate(1, z2)
        self.u1, self.u2 = u1, u2
        self._memory_tick(o, eps, intero, z1_up, m, ant)
        self.z1_prev, self.z1, self.z1_up, self.eps, self.m = self.z1, z1, z1_up, eps, m
        self.z2_prev, self.z2, self.eps2 = self.z2, z2, eps2
        self.o_prev, self.l_prev = o, intero
        self.ant = ant
        self._t += 1
        if not (bool(torch.isfinite(z1).all()) and bool(torch.isfinite(m).all())
                and all(bool(torch.isfinite(t).all()) for t in (self.C, self.K, self.Km, self.A, self.W2, self.Ca))):
            self.diverged = True
        self.info["gate"] = self.gate
        self.info["anticipation_norm"] = float(torch.linalg.vector_norm(self.Ca, dim=-1).mean())
        self.info["firing_units"] = float((z1 > 0).to(z1.dtype).sum(-1).mean())
        return m.detach().clone()

    # ------------------------------------------------------------------ checkpoints
    _TENSORS = ("C", "c0", "K", "Km", "A", "Ka", "W2", "K2", "A2", "Wv", "bv", "Ca", "from_level")
    _OPTIONAL = ("var_ebar", "hold_mbar", "level_lo", "level_hi", "news_q")
    _SCALARS = ("ant_power", "gate_fast", "gate_slow")

    def state_dict(self) -> dict:
        """The learned weights and running statistics (not the per-body state)."""
        sd = {k: getattr(self, k).clone() for k in self._TENSORS}
        sd["theta"] = [t.clone() for t in self.theta]
        sd["skill_stats"] = [t.clone() for t in self.skill_stats]
        for k in self._OPTIONAL:
            v = getattr(self, k)
            sd[k] = None if v is None else v.clone()
        for k in self._SCALARS:
            sd[k] = getattr(self, k)
        sd["outcome_stats"] = None if self.outcome_stats is None else [t.clone() for t in self.outcome_stats]
        sd["widths"] = self.widths
        return sd

    def load_state_dict(self, sd: dict) -> None:
        if tuple(sd["widths"]) != self.widths:
            raise ValueError("checkpoint widths %r do not match %r" % (sd["widths"], self.widths))
        for k in self._TENSORS:
            if tuple(sd[k].shape) != tuple(getattr(self, k).shape):
                raise ValueError("checkpoint %s%r does not match %r"
                                 % (k, tuple(sd[k].shape), tuple(getattr(self, k).shape)))
            setattr(self, k, sd[k].clone().to(self.device))
        self.theta = [t.clone().to(self.device) for t in sd["theta"]]
        self.skill_stats = [t.clone().to(self.device) for t in sd["skill_stats"]]
        _, v, e = self.skill_stats
        self.skill = (v - e).clamp_min(0.0) / (v + 1e-08)
        for k in self._OPTIONAL:
            setattr(self, k, None if sd[k] is None else sd[k].clone().to(self.device))
        for k in self._SCALARS:
            setattr(self, k, sd[k])
        self.outcome_stats = None if sd["outcome_stats"] is None \
            else [t.clone().to(self.device) for t in sd["outcome_stats"]]
