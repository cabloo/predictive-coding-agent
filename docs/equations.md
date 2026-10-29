# The model in equations

Every equation here is implemented, by hand, in [`pcagent/agent.py`](../pcagent/agent.py) and
[`pcagent/memory.py`](../pcagent/memory.py); the function is named beside each one. The prose account is in
[how-it-works.md](how-it-works.md). Angle brackets are averages over the 32 bodies, which share every weight.

- [Notation](#notation)
- [The forecast and its error](#the-forecast-and-its-error)
- [Settling](#settling)
- [The two preferences](#the-two-preferences)
- [The movement](#the-movement)
- [Each error's precision](#each-errors-precision)
- [Learning on every tick](#learning-on-every-tick)
- [Events, outcomes and the hindsight error](#events-outcomes-and-the-hindsight-error)
- [Gains and running statistics](#gains-and-running-statistics)
- [What is not differentiated](#what-is-not-differentiated)

## Notation

| Symbol | Meaning |
|---|---|
| `o_t` | what is sensed on tick `t`: sight, the last movement, fullness `L_t`, and the change `ΔL_t = L_t − L_{t−1}` |
| `z_t`, `z↑_t` | level-1 state (256 units, non-negative, sparse) from the settle with the goal, and from the goal-free settle |
| `y_t` | level-2 state (64 units) |
| `m_t` | the movement, a unit direction (or zero) |
| `C`, `c0` | forecast weights and their constant term; `C_Δ` are the rows that forecast `ΔL` |
| `K`, `K_m`, `A`, `K_a` | input weights of level 1: from the prediction error, the movement, the last state and the anticipation |
| `C_a` | the anticipation's weights; `ℓ(L) = αL + β` is the line that forecasts an outcome from the fullness |
| `W_2`, `K_2`, `A_2` | level 2's prediction of level 1, and its input weights |
| `ε`, `δ` | prediction errors: `ε` drives the state, `δ` trains weights; `δ_G` is the hindsight error |
| `π`, `π_G` | the precisions: of each sensed channel's one-tick error, and of the hindsight error |
| `ω`, `λ` | two of the regulators: attention (per channel) and the hold (per body) |
| `p`, `l` | the preferences' push on the units, and the lean: the push passed through the units' time constant |

## The forecast and its error

The forecast of the next tick's senses is linear in the state (`Agent.step`):

```math
\hat o_{t+1} = C z_t + c_0
```

Two details. The rows that forecast the change in fullness read the state centered over its active units,
$`\tilde z = z - \bar z\,[z>0]`$ with $`\bar z`$ the mean of the active units. The last movement is known, so
its entries are copied, not forecast.

Two errors come from it. The first drives the next state and leaves the constant term out; the second trains
the weights:

```math
\varepsilon_t = o_t - C z_{t-1}, \qquad \delta_t = o_t - \hat o_t
```

The code's comments call $`\varepsilon`$ the surprise. It is a signed prediction error, not $`-\ln p(o)`$.

## Settling

Level 1 is driven by (`Agent.step`, `_settle_move`, `_settle_fixed`)

```math
b = A z_{t-1} + K(\omega \odot \varepsilon_t) + K_a a_t + K_m m \;\; \big(+\, W_2\, y_{t-1} \text{ in the first settle}\big)
```

where $`\omega`$ is the attention gain and $`a_t`$ the anticipation fed back from the last tick. Potentials relax
from zero for 40 steps with rate $`\eta_s = 0.1`$, and a unit is active while its potential is over its
threshold:

```math
u \leftarrow u + \eta_s\Big((1-\lambda)\,(b - \Lambda z) + \lambda\, u_{t-1} - u\Big), \qquad z = u \cdot [u > \theta]
```

$`\Lambda`$ is lateral inhibition between units whose input weights overlap (the positive part of the cosine
between their rows of $`[K \mid K_m]`$, zero on the diagonal), and $`\lambda`$ is the hold on ticks with small
errors. The settle runs twice. In the first, the movement $`m`$ is free and is inferred as the units relax
([below](#the-movement)), and level 2's prediction is part of the drive: it gives $`m_t`$ and $`z_t`$. In the
second, the movement is fixed at $`m_t`$ and level 2 takes no part: it gives $`z^{\uparrow}_t`$. The preferences
are in the drive of neither. Both settles hold toward the same $`u_{t-1}`$, the potentials the first settle
ended on at the previous tick.

Level 2 is driven by the error of its own prediction of level 1 (`_settle2`):

```math
\varepsilon^{(2)}_t = z^{\uparrow}_t - W_2\, y_{t-1}, \qquad b^{(2)} = A_2\, y_{t-1} + K_2\, \varepsilon^{(2)}_t
```

Its relaxation has the same form with three differences: it starts from last tick's potentials and not from
zero, its hold is a constant 0.9, and its inhibition comes from the rows of $`K_2`$ alone.

Thresholds move toward a firing rate of 5.6% and are kept at or above zero (`_regulate`). At level 1 each
body's tick is weighted by $`n = 1 - \lambda/0.9`$, which is 1 on a tick with the usual error and 0 on a fully
held one, and the rule reads the state from the first settle:

```math
\theta \leftarrow \max\Big(0,\ \theta + 0.01\,\big(\langle n\,[z_t > 0] \rangle - 0.056\,\langle n \rangle\big)\Big), \qquad \theta^{(2)} \leftarrow \max\Big(0,\ \theta^{(2)} + 0.01\,\big(\langle [y_t > 0] \rangle - 0.056\big)\Big)
```

## The two preferences

**A range, one tick ahead** (`_expected_slope`). Outside the comfort band $`[x_{lo}, x_{hi}]`$ of width $`R`$
the cost of a fullness at distance $`d`$ past the nearer edge is

```math
D(x) = d + \frac{d^2}{R}
```

and it is zero inside. The agent never receives $`D`$. It reads the slope of $`D`$, averaged over a normal
density for the fullness it forecasts for the next tick:

```math
\phi(z) = \mathbb{E}\big[D'(X)\big], \qquad X \sim \mathcal{N}\big(L_t + C_\Delta \tilde z + c_{0,\Delta},\ \sigma^2(z)\big), \qquad \ln \sigma^2(z) = W_v z + b_v
```

The expectation has a closed form in the normal distribution's density and cumulative function.
$`\ln \sigma^2`$ is kept in $`[\ln 10^{-8},\ 0]`$.

**A point, at the next eventful moment** (`_push`). The fullness the agent expects at the next eventful moment
is the fullness now plus the forecast outcome, and its setpoint $`x^*`$ is the middle of the band. The error is
counted in half-widths $`h = R/2`$:

```math
X^{ev}(z) = L_t + \ell(L_t) + C_a z, \qquad e(z) = \frac{x^* - X^{ev}(z)}{h}
```

This read is a point forecast: no variance enters it.

## The movement

**The push** (`_push`) says how each unit bears on the two preferences:

```math
p(z) = -\,w_1\, \mathrm{center}\big(\phi(z)\, C_\Delta\big) \;+\; g_a\, e(z)\, C_a, \qquad g_a = 8
```

It is minus the gradient, with respect to the state, of the preferences' cost

```math
G_0(z) = w_1\,\mathbb{E}\big[D(X)\big] + \tfrac{1}{2}\, g_a h\, e(z)^2
```

with the variance held fixed. `center` removes the mean over the active units.

Both terms are negative log preferences over a predicted fullness. $`\mathbb{E}[D(X)]`$ is, up to a constant,
the expected negative log of a preferred density proportional to $`e^{-D(x)}`$, which is flat over the band. The
second term is read at a point forecast and is not an expectation: its expectation under the forecast's error
would add $`\tfrac{1}{2}\kappa\,\sigma_o^2`$, with $`\sigma_o^2`$ the forecast's mean squared error, which does
not depend on the state and is left out. No uncertainty about the anticipation's weights is kept. It is
$`\tfrac{1}{2}\kappa\,(x^* - X^{ev})^2`$ with $`\kappa = g_a/h = 32`$, the negative log of a Gaussian centered
on the setpoint, so the fixed gain acts as a fixed precision of 32 (a standard deviation of about 0.18 in units
of fullness). The weight of the first term is the one-tick forecast's share of the two forecasts' skill:

```math
w_1 = \frac{s_1 + 0.05}{s_1 + s_G + 0.05}
```

with $`s_1`$ and $`s_G`$ the running fractions of variance explained by the forecast of $`\Delta L`$ and by the
forecast of the outcome (`_one_tick_weight`).

**One function.** The push is never added to the units' drive. The movement descends a function of free-energy
form in which the preferences are priors, taken in a limit. Let $`\hat z`$ be a second copy of the state, with
potentials $`\hat u`$, that the preferences may pull on. Write $`w = 1 - \lambda`$ for the weight the hold
leaves this tick, and $`c = \Lambda z`$ for the inhibition the state has settled. With the units' energy

```math
E(\zeta; m) = \tfrac{1}{2}|\zeta|^2 - \big(w\, b(m) + \lambda\, u_{t-1}\big)\cdot \zeta + w\, \zeta \cdot c + \sum_i \tfrac{1}{2}\theta_i^2\,[\zeta_i > 0]
```

the function is

```math
F(\hat z, m) = \Pi\,\Big[E(\hat z; m) - \min_{\zeta \ge 0} E(\zeta; m)\Big] + w\, G_0(\hat z)
```

The bracket is zero where the copy is the state that the drive alone settles on, and positive elsewhere, and
$`\Pi`$ is how steeply that difference is charged. The second term is the preferences' cost at the copy. The
inhibition $`c`$ and the forecast's variance are parameters of $`F`$ that perception supplies. They are held
fixed, and $`F`$ is not differentiated through them.

Descending $`F/\Pi`$ in the copy gives the relaxation
$`\hat u \leftarrow \hat u + \eta_s\big(w\,(b + p(\hat z)/\Pi - c) + \lambda u_{t-1} - \hat u\big)`$, which is the
state's own relaxation with $`p/\Pi`$ added to the drive. Its slope in the movement is
$`-\partial F/\partial m = \Pi\, w\, K_m^{\top}(\hat z - z)`$, because the minimum's own dependence on $`m`$ is
read at its minimizer. The relaxing state $`z`$ stands for that minimizer.

**The limit.** The agent takes $`\Pi \to \infty`$. The copy then cannot leave the state, $`\hat z = z`$, and the
difference of the two relaxations, scaled by $`\Pi`$, stays finite. It is the lean, $`l = \lim \Pi\,(\hat u - u)`$.
Per settle step $`k = 0 \dots 39`$, with the state, the lean and the movement all starting at zero
(`_settle_move`):

```math
q = w\, K_m^{\top}\big([z>0] \odot l\big), \qquad l \leftarrow l + \eta_s\big(w\, p(z) - l\big)
```

$`q`$ is the limit of $`-\partial F/\partial m`$ with the relaxing state $`z`$ in place of the minimizer. The two
agree once the relaxation has converged at a fixed movement, which the settle does not wait for. It uses the lean
from before this step's update. In this limit there is one state, formed by its drive alone, and the preferences
reach only the movement. Once the lean has converged, $`q = w^2 K_m^{\top}([z>0] \odot p)`$, which is exactly
$`-\,\mathrm{d}\big[w\,G_0(z(m))\big]/\mathrm{d}m`$ with the inhibition held fixed: the gradient of the preferences'
cost through the agent's own mapping from movement to state.

**The turn** (`_turn`). The movement turns toward $`q`$ on the unit circle: fast early in the settle and slowly
late. It does not turn on step $`k = 0`$:

```math
m \leftarrow \mathrm{unit}\Big(m + \xi_{k-1}\, \eta_s\,\big(q - (q \cdot m)\, m\big)\Big) \quad (k \ge 1), \qquad \xi_j = \frac{10}{1 + j/10}
```

The first step on which $`q`$ is not zero ($`k = 1`$ at the earliest) sets the movement to its direction, so
the movement turns on at most 39 of the settle's 40 steps.
The movement returned for the tick is $`m`$ after the last step. $`K_m`$ is thus used in both directions:
forward, the movement drives the state through it, and backward, the lean reaches the movement through its
transpose. It learns from the prediction errors below, like the other input weights.

## Each error's precision

The errors that train the forecast weights, the first level's input weights and the anticipation's weights are first
multiplied by their own precision (`_channel_precision`, `_closing_fold`). The second level, the variance weights,
the line and the thresholds learn without it. For the one-tick forecast there is one precision per sensed channel
$`c`$, from the running mean (rate 0.01) of that channel's squared error over the bodies:

```math
\pi_{c,t} = \mathrm{clip}\Big(\frac{1}{\overline{\delta_c^2}_{\,t-1}},\ 1,\ \pi_{max}\Big), \qquad \overline{\delta_c^2}_{\,t} = 0.99\,\overline{\delta_c^2}_{\,t-1} + 0.01\,\big\langle \delta_{c,t}^2 \big\rangle
```

The precision is read before tick $`t`$'s error enters the mean, so an error never sets its own weight. On the
first tick that learns, $`\pi = 1`$ and the mean starts at that tick's value. Below, $`\delta'_t = \pi_t \odot
\delta_t`$ is the weighted error.

The hindsight error has one precision, $`\pi_G`$, made in the same way from the running mean of
$`\langle \delta_G^2 \rangle`$ over the closing bodies and the needs, which advances once on each tick on which
some body closes a moment.

The ceiling is $`\pi_{max} = \min\big(10,\ 1/\max(\eta_f, \eta_p)\big) = 10`$, so $`\eta_f\, \pi \le 1`$.
Inside the clip, $`\pi = 1/\overline{\delta^2}`$ is where the free energy of a Gaussian error,
$`\tfrac{1}{2}(\pi\,\delta^2 - \ln \pi)`$, is lowest in $`\pi`$, with $`\delta^2`$ replaced by its running
mean. The floor of 1 and the ceiling are set by hand. The error $`\varepsilon`$ that drives the state is not
multiplied by a precision.

## Learning on every tick

All of these use the error of the forecast made on the previous tick (`_learn_level1`, `_learn_level2`), and
the first three take it weighted by precision, $`\delta'_t`$ ([above](#each-errors-precision)).

**Forecast weights**, by a normalized delta rule ($`\eta_f = 0.1`$; entries are kept in $`[-1, 1]`$, and $`c_0`$
is not clipped):

```math
\Delta C = \eta_f\, \frac{\big\langle \delta'_t\, z_{t-1}^{\top} \big\rangle}{10^{-6} + \big\langle |z_{t-1}|^2 \big\rangle}, \qquad \Delta c_0 = \eta_f\, \langle \delta'_t \rangle
```

The rows for $`\Delta L`$ use $`\tilde z_{t-1}`$ in the numerator; the denominator is the same for every row.

**Input weights** $`W = [K \mid K_m \mid A]`$ ($`\eta_p = 0.03`$). The error is sent back one step through the
forecast weights to the units that were active. Because the rows for $`\Delta L`$ read the centered state, their
part loses its mean over the active units (the bar below):

```math
g = [z_{t-1} > 0] \odot \Big(C^{\top} \delta'_t - \overline{C_\Delta^{\top}\, \delta'_{\Delta,t}}\Big)
```

Each unit then steps by that error times the input $`x_{t-1} = [\varepsilon_{t-1},\ m_{t-1},\ z_{t-2}]`$ that
formed its state (the error without the attention gain):

```math
\Delta W = \eta_p \left\langle \frac{g}{\sum_{\text{rows}} C^2 + 10^{-3}}\ \frac{x_{t-1}^{\top}}{|x_{t-1}|^2 + 10^{-6}} \right\rangle
```

$`g`$ uses $`C`$ from before its step and the sum of squares uses $`C`$ from after it. After the
step each unit's sight weights, and the rest of its input weights, are rescaled to the lengths they had before
it.

**The anticipation's input weights** take the same $`g`$ with the anticipation as input, and entries are kept
in $`[-1, 1]`$ ($`\overline{|a|^2}`$ is a running mean, rate 0.01):

```math
\Delta K_a = \eta_p \left\langle \frac{g}{\sum_{\text{rows}} C^2 + 10^{-3}}\ \frac{a_{t-1}^{\top}}{|a_{t-1}|^2 + \overline{|a|^2}} \right\rangle
```

**The variance weights** move the predicted variance toward the squared error of the change forecast. They learn
a variance, so their error is not weighted by precision ($`\overline{\delta_\Delta^2}`$ is a running mean, rate
0.01):

```math
\chi = \mathrm{clip}\!\left(\frac{\delta_{\Delta,t}^2 - \sigma^2(z_{t-1})}{\max\big(\sigma^2(z_{t-1}),\ \overline{\delta_\Delta^2}\big)},\ -1,\ 1000\right), \qquad \Delta W_v = 0.02\, \frac{\langle \chi\, z_{t-1}^{\top} \rangle}{10^{-6} + \langle |z_{t-1}|^2 \rangle}, \qquad \Delta b_v = 0.02\, \langle \chi \rangle
```

with $`\chi`$ set to zero where $`\ln \sigma^2`$ sits at an edge of its range and $`\chi`$ would push it further
out.

**Level 2** (`_learn_level2`) learns in the same form, with these differences: no precision, no constant term and
no clip on $`W_2`$, and each unit's row of $`[K_2 \mid A_2]`$ is rescaled to length 1 after the step.

```math
\delta^{(2)}_t = z^{\uparrow}_t - W_2\, y_{t-1}, \qquad \Delta W_2 = 0.02\, \frac{\big\langle \delta^{(2)}_t\, y_{t-1}^{\top} \big\rangle}{10^{-6} + \big\langle |y_{t-1}|^2 \big\rangle}
```

```math
g^{(2)} = [y_{t-1} > 0] \odot W_2^{\top} \delta^{(2)}_t, \qquad \Delta [K_2 \mid A_2] = 0.03 \left\langle \frac{g^{(2)}}{\sum_{\text{rows}} W_2^2 + 10^{-3}}\ \frac{x^{(2)\top}}{|x^{(2)}|^2 + 10^{-6}} \right\rangle, \qquad x^{(2)} = \big[\varepsilon^{(2)}_{t-1},\ y_{t-2}\big]
```

The error that trains uses $`W_2`$ from before its step. The $`\varepsilon^{(2)}_t`$ that drives level 2's settle
is computed after it.

## Events, outcomes and the hindsight error

**When a tick is eventful** (`memory.news`). For each body, over the sight and movement channels $`c`$ (the
fullness channels are left out):

```math
\mathrm{news} = \frac{\sum_c n_c}{\sum_c \bar n_c + 10^{-12}}, \qquad n_c = \min\big(\varepsilon_c^2,\ (o_c - \mu_c)^2\big)
```

where $`\mu_c`$ is the channel's running mean and $`\bar n_c`$ the running mean of $`\langle n_c \rangle`$ (rate
0.01). The tick is eventful for that body when $`\mathrm{news} \ge 0.5`$: news is what neither the forecast nor the
running mean predicted.

**The outcome** of an eventful moment at tick $`\tau`$, measured at the next one, $`\tau'`$ (`_memory_tick`):

```math
G = \sum_{t=\tau+1}^{\tau'} \Delta L_t = L_{\tau'} - L_{\tau}
```

**The hindsight error** is that outcome against what the remembered moment forecast, read with the weights as
they are at $`\tau'`$:

```math
\delta_G = G - \ell(L_\tau) - C_a\, z^{\uparrow}_\tau
```

It makes four corrections at $`\tau'`$ and changes nothing else (`_correct_anticipation`,
`_correct_perception`, `memory.fit_step`). Below, the sums run over the bodies that close a moment on this tick,
$`B = 32`$ is the number of bodies, and $`\pi_G`$ is the hindsight error's precision
([above](#each-errors-precision)), read before this tick's error enters its mean.

The anticipation's weights step by the delta rule at three times the forecast's rate, with a limit on the step
for each need:

```math
\Delta C_a = 3\,\pi_G\, \eta_f\, \frac{\tfrac{1}{B}\sum_{\text{closing}} \delta_G\, z^{\uparrow\top}_\tau}{10^{-6} + \big\langle |z^{\uparrow}_\tau|^2 \big\rangle_{\text{closing}}}, \qquad \|\Delta C_a\| \le 0.005\,\big(\|C_a\| + 1\big)
```

Where the limit binds it scales the whole step down, so it, and not $`\pi_G`$, sets the step's size.

The input weights that formed the remembered state step by the same error sent back one step, with the
remembered inputs $`x_\tau = [\varepsilon_\tau,\ m_\tau,\ z_{\tau-1}]`$ ($`C_a`$ from before its step):

```math
g_G = [z^{\uparrow}_\tau > 0] \odot C_a^{\top} \delta_G, \qquad \Delta W = 3\,\pi_G\, \eta_p\, \frac{1}{B} \sum_{\text{closing}} \frac{g_G}{\sum_{\text{rows}} C_a^2 + 10^{-3}}\ \frac{x_\tau^{\top}}{|x_\tau|^2 + 10^{-6}}
```

After this step each unit's input weights are rescaled as after the every-tick step.

The anticipation's input weights step by the same $`g_G`$ with the anticipation that was fed in at the
remembered moment (entries kept in $`[-1, 1]`$):

```math
\Delta K_a = 3\,\pi_G\, \eta_p\, \frac{1}{B} \sum_{\text{closing}} \frac{g_G}{\sum_{\text{rows}} C_a^2 + 10^{-3}}\ \frac{a_\tau^{\top}}{|a_\tau|^2 + \overline{|a|^2}}
```

The line moves a twentieth of the way toward the least-squares fit, across the bodies that close, of $`G`$ on
$`[L_\tau, 1]`$. With $`\rho = G - \ell(L_\tau)`$ and bars for means over the $`n`$ closing bodies:

```math
\psi = \frac{\sum (\rho - \bar\rho)(L_\tau - \bar L)}{\sum (L_\tau - \bar L)^2 + 10^{-4}\, n}, \qquad \Delta\alpha = 0.05\, \psi, \qquad \Delta\beta = 0.05\,\big(\bar\rho - \psi\, \bar L\big)
```

The small constant in the denominator means the slope barely moves when the bodies' fullness agree. The line is
fitted to the outcomes themselves, so $`C_a`$ is left with what the fullness alone does not predict. It is a
baseline fitted across the bodies.

The forecast is trained only at eventful moments but read on every tick: $`X^{ev}`$ above uses it at every step
of each tick's first settle, and its value is fed back into the next tick's drive, limited to
the range of fullness seen so far (`_anticipation_input`):

```math
a_{t+1} = \mathrm{clip}\big(C_a\, z^{\uparrow}_t,\ \pm(L_{max} - L_{min})\big)
```

## Gains and running statistics

None of these is an estimated inverse variance. The precisions are, and are [above](#each-errors-precision).

- **Attention** $`\omega`$ (`_attend`): between 1 and 10 per sensed channel, larger for a channel whose error
  would move the agent's other skilled forecasts. With $`S`$ the diagonal matrix of the channels' skills and
  $`K_{\cdot c}`$ the input weights from channel $`c`$, let $`v_c = S\,C\,K_{\cdot c}`$ be what a unit error on
  channel $`c`$ does, through the state, to every channel's forecast. Then

  ```math
  \mathrm{lev}_c = \sqrt{\max\big(0,\ |v_c|^2 - v_{c,c}^2\big)}, \qquad \omega_c = \min\Big(10,\ 1 + 3\,\max\big(0,\ \mathrm{lev}_c / \overline{\mathrm{lev}} - 1\big)\Big)
  ```

  with $`\overline{\mathrm{lev}}`$ the mean over channels. It is fixed at 1 for the fullness channels (the level
  and its change, for each need).
- **Hold** $`\lambda`$ (`_update_hold`), per body and tick. With $`\bar a_c`$ the running mean (rate 0.01) of
  $`\langle |\varepsilon_c| \rangle`$ and the mean below taken over all sensed channels:

  ```math
  \mathrm{sal} = \mathrm{mean}_c\ \frac{|\varepsilon_c| + 10^{-3}}{\bar a_c + 10^{-3}}, \qquad \lambda = 0.9\, \min\big(1,\ 3\,\max(0,\ 1 - \mathrm{sal})\big)
  ```

- **Skill** $`s_1`$, $`s_G`$: $`\max(0,\ v - \overline{e^2}) / v`$, from running means (rate 0.01) of a target's
  variance $`v`$ and of its forecast's squared error. The outcome's means advance once on each tick on which
  some body closes a moment.
- **The limit on one sample** (`_update_outcome_stats`): the running mean of $`\delta_G^2`$ behind $`s_G`$ takes
  one tick's sample at no more than 10 times its own value. The running mean behind $`\pi_G`$ has no such
  limit.

## What is not differentiated

Nothing is differentiated automatically, and no error is sent back through the recurrent weights (the hindsight
error is applied to one stored state). The corrections above send an
error back through exactly one set of weights ($`C`$, $`C_a`$ or $`W_2`$) to the units that were active, and
stop there. The lean reaches the movement through $`K_m`$ on the active units only. The function $`F`$ holds the
inhibition the state has settled and the forecast's variance as parameters. The precisions multiply errors and
are not differentiated.
