# Results

Every number in the tables and in the sections about the runs comes from
[`results/acceptance.json`](../results/acceptance.json), and a test holds the page to that file. (One exception:
the tick at which a control run stopped, 24,976, lies inside the window after that file's last point.)
`python -m pcagent.report results/acceptance.json` prints both tables.

- [How a run is judged](#how-a-run-is-judged)
- [All 27 runs](#all-27-runs)
- [The control](#the-control)
- [Every curve](#every-curve)
- [Reproducing a row](#reproducing-a-row)
- [How the numbers were checked](#how-the-numbers-were-checked)

## How a run is judged

A score is the mean share of bodies in the comfort band over a window of 1,000 ticks. Each combination of game
and delay has two reference scores, measured on that game:

- the *oracle*, a hand-written policy that knows the rules (1.000 in every case here);
- the *floor*, the best score among policies that ignore what the game is about: a fixed movement and a random
  one, which ignore the cue, and, in the two-need game, twelve policies that read the cues but never the body's
  fullness.

The **pass bar** is 90% of the way from the floor to the oracle. A run has **mastered** the game at the first of
three windows in a row at or above the bar. It then continues for at least 100,000 ticks, or 20 times its time to
mastery if that is longer (runs are capped at 200,000 ticks). The **hold ratio** `R` is how long mastery lasted,
divided by how long it took to reach; mastery ends at three windows in a row below the bar. A run **holds** if
`R ≥ 10`. "≥" means the run ended while still holding.

## All 27 runs

**All 27 runs mastered and held. None lost mastery.** Time to mastery ranged from 1,000 to 13,000 ticks, with a
median of 2,000.

<!-- results:begin -->
| Game | Delay (ticks) | Pass bar | Mastered at tick, by seed | Hold ratio R, lowest seed | Windows below the bar after mastery | Lowest window after mastery |
|---|---|---|---|---|---|---|
| cue | 0 | 0.900 | 1,000, 6,000, 1,000 | ≥ 20.0 | 0 of 323 | 0.907 |
| cue | 1 | 0.900 | 1,000, 1,000, 1,000 | ≥ 100.0 | 0 of 303 | 0.956 |
| cue | 2 | 0.900 | 2,000, 2,000, 2,000 | ≥ 50.0 | 0 of 303 | 0.996 |
| cue | 4 | 0.900 | 3,000, 2,000, 1,000 | ≥ 33.3 | 0 of 303 | 0.915 |
| cue | 8 | 0.900 | 4,000, 4,000, 3,000 | ≥ 25.0 | 0 of 303 | 0.948 |
| cue | 12 | 0.900 | 6,000, 6,000, 4,000 | ≥ 20.0 | 0 of 343 | 0.929 |
| cue | 16 | 0.900 | 13,000, 6,000, 11,000 | ≥ 14.4 | 1 of 499 | 0.871 |
| two-need | 1 | 0.944 | 1,000, 1,000, 1,000 | ≥ 100.0 | 0 of 303 | 0.955 |
| two-need | 2 | 0.943 | 1,000, 1,000, 1,000 | ≥ 100.0 | 0 of 303 | 0.969 |
<!-- results:end -->

Reading the table:

- **A longer delay takes longer to learn, in ticks and in trials.** Mastery takes 1,000 to 3,000 ticks at delays
  of 1 to 4, and 3,000 to 13,000 ticks at delays of 8 to 16. That measure is too coarse to count trials with: it
  is taken to the nearest 1,000 ticks, and 12 of the 27 runs mastered inside the first window. A finer one comes
  from the same curves. One minus the score, summed over the windows up to mastery, is the average number of
  ticks a body spent outside the band before mastery, and a trial is `delay + 1` ticks. In the cue game a body
  spent 16 to 22 trials outside the band at a delay of 1, 36 to 58 at 2, 17 to 67 at 4, 81 to 108 at 8,
  92 to 120 at 12, and 89 to 160 at 16. (There every body starts inside the band, at 0.75, and leaves it within
  two trials if it eats nothing.)
- **With no delay, one seed was slow.** At a delay of 0 a trial is a single tick. Seeds 1 and 3 spent 29 and 40
  trials outside the band, in line with the short delays. Seed 2 spent 845.
- **After mastery the score is steady.** At a delay of 16, a single window fell below the bar (1 of 2,983
  windows over all runs, the lowest 0.871), and the score came back in the next window.
- **The seeds differ most at the two ends.** At a delay of 0, seed 2 mastered at 6,000 ticks, where the other
  two seeds took 1,000. At a delay of 16 the three seeds took 13,000, 6,000 and 11,000.

## The control

**With hindsight switched off, the same agent mastered nothing at the five combinations where it was run (one
run each, with seeds 1, 1, 2, 3 and 1 in the table's order).** Memory still records moments and outcomes, but
corrects no weight, so the anticipation and the line stay at zero.

<!-- control:begin -->
| Game | Delay (ticks) | Pass bar | Best window | Mean score | Ran for (ticks) | Outcome |
|---|---|---|---|---|---|---|
| cue | 0 | 0.900 | 0.084 | 0.002 | 200,000 | never mastered |
| cue | 2 | 0.900 | 0.022 | 0.001 | 200,000 | never mastered |
| cue | 8 | 0.900 | 0.070 | 0.002 | 200,000 | never mastered |
| cue | 16 | 0.900 | 0.034 | 0.002 | 24,000 | never mastered; weights became non-finite |
| two-need | 2 | 0.943 | 0.625 | 0.354 | 100,000 | never mastered |
<!-- control:end -->

Three of the five control runs went the full 200,000 ticks. The one at a delay of 16 stopped at tick 24,976,
when its weights were no longer finite, and the two-need run was stopped at 100,000 ticks, after its score had
fallen from a peak of 0.625 to about 0.2, below that game's floor.

What the control does and does not show:

- **It removes the event path as a whole.** With the anticipation's weights at zero, the second preference has
  nothing to act through, so the control acts on the one-tick read of the comfort band alone. Several things
  go with it: the fixed gain of 8, the point setpoint (the band's cost is flat inside the band, and the
  setpoint's error is not), the tripled rate, the line, and the hindsight step into the input weights. The
  control does not say which of these matters.
- **The gate is not removed.** The learning-rate gate keeps running in the control. With the anticipation's
  weights and the line at zero it is driven by the size of the outcomes themselves, and it still scales the
  ordinary learning rates between 1 and 10.
- **It fails even with no delay, where memory is not needed.** In the cue game the control stays near zero at a
  delay of 0. There a cue arrives on every tick, nearly every tick is an eventful moment, and the outcome is the
  next tick's change in fullness: the one-tick forecast and the outcome forecast have the same target. So the
  control shows that this agent needs the event path as it is built. It does not show that reaching across the
  delay is what that path contributes. Telling the two apart needs controls that were not run: the differences
  listed above switched one at a time at a delay of 0, with the gate held at 1.
- **One run each.** Each control combination was run once. The run at a delay of 16 ended with weights that were
  no longer finite, and why was not tracked down.
- **In the two-need game it rises and falls back.** The control scores 0.354 on average, below that
  game's floor of 0.427. Its best window, 0.625, is above the floor and far below the bar of 0.943.

## Every curve

![Learning curves for every game and delay](acceptance.svg)

*Each panel is one game at one delay. Each point is the mean score over 1,000 ticks, so where a curve starts
above the bar, the learning happened inside the first window. Blue: three runs from different random weights.
Gray (five panels): the control, one run each. Dotted: the pass bar.*

## Reproducing a row

Run `python -m pcagent.run --game cue --delay 8 --seed 2` (seeds 1 to 3; `--game two-need` for the other game;
`--no-hindsight` for the control). Results are exactly reproducible on one machine. Across machines,
floating-point differences can shift a curve slightly, because small differences grow during learning.

## How the numbers were checked

The curves were produced by the research version of this agent. The code in this repository is that program
reduced to the one configuration reported here, and the reduction was checked:

- **The agent is the same program.** The research agent and this one were given the same inputs, tick by tick,
  and after every tick each weight, each state value, each running average and the movement were compared. They
  were bit-identical in 17 runs of 4,000 ticks with hindsight on (both games, every delay in the table, all
  three seeds at some) and in 5 runs of 4,000 ticks with hindsight off (the five control combinations):
  88,000 ticks in all.
- **The comparison can fail.** With one constant of this agent changed by one part in a thousand (the gain of 8
  set to 8.008), the first difference appeared at tick 4.
- **The whole pipeline gives the same curve.** The research harness and `pcagent.run` (this repository's games,
  reference policies, scoring and stopping rule) were run on the same machine. The floor, the oracle, the pass
  bar and every point of the curve were equal: over 20,000 ticks at three combinations with hindsight on and at
  one with it off, and over one complete run (cue game, delay 2, seed 1), which both programs ended at tick
  102,000 with the same result.
- **The games are the same.** For the same movements, both games return the same senses, credits and scores as
  the research versions at five delays each, and the reference scores are equal.

These comparisons need the research program, so they are not among this repository's tests.
