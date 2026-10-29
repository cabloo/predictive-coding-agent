# Results

Every number in the tables and in the sections about the runs comes from
[`results/acceptance.json`](../results/acceptance.json), and a test holds the page to that file.
`python -m pcagent.report results/acceptance.json` prints the tables.

- [How a run is judged](#how-a-run-is-judged)
- [All 27 runs](#all-27-runs)
- [Where the precisions sat](#where-the-precisions-sat)
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
mastery if that is longer. Runs are capped at 200,000 ticks, and one run was continued past the cap
([below](#all-27-runs)). The **hold ratio** `R` is how long mastery lasted, divided by how long it took to reach;
mastery ends at three windows in a row below the bar. A run **holds** if `R ≥ 10`. "≥" means the run ended while
still holding.

## All 27 runs

**All 27 runs mastered and held, and none lost mastery.** Time to mastery ranged from 1,000 to 59,000 ticks,
with a median of 2,000.

<!-- results:begin -->
| Game | Delay (ticks) | Pass bar | Mastered at tick, by seed | Hold ratio R, lowest seed | Windows below the bar after mastery | Lowest window after mastery |
|---|---|---|---|---|---|---|
| cue | 0 | 0.900 | 1,000, 1,000, 1,000 | ≥ 100.0 | 0 of 303 | 0.984 |
| cue | 1 | 0.900 | 1,000, 1,000, 1,000 | ≥ 100.0 | 0 of 303 | 0.946 |
| cue | 2 | 0.900 | 2,000, 2,000, 2,000 | ≥ 50.0 | 0 of 303 | 0.992 |
| cue | 4 | 0.900 | 3,000, 3,000, 4,000 | ≥ 25.0 | 0 of 303 | 0.990 |
| cue | 8 | 0.900 | 7,000, 2,000, 3,000 | ≥ 20.0 | 1 of 343 | 0.884 |
| cue | 12 | 0.900 | 6,000, 7,000, 59,000 | ≥ 10.2 | 0 of 864 | 0.910 |
| cue | 16 | 0.900 | 6,000, 15,000, 5,000 | ≥ 12.3 | 1 of 408 | 0.893 |
| two-need | 1 | 0.944 | 1,000, 1,000, 1,000 | ≥ 100.0 | 0 of 303 | 0.957 |
| two-need | 2 | 0.943 | 1,000, 1,000, 1,000 | ≥ 100.0 | 0 of 303 | 0.948 |
<!-- results:end -->

Reading the table:

- **A longer delay mostly takes longer to learn.** In the cue game, mastery takes 1,000 to 4,000 ticks at
  delays of 0 to 4, and 2,000 to 15,000 ticks at delays of 8 to 16, apart from one run at 59,000. That measure
  is coarse: it is taken to the nearest 1,000 ticks, and 12 of the 27 runs mastered inside the first window. A
  finer one comes from the same curves. One minus the score, summed over the windows up to mastery, is the
  average number of ticks a body spent outside the band before mastery, and a trial is `delay + 1` ticks. In
  the cue game a body spent 9 to 16 trials outside the band at a delay of 0, 16 to 27 at 1, 36 to 139 at 2,
  114 to 232 at 4, 52 to 445 at 8, 117 to 3,278 at 12, and 88 to 509 at 16. Counted that way, the cost rises
  from a delay of 0 to a delay of 4, and the spans at delays of 4 to 16 overlap. (In the cue game every body
  starts inside the band, at 0.75, and leaves it within two trials if it eats nothing.)
- **One run was slow.** In the cue game at a delay of 12, seed 3 mastered at 59,000 ticks, where the other two seeds
  took 6,000 and 7,000, and spent 3,278 trials outside the band, where they spent 165 and 117. Its score stayed
  between 0.034 and 0.606 for its first 52 windows, reached 0.968 at tick 53,000, fell to 0.030 at tick 56,000, and
  rose past the bar at tick 59,000. Why it was slow was not tracked down, but the seed alone does not make it slow:
  run again on a machine where rounding differs, the same seed mastered at tick 6,000
  ([below](#seven-re-runs-are-not-copies)).
- **After mastery the score fell below the bar twice.** 2 of 3,433 windows over all runs fell below the bar
  after mastery, each a single window: in the cue game at a delay of 8, seed 2 at tick 5,000 (0.884), and at a
  delay of 16, seed 1 at tick 30,000 (0.893). No run lost mastery, so every `R` in the table is marked "≥".
- **The slow run was continued past the cap.** At the cap of 200,000 ticks it had held mastery for 141,000
  ticks, so its `R` was 2.4: still holding, but short of the 10 that counts as held. It was continued to 660,000
  ticks, past ten times its time to mastery, from the weights and running statistics it had at the cap, with
  the game and each body's state started afresh (the research program saves those and not each body's state or
  memory). In the 460 windows after the join the score never fell below the bar, the lowest being 0.916, and the
  run ended still holding, with `R` ≥ 10.2. It is the only run longer than 200,000 ticks, and the only one whose
  curve is joined from two parts.

## Where the precisions sat

Each of the 27 runs was run again from its start, up to its mastery or to tick 5,000 if it mastered sooner, with
every precision it used recorded (each channel's on every tick, memory's at every closing) and summed per 1,000-tick
window ([how a precision is made](how-it-works.md#each-error-is-weighted-by-its-own-precision)). 20 of these re-runs
scored exactly as their published runs did in every window. The other 7 had first been run on a machine the re-runs
did not use, and rounding differs between machines, so they are other runs of the same agent and seed
([below](#seven-re-runs-are-not-copies)). The first table covers each re-run up to its own mastery: the mean
precision used and, in brackets, the share of uses at the ceiling of 10, over the three seeds. A re-run equal to its
published run is dated by that run's mastery tick, which the published run's later windows confirm.

<!-- learning:begin -->
| Game | Delay (ticks) | Re-runs equal to the published run | Re-run mastered at tick, by seed | Sight | Fullness | Change | Memory's |
|---|---|---|---|---|---|---|---|
| cue | 0 | 3 of 3 | 1,000, 1,000, 1,000 | 2.0 (0%) | 9.4 (87%) | 10.0 (>99%) | 10.0 (>99%) |
| cue | 1 | 3 of 3 | 1,000, 1,000, 1,000 | 4.6 (5%) | 9.2 (84%) | 10.0 (>99%) | 10.0 (>99%) |
| cue | 2 | 3 of 3 | 2,000, 2,000, 2,000 | 6.1 (4%) | 9.6 (92%) | 10.0 (>99%) | 10.0 (>99%) |
| cue | 4 | 3 of 3 | 3,000, 3,000, 4,000 | 9.2 (7%) | 9.7 (95%) | 10.0 (>99%) | 10.0 (>99%) |
| cue | 8 | 3 of 3 | 7,000, 2,000, 3,000 | 10.0 (>99%) | 9.8 (96%) | 10.0 (>99%) | 10.0 (>99%) |
| cue | 12 | 2 of 3 | 6,000, 7,000, 6,000 | 10.0 (>99%) | 9.9 (97%) | 10.0 (>99%) | 10.0 (>99%) |
| cue | 16 | 2 of 3 | 6,000, 15,000, not by 5,000 | 10.0 (>99%) | 9.9 (98%) | 10.0 (>99%) | 10.0 (>99%) |
| two-need | 1 | 0 of 3 | 1,000, 1,000, 1,000 | 4.6 (5%) | 9.1 (82%) | 10.0 (>99%) | 10.0 (>99%) |
| two-need | 2 | 1 of 3 | 1,000, 1,000, 1,000 | 6.5 (8%) | 9.1 (82%) | 10.0 (>99%) | 10.0 (>99%) |
<!-- learning:end -->

Each run also saved, every 5,000 ticks, the running averages its precisions are read from. The second table gives,
at each of those readings, the precision each channel would use on its next tick and the one memory would use at
its next closing.

<!-- precision:begin -->
| Game | Delay (ticks) | Readings | Sight's precision, median, by seed | Sight's readings at the ceiling | Fullness, change and memory: readings at the ceiling |
|---|---|---|---|---|---|
| cue | 0 | 60 | 2.0, 2.0, 2.0 | 0% | 100% |
| cue | 1 | 60 | 4.0, 4.0, 4.0 | 0% | 100% |
| cue | 2 | 60 | 5.9, 5.9, 5.9 | 0% | 100% |
| cue | 4 | 60 | 9.8, 9.8, 9.8 | 38% | 100% |
| cue | 8 | 69 | 10.0, 10.0, 10.0 | 100% | 100% |
| cue | 12 | 186 | 10.0, 10.0, 10.0 | 100% | 100% |
| cue | 16 | 85 | 10.0, 10.0, 10.0 | 100% | 100% |
| two-need | 1 | 60 | 4.0, 4.0, 3.9 | 0% | 100% |
| two-need | 2 | 60 | 5.8, 5.8, 5.8 | 0% | 100% |
<!-- precision:end -->

- **The fullness, its change and memory sat at the ceiling, apart from the start.** The fullness's precision was
  below the ceiling only in its first 130 to 180 ticks, while its running average still held the large errors of
  an untrained forecast: between 1% and 18% of a run's ticks up to mastery, the most where mastery came
  soonest. The change's and memory's precisions were below the ceiling only at their first use, which is weighted
  at 1. After that all three were at the ceiling on every tick recorded and in every reading of every run. Their
  errors are small in the units the games use, so for them the ceiling, not the estimate, sets the rate.
- **The movement's precision changes nothing.** The movement is copied, not forecast, so its error is zero and its
  precision scales no step. The tables leave it out.
- **Sight's precision follows how often a cue is shown.** The forecast cannot know which way the next cue will
  point, so on a cue tick its error is about the whole cue, a mean square of about 0.5 per channel, and between cues
  it is near zero. One precision covers every tick, so it sat near 2 × (delay + 1): at the readings, 2.0 at a delay
  of 0, 4.0 at 1, 5.9 at 2 and 9.8 at 4 in the cue game, and at the ceiling from a delay of 8. Up to mastery it was
  much the same: a little higher in the first 1,000 ticks at delays of 1 and 2, and a little lower at a delay of 4
  (9.2 against 9.8). At a long delay, then, a cue tick's error is learned from at a precision set by the blank ticks
  around it.
- **So in these games precision acts mostly as a tenfold rate.** On every tick recorded after a run's first 180, and
  at every reading, a fixed precision of 10 would have differed from this agent only in sight's precision at delays
  of 0 to 4. No run here compares the two. The precision also depends on the units of the senses: sight scaled up a
  hundredfold would put its precision at the floor.

### Seven re-runs are not copies

These 7 runs had first been run on a machine the re-runs did not use. In five of the two-need runs the re-run's
score differed from the published one by less than 0.008 in any window, and it mastered at the same tick. In the cue
game at a delay of 16, seed 3, the re-run had not mastered by tick 5,000, where the published run did. At a delay
of 12, seed 3, the slow run, the re-run mastered at tick 6,000 instead of 59,000. Those rows of the first table
describe the re-runs, up to their own mastery.

## The control

**With the hindsight correction switched off (`hindsight=False`), the same agent mastered nothing at the five
combinations where it was run (one run each, with seeds 1, 1, 2, 3 and 1 in the table's order).** Memory still
records moments and outcomes, but corrects no weight, so the anticipation's weights and the line stay at zero.

<!-- control:begin -->
| Game | Delay (ticks) | Pass bar | Best window | Mean score | Ran for (ticks) | Outcome |
|---|---|---|---|---|---|---|
| cue | 0 | 0.900 | 0.297 | 0.036 | 100,000 | never mastered |
| cue | 2 | 0.900 | 0.164 | 0.002 | 200,000 | never mastered |
| cue | 8 | 0.900 | 0.090 | 0.006 | 200,000 | never mastered |
| cue | 16 | 0.900 | 0.038 | 0.001 | 200,000 | never mastered |
| two-need | 2 | 0.943 | 0.240 | 0.070 | 200,000 | never mastered |
<!-- control:end -->

Four of the five control runs went the full 200,000 ticks. The research program stopped the one at a delay of 0 at
tick 100,000, under a rule that `pcagent.run` does not have: from tick 100,000 on, a run that has not mastered is
stopped once its best window has come a fifth of the way from the floor to the pass bar and its latest three windows
or more have all fallen back to within a tenth of that way. `pcagent.run` would have run it to 200,000 ticks. None
of the five ended with weights that were no longer finite.

What the control does and does not show:

- **It removes the event path as a whole.** With the anticipation's weights at zero, the second preference has
  nothing to act through, so the control acts on the one-tick read of the comfort band alone. Several things go
  with it: the fixed gain of 8, the point setpoint (the band's cost is flat inside the band, and the setpoint's
  error is not), the tripled rate, the line and the hindsight step into the input weights. The control does not
  say which of these matters.
- **Precision is not removed.** Each channel's precision is kept, and it scales the same steps as in the agent.
  Memory's precision is kept too, but as memory corrects no weight, it scales no step.
- **It fails even with no delay, where memory is not needed.** In the cue game at a delay of 0, the control's score
  rose from 0.053 in its first window to 0.297 at tick 22,000, then fell back: after tick 43,000 no window was above
  0.02. There a cue arrives on every tick, nearly every tick is an eventful moment, and the outcome is the next
  tick's change in fullness: the one-tick forecast and the outcome forecast have the same target. So the control
  shows that this agent needs the event path as it is built. It does not show that reaching across the delay is
  what that path contributes. Telling the two apart needs controls that were not run: the differences listed above
  switched one at a time at a delay of 0.
- **One run each.** Each control combination was run once.
- **In the cue game every run fell to zero.** At delays of 2 and 8 the best window came early (0.164 at tick 3,000
  and 0.090 at tick 11,000). At a delay of 16 it was the first one (0.038), where every body starts inside the
  band. All four runs scored 0.000 over their last 40,000 ticks. With hindsight on, the same seeds mastered these
  four combinations at ticks 1,000, 2,000, 2,000 and 5,000.
- **In the two-need game it stays below the floor.** The control scores 0.070 on average and 0.240 in its best
  window, at tick 187,000, and it was still rising at the end: 0.147 on average over the last 40,000 ticks, against
  0.073 over the 40,000 before. No window reaches that game's floor of 0.427, the best score of the reference
  policies that ignore the body's fullness. With hindsight on, the same seed mastered it at tick 1,000.

## Every curve

![Learning curves for every game and delay](acceptance.svg)

*Each panel is one game at one delay, with ticks on a log scale. Each point is the mean score over 1,000 ticks,
so where a curve starts above the bar, the learning happened inside the first window. Blue: three runs from
different random weights. Gray, dashed (five panels): the control, one run each. Dotted, across: the pass bar.
In the panel for the cue game at a delay of 12, which runs to 660,000 ticks, a dotted line upright marks tick
200,000, where the slow run was continued ([above](#all-27-runs)).*

## Reproducing a row

Run `python -m pcagent.run --game cue --delay 8 --seed 2` (seeds 1 to 3; `--game two-need` for the other game;
`--no-hindsight` for the control). Results are exactly reproducible on one machine. Across machines,
floating-point differences grow during learning and can move a curve a long way: the slow run, run again on
another machine, mastered at tick 6,000 instead of 59,000 ([the re-runs](#seven-re-runs-are-not-copies)). The
runner stops at the cap of 200,000 ticks and cannot continue a saved run, so the part of the slow run after the
cap is not reproducible with it.

## How the numbers were checked

The curves were produced by the research version of this agent. The code in this repository is that program
reduced to the one configuration reported here, and the reduction was checked:

- **The agent is the same program.** The research agent and this one were given the same inputs, tick by tick,
  and after every tick each weight, each state value, each running average, each precision and the movement
  were compared. They were bit-identical in 24 runs: 19 with hindsight on (17 of 4,000 ticks, in both games, at
  every delay in the table and with all three seeds at some, and 2 of 20,000 ticks) and 5 of 4,000 ticks with
  hindsight off (the five control combinations), 128,000 ticks in all. Of the files compared, the published ones
  differ only in the agent's comments and documentation, and the comparison was repeated on them in 6 more runs
  of 4,000 ticks (4 with hindsight on, 2 off): again bit-identical, with both planted changes below seen again.
- **The comparison can fail.** Two changes were planted, and each was seen. With one constant changed by one
  part in a thousand (the gain of 8 set to 8.008), the first difference appeared at tick 4. With each precision
  read after this tick's error had entered its running mean square, where the agent reads it before, the first
  difference appeared at tick 1.
- **The precisions were exercised.** In every compared run memory's precision left its starting value of 1
  within 2 to 34 ticks and reached the ceiling of 10. Sight's precision ranged from 1 to the ceiling in 22 of the
  24 runs; in the other 2, at a delay of 0, it reached 2.1. So the comparison covers the precisions moving and
  being clipped at the top.
- **The whole pipeline gives the same curve.** The research harness and `pcagent.run` (this repository's games,
  reference policies, scoring and stopping rule) were run on the same machine. The floor, the oracle, the pass bar
  and every point of the curve were equal: over 20,000 ticks at three combinations with hindsight on and at one
  with it off, over three shorter runs, and over one complete run (cue game, delay 2, seed 1), which both programs
  stopped at tick 102,000, 100,000 ticks after mastery at tick 2,000.
- **The games are the same.** For the same movements, both games return the same senses, credits and scores as
  the research versions at five delays each, and the reference scores are equal.

These comparisons need the research program, so they are not among this repository's tests.
