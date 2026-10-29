# Limits

What this repository does not show, what is built into the agent by hand, and what would test it further. For
what it does show, see [results.md](results.md); for the short account, the [README](../README.md).

- [What was not shown](#what-was-not-shown)
- [What is built in](#what-is-built-in)
- [What would test it further](#what-would-test-it-further)

## What was not shown

- **Two small games.** Senses are 5 to 8 numbers and the movement is a direction in a plane. Nothing here shows
  that the method scales to images, long sequences of decisions or many needs.
- **No baseline.** These games are easy for reinforcement learning when a reward is given: a
  temporal-difference learner with eligibility traces would be expected to solve them. None was run, so nothing
  here compares speed or reliability with a standard method.
- **The games hand the agent its events.** Sight is blank except on cue ticks, and the movement is never news, so
  the agent's eventful moments are cue ticks only, and nearly all of them (99.97% of a body's cue ticks in the first
  30,000 ticks of every run). Nearly every interval between two of them holds one decision and one outcome. Finding
  events by prediction error was therefore not put to a test: a detector that fired whenever sight was not blank
  would divide these games almost the same way.
- **The control removes several things at once.** Switching the hindsight correction off also leaves the second
  preference with nothing to act through, and the control fails even with no delay, where the one-tick forecast
  and the outcome forecast have the same target. It shows that the agent needs the event path as it is built.
  It does not show that reaching across the delay is what that path contributes
  ([the control](results.md#the-control)).
- **No exploration.** Nothing in the agent seeks information, and no noise is added to its movement. The games
  do not ask for exploration: a movement that is partly right is partly fed. No task here shows what the agent
  does where a movement that pays a little hides one that pays more.
- **Precision mostly sits at its ceiling.** In these games every precision but sight's is at the ceiling of 10 in
  every reading, and on every tick after a run's first 180 while it learns, and sight's follows how often a cue is
  shown ([results](results.md#where-the-precisions-sat)). So the precision acts mostly as a fixed tenfold rate, and
  no run here compares it with one. It also depends on the units of the senses: sight scaled up a hundredfold would
  put its precision at the floor. And because a running mean square mixes the forecast's misfit with noise, the
  precision falls when the forecast is wrong. The agent keeps no estimate of its uncertainty about the weights,
  which is what raises a Bayesian learning rate at such times.
- **One slow run, not explained.** One run of the 27 (cue game, delay 12, seed 3) took 59,000 ticks to master, where
  the other two at that delay took 6,000 and 7,000, and its score stayed below 0.61 for its first 52,000 ticks
  ([results](results.md#all-27-runs)). Why was not tracked down. The seed alone does not make it slow: run again on
  a machine where rounding differs, the same seed mastered at tick 6,000. So rounding alone can move a time to
  mastery far, as it did for one other run, at a delay of 16, which had not mastered by the tick its published run
  mastered at ([results](results.md#seven-re-runs-are-not-copies)), and each time in the tables is one such outcome.
- **One run was continued past the cap.** That slow run reached the cap of 200,000 ticks still holding, but short of
  ten times its time to mastery, so it was continued to 660,000 ticks from the weights and running statistics it had
  at the cap, with the game and each body's state started afresh. It is the only run whose curve is joined from two
  parts ([results](results.md#all-27-runs)).
- **Most parts were not tested one by one.** One switch was run: the hindsight correction, at five of the nine
  combinations with one run each. The second level, the one-tick preference, the attention gain, the hold, the
  precisions and their bounds of 1 and 10, and the movement's one-step wait have no test of their own.
- **Holding back is barely tested.** In the cue game a body that always answers correctly sits between 0.60 and
  0.75, and 0.75 is the setpoint, so the fullness never rises past the setpoint and more food is always right.
  There, steering toward the setpoint and simply eating as much as possible are the same behavior. Only the
  two-need game asks the agent to decline a bite, and it was run only at delays of 1 and 2 ticks.
- **One delay per run, and a blank wait.** Each run has a fixed delay of at most 16 ticks, with nothing shown
  during the wait. Delays that vary within a run and longer delays were not tested. Neither were distracting
  inputs: one that the agent could not predict would count as an eventful moment and close the remembered
  moment early.
- **How learning slows past 16 ticks.** Within the tested range a longer delay takes more ticks and, less
  clearly, more trials ([results](results.md#all-27-runs)). Nothing here says how that continues.
- **A graded outcome.** The bite shrinks smoothly as the answer turns away from the cue. That slope is plausibly
  what lets the agent find the right movement with nothing added for exploration. All-or-nothing outcomes were
  not tested.
- **One remembered moment.** Each body remembers only its latest eventful moment, and the outcome is credited to
  that moment. Outcomes that overlap in time, or that depend on a moment before the latest one, are outside what
  this memory can represent.
- **A point, not only a range.** The second read asks for the middle of the band, so the agent keeps steering
  when its fullness is already inside the band. The score counts only whether fullness is inside the band, so
  it does not show how well the agent holds back a bite that would overfill a need.
- **Three seeds.** Each combination was run with three sets of random weights.

## What is built in

- **The goal.** The comfort band and its middle are given to the agent, and the score is computed from the same
  band. The shape of the cost outside the band is chosen by hand.
- **One privileged sense.** The fullness is the only sense with preferences and the only one summed into the
  delayed target. It is left out of the test for eventful moments, its attention gain is fixed at 1, and its
  change from tick to tick is given to the agent as a channel of its own.
- **32 bodies that share weights.** Several running estimates are fitted across the bodies. One of them is the
  line that forecasts an outcome from the fullness: it is a baseline whose slope one body alone could not fit.
  Other numbers of bodies, including one, were not tested.
- **Constants set by hand.** The agent has about 40 constants. 28 numbers are set through 27 arguments of
  `Agent` (`widths` holds two, and the band's two edges are among them; the others are rates, the numbers of
  units, the gain of 8, the step limit, the limit of 10 on one outcome's sample, the movement's one-step wait
  and the like). The rest are fixed in the code: the bar for an eventful moment, the line's rate, the rates of
  the running averages, the bounds of 1 and 10 on the precisions, and a few floors and clip ranges. The
  constants are the same for all 27 runs and were never set per game or per delay, but they were chosen while
  working on these two games. There is no held-out task.
- **It is a model of an idea, not of a brain.** The parts are named after what they do. No claim is made that
  they correspond to particular neural structures.

## What would test it further

None of these was run.

1. **A change of goal after learning.** The weights learn to predict the fullness, and the setpoint is applied
   only when the agent acts. So if the band is moved once a game is mastered, the movement should change at
   once, before anything is relearned. A learned value would not do that.
2. **A task where events and trials differ.** Distractions during the wait, gaps of varying length, or two
   decisions before one outcome.
3. **The parts of the event path, one at a time.** At a delay of 0 the two forecasts have the same target, so
   the differences between the two paths (the gain, the point setpoint, the rate, the line, the step into the
   input weights) can be switched singly there.
4. **A matched baseline.** A Monte Carlo or temporal-difference learner on the same state, with the change in
   fullness as its reward.
5. **Outcomes shuffled across bodies.** If the agent still learned, the extra updates and not their target
   would be doing the work.
6. **A task that needs exploration.** All-or-nothing outcomes, or a game in which a movement that pays a little
   hides one that pays more. This agent has nothing that seeks information, so there it would be expected to
   get stuck.
