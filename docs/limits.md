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
- **The games hand the agent its events.** Sight is blank except on cue ticks, and the movement is never news,
  so the agent's eventful moments are exactly the cue ticks, and every interval between two of them holds one
  decision and one outcome. Finding events by prediction error was therefore not put to a test: a detector that
  fired whenever sight was not blank would divide these games the same way.
- **The control removes several things at once.** Switching the hindsight correction off also leaves the second
  preference with nothing to act through, and the control fails even with no delay, where the one-tick forecast
  and the outcome forecast have the same target. It shows that the agent needs the event path as it is built.
  It does not show that reaching across the delay is what that path contributes
  ([the control](results.md#the-control)).
- **The parts were not tested one by one.** Apart from the hindsight switch, which was run at five of the nine
  combinations with one run each, this repository contains no comparison of the parts. The second level, the
  one-tick preference, the attention gain, the hold and the learning-rate gate have no test of their own.
- **Holding back is barely tested.** In the cue game a body that always answers correctly sits between 0.60 and
  0.75, and 0.75 is the setpoint, so the fullness never rises past the setpoint and more food is always right.
  There, steering toward the setpoint and simply eating as much as possible are the same behavior. Only the
  two-need game asks the agent to decline a bite, and it was run only at delays of 1 and 2 ticks.
- **One delay per run, and a blank wait.** Each run has a fixed delay of at most 16 ticks, with nothing shown
  during the wait. Delays that vary within a run and longer delays were not tested. Neither were distracting
  inputs: one that the agent could not predict would count as an eventful moment and close the remembered
  moment early.
- **Learning is slower at longer delays.** A longer delay takes more trials, not only more ticks
  ([results](results.md#all-27-runs)).
- **A graded outcome.** The bite shrinks smoothly as the answer turns away from the cue. That slope is plausibly
  what lets an agent with no exploration find the right movement. All-or-nothing outcomes were not tested.
- **One remembered moment.** Each body remembers only its latest eventful moment, and the outcome is credited to
  that moment. Outcomes that overlap in time, or that depend on a moment before the latest one, are outside what
  this memory can represent.
- **A point, not only a range.** The second read asks for the middle of the band, so the agent keeps steering
  when its fullness is already inside the band. In the two-need game, on trials where a whole bite would
  overfill a need, the research runs show it still taking 5% to 6% of a bite on average. The score counts only
  whether fullness is inside the band, so it does not show this.
- **Three seeds.** Each combination was run with three sets of random weights.

## What is built in

- **The goal.** The comfort band and its middle are given to the agent, and the score is computed from the same
  band. The shape of the cost outside the band is chosen by hand.
- **One privileged sense.** The fullness is the only sense with preferences and the only one summed into the
  delayed target. It is left out of the test for eventful moments, its attention gain is fixed at 1, and its
  change from tick to tick is given to the agent as a channel of its own.
- **32 bodies that share weights.** Several running estimates are fitted across the bodies. One of them is the
  line that forecasts an outcome from the fullness: it is a baseline that one body alone could not compute.
  Other numbers of bodies, including one, were not tested.
- **Constants set by hand.** The agent has about 40 constants: 28 are set through arguments of `Agent` (rates,
  the numbers of units, the gain of 8, the step limit, the limit of 10 on one outcome's sample), and the rest
  are fixed in the code (the threshold for an eventful moment, the rates of the running averages). They are the
  same for all 27 runs and were never set per game or per delay, but they were chosen while working on these two
  games. There is no held-out task.
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
   input weights) can be switched singly there, with the learning-rate gate held at 1.
4. **A matched baseline.** A Monte Carlo or temporal-difference learner on the same state, with the change in
   fullness as its reward.
5. **Outcomes shuffled across bodies.** If the agent still learned, the extra updates and not their target
   would be doing the work.
