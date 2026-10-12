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
- **The control is not in yet, and it removes several things at once.** The runs with the hindsight
  correction switched off had not finished when this page was written. Switching it off also leaves the second
  preference with nothing to act through. So the control can show that the agent needs the event path as it is
  built. It cannot show that reaching across the delay is what that path contributes
  ([the control](results.md#the-control)).
- **The information term is not shown to help.** In these two games it is a cost: the same agent without it
  masters every game with less than a third of the time outside the band on the way, and more steadily
  ([results](results.md#what-the-information-term-costs)). The games do not ask for exploration, and no task
  here shows what the term is worth where an agent without it gets stuck. Whether its avoiding side was ever
  active in these runs was not recorded, and it was run at one weighting against the preferences only.
- **One late loss of mastery, not explained.** One run of the 27 fell below the bar for four windows in a row
  after 138,000 ticks of holding, and came back by itself ([results](results.md#all-27-runs)). Why was not
  tracked down.
- **Most parts were not tested one by one.** One switch was run, the information term, at every combination
  with three seeds; the hindsight correction's control is not in yet. The second level, the one-tick
  preference, the attention gain, the hold, the learning-rate gate (two-sided, and its falling side alone for
  the input weights), and the movement's one-step wait have no test of their own.
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
  what lets the agent find the right movement: the agent without the information term finds it too, and
  sooner. All-or-nothing outcomes were not tested.
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
  line that forecasts an outcome from the fullness: it is a baseline that one body alone could not compute.
  Other numbers of bodies, including one, were not tested.
- **Constants set by hand.** The agent has about 40 constants: 29 are set through arguments of `Agent` (rates,
  the numbers of units, the gain of 8, the step limit, the limit of 10 on one outcome's sample, the movement's
  one-step wait), and the rest are fixed in the code (the threshold for an eventful moment, the rates of the
  running averages). The information term adds none: its sizes come from running statistics. The constants are
  the same for all 27 runs and were never set per game or per delay, but they were chosen while working on
  these two games. There is no held-out task.
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
6. **A task that needs exploration.** All-or-nothing outcomes, or a game in which a movement that pays a little
   hides one that pays more. There the information term should earn what it costs here, and the agent without
   it should get stuck.
