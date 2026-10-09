# How it relates to predictive coding, active inference and reinforcement learning

The agent borrows its outline from predictive coding and active inference, departs from the textbook form of
each, and is easy to place in reinforcement-learning terms. The departures are listed so that nothing is claimed
that the code does not do. The mechanisms themselves are in [how-it-works.md](how-it-works.md) and, as update
rules, in [equations.md](equations.md).

One thing first. The two games here are easy for reinforcement learning when a reward is given: a
temporal-difference learner with eligibility traces would be expected to solve them, and none was run for
comparison. What is unusual is the set of constraints the agent works under. No reward is delivered, no value is
bootstrapped, no noise is added to its actions, and no error is sent back through time. It is not free of an
objective: it is built with a cost on one sensed variable, which stands where a reward function would (Friston,
Daunizeau and Kiebel, 2009, discuss how the two framings relate).

- [Predictive coding](#predictive-coding)
- [Active inference](#active-inference)
- [Events and retrospective learning](#events-and-retrospective-learning)
- [Reinforcement learning](#reinforcement-learning)
- [References](#references)

## Predictive coding

Predictive coding (Rao and Ballard, 1999) describes a hierarchy in which each level predicts the level below
and receives the error of that prediction.

**What the agent keeps.** The senses drive the state only as prediction errors, every weight learns from a
prediction error, and a higher level predicts a lower one and sends its prediction down.

**Where it differs.**

- Predictions run across time, not only down the hierarchy, as in temporal forms of predictive coding (Rao and
  Ballard, 1997). A state predicts the *next* tick's senses, and the error of that forecast is the input to the
  next state.
- Settling is a relaxation of thresholded units, with inhibition between units whose input weights overlap, in
  the style of sparse coding by local competition (Olshausen and Field, 1996; Rozell et al., 2008). The
  forecast's error is the input to this circuit. The relaxation does not itself reduce that error, and there
  are no separate error units.
- The input weights learn by one step of error backpropagation through the forecast weights. Whittington and
  Bogacz (2017) show that a predictive-coding network for supervised learning, by relaxing its error units,
  arrives at approximately this quantity, so that the weight change needs only local Hebbian plasticity. This
  agent computes it directly, with the transpose of the forecast weights.
- Ordinary learning is local in time: each update uses one tick's error. The hindsight step is the exception.
  It applies an error that is known only later to a stored copy of one earlier moment. Seen from a synapse,
  that step has three parts: a scalar error that arrives later, the unit's own anticipation weight, and the
  stored input. A neuroscientist may read it as a three-factor rule with a one-slot eligibility trace that does
  not decay (Gerstner et al., 2018). In the model's own terms it is a prediction error sent back one step
  through the weights that made the prediction.
- Errors are not weighted by an estimated precision (an inverse variance). Three hand-shaped gains stand where
  precision would, and none of them is one. An attention gain multiplies the sensory prediction errors that
  drive the state; it measures how much a channel's error would change the agent's other forecasts, not the
  error's variance, so it is not attention as precision in the sense of Feldman and Friston (2010). A skill
  weight sets the balance between the two preferences. A fixed gain of 8 multiplies the setpoint's error; in the
  cost it is a fixed precision of 32 on the setpoint, set by hand. The learned variance is used in one place
  only, where the comfort band is read.
- One global number, the learning-rate gate, multiplies the learning rate of the first level's forecast weights,
  most of its input weights, and memory's corrections of the anticipation's weights and of the input weights (the
  fit of the line is not gated). It changes the size of a step and never its direction. It is a global signal,
  not a local one, and can be read as a global third factor on the learning rate.
- The fullness is not one sense among equals. It is the only sense with preferences, the only one summed into
  the delayed target, it is left out of event detection, its attention gain is fixed at 1, and its change is
  given to the agent as a channel of its own. The current fullness is also read directly in three places, none
  of which drives the state with it: by the preferences, which act on the movement; by memory, which measures
  outcomes; and to bound the anticipation that is fed back.

## Active inference

Active inference (Friston, 2010; Friston et al., 2017) treats goals as prior preferences over what the agent
senses, and action as a way of making the senses match predictions.

**What the agent keeps.** The preferences are over an interoceptive observation, the fullness (compare Pezzulo,
Rigoli and Friston, 2015; Seth and Friston, 2016): a range for the fullness one tick ahead, and a point, the
middle of that range, for the fullness expected at the next eventful moment. No reward enters. The movement is
found while the state settles, with no separate actor and no comparison of policies.

Both preferences are prior preferences in form. The cost the movement descends
([equations](equations.md#the-movement)) is the expected negative log of a preferred density over the fullness
the agent predicts, which is the negative of what Friston et al. (2015) call extrinsic or pragmatic value. For
the range, the preferred density is flat over the comfort band and falls off outside it, and the expectation is
taken under the forecast's learned normal density. For the point, it is a Gaussian centered on the band's middle
with a precision of 32 (a standard deviation of about 0.18 in units of fullness), read at a point forecast.

**Where it differs.**

- The preferences steer only the movement. In active inference a prior belief about what will be sensed is part
  of the free energy that perception minimizes too, so it also shapes the state estimate, and prior beliefs can
  stand in for a cost function altogether (Friston, Samothrakis and Montague, 2012). Here the preferences are
  not added to the units' drive, and what the agent perceives comes from a second settle that leaves them out.
- No free-energy functional is defined. The settle is a relaxation driven by prediction errors, with competition
  between units, and the preferences are not a term in what the state minimizes. Perception and action share no
  single functional.
- The shape of the band's preferred density, the precision of 32 and the weight between the two preferences are
  set by hand. None is estimated.
- Action descends the expected negative log preference, not free energy. In the continuous formulation (Friston et al., 2010) action sees
  only sensory prediction errors, and reflexes fulfill a predicted sensation. Here the movement is found by
  approximate gradient descent on the cost through a learned forward mapping. That is the forward-model
  construction of Jordan and Rumelhart (1992), with the gradient used to set the action itself and not to train
  a controller. The only thing it shares with the continuous formulation is that no policies are compared. The descent is 40 annealed steps on the unit circle, with the inhibition between units, the hold
  and the thresholds treated as fixed.
- Nothing is epistemic. The cost has the pragmatic term only, and no term seeks information.
- The movement is a command that the body executes and reports back, not a predicted sensation.
- No distribution over movements or over hidden causes is formed. The state and the movement are point
  estimates.

**Anticipation.** The second preference makes the regulation anticipatory: the movement at a cue is set by a
learned forecast of where the fullness will be at the next eventful moment
([how](how-it-works.md#anticipation-corrected-in-hindsight)), before any shortfall is sensed. Sterling (2012)
argues that efficient regulation works by anticipating needs in this way, and calls it allostasis. That name is
not claimed here, for two reasons. The setpoint is fixed. And in the cue game the fullness never rises past the
setpoint, so steering toward the setpoint and simply eating more are the same behavior; only the two-need game
asks for anything else ([limits](limits.md)). Pezzulo, Rigoli and Friston (2015) and Tschantz et al. (2022) lay
out homeostatic, allostatic and goal-directed control within active inference.

## Events and retrospective learning

The agent's own prediction errors divide time into events: a tick is eventful when its sight or movement was
predicted neither by the forecast nor by a running mean. Event segmentation theory (Zacks et al., 2007) places
the boundaries of perceived events at transient rises in prediction error, and the agent's rule is a crude form
of that idea. The agent does not model when events will come. It detects them.

In the two games here that rule was not put to a test. Sight is blank except on cue ticks, and the movement is
never news, so the eventful ticks are exactly the cue ticks, and every interval between two of them holds one
decision and one outcome. A detector that fired whenever sight was not blank would segment these games the same
way. An event clock of this kind is, in reinforcement-learning terms, a semi-Markov step (Sutton, Precup and
Singh, 1999).

Learning then runs backward from one event to the one before it: the outcome measured at the later event
becomes the target for a forecast made at the earlier one. Learning that looks back from an outcome to what
preceded it has been proposed as an account of mesolimbic dopamine (Jeong et al., 2022), as an alternative to
the temporal-difference account (Montague, Dayan and Sejnowski, 1996). This agent's version is much narrower: one
stored state per body, one kind of outcome, and attribution to the latest event only.

## Reinforcement learning

**What the forecast is.** The outcome forecast, the line plus the anticipation, is a general value function
(Sutton et al., 2011). The signal it sums is the change in fullness, the sum ends at the next eventful moment,
and it is learned by regression on the measured sum under the agent's own behavior: a Monte Carlo target, with
no bootstrapping from the next prediction (compare Sutton, 1988). The line is a baseline fitted across the 32
bodies, which leaves the anticipation with the residual that depends on the cue and the movement. The
remembered moment is a one-slot eligibility trace.

**What the action is.** A point estimate found by gradient descent on a quadratic cost of that forecast, at the
moment of acting, through the learned forward mapping (compare Jordan and Rumelhart, 1992). There is no separate
policy. The mapping from cue to movement lives in the forward weights:
the input weights from the movement, whose transpose carries the gradient back to the movement, and the
anticipation's weights. The hindsight error trains both. No noise is added to the movement. The first movements
are whatever the random initial weights produce, and the graded outcome is plausibly what lets learning start
from there ([limits](limits.md)).

**Is it a value?** In the cue game, where more food is always right, the forecast is functionally a value, and
the behavior is the one a reward-maximizing agent would learn. What differs is where the preference enters. The
weights learn to predict the fullness, and the setpoint is applied only when the agent acts. This is the split
between prediction and preference that the successor representation makes for states (Dayan, 1993). It predicts
a property that a cached value does not have: after a change of setpoint, behavior should change before
anything is relearned. That was not tested. Homeostatic reinforcement learning (Keramati and Gutkin, 2014) also
derives what is good from an interoceptive variable, but in the other order: there the reward is the reduction
of a drive, and its value is learned.

**The delay.** Learning takes more trials at longer delays here ([results](results.md#all-27-runs)). The games
do not lengthen the gap between trials along with the delay, so they do not test the timescale invariance of
acquisition found in animal conditioning (Gallistel and Gibbon, 2000).

**The name.** "Hindsight" names one correction and should not be read as either of two established methods.
Hindsight credit assignment (Harutyunyan et al., 2019) assigns credit to past decisions by how likely they were
to have led to the observed outcome; this agent learns no such likelihood, and uses the measured outcome as a
regression target for one forecast at one remembered state. Hindsight experience replay (Andrychowicz et al.,
2017) relabels goals in stored transitions; this agent replays nothing.

## References

- Andrychowicz, M., Wolski, F., Ray, A., Schneider, J., Fong, R., Welinder, P., McGrew, B., Tobin, J., Abbeel,
  P., and Zaremba, W. (2017). Hindsight experience replay. *Advances in Neural Information Processing Systems*,
  30. <https://arxiv.org/abs/1707.01495>
- Dayan, P. (1993). Improving generalization for temporal difference learning: the successor representation.
  *Neural Computation*, 5(4), 613–624. <https://doi.org/10.1162/neco.1993.5.4.613>
- Feldman, H., and Friston, K. J. (2010). Attention, uncertainty, and free-energy. *Frontiers in Human
  Neuroscience*, 4, 215. <https://doi.org/10.3389/fnhum.2010.00215>
- Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11,
  127–138. <https://doi.org/10.1038/nrn2787>
- Friston, K. J., Daunizeau, J., and Kiebel, S. J. (2009). Reinforcement learning or active inference? *PLoS
  ONE*, 4(7), e6421. <https://doi.org/10.1371/journal.pone.0006421>
- Friston, K. J., Daunizeau, J., Kilner, J., and Kiebel, S. J. (2010). Action and behavior: a free-energy
  formulation. *Biological Cybernetics*, 102(3), 227–260. <https://doi.org/10.1007/s00422-010-0364-z>
- Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P., and Pezzulo, G. (2017). Active inference: a process
  theory. *Neural Computation*, 29(1), 1–49. <https://doi.org/10.1162/NECO_a_00912>
- Friston, K., Rigoli, F., Ognibene, D., Mathys, C., FitzGerald, T., and Pezzulo, G. (2015). Active inference and
  epistemic value. *Cognitive Neuroscience*, 6(4), 187–214. <https://doi.org/10.1080/17588928.2015.1020053>
- Friston, K., Samothrakis, S., and Montague, R. (2012). Active inference and agency: optimal control without
  cost functions. *Biological Cybernetics*, 106(8–9), 523–541. <https://doi.org/10.1007/s00422-012-0512-8>
- Gallistel, C. R., and Gibbon, J. (2000). Time, rate, and conditioning. *Psychological Review*, 107(2),
  289–344. <https://doi.org/10.1037/0033-295X.107.2.289>
- Gerstner, W., Lehmann, M., Liakoni, V., Corneil, D., and Brea, J. (2018). Eligibility traces and plasticity on
  behavioral time scales: experimental support of neoHebbian three-factor learning rules. *Frontiers in Neural
  Circuits*, 12, 53. <https://doi.org/10.3389/fncir.2018.00053>
- Harutyunyan, A., Dabney, W., Mesnard, T., Azar, M., Piot, B., Heess, N., van Hasselt, H., Wayne, G., Singh, S.,
  Precup, D., and Munos, R. (2019). Hindsight credit assignment. *Advances in Neural Information Processing
  Systems*, 32. <https://arxiv.org/abs/1912.02503>
- Jeong, H., Taylor, A., Floeder, J. R., Lohmann, M., Mihalas, S., Wu, B., Zhou, M., Burke, D. A., and
  Namboodiri, V. M. K. (2022). Mesolimbic dopamine release conveys causal associations. *Science*, 378(6626),
  eabq6740. <https://doi.org/10.1126/science.abq6740>
- Jordan, M. I., and Rumelhart, D. E. (1992). Forward models: supervised learning with a distal teacher.
  *Cognitive Science*, 16(3), 307–354. <https://doi.org/10.1207/s15516709cog1603_1>
- Keramati, M., and Gutkin, B. (2014). Homeostatic reinforcement learning for integrating reward collection and
  physiological stability. *eLife*, 3, e04811. <https://doi.org/10.7554/eLife.04811>
- Montague, P. R., Dayan, P., and Sejnowski, T. J. (1996). A framework for mesencephalic dopamine systems based
  on predictive Hebbian learning. *Journal of Neuroscience*, 16(5), 1936–1947.
  <https://doi.org/10.1523/JNEUROSCI.16-05-01936.1996>
- Olshausen, B. A., and Field, D. J. (1996). Emergence of simple-cell receptive field properties by learning a
  sparse code for natural images. *Nature*, 381, 607–609. <https://doi.org/10.1038/381607a0>
- Pezzulo, G., Rigoli, F., and Friston, K. (2015). Active Inference, homeostatic regulation and adaptive
  behavioural control. *Progress in Neurobiology*, 134, 17–35. <https://doi.org/10.1016/j.pneurobio.2015.09.001>
- Rao, R. P. N., and Ballard, D. H. (1997). Dynamic model of visual recognition predicts neural response
  properties in the visual cortex. *Neural Computation*, 9(4), 721–763.
  <https://doi.org/10.1162/neco.1997.9.4.721>
- Rao, R. P. N., and Ballard, D. H. (1999). Predictive coding in the visual cortex: a functional interpretation
  of some extra-classical receptive-field effects. *Nature Neuroscience*, 2, 79–87. <https://doi.org/10.1038/4580>
- Rozell, C. J., Johnson, D. H., Baraniuk, R. G., and Olshausen, B. A. (2008). Sparse coding via thresholding and
  local competition in neural circuits. *Neural Computation*, 20(10), 2526–2563.
  <https://doi.org/10.1162/neco.2008.03-07-486>
- Seth, A. K., and Friston, K. J. (2016). Active interoceptive inference and the emotional brain. *Philosophical
  Transactions of the Royal Society B*, 371(1708), 20160007. <https://doi.org/10.1098/rstb.2016.0007>
- Sterling, P. (2012). Allostasis: a model of predictive regulation. *Physiology & Behavior*, 106(1), 5–15.
  <https://doi.org/10.1016/j.physbeh.2011.06.004>
- Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. *Machine Learning*, 3(1),
  9–44. <https://doi.org/10.1007/BF00115009>
- Sutton, R. S., Modayil, J., Delp, M., Degris, T., Pilarski, P. M., White, A., and Precup, D. (2011). Horde: a
  scalable real-time architecture for learning knowledge from unsupervised sensorimotor interaction.
  *Proceedings of the 10th International Conference on Autonomous Agents and Multiagent Systems*, 761–768.
- Sutton, R. S., Precup, D., and Singh, S. (1999). Between MDPs and semi-MDPs: a framework for temporal
  abstraction in reinforcement learning. *Artificial Intelligence*, 112(1–2), 181–211.
  <https://doi.org/10.1016/S0004-3702(99)00052-1>
- Tschantz, A., Barca, L., Maisto, D., Buckley, C. L., Seth, A. K., and Pezzulo, G. (2022). Simulating
  homeostatic, allostatic and goal-directed forms of interoceptive control using active inference. *Biological
  Psychology*, 169, 108266. <https://doi.org/10.1016/j.biopsycho.2022.108266>
- Whittington, J. C. R., and Bogacz, R. (2017). An approximation of the error backpropagation algorithm in a
  predictive coding network with local Hebbian synaptic plasticity. *Neural Computation*, 29(5), 1229–1262.
  <https://doi.org/10.1162/NECO_a_00949>
- Zacks, J. M., Speer, N. K., Swallow, K. M., Braver, T. S., and Reynolds, J. R. (2007). Event perception: a
  mind-brain perspective. *Psychological Bulletin*, 133(2), 273–293.
  <https://doi.org/10.1037/0033-2909.133.2.273>
