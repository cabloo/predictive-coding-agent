import os

import pytest
import torch

from pcagent import Agent, CueGame, TwoNeedGame
from pcagent.run import run

WEIGHTS = ("C", "c0", "K", "Km", "A", "Ka", "W2", "K2", "A2", "Wv", "bv", "Ca", "from_level")


def play(agent, game, ticks):
    cmds = []
    for _ in range(ticks):
        cmd = agent.step(game.extero(), game.proprio(), game.intero())
        game.step(cmd)
        cmds.append(cmd)
    return torch.stack(cmds)


def make(cls=CueGame, seed=1, delay=1, bodies=16, **kw):
    game = cls(bodies, seed=seed, delay=delay)
    return Agent(game.n_extero, game.n_proprio, game.n_intero, seed=seed, **kw), game


def test_the_same_seed_gives_the_same_run():
    a = play(*make(seed=4), 40)
    b = play(*make(seed=4), 40)
    c = play(*make(seed=5), 40)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


@pytest.mark.parametrize("cls", [CueGame, TwoNeedGame])
def test_the_movement_is_a_unit_direction_or_nothing(cls):
    cmds = play(*make(cls), 60)
    norms = torch.linalg.vector_norm(cmds, dim=-1)
    assert bool(((norms - 1.0).abs() < 1e-5).logical_or(norms == 0).all())
    assert bool(torch.isfinite(cmds).all())


@pytest.mark.parametrize("cls", [CueGame, TwoNeedGame])
def test_every_weight_learns(cls):
    """A weight no rule reaches would make every result downstream of it meaningless: each tensor must move."""
    agent, game = make(cls)
    born = {k: getattr(agent, k).clone() for k in WEIGHTS}
    theta = [t.clone() for t in agent.theta]
    play(agent, game, 150)
    still = [k for k in WEIGHTS if torch.equal(born[k], getattr(agent, k))]
    assert still == []
    assert not torch.equal(theta[0], agent.theta[0]) and not torch.equal(theta[1], agent.theta[1])
    assert not agent.diverged


def test_nothing_uses_autograd():
    """Every update is written out by hand: the source never asks torch for a gradient."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("agent.py", "memory.py"):
        src = open(os.path.join(here, "pcagent", name), encoding="utf-8").read()
        for word in ("backward(", "autograd", "requires_grad", "enable_grad", "torch.optim", "torch.func",
                     ".grad(", ".grad "):
            assert word not in src, (name, word)


def test_the_movement_is_never_forecast():
    """The last movement is known to the agent, so its forecast rows stay exactly zero."""
    agent, game = make()
    play(agent, game, 100)
    assert float(agent.C[agent.i_p].abs().max()) == 0.0
    assert float(agent.c0[agent.i_p].abs().max()) == 0.0


def test_with_hindsight_off_the_anticipation_stays_at_zero():
    agent, game = make(hindsight=False)
    play(agent, game, 150)
    assert float(agent.Ca.abs().max()) == 0.0 and float(agent.from_level.abs().max()) == 0.0
    assert float(agent.ant.abs().max()) == 0.0
    assert agent.outcome_stats is not None            # memory still records outcomes


def test_a_checkpoint_restores_the_weights():
    agent, game = make()
    play(agent, game, 60)
    other, _ = make(seed=9)
    other.load_state_dict(agent.state_dict())
    for k in WEIGHTS + ("skill",):
        assert torch.equal(getattr(agent, k), getattr(other, k)), k
    assert all(torch.equal(a, b) for a, b in zip(agent.theta, other.theta))
    assert other.gate_fast == agent.gate_fast and other.gate_slow == agent.gate_slow


def test_a_checkpoint_of_another_size_is_refused():
    agent, _ = make()
    small = Agent(2, 2, 1, widths=(64, 16))
    with pytest.raises(ValueError):
        small.load_state_dict(agent.state_dict())


def test_wrong_input_shapes_are_refused():
    agent, game = make()
    with pytest.raises(ValueError):
        agent.step(torch.zeros(16, 3), game.proprio(), game.intero())
    with pytest.raises(ValueError):
        agent.step(game.extero()[0], game.proprio(), game.intero())
    with pytest.raises(ValueError):
        Agent(2, 2, 1, band_lo=1.0, band_hi=0.5)


@pytest.mark.slow
def test_it_learns_a_delayed_outcome_from_scratch():
    """About a minute on one core: the cue game with the outcome one tick after the answer."""
    torch.set_num_threads(1)
    res = run("cue", delay=1, seed=1, ticks=3000, log=lambda *_: None)
    assert res["anchors"]["floor"] < 0.05
    assert [c["score"] >= res["bar"] for c in res["curve"][1:]] == [True, True]


@pytest.mark.slow
def test_without_hindsight_it_does_not():
    torch.set_num_threads(1)
    res = run("cue", delay=2, seed=1, ticks=2000, hindsight=False, log=lambda *_: None)
    assert max(c["score"] for c in res["curve"]) < 0.5
