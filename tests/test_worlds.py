import math

import pytest
import torch

from pcagent import scoring
from pcagent.worlds import BAND_HI, BAND_LO, CueGame, TwoNeedGame, anchors, grade


def test_grade_falls_linearly_with_the_angle():
    cue = torch.tensor([[1.0, 0.0]] * 5)
    cmd = torch.tensor([[1.0, 0.0], [math.cos(math.pi / 4), math.sin(math.pi / 4)], [0.0, 1.0], [-1.0, 0.0],
                        [0.0, 0.0]])
    got = grade(cmd, cue)
    assert got.tolist() == pytest.approx([1.0, 0.5, 0.0, 0.0, 0.0], abs=1e-4)


def test_grade_ignores_the_size_of_the_movement():
    cue = torch.tensor([[0.0, 1.0]])
    assert float(grade(torch.tensor([[0.0, 7.0]]), cue)) == pytest.approx(1.0)


@pytest.mark.parametrize("delay", [0, 3])
def test_the_cue_is_shown_on_the_cue_tick_only(delay):
    game = CueGame(8, seed=1, delay=delay)
    for t in range(4 * (delay + 1)):
        shown = bool(game.extero().abs().sum() > 0)
        assert shown == (t % (delay + 1) == 0)
        game.step(torch.zeros(8, 2))


@pytest.mark.parametrize("delay", [0, 1, 5])
def test_the_bite_lands_after_the_delay(delay):
    game = CueGame(4, seed=3, delay=delay)
    levels = [float(game.intero()[0, 0])]
    for t in range(delay + 1):
        game.step(game.oracle_cmd() if t == 0 else torch.zeros(4, 2))
        levels.append(float(game.intero()[0, 0]))
    falls = [b < a for a, b in zip(levels, levels[1:])]
    assert falls == [True] * delay + [False]          # leaks through the delay, then the bite


@pytest.mark.parametrize("cls", [CueGame, TwoNeedGame])
def test_a_game_is_reproducible_by_seed(cls):
    a, b, c = cls(16, seed=5, delay=2), cls(16, seed=5, delay=2), cls(16, seed=6, delay=2)
    assert torch.equal(a.extero(), b.extero())
    assert not torch.equal(a.extero(), c.extero())


def test_the_cue_game_oracle_stays_in_the_band_and_guessing_does_not():
    a = anchors(lambda B, s: CueGame(B, seed=s, delay=4), B=32, ticks=600)
    assert a["oracle"] == 1.0
    assert a["floor"] < 0.1
    assert a["floor"] == max(a["blind_fixed"], a["blind_random"])
    assert 0.9 < scoring.solved_bar(a) < 1.0


def test_the_two_need_game_cannot_be_solved_without_reading_the_body():
    a = anchors(lambda B, s: TwoNeedGame(B, seed=s, delay=1), B=32, ticks=3000)
    assert a["oracle"] == 1.0
    blind = {k: v for k, v in a.items() if k.startswith("blind_")}
    assert len(blind) == 14                              # fixed, random, four cue policies, eight schedules
    assert a["floor"] == max(blind.values())
    assert a["floor"] < 0.5


def test_the_two_need_cues_point_in_different_directions():
    game = TwoNeedGame(64, seed=2, delay=0)
    for _ in range(20):
        f, w = game.cues()
        assert bool(((f - w).abs().sum(-1) > 0).all())
        game.step(game.oracle_cmd())


def test_two_need_levels_are_capped():
    game = TwoNeedGame(8, seed=1, delay=0)
    for _ in range(40):
        game.step(game.cues()[0])                        # always eat: food overshoots the band
    assert float(game.intero()[:, 0].max()) <= TwoNeedGame.LEVEL_MAX
    assert float(game.intero()[:, 0].max()) > BAND_HI
    assert float(game.intero()[:, 1].max()) < BAND_LO    # and water is neglected
    assert game.score() == 0.0


def test_a_negative_delay_is_refused():
    with pytest.raises(ValueError):
        CueGame(4, delay=-1)
    with pytest.raises(ValueError):
        TwoNeedGame(4, delay=-1)
