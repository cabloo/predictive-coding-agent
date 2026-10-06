import pytest

from pcagent import scoring


def curve(scores, every=1000):
    return [{"tick": (i + 1) * every, "score": s} for i, s in enumerate(scores)]


def test_the_bar_sits_nine_tenths_of_the_way_to_the_oracle():
    assert scoring.solved_bar({"floor": 0.4, "oracle": 1.0}) == pytest.approx(0.94)


def test_one_point_above_the_bar_is_not_mastery():
    assert scoring.ticks_to_mastery(curve([0.2, 0.95, 0.3, 0.95, 0.95, 0.4]), 0.9) is None
    assert scoring.ticks_to_mastery(curve([0.2, 0.95, 0.3, 0.95, 0.95, 0.91]), 0.9) == 4000


def test_a_curve_that_never_masters_has_no_hold():
    assert scoring.hold_ratio(curve([0.1] * 10), 0.9) is None


def test_a_run_that_ends_while_holding_is_censored():
    h = scoring.hold_ratio(curve([0.5, 0.95] + [0.97] * 30), 0.9)
    assert (h["ttm"], h["hold"], h["censored"]) == (2000, 30000, True)
    assert h["R"] == pytest.approx(15.0)
    assert h["held"] is True


def test_a_short_run_that_is_still_holding_is_undetermined():
    h = scoring.hold_ratio(curve([0.5, 0.95] + [0.97] * 6), 0.9)
    assert h["censored"] and h["R"] == pytest.approx(3.0)
    assert h["held"] is None


def test_a_sustained_loss_ends_the_hold_and_a_brief_dip_does_not():
    scores = [0.95] * 5 + [0.5] + [0.95] * 4 + [0.5, 0.5, 0.5] + [0.95] * 3
    h = scoring.hold_ratio(curve(scores), 0.9)
    assert h["ttm"] == 1000
    assert (h["hold"], h["censored"]) == (10000, False)      # the three points below the bar start at 11,000
    assert h["held"] is True                                # R = 10
    assert (h["hold_first"], h["censored_first"]) == (5000, False)
    early = scoring.hold_ratio(curve([0.95] * 3 + [0.5] * 3), 0.9)
    assert early["held"] is False


def test_a_mastered_run_continues_for_the_longer_of_the_two_holds():
    c = curve([0.95] * 3)
    assert not scoring.run_until(c, 0.9, tick=3000)
    assert not scoring.run_until(c, 0.9, tick=100_000)
    assert scoring.run_until(c, 0.9, tick=101_000)           # 1,000 + max(20 * 1,000, 100,000)
    late = curve([0.1] * 9 + [0.95] * 3)                     # mastered at 10,000
    assert not scoring.run_until(late, 0.9, tick=209_000)
    assert scoring.run_until(late, 0.9, tick=210_000)        # 10,000 + 20 * 10,000
    assert not scoring.run_until(curve([0.1] * 5), 0.9, tick=10 ** 9)
