"""Which ticks the agent treats as eventful in the two games, as docs/how-it-works.md and docs/results.md say."""
import pytest
import torch

from pcagent import Agent, CueGame, TwoNeedGame, memory


def eventful(game, ticks, monkeypatch):
    """Run a fresh agent and return a (ticks, bodies) table: was the tick eventful for the body?"""
    agent = Agent(game.n_extero, game.n_proprio, game.n_intero, seed=1)
    real, seen = memory.news, []

    def spy(*args):
        s, q = real(*args)
        seen.append(s >= memory.WRITE_BAR)
        return s, q
    monkeypatch.setattr(memory, "news", spy)
    for _ in range(ticks):
        game.step(agent.step(game.extero(), game.proprio(), game.intero()))
    return torch.stack(seen)


@pytest.mark.parametrize("cls, delay", [(CueGame, 4), (TwoNeedGame, 2)])
def test_the_eventful_ticks_are_exactly_the_cue_ticks(cls, delay, monkeypatch):
    table = eventful(cls(8, seed=1, delay=delay), 240, monkeypatch)
    cue = torch.tensor([t % (delay + 1) == 0 for t in range(240)])
    assert bool(table[cue].all()) and not bool(table[~cue].any())


def test_with_no_delay_nearly_every_tick_is_eventful(monkeypatch):
    table = eventful(CueGame(8, seed=1, delay=0), 240, monkeypatch)
    assert float(table.float().mean()) > 0.95
