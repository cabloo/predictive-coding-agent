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
def test_only_cue_ticks_are_eventful_and_nearly_all_of_them_are(cls, delay, monkeypatch):
    """No tick between two cues is eventful. A body can let a cue tick pass, when its forecast had already partly
    predicted that cue, so the cue ticks are held to a share and not to all of them."""
    table = eventful(cls(8, seed=1, delay=delay), 240, monkeypatch)
    cue = torch.tensor([t % (delay + 1) == 0 for t in range(240)])
    assert not bool(table[~cue].any())
    assert float(table[cue].float().mean()) > 0.95


def test_with_no_delay_nearly_every_tick_is_eventful(monkeypatch):
    table = eventful(CueGame(8, seed=1, delay=0), 240, monkeypatch)
    assert float(table.float().mean()) > 0.95
