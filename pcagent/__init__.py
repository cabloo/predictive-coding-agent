"""A predictive-coding agent that learns to act for a delayed outcome by hindsight anticipation."""
from .agent import Agent
from .worlds import BAND_HI, BAND_LO, GAMES, CueGame, TwoNeedGame, anchors

__all__ = ["Agent", "CueGame", "TwoNeedGame", "GAMES", "anchors", "BAND_LO", "BAND_HI"]
__version__ = "0.1.0"
