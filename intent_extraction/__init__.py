"""Syntax-based intent extraction for agent execution traces."""
from .trf import TrfIntentExtractor
from .string_screen import StringStepScreen
from .single_tree import SingleTreeIntentExtractor
from .probabilistic import PCFGIntentExtractor
from .stanza_beam import StanzaBeamExtractor

DefaultIntentExtractor = TrfIntentExtractor

__all__ = [
    "StringStepScreen", "DefaultIntentExtractor", "TrfIntentExtractor",
    "SingleTreeIntentExtractor", "PCFGIntentExtractor", "StanzaBeamExtractor",
    "IntentExtractor", "chain_frames", "project_frames",
]


def __getattr__(name):
    # The legacy Stanza dependency is only needed by existing chain experiments.
    if name in {"IntentExtractor", "chain_frames", "project_frames"}:
        from . import extractor
        return getattr(extractor, name)
    raise AttributeError(name)
