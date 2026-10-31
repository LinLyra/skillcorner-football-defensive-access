"""Defensive Access: auditable football tracking metrics."""

from .features import AccessConfig, analyse_frame
from .intervals import labels_to_spells, spell_f1
from .temporal import DefensiveStateDecoder

__all__ = [
    "AccessConfig",
    "DefensiveStateDecoder",
    "analyse_frame",
    "labels_to_spells",
    "spell_f1",
]

__version__ = "0.1.0"

