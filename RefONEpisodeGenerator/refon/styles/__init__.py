"""Instruction style registry.

NORMAL_STYLES   : normal styles (do not terminate the episode)
TERMINAL_STYLES : goal-absent (terminal) styles
STYLE_BY_NAME   : name -> instance

To add a new style:
  1. Define a class that subclasses InstructionStyle (anywhere).
  2. Add one register(MyStyle()) line below.
The generator/balancer/builder only reference this registry, so nothing else changes.
"""
from __future__ import annotations

from typing import Dict, List

from .base import InstructionStyle, Resolved, BuildContext
from .alias_binding import AliasBindingPost, AliasBindingPre
from .anaphoric import AnaphoricPost, AnaphoricPre
from .goal_absent import AbsentObject, InvalidOrdinal, UnboundAlias
from .multi_role import AliasOrdinal
from .ordinal import OrdinalPost
from .plain import PlainGoal

NORMAL_STYLES: List[InstructionStyle] = []
TERMINAL_STYLES: List[InstructionStyle] = []
STYLE_BY_NAME: Dict[str, InstructionStyle] = {}


def register(style: InstructionStyle) -> InstructionStyle:
    if style.name in STYLE_BY_NAME:
        raise ValueError(f"duplicate style name: {style.name}")
    STYLE_BY_NAME[style.name] = style
    (TERMINAL_STYLES if style.is_terminal else NORMAL_STYLES).append(style)
    return style


# --- register normal styles ---
register(PlainGoal())
register(AliasBindingPre())
register(AliasBindingPost())
register(OrdinalPost())
register(AnaphoricPre())
register(AnaphoricPost())
register(AliasOrdinal())

# --- register goal-absent (terminal) styles ---
register(UnboundAlias())
register(AbsentObject())
register(InvalidOrdinal())


def all_styles() -> List[InstructionStyle]:
    return NORMAL_STYLES + TERMINAL_STYLES


def admissible_normal(ctx) -> List[InstructionStyle]:
    return [s for s in NORMAL_STYLES if s.admissible(ctx)]


def admissible_terminal(ctx) -> List[InstructionStyle]:
    return [s for s in TERMINAL_STYLES if s.admissible(ctx)]


__all__ = [
    "InstructionStyle",
    "Resolved",
    "BuildContext",
    "NORMAL_STYLES",
    "TERMINAL_STYLES",
    "STYLE_BY_NAME",
    "register",
    "all_styles",
    "admissible_normal",
    "admissible_terminal",
]
