"""InstructionStyle — abstract class representing one instruction style (role).

Per the requirements (description.md), 'each subgoal kind is a class'. A single
class is responsible for:
  (a) admissible(ctx)   : can this style appear at this position (automaton state)? (constraints)
  (b) apply(ctx, rng)   : automaton transition + fix structural params -> StyleToken (command 1)
  (c) resolve(token, b) : map the token's reference structure onto a real scene object (command 2)
  (d) render(resolved)  : produce the actual natural-language instruction (command 2)

To add a new style, subclass this class and register it in styles/__init__.py.
The generator/balancer/builder only consult the registry, so no other code changes.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Protocol

from ..automaton import GenContext
from ..style_token import StyleToken


@dataclass
class Resolved:
    """Result of resolving one token onto a scene in command 2."""
    role: str
    order: int
    category: Optional[str] = None     # object category to display (None for goal-absent unbound_alias)
    object_id: Optional[str] = None    # resolved instance handle (None for goal-absent)
    alias: Optional[str] = None
    ref_order: Optional[int] = None
    ordinal_k: Optional[int] = None
    goal_absent: bool = False
    reason: Optional[str] = None       # reason for goal-absent
    instruction: str = ""              # render() output
    extra: dict = field(default_factory=dict)


class BuildContext(Protocol):
    """Scene-level build context used by resolve() (implemented by episode_builder)."""

    rng: random.Random

    def choose_new_object(self) -> "tuple[str, str]":
        """Pick an unused scene object and return (object_id, category)."""
        ...

    def object_at(self, order: int) -> "tuple[Optional[str], Optional[str]]":
        """Return the (object_id, category) at a visit order (for back-ref resolution)."""
        ...

    def bind_alias(self, alias: str, object_id: Optional[str], category: Optional[str]) -> None:
        ...

    def sample_absent_category(self) -> str:
        """A plausible category name that does NOT exist in the scene (for absent_object goal-absent)."""
        ...


class InstructionStyle:
    name: str = ""
    #: whether this style terminates the episode (like goal-absent)
    is_terminal: bool = False
    #: base sampling weight used by the balancer
    base_weight: float = 1.0

    # ------------------------------------------------------------------ #
    # command 1: structure generation
    # ------------------------------------------------------------------ #
    def admissible(self, ctx: GenContext) -> bool:
        raise NotImplementedError

    def apply(self, ctx: GenContext, rng: random.Random) -> StyleToken:
        """Transition ctx and return a StyleToken. Must increment ctx.v by 1."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # command 2: object resolution + rendering
    # ------------------------------------------------------------------ #
    def resolve(self, token: StyleToken, b: BuildContext) -> Resolved:
        raise NotImplementedError

    def render(self, r: Resolved, rng: random.Random) -> str:
        raise NotImplementedError

    # convenience: return a Resolved with render() already applied
    def resolve_and_render(self, token: StyleToken, b: BuildContext) -> Resolved:
        r = self.resolve(token, b)
        r.instruction = self.render(r, b.rng)
        return r


# ---------------------------------------------------------------------- #
# shared rendering helpers
# ---------------------------------------------------------------------- #
def article_phrase(category: str) -> str:
    """'chair' -> 'the chair', 'tv_monitor' -> 'the tv monitor'."""
    return f"the {category.replace('_', ' ')}"


_ORDINALS = {
    1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th", 6: "6th", 7: "7th",
    8: "8th", 9: "9th", 10: "10th", 11: "11th", 12: "12th",
}


def ordinal_word(k: int) -> str:
    return _ORDINALS.get(k, f"{k}th")
