"""Goal-absent styles — 'nonsensical' instructions.

Instructions that the agent must fail to resolve and stop on. They only ever appear
as the LAST subgoal of an episode (no subgoals after), and with is_terminal=True
they set ctx.terminated. They are restricted to ar_mode==NONE so they do not break
AR-block grammar.

Three kinds:
  UnboundAlias   : asks for an alias that was never bound ("Find Z3." with Z3 undefined)
  AbsentObject   : asks for a category that does not exist in the scene ("Find the dragon.")
  InvalidOrdinal : points at an ordinal that does not exist yet (visited 2, "Find the 4th one again.")
"""
from __future__ import annotations

import random

from ..automaton import AR_NONE, GenContext
from ..style_token import StyleToken
from .base import BuildContext, InstructionStyle, Resolved, article_phrase, ordinal_word


class _GoalAbsentBase(InstructionStyle):
    is_terminal = True
    base_weight = 1.0
    absent_kind = ""

    def admissible(self, ctx: GenContext) -> bool:
        return ctx.ar_mode == AR_NONE and not ctx.terminated


class UnboundAlias(_GoalAbsentBase):
    name = "GA_unbound_alias"
    absent_kind = "unbound_alias"

    def apply(self, ctx: GenContext, rng: random.Random) -> StyleToken:
        order = ctx.order
        ghost = ctx.fresh_ghost_alias()  # a name that was never bound
        ctx.terminated = True
        ctx.v += 1
        return StyleToken(
            role=self.name, order=order, alias=ghost,
            goal_absent=True, absent_kind=self.absent_kind,
        )

    def resolve(self, token: StyleToken, b: BuildContext) -> Resolved:
        return Resolved(
            role=self.name, order=token.order, category=None, object_id=None,
            alias=token.alias, goal_absent=True,
            reason=f"alias '{token.alias}' was never bound",
        )

    def render(self, r: Resolved, rng: random.Random) -> str:
        return f"Find {r.alias}."


class AbsentObject(_GoalAbsentBase):
    name = "GA_absent_object"
    absent_kind = "absent_object"

    def apply(self, ctx: GenContext, rng: random.Random) -> StyleToken:
        order = ctx.order
        ctx.terminated = True
        ctx.v += 1
        return StyleToken(
            role=self.name, order=order,
            goal_absent=True, absent_kind=self.absent_kind,
        )

    def resolve(self, token: StyleToken, b: BuildContext) -> Resolved:
        cat = b.sample_absent_category()
        return Resolved(
            role=self.name, order=token.order, category=cat, object_id=None,
            goal_absent=True, reason=f"no '{cat}' exists in this scene",
        )

    def render(self, r: Resolved, rng: random.Random) -> str:
        return f"Find {article_phrase(r.category)}."


class InvalidOrdinal(_GoalAbsentBase):
    name = "GA_invalid_ordinal"
    absent_kind = "invalid_ordinal"

    def admissible(self, ctx: GenContext) -> bool:
        # for an 'ordinal' to be plausible there must be at least one prior visit
        return super().admissible(ctx) and ctx.v >= 1

    def apply(self, ctx: GenContext, rng: random.Random) -> StyleToken:
        order = ctx.order
        # a non-existent future/overshoot ordinal
        k = ctx.v + rng.randint(1, 3)
        ctx.terminated = True
        ctx.v += 1
        return StyleToken(
            role=self.name, order=order, ordinal_k=k,
            goal_absent=True, absent_kind=self.absent_kind,
        )

    def resolve(self, token: StyleToken, b: BuildContext) -> Resolved:
        return Resolved(
            role=self.name, order=token.order, category=None, object_id=None,
            ordinal_k=token.ordinal_k, goal_absent=True,
            reason=f"only {token.order - 1} object(s) visited before, "
                   f"no {ordinal_word(token.ordinal_k)} one exists",
        )

    def render(self, r: Resolved, rng: random.Random) -> str:
        return f"Find the {ordinal_word(r.ordinal_k)} one again."
