"""S — plain goal (direct navigation, no reference). role set = empty."""
from __future__ import annotations

import random

from ..automaton import AR_EXPECT_POST, GenContext
from ..style_token import StyleToken
from .base import BuildContext, InstructionStyle, Resolved, article_phrase


class PlainGoal(InstructionStyle):
    name = "S"
    base_weight = 3.0

    def admissible(self, ctx: GenContext) -> bool:
        # in EXPECT_POST only AR_post is allowed
        return ctx.ar_mode != AR_EXPECT_POST

    def apply(self, ctx: GenContext, rng: random.Random) -> StyleToken:
        order = ctx.order
        ctx.advance_ar_after_normal()
        ctx.v += 1
        return StyleToken(role=self.name, order=order)

    def resolve(self, token: StyleToken, b: BuildContext) -> Resolved:
        obj_id, cat = b.choose_new_object()
        return Resolved(role=self.name, order=token.order, category=cat, object_id=obj_id)

    def render(self, r: Resolved, rng: random.Random) -> str:
        return f"Find {article_phrase(r.category)}."
