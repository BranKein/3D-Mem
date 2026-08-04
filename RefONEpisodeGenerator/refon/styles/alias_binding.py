"""Alias Binding (AB) — balanced mechanism.

  AB_pre  : direct navigation + bind a new alias to that target  (n_AB += 1)
  AB_post : close one open alias (LIFO) and re-navigate to its object  (n_AB -= 1)
"""
from __future__ import annotations

import random

from ..automaton import AR_EXPECT_POST, GenContext
from ..style_token import StyleToken
from .base import BuildContext, InstructionStyle, Resolved, article_phrase


class AliasBindingPre(InstructionStyle):
    name = "AB_pre"
    base_weight = 2.0

    def admissible(self, ctx: GenContext) -> bool:
        return ctx.ar_mode != AR_EXPECT_POST

    def apply(self, ctx: GenContext, rng: random.Random) -> StyleToken:
        order = ctx.order
        alias = ctx.fresh_alias()
        ctx.alias_stack.append((alias, order))
        ctx.n_AB += 1
        ctx.advance_ar_after_normal()
        ctx.v += 1
        return StyleToken(role=self.name, order=order, alias=alias)

    def resolve(self, token: StyleToken, b: BuildContext) -> Resolved:
        obj_id, cat = b.choose_new_object()
        b.bind_alias(token.alias, obj_id, cat)
        return Resolved(
            role=self.name, order=token.order, category=cat,
            object_id=obj_id, alias=token.alias,
        )

    def render(self, r: Resolved, rng: random.Random) -> str:
        return f"Find {article_phrase(r.category)}. Let's call it {r.alias}."


class AliasBindingPost(InstructionStyle):
    name = "AB_post"
    base_weight = 2.0

    def admissible(self, ctx: GenContext) -> bool:
        return ctx.ar_mode != AR_EXPECT_POST and ctx.n_AB > 0

    def apply(self, ctx: GenContext, rng: random.Random) -> StyleToken:
        order = ctx.order
        alias, bound_order = ctx.alias_stack.pop()  # LIFO
        ctx.n_AB -= 1
        ctx.advance_ar_after_normal()
        ctx.v += 1
        return StyleToken(role=self.name, order=order, alias=alias, ref_order=bound_order)

    def resolve(self, token: StyleToken, b: BuildContext) -> Resolved:
        obj_id, cat = b.object_at(token.ref_order)
        return Resolved(
            role=self.name, order=token.order, category=cat,
            object_id=obj_id, alias=token.alias, ref_order=token.ref_order,
        )

    def render(self, r: Resolved, rng: random.Random) -> str:
        return f"Find {r.alias}."
