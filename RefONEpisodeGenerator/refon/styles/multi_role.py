"""{AB_pre, OR_post} — the only multi-role token.

Re-navigate to a past visit slot k via OR_post, then bind a new alias to that
target. Processing order (spec 3.3): resolve target via OR_post, then apply AB_pre
to bind an alias to it. The automaton effect equals AB_pre (n_AB += 1); OR_post is a no-op.
"""
from __future__ import annotations

import random

from ..automaton import AR_EXPECT_POST, GenContext
from ..style_token import StyleToken
from .base import BuildContext, InstructionStyle, Resolved, ordinal_word


class AliasOrdinal(InstructionStyle):
    name = "AB_pre+OR_post"
    base_weight = 0.8

    def admissible(self, ctx: GenContext) -> bool:
        return ctx.ar_mode != AR_EXPECT_POST and ctx.v >= 1

    def apply(self, ctx: GenContext, rng: random.Random) -> StyleToken:
        order = ctx.order
        k = rng.randint(1, ctx.v)
        alias = ctx.fresh_alias()
        ctx.alias_stack.append((alias, order))
        ctx.n_AB += 1
        ctx.advance_ar_after_normal()
        ctx.v += 1
        return StyleToken(role=self.name, order=order, alias=alias, ordinal_k=k, ref_order=k)

    def resolve(self, token: StyleToken, b: BuildContext) -> Resolved:
        obj_id, cat = b.object_at(token.ordinal_k)
        b.bind_alias(token.alias, obj_id, cat)
        return Resolved(
            role=self.name, order=token.order, category=cat, object_id=obj_id,
            alias=token.alias, ordinal_k=token.ordinal_k, ref_order=token.ordinal_k,
        )

    def render(self, r: Resolved, rng: random.Random) -> str:
        return f"Find the {ordinal_word(r.ordinal_k)} one again. Let's call it {r.alias}."
