"""Ordinal Referencing (OR) — free back-reference.

There is no OR_pre. Every instruction occupies one visit slot, and OR_post freely
references any past slot k (1..v). Structurally it behaves exactly like S (no-op).
"""
from __future__ import annotations

import random

from ..automaton import AR_EXPECT_POST, GenContext
from ..style_token import StyleToken
from .base import BuildContext, InstructionStyle, Resolved, ordinal_word


class OrdinalPost(InstructionStyle):
    name = "OR_post"
    base_weight = 1.5

    def admissible(self, ctx: GenContext) -> bool:
        # at least one prior visit must exist to point back to
        return ctx.ar_mode != AR_EXPECT_POST and ctx.v >= 1

    def apply(self, ctx: GenContext, rng: random.Random) -> StyleToken:
        order = ctx.order
        k = rng.randint(1, ctx.v)  # uniform over 1..v
        ctx.advance_ar_after_normal()
        ctx.v += 1
        return StyleToken(role=self.name, order=order, ordinal_k=k, ref_order=k)

    def resolve(self, token: StyleToken, b: BuildContext) -> Resolved:
        obj_id, cat = b.object_at(token.ordinal_k)
        return Resolved(
            role=self.name, order=token.order, category=cat,
            object_id=obj_id, ordinal_k=token.ordinal_k, ref_order=token.ordinal_k,
        )

    def render(self, r: Resolved, rng: random.Random) -> str:
        return f"Find the {ordinal_word(r.ordinal_k)} one again."
