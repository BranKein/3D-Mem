"""Anaphoric Referencing (AR) — fixed-gap (gap=1) balanced mechanism.

  AR_pre  : direct navigation + mark that target as the antecedent  (ar_mode -> EXPECT_ONE)
  (one intervening non-AR instruction)                              (ar_mode -> EXPECT_POST)
  AR_post : re-navigate to the antecedent (= the AR_pre target)      (ar_mode -> NONE)

AR_pre/AR_post do not combine with other roles; they appear standalone (spec 3.2).
"""
from __future__ import annotations

import random

from ..automaton import AR_EXPECT_ONE, AR_EXPECT_POST, AR_NONE, GenContext
from ..style_token import StyleToken
from .base import BuildContext, InstructionStyle, Resolved, article_phrase


class AnaphoricPre(InstructionStyle):
    name = "AR_pre"
    base_weight = 1.5

    def admissible(self, ctx: GenContext) -> bool:
        # AR blocks cannot nest -> only start from NONE
        return ctx.ar_mode == AR_NONE

    def apply(self, ctx: GenContext, rng: random.Random) -> StyleToken:
        order = ctx.order
        ctx.antecedent_order = order
        ctx.ar_mode = AR_EXPECT_ONE
        ctx.v += 1
        return StyleToken(role=self.name, order=order)

    def resolve(self, token: StyleToken, b: BuildContext) -> Resolved:
        obj_id, cat = b.choose_new_object()
        return Resolved(role=self.name, order=token.order, category=cat, object_id=obj_id)

    def render(self, r: Resolved, rng: random.Random) -> str:
        return f"Find {article_phrase(r.category)}."


# AR_post phrasing variants (spec 2.3: must denote 'the one before the most recent', not 'the most recent')
_AR_POST_PHRASINGS = {
    "previous": "Go back to the previous one.",
    "before_that": "Find the one before that.",
    "before_last": "Go back to the one before the last.",
}


class AnaphoricPost(InstructionStyle):
    name = "AR_post"
    base_weight = 1.5

    def admissible(self, ctx: GenContext) -> bool:
        return ctx.ar_mode == AR_EXPECT_POST

    def apply(self, ctx: GenContext, rng: random.Random) -> StyleToken:
        order = ctx.order
        ref = ctx.antecedent_order
        ctx.antecedent_order = None
        ctx.ar_mode = AR_NONE
        ctx.v += 1
        phrasing = rng.choice(list(_AR_POST_PHRASINGS.keys()))
        return StyleToken(role=self.name, order=order, ref_order=ref, phrasing=phrasing)

    def resolve(self, token: StyleToken, b: BuildContext) -> Resolved:
        obj_id, cat = b.object_at(token.ref_order)
        return Resolved(
            role=self.name, order=token.order, category=cat,
            object_id=obj_id, ref_order=token.ref_order,
            extra={"phrasing": token.phrasing},
        )

    def render(self, r: Resolved, rng: random.Random) -> str:
        key = r.extra.get("phrasing") or "previous"
        return _AR_POST_PHRASINGS.get(key, _AR_POST_PHRASINGS["previous"])
