"""Single style list generator — drives the automaton, picking admissible tokens.

The procedural algorithm of spec section 7 is ignored (per requirements); at each
step we gather admissible styles from the registry and use weighted sampling plus
'closing pressure' to hit a target length. Structural transitions and reference
parameters are handled by each style.apply().

Generation here is FREE (no distance/style biasing). All ratio balancing — length,
AB/OR reference distance, style evenness — is handled afterwards by the balancer's
generate-measure-prune-refill loop (refon.balancer), not at generation time.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional

from .automaton import AR_NONE, AR_EXPECT_POST, GenContext
from .style_token import StyleList, StyleToken
from .styles import NORMAL_STYLES, admissible_normal, admissible_terminal
from .styles.base import InstructionStyle


def _weight(style: InstructionStyle, weights: Optional[Dict[str, float]]) -> float:
    if weights and style.name in weights:
        return max(0.0, float(weights[style.name]))
    return style.base_weight


def _closing_progress_styles(
    ctx: GenContext, cands: List[InstructionStyle]
) -> List[InstructionStyle]:
    """Keep only candidates that make 'progress' toward the accepting state (0, NONE).

      - ar_mode==EXPECT_POST : AR_post only (already forced at the admissibility step)
      - ar_mode==NONE, n_AB>0: prefer AB_post (close an alias)
      - otherwise            : unchanged
    """
    if ctx.ar_mode == AR_EXPECT_POST:
        return [s for s in cands if s.name == "AR_post"] or cands
    if ctx.ar_mode == AR_NONE and ctx.n_AB > 0:
        closers = [s for s in cands if s.name == "AB_post"]
        if closers:
            return closers
    return cands


def _weighted_pick(
    cands: List[InstructionStyle],
    rng: random.Random,
    weights: Optional[Dict[str, float]],
    boost: Optional[set] = None,
    boost_factor: float = 1.0,
) -> InstructionStyle:
    w = []
    for s in cands:
        base = _weight(s, weights)
        if boost and s.name in boost:
            base *= boost_factor
        w.append(max(1e-9, base))
    return rng.choices(cands, weights=w, k=1)[0]


def generate_normal(
    target_length: int,
    rng: random.Random,
    weights: Optional[Dict[str, float]] = None,
    max_length: int = 12,
) -> StyleList:
    """Generate one normal (accepting) style list (free generation).

    target_length is only a 'target'; it may be exceeded while closing alias/AR
    blocks (spec acknowledges this). When max_length is reached, only closing
    progress is allowed to force termination.
    """
    ctx = GenContext()
    tokens: List[StyleToken] = []

    while True:
        cands = admissible_normal(ctx)
        if not cands:
            break  # in theory unreachable (S is almost always admissible)

        over_budget = ctx.v >= max_length
        need_close = ctx.v >= target_length

        if over_budget:
            cands = _closing_progress_styles(ctx, cands)
            chosen = _weighted_pick(cands, rng, weights)
        elif need_close:
            # soft closing pressure: boost the weight of closing-progress candidates
            progress = {s.name for s in _closing_progress_styles(ctx, cands)}
            chosen = _weighted_pick(cands, rng, weights, boost=progress, boost_factor=4.0)
        else:
            chosen = _weighted_pick(cands, rng, weights)

        tokens.append(chosen.apply(ctx, rng))

        if ctx.accepting and ctx.v >= target_length:
            break

    return StyleList(tokens=tokens, goal_absent=False)


def generate_goal_absent(
    target_length: int,
    rng: random.Random,
    weights: Optional[Dict[str, float]] = None,
    max_length: int = 12,
) -> StyleList:
    """Generate one goal-absent style list (free generation).

    Build a normal prefix of length target_length-1 (need not be accepting — the agent
    stops), then append one terminal (goal-absent) style at a point where ar_mode==NONE.
    """
    prefix_len = max(0, target_length - 1)
    ctx = GenContext()
    tokens: List[StyleToken] = []

    while ctx.v < prefix_len or ctx.ar_mode != AR_NONE:
        cands = admissible_normal(ctx)
        if not cands:
            break
        if ctx.v >= max_length and ctx.ar_mode == AR_NONE:
            break
        if ctx.ar_mode != AR_NONE:
            cands = _closing_progress_styles(ctx, cands)
        chosen = _weighted_pick(cands, rng, weights)
        tokens.append(chosen.apply(ctx, rng))

    term_cands = admissible_terminal(ctx)
    if not term_cands:
        # e.g. v==0 so only invalid_ordinal is blocked — lay down one S and retry
        s = NORMAL_STYLES[0]
        tokens.append(s.apply(ctx, rng))
        term_cands = admissible_terminal(ctx)

    term = _weighted_pick(term_cands, rng, weights)
    tokens.append(term.apply(ctx, rng))
    return StyleList(tokens=tokens, goal_absent=True)
