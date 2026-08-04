"""StyleList validation — re-check a style list against the automaton grammar by replay.

The generator trusts what it produces, but to trust a hand-written/edited list or a
list after a serialization round-trip, an independent check is needed. Here we look
only at each token's role, re-run the automaton, and verify each token's
admissibility and the final acceptance.

  - normal list (goal_absent=False): every token admissible, ends in accepting (0, NONE).
  - goal-absent list: the last token is a terminal style, and everything before it is
    admissible. (Open aliases may remain — the agent stops, so they need not close.)
"""
from __future__ import annotations

from typing import List, Tuple

from .automaton import AR_NONE, GenContext
from .style_token import StyleList
from .styles import STYLE_BY_NAME


class ValidationError(ValueError):
    pass


def validate(style_list: StyleList, *, raise_on_error: bool = False) -> Tuple[bool, List[str]]:
    """Return (ok, errors)."""
    errors: List[str] = []
    ctx = GenContext()
    tokens = style_list.tokens
    n = len(tokens)

    if n == 0:
        errors.append("empty style list")

    for i, tok in enumerate(tokens):
        style = STYLE_BY_NAME.get(tok.role)
        if style is None:
            errors.append(f"#{i+1}: unknown role '{tok.role}'")
            break
        is_last = i == n - 1
        if style.is_terminal and not is_last:
            errors.append(f"#{i+1}: terminal style '{tok.role}' is not last")
        if not style.is_terminal and tok.goal_absent:
            errors.append(f"#{i+1}: non-terminal style marked goal_absent")
        if not style.admissible(ctx):
            errors.append(
                f"#{i+1}: '{tok.role}' inadmissible at "
                f"(n_AB={ctx.n_AB}, ar_mode={ctx.ar_mode}, v={ctx.v})"
            )
            # no point continuing
            break
        # replay the structural transition (apply uses rng, but validation only needs
        # the resulting structure -> dummy rng)
        import random as _r
        style.apply(ctx, _r.Random(0))

    if not errors:
        if style_list.goal_absent:
            if not ctx.terminated:
                errors.append("goal_absent list but no terminal style emitted")
        else:
            if not ctx.accepting:
                errors.append(
                    f"not in accepting state at end (n_AB={ctx.n_AB}, ar_mode={ctx.ar_mode})"
                )
            if ctx.terminated:
                errors.append("normal list but a terminal style was emitted")

    ok = not errors
    if not ok and raise_on_error:
        raise ValidationError("; ".join(errors))
    return ok, errors
