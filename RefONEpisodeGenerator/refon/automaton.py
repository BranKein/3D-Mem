"""State of the one-counter pushdown automaton (GenContext).

Follows episode_generator_spec.md section 4. The procedural generate_episode of
spec section 7 is intentionally ignored (per requirements); transitions are
performed by each InstructionStyle class via self.apply(ctx). This module only
holds the 'state container' and shared helpers.

Structural state:
  n_AB     : number of currently open alias bindings (+1 on AB_pre, -1 on AB_post) — pushdown counter
  ar_mode  : anaphoric block mode (NONE / EXPECT_ONE / EXPECT_POST)
  v        : number of instructions emitted so far (length of the visit order)

Extra bookkeeping for object resolution (not needed for structural validity, but
needed to build reference links):
  alias_stack       : LIFO stack [(alias_id, bound_order), ...]
  antecedent_order  : visit order of the current AR block's antecedent (= the AR_pre target)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ar_mode values
AR_NONE = "NONE"
AR_EXPECT_ONE = "EXPECT_ONE"
AR_EXPECT_POST = "EXPECT_POST"


@dataclass
class GenContext:
    n_AB: int = 0
    ar_mode: str = AR_NONE
    v: int = 0
    terminated: bool = False  # True once a goal-absent terminal style is emitted

    alias_stack: List[Tuple[str, int]] = field(default_factory=list)
    antecedent_order: Optional[int] = None
    _next_alias: int = 1
    _next_ghost: int = 1

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def fresh_alias(self) -> str:
        """A new real alias identifier (A1, A2, ...)."""
        a = f"A{self._next_alias}"
        self._next_alias += 1
        return a

    def fresh_ghost_alias(self) -> str:
        """A fake alias that was never bound (for goal-absent: Z1, Z2, ...).

        Uses a separate namespace from real aliases (A*) to guarantee it is a name
        that was never introduced.
        """
        a = f"Z{self._next_ghost}"
        self._next_ghost += 1
        return a

    @property
    def order(self) -> int:
        """1-based visit order of the next token to be emitted."""
        return self.v + 1

    def advance_ar_after_normal(self) -> None:
        """Advance ar_mode after consuming the single intervening (non-AR) token of an AR block."""
        if self.ar_mode == AR_EXPECT_ONE:
            self.ar_mode = AR_EXPECT_POST

    @property
    def accepting(self) -> bool:
        """Normal-termination-ready state: all aliases closed and outside any AR block."""
        return self.n_AB == 0 and self.ar_mode == AR_NONE
