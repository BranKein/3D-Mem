"""StyleToken / StyleList — the output of command 1 (structure only).

A StyleToken holds one instruction's 'style + structural parameters'. There is no
scene/object yet, so object_id is not included; instead the reference structure
(e.g. which earlier visit this points back to: ref_order, ordinal_k, alias) is
fixed here. Command 2 maps this reference structure onto real scene objects.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class StyleToken:
    role: str                          # 'S','AB_pre','AB_post','OR_post','AR_pre','AR_post','AB_pre+OR_post', and goal-absent roles
    order: int                         # 1-based visit order
    alias: Optional[str] = None        # AB_pre / AB_post / multi-role / unbound_alias
    ref_order: Optional[int] = None    # earlier visit order a back-ref points to (AB_post/OR_post/AR_post/multi)
    ordinal_k: Optional[int] = None    # k of OR_post / invalid_ordinal
    goal_absent: bool = False
    absent_kind: Optional[str] = None  # 'unbound_alias' | 'absent_object' | 'invalid_ordinal'
    phrasing: Optional[str] = None     # rendering phrasing-variant key (fixed at generation for reproducibility)

    def to_dict(self) -> Dict:
        d = {k: v for k, v in asdict(self).items() if v is not None and v is not False}
        d["role"] = self.role
        d["order"] = self.order
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "StyleToken":
        return cls(
            role=d["role"],
            order=d["order"],
            alias=d.get("alias"),
            ref_order=d.get("ref_order"),
            ordinal_k=d.get("ordinal_k"),
            goal_absent=bool(d.get("goal_absent", False)),
            absent_kind=d.get("absent_kind"),
            phrasing=d.get("phrasing"),
        )


@dataclass
class StyleList:
    """A style list (= instruction style sequence) for one episode."""
    tokens: List[StyleToken] = field(default_factory=list)
    goal_absent: bool = False  # whether this episode is a goal-absent (terminal) episode

    @property
    def length(self) -> int:
        return len(self.tokens)

    def role_sequence(self) -> List[str]:
        return [t.role for t in self.tokens]

    def to_dict(self) -> Dict:
        return {
            "length": self.length,
            "goal_absent": self.goal_absent,
            "tokens": [t.to_dict() for t in self.tokens],
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "StyleList":
        return cls(
            tokens=[StyleToken.from_dict(t) for t in d["tokens"]],
            goal_absent=bool(d.get("goal_absent", False)),
        )
