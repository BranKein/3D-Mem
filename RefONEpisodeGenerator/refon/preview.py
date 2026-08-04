"""Human-readable console preview of generated style lists.

command 1 output has no scene/objects, so instructions are rendered with symbolic object
labels (O1, O2, ...) using each style's own render(). Back-references are shown with an
arrow to the subgoal they resolve to, so the referential structure is visible at a glance.

Used by the `sample` CLI command: pick one random episode per length and print them.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from .storage import list_scene_slots, load_scene_file
from .style_token import StyleList
from .styles import STYLE_BY_NAME
from .styles.base import Resolved

# ANSI (kept minimal; degrade gracefully if the terminal ignores them)
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_YEL = "\033[33m"
_RED = "\033[31m"
_RST = "\033[0m"


class _SymbolicContext:
    """BuildContext that assigns symbolic object labels instead of real scene objects."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.visit: List[Tuple[Optional[str], Optional[str]]] = []
        self.aliases: Dict[str, Tuple[str, str]] = {}
        self._n = 0

    def choose_new_object(self) -> Tuple[str, str]:
        self._n += 1
        lbl = f"O{self._n}"
        return lbl, lbl

    def object_at(self, order: int) -> Tuple[Optional[str], Optional[str]]:
        return self.visit[order - 1]

    def bind_alias(self, alias: str, object_id, category) -> None:
        self.aliases[alias] = (object_id, category)

    def sample_absent_category(self) -> str:
        return self.rng.choice(["dragon", "spaceship", "unicorn", "dinosaur"])


def _resolve_list(sl: StyleList, rng: random.Random) -> List[Resolved]:
    ctx = _SymbolicContext(rng)
    out: List[Resolved] = []
    for tok in sl.tokens:
        style = STYLE_BY_NAME[tok.role]
        r = style.resolve_and_render(tok, ctx)
        ctx.visit.append((r.object_id, r.category))
        out.append(r)
    return out


def format_episode(sl: StyleList, header: str, rng: random.Random, color: bool = True) -> str:
    def c(s, code):
        return f"{code}{s}{_RST}" if color else s

    resolved = _resolve_list(sl, rng)
    labels = {r.order: (r.category or r.alias or "-") for r in resolved}
    lines = [c(header, _BOLD + _CYAN)]
    for r in resolved:
        # annotation column: what this subgoal binds / refers to
        ann = ""
        if r.goal_absent:
            ann = c(f"✗ STOP — {r.reason}", _RED)
        elif r.ref_order:
            tgt = labels.get(r.ref_order, "?")
            kind = f"k={r.ordinal_k}" if r.ordinal_k else ""
            ann = c(f"↩ refers to #{r.ref_order} ({tgt}) {kind}".rstrip(), _YEL)
        elif r.alias:
            ann = c(f"binds {r.alias}", _DIM)
        elif r.role == "AR_pre":
            ann = c("antecedent", _DIM)
        role = r.role if not r.goal_absent else r.role
        lines.append(f"  {c(f'#{r.order:<2}', _DIM)} {role:<16} {r.instruction:<46} {ann}")
    return "\n".join(lines)


def sample_one_per_length(
    in_dir: str, rng: random.Random, max_scenes: int = 12
) -> Dict[int, Tuple[StyleList, int]]:
    """Reservoir-sample one random style list per length across (up to max_scenes) random
    scene files. Returns {length: (style_list, scene_slot)}."""
    slots = list_scene_slots(in_dir)
    rng.shuffle(slots)
    slots = slots[:max_scenes] if max_scenes else slots

    counts: Dict[int, int] = {}
    picked: Dict[int, Tuple[StyleList, int]] = {}
    for slot in slots:
        for sl in load_scene_file(in_dir, slot):
            L = sl.length
            counts[L] = counts.get(L, 0) + 1
            if rng.random() < 1.0 / counts[L]:
                picked[L] = (sl, slot)
    return picked


def print_samples(in_dir: str, rng: random.Random, color: bool = True, max_scenes: int = 12) -> int:
    picked = sample_one_per_length(in_dir, rng, max_scenes)
    if not picked:
        print(f"[sample] no style lists found under {in_dir}")
        return 1
    bar = "=" * 78
    print(bar)
    print(f"one random episode per length — {in_dir}")
    print(bar)
    for L in sorted(picked):
        sl, slot = picked[L]
        tag = "  (goal-absent)" if sl.goal_absent else ""
        header = f"length {L}{tag}   [scene slot {slot}]"
        print()
        print(format_episode(sl, header, rng, color=color))
    print()
    print(bar)
    print(f"lengths shown: {sorted(picked)}")
    return 0
