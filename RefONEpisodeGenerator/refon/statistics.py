"""Statistics over style lists — incremental accumulator + finalize.

To support resumable, per-scene generation (memory stays per-scene), statistics are
computed by accumulating raw counters scene-by-scene and finalized at the end. The
accumulator is plain JSON-serializable dicts so it can be checkpointed in the progress
file and reloaded after an interruption.

  new_accumulator() -> acc
  accumulate(acc, style_lists) -> acc   (in place, returns it)
  finalize_accumulator(acc) -> stats dict
  compute_statistics(style_lists) = finalize_accumulator(accumulate(new_acc, lists))
"""
from __future__ import annotations

from typing import Dict, List

from .style_token import StyleList
from .styles import all_styles


# ---------------------------------------------------------------------- #
# accumulator
# ---------------------------------------------------------------------- #
def new_accumulator() -> Dict:
    return {
        "num_lists": 0,
        "num_tokens": 0,
        "goal_absent_lists": 0,
        "length": {},        # int length -> count
        "role": {},          # role name -> count
        "absent_kind": {},   # kind -> count
        "ab": {},            # int distance -> count (AB close span)
        "or": {},            # int distance -> count (OR back-ref span)
    }


def _inc(d: Dict, key, by: int = 1) -> None:
    d[key] = d.get(key, 0) + by


def normalize_accumulator(acc: Dict) -> Dict:
    """Coerce int-keyed sub-counters back to int keys (JSON turns them into str)."""
    for k in ("length", "ab", "or"):
        acc[k] = {int(kk): v for kk, v in acc.get(k, {}).items()}
    acc.setdefault("role", {})
    acc.setdefault("absent_kind", {})
    return acc


def merge_accumulators(acc: Dict, other: Dict) -> Dict:
    """Merge `other` into `acc` in place (for combining per-scene accumulators built by
    parallel workers). Returns acc."""
    acc["num_lists"] += other["num_lists"]
    acc["num_tokens"] += other["num_tokens"]
    acc["goal_absent_lists"] += other["goal_absent_lists"]
    for key in ("length", "role", "absent_kind", "ab", "or"):
        dst = acc[key]
        for k, v in other[key].items():
            dst[k] = dst.get(k, 0) + v
    return acc


def accumulate(acc: Dict, style_lists: List[StyleList]) -> Dict:
    for sl in style_lists:
        acc["num_lists"] += 1
        _inc(acc["length"], sl.length)
        if sl.goal_absent:
            acc["goal_absent_lists"] += 1
        for t in sl.tokens:
            acc["num_tokens"] += 1
            _inc(acc["role"], t.role)
            if t.absent_kind:
                _inc(acc["absent_kind"], t.absent_kind)
            if t.role == "AB_post" and t.ref_order is not None:
                _inc(acc["ab"], t.order - t.ref_order)
            if t.role in ("OR_post", "AB_pre+OR_post") and t.ordinal_k is not None:
                _inc(acc["or"], t.order - t.ordinal_k)
    return acc


# ---------------------------------------------------------------------- #
# finalize
# ---------------------------------------------------------------------- #
def _summary_from_counter(counter: Dict[int, int]) -> Dict:
    c = {int(k): v for k, v in counter.items()}
    total = sum(c.values())
    return {
        "count": total,
        "min": min(c) if c else 0,
        "max": max(c) if c else 0,
        "avg": round(sum(k * v for k, v in c.items()) / total, 3) if total else 0.0,
        "distribution": {k: c[k] for k in sorted(c)},
    }


def finalize_accumulator(acc: Dict) -> Dict:
    n = acc["num_lists"]
    total_tokens = acc["num_tokens"]
    length = {int(k): v for k, v in acc["length"].items()}
    role = dict(acc["role"])
    ab = {int(k): v for k, v in acc["ab"].items()}
    orr = {int(k): v for k, v in acc["or"].items()}
    combined: Dict[int, int] = {}
    for src in (ab, orr):
        for k, v in src.items():
            combined[k] = combined.get(k, 0) + v

    role_freq = {s.name: role.get(s.name, 0) for s in all_styles()}
    lengths = sorted(length)
    return {
        "num_lists": n,
        "num_tokens": total_tokens,
        "avg_length": round(total_tokens / n, 3) if n else 0.0,
        "goal_absent_lists": acc["goal_absent_lists"],
        "goal_absent_ratio": round(acc["goal_absent_lists"] / n, 4) if n else 0.0,
        "length_distribution": {int(k): length[k] for k in lengths},
        "length_ratio": {int(k): round(length[k] / n, 4) for k in lengths} if n else {},
        "role_frequency": role_freq,
        "role_ratio": {
            name: round(c / total_tokens, 4) for name, c in role_freq.items()
        } if total_tokens else {},
        "absent_kind_distribution": dict(acc["absent_kind"]),
        "ab_pre_post_distance": _summary_from_counter(ab),
        "or_reference_distance": _summary_from_counter(orr),
        "ref_distance_combined": _summary_from_counter(combined),
    }


def compute_statistics(style_lists: List[StyleList]) -> Dict:
    return finalize_accumulator(accumulate(new_accumulator(), style_lists))


# ---------------------------------------------------------------------- #
# formatting
# ---------------------------------------------------------------------- #
def format_statistics(stats: Dict, title: str = "STYLE LIST STATISTICS") -> str:
    lines = []
    bar = "=" * 60
    lines.append(bar)
    lines.append(title)
    lines.append(bar)
    lines.append(f"lists           : {stats['num_lists']}")
    lines.append(f"tokens          : {stats['num_tokens']}")
    lines.append(f"avg length      : {stats['avg_length']}")
    lines.append(
        f"goal-absent     : {stats['goal_absent_lists']} "
        f"({stats['goal_absent_ratio'] * 100:.1f}%)"
    )
    lines.append("")
    lines.append("length distribution:")
    for length, count in stats["length_distribution"].items():
        ratio = stats["length_ratio"].get(length, 0.0)
        bar_str = "#" * int(ratio * 40)
        lines.append(f"  len {length:>2}: {count:>5}  {ratio * 100:5.1f}%  {bar_str}")
    lines.append("")
    lines.append("role frequency:")
    for name, count in sorted(stats["role_frequency"].items(), key=lambda kv: -kv[1]):
        ratio = stats["role_ratio"].get(name, 0.0)
        bar_str = "#" * int(ratio * 40)
        lines.append(f"  {name:<16}: {count:>6}  {ratio * 100:5.1f}%  {bar_str}")
    if stats["absent_kind_distribution"]:
        lines.append("")
        lines.append("goal-absent kinds:")
        for kind, count in stats["absent_kind_distribution"].items():
            lines.append(f"  {kind:<18}: {count}")
    _append_distance_section(lines, "AB pre->post distance", stats.get("ab_pre_post_distance"))
    _append_distance_section(lines, "OR reference distance", stats.get("or_reference_distance"))
    _append_distance_section(lines, "AB+OR reference distance (combined)",
                             stats.get("ref_distance_combined"))
    lines.append(bar)
    return "\n".join(lines)


def _append_distance_section(lines: List[str], title: str, summary: Dict) -> None:
    if not summary or not summary.get("count"):
        return
    total = summary["count"]
    lines.append("")
    lines.append(
        f"{title} (count={total}, min={summary['min']}, "
        f"max={summary['max']}, avg={summary['avg']}):"
    )
    target_ratio = summary.get("target_ratio") or {}
    for dist, count in summary["distribution"].items():
        ratio = count / total
        bar_str = "#" * int(ratio * 40)
        tgt = target_ratio.get(dist, target_ratio.get(str(dist)))
        tgt_str = f"  (target {tgt * 100:4.1f}%)" if tgt is not None else ""
        lines.append(f"  dist {dist:>2}: {count:>6}  {ratio * 100:5.1f}%  {bar_str}{tgt_str}")
