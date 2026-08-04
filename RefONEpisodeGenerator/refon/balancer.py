"""Balancer — generate freely, measure, prune the misfits, refill, repeat.

Philosophy (description.md, refined per user request): do NOT bias generation. Just
generate lists as they come, measure how the aggregate distributions compare to the
target ratios, delete the lists that hurt the fit the most, regenerate, keep the ones
that improve the fit, and repeat several rounds. The SAME mechanism balances every
dimension at once:

  - length        : per-list scalar      (target = length_ratios)
  - ab_distance   : per AB-close span     (target = ab_distance_ratios, if given)
  - or_distance   : per OR back-ref span  (target = or_distance_ratios, if given)
  - styles        : per token role        (target = uniform, if even_styles)

Hard rejection (candidate never enters the pool): a list whose length is outside the
configured length set, or whose AB/OR reference distance falls outside the configured
distance set, is discarded at generation time — so out-of-target lengths/distances do
not appear at all. The soft prune/refill loop then matches the in-target proportions.

`balance_lists(cfg, n, rng)` balances one pool of n lists; this is the unit used by the
resumable per-scene runner (refon.runner). `balance(cfg)` keeps the old whole-dataset
API (used programmatically/tests).
"""
from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Tuple

from .config import GeneratorConfig
from .generator import generate_goal_absent, generate_normal
from .statistics import compute_statistics
from .style_token import StyleList
from .styles import NORMAL_STYLES
from .validate import validate


# ---------------------------------------------------------------------- #
# targets / metrics
# ---------------------------------------------------------------------- #
def _normalize(ratios: Dict) -> Dict[int, float]:
    s = sum(ratios.values()) or 1.0
    return {k: v / s for k, v in ratios.items()}


def _list_bins(sl: StyleList) -> Dict[str, list]:
    """The per-dimension bin contributions of a single list."""
    ab, orr, styles = [], [], []
    for t in sl.tokens:
        styles.append(t.role)
        if t.role == "AB_post" and t.ref_order is not None:
            ab.append(t.order - t.ref_order)
        if t.role in ("OR_post", "AB_pre+OR_post") and t.ordinal_k is not None:
            orr.append(t.order - t.ordinal_k)
    return {"length": [sl.length], "ab_distance": ab, "or_distance": orr, "styles": styles}


def _build_targets(cfg: GeneratorConfig) -> Tuple[Dict[str, Dict], Dict[str, float]]:
    """Active target distributions and their weights, given the config."""
    targets: Dict[str, Dict] = {"length": _normalize(cfg.length_ratios)}
    if cfg.ab_distance_ratios:
        targets["ab_distance"] = _normalize(cfg.ab_distance_ratios)
    if cfg.or_distance_ratios:
        targets["or_distance"] = _normalize(cfg.or_distance_ratios)
    if cfg.even_styles:
        names = [s.name for s in NORMAL_STYLES]
        targets["styles"] = {n: 1.0 / len(names) for n in names}
    weights = {d: cfg.balance_weights.get(d, 1.0) for d in targets}
    return targets, weights


def _aggregate(pool: List[StyleList], dims: List[str]) -> Dict[str, Dict]:
    counts: Dict[str, Dict] = {d: {} for d in dims}
    for sl in pool:
        bins = _list_bins(sl)
        for d in dims:
            for b in bins[d]:
                counts[d][b] = counts[d].get(b, 0) + 1
    return counts


def _ratios_from_counts(counts: Dict[str, Dict]) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for d, c in counts.items():
        total = sum(c.values()) or 1
        out[d] = {b: n / total for b, n in c.items()}
    return out


def _error(counts: Dict[str, Dict], targets: Dict[str, Dict], weights: Dict[str, float]) -> float:
    e = 0.0
    for d, tr in targets.items():
        total = sum(counts[d].values()) or 1
        bins = set(counts[d]) | set(tr)
        e += weights[d] * sum(abs(counts[d].get(b, 0) / total - tr.get(b, 0.0)) for b in bins)
    return e


def _harm(sl: StyleList, obs: Dict[str, Dict], targets: Dict[str, Dict],
          weights: Dict[str, float]) -> float:
    """How much this list pushes toward OVER-represented bins (higher = better to drop)."""
    bins = _list_bins(sl)
    h = 0.0
    for d, tr in targets.items():
        ob = obs[d]
        for b in bins[d]:
            h += weights[d] * (ob.get(b, 0.0) - tr.get(b, 0.0))
    return h


# ---------------------------------------------------------------------- #
# refinement
# ---------------------------------------------------------------------- #
def _refine_pool(
    selected: List[StyleList],
    target_size: int,
    gen: Callable[[], StyleList],
    all_pool: Callable[[], List[StyleList]],
    targets: Dict[str, Dict],
    weights: Dict[str, float],
    cfg: GeneratorConfig,
    tag: str,
    progress: bool,
) -> List[StyleList]:
    """Prune-and-refill one typed sub-pool (normal or goal-absent)."""
    dims = list(targets)

    def cur_error() -> float:
        return _error(_aggregate(all_pool(), dims), targets, weights)

    best = list(selected)
    best_err = cur_error()

    for it in range(cfg.refine_iterations):
        obs = _ratios_from_counts(_aggregate(all_pool(), dims))
        k = max(1, int(round(target_size * cfg.prune_fraction)))
        ranked = sorted(selected, key=lambda s: _harm(s, obs, targets, weights), reverse=True)
        selected[:] = ranked[k:]

        obs = _ratios_from_counts(_aggregate(all_pool(), dims))
        need = target_size - len(selected)
        cands = [gen() for _ in range(max(1, need) * cfg.refill_oversample)]
        cands.sort(key=lambda s: _harm(s, obs, targets, weights))  # lowest harm first
        selected.extend(cands[:need])

        err = cur_error()
        if err < best_err:
            best_err = err
            best = list(selected)
        if progress and (it % 10 == 0 or it == cfg.refine_iterations - 1):
            print(f"[balance] {tag}: round {it} error={err:.4f} (best={best_err:.4f})")

    selected[:] = best
    return best


def _make_gen(cfg: GeneratorConfig, rng: random.Random) -> Callable[[bool], StyleList]:
    """Return a free-generation function _gen(goal_absent) with hard rejection of
    out-of-target lengths and out-of-target AB/OR reference distances."""
    allowed_lengths = set(cfg.length_ratios)
    ab_keys = set(cfg.ab_distance_ratios) if cfg.ab_distance_ratios else None
    or_keys = set(cfg.or_distance_ratios) if cfg.or_distance_ratios else None
    length_keys = list(cfg.length_ratios)
    length_w = [max(0.0, cfg.length_ratios[L]) for L in length_keys]
    base_w = {s.name: s.base_weight for s in NORMAL_STYLES}
    base_w.update(cfg.style_weights or {})

    def _clean(sl: StyleList) -> bool:
        if ab_keys is None and or_keys is None:
            return True
        for t in sl.tokens:
            if ab_keys is not None and t.role == "AB_post" and t.ref_order is not None:
                if (t.order - t.ref_order) not in ab_keys:
                    return False
            if or_keys is not None and t.role in ("OR_post", "AB_pre+OR_post") \
                    and t.ordinal_k is not None:
                if (t.order - t.ordinal_k) not in or_keys:
                    return False
        return True

    def _gen(goal_absent: bool) -> StyleList:
        gen_fn = generate_goal_absent if goal_absent else generate_normal
        sl = None
        for _ in range(500):
            L = rng.choices(length_keys, weights=length_w, k=1)[0]
            sl = gen_fn(max(2, L), rng, base_w, cfg.max_length)
            if sl.length in allowed_lengths and _clean(sl) and validate(sl)[0]:
                return sl
        return sl  # fall back to last (rare; keeps progress)

    return _gen


def balance_lists(
    cfg: GeneratorConfig,
    n_total: int,
    rng: random.Random,
    progress: bool = False,
    tag: str = "",
) -> List[StyleList]:
    """Balance one pool of n_total lists to the configured targets. Returns the lists."""
    if n_total <= 0:
        return []
    goal_absent_total = round(n_total * cfg.goal_absent_ratio)
    normal_total = n_total - goal_absent_total
    targets, weights = _build_targets(cfg)
    gen = _make_gen(cfg, rng)

    normal = [gen(False) for _ in range(normal_total)]
    ga = [gen(True) for _ in range(goal_absent_total)]

    def combined() -> List[StyleList]:
        return normal + ga

    if normal_total:
        _refine_pool(normal, normal_total, lambda: gen(False), combined,
                     targets, weights, cfg, f"{tag} normal".strip(), progress)
    if goal_absent_total:
        _refine_pool(ga, goal_absent_total, lambda: gen(True), combined,
                     targets, weights, cfg, f"{tag} goal-absent".strip(), progress)

    out = normal[:normal_total] + ga[:goal_absent_total]
    rng.shuffle(out)
    return out


def attach_targets(stats: Dict, cfg: GeneratorConfig, total: int) -> Dict:
    """Add target distributions (for stats/plots) to a finalized stats dict."""
    stats["target_length_distribution"] = {
        int(L): round(total * r) for L, r in _normalize(cfg.length_ratios).items()
    }
    if cfg.ab_distance_ratios:
        stats["ab_pre_post_distance"]["target_ratio"] = {
            int(k): round(v, 4) for k, v in sorted(_normalize(cfg.ab_distance_ratios).items())
        }
    if cfg.or_distance_ratios:
        stats["or_reference_distance"]["target_ratio"] = {
            int(k): round(v, 4) for k, v in sorted(_normalize(cfg.or_distance_ratios).items())
        }
    return stats


def balance(
    cfg: GeneratorConfig,
    num_scenes: Optional[int] = None,
    progress: bool = False,
) -> Tuple[List[List[StyleList]], Dict]:
    """Whole-dataset balance (non-resumable). Returns (scene groups, statistics).

    The resumable per-scene path lives in refon.runner; this remains for programmatic
    use and tests.
    """
    rng = random.Random(cfg.seed)
    n_scenes = num_scenes if num_scenes is not None else cfg.num_scenes
    total = n_scenes * cfg.episodes_per_scene
    if total <= 0:
        return [], compute_statistics([])

    all_lists = balance_lists(cfg, total, rng, progress=progress)
    rng.shuffle(all_lists)

    groups: List[List[StyleList]] = []
    per = cfg.episodes_per_scene
    for i in range(n_scenes):
        groups.append(all_lists[i * per:(i + 1) * per])

    stats = compute_statistics(all_lists)
    stats = attach_targets(stats, cfg, total)
    stats["num_scenes"] = n_scenes
    stats["episodes_per_scene"] = per
    return groups, stats
