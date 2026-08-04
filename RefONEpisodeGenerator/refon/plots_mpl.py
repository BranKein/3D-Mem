"""matplotlib-based statistics visualization — saves PNGs.

Draws the statistics charts with matplotlib and writes them as PNG. If matplotlib is
missing, the import fails and the CLI reports it gracefully. Chart titles/labels are
in English.

Generated PNGs:
  role_pie.png        : instruction style (role) distribution
  length_bar.png      : length distribution (target vs actual)
  goal_absent_pie.png : goal-absent vs normal
  absent_kind_pie.png : the 3 goal-absent kinds
  dashboard.png       : all 4 on a single sheet
"""
from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # save to files only, no GUI
import matplotlib.pyplot as plt  # noqa: E402

_PALETTE = [
    "#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
    "#86bcb6", "#d37295",
]


def _pie(ax, title: str, items: Sequence[Tuple[str, float]]) -> None:
    items = [(l, float(v)) for l, v in items if v > 0]
    ax.set_title(title, fontsize=13, fontweight="bold")
    if not items:
        ax.text(0.5, 0.5, "(no data)", ha="center", va="center", color="#999")
        ax.axis("off")
        return
    labels = [l for l, _ in items]
    vals = [v for _, v in items]
    total = sum(vals)
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(items))]
    wedges, _texts = ax.pie(
        vals, colors=colors, startangle=90, counterclock=False,
        wedgeprops={"edgecolor": "white", "linewidth": 1},
    )
    legend_labels = [
        f"{l} — {v:g} ({100*v/total:.1f}%)" for l, v in items
    ]
    ax.legend(
        wedges, legend_labels, loc="center left",
        bbox_to_anchor=(1.0, 0.5), fontsize=9, frameon=False,
    )
    ax.axis("equal")


def _length_bar(ax, stats: Dict) -> None:
    actual = stats.get("length_distribution", {})
    target = stats.get("target_length_distribution", {})
    cats = sorted(set(int(k) for k in actual) | set(int(k) for k in target))
    actual_vals = [actual.get(c, actual.get(str(c), 0)) for c in cats]
    target_vals = [target.get(c, target.get(str(c), 0)) for c in cats]

    import numpy as np
    x = np.arange(len(cats))
    has_target = any(target_vals)
    w = 0.38 if has_target else 0.6
    ax.set_title("style list length distribution", fontsize=13, fontweight="bold")
    if has_target:
        ax.bar(x - w / 2, actual_vals, w, label="actual", color=_PALETTE[0])
        ax.bar(x + w / 2, target_vals, w, label="target", color=_PALETTE[1])
        ax.legend(fontsize=9, frameon=False)
    else:
        ax.bar(x, actual_vals, w, label="actual", color=_PALETTE[0])
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in cats])
    ax.set_xlabel("length")
    ax.set_ylabel("lists")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#eee")
    ax.set_axisbelow(True)


def _distance_bar(ax, title: str, summary: Dict, color_idx: int = 2) -> None:
    """Generic reference-distance bar chart with an optional target overlay."""
    import numpy as np
    summary = summary or {}
    dist = summary.get("distribution", {})
    target_ratio = summary.get("target_ratio") or {}
    ax.set_title(title, fontsize=13, fontweight="bold")
    if not dist:
        ax.text(0.5, 0.5, "(no references)", ha="center", va="center", color="#999")
        ax.axis("off")
        return
    count = summary.get("count", 0) or 0
    keys = sorted(set(int(k) for k in dist) | set(int(k) for k in target_ratio))
    actual = [dist.get(k, dist.get(str(k), 0)) for k in keys]
    x = np.arange(len(keys))
    if target_ratio:
        # express the target as expected counts (target_ratio * count) to overlay
        target = [round(target_ratio.get(k, target_ratio.get(str(k), 0)) * count)
                  for k in keys]
        w = 0.38
        ax.bar(x - w / 2, actual, w, label="actual", color=_PALETTE[color_idx])
        ax.bar(x + w / 2, target, w, label="target", color=_PALETTE[1])
        ax.legend(fontsize=9, frameon=False)
    else:
        ax.bar(x, actual, 0.7, color=_PALETTE[color_idx])
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in keys])
    ax.set_xlabel(f"distance (count={count}, avg={summary.get('avg', 0)})")
    ax.set_ylabel("references")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#eee")
    ax.set_axisbelow(True)


def write_plots(stats: Dict, out_dir: str, title: str = "RefON statistics", dpi: int = 130) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    written: List[str] = []

    role_items = sorted(stats.get("role_frequency", {}).items(), key=lambda kv: -kv[1])
    n = stats.get("num_lists", 0)
    ga = stats.get("goal_absent_lists", 0)
    akd = sorted(stats.get("absent_kind_distribution", {}).items(), key=lambda kv: -kv[1])

    # individual PNGs
    def _save_pie(fname, ttl, items):
        fig, ax = plt.subplots(figsize=(6.2, 3.4))
        _pie(ax, ttl, items)
        fig.tight_layout()
        path = os.path.join(out_dir, fname)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    _save_pie("role_pie.png", "Instruction style (role) distribution", role_items)
    _save_pie("goal_absent_pie.png", "Episodes: goal-absent vs normal",
              [("normal", n - ga), ("goal-absent", ga)])
    _save_pie("absent_kind_pie.png", "goal-absent kind distribution", akd)

    ab = stats.get("ab_pre_post_distance", {})
    orr = stats.get("or_reference_distance", {})
    comb = stats.get("ref_distance_combined", {})

    def _save_fig(fname, draw):
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        draw(ax)
        fig.tight_layout()
        path = os.path.join(out_dir, fname)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    _save_fig("length_bar.png", lambda ax: _length_bar(ax, stats))
    _save_fig("ab_distance_bar.png", lambda ax: _distance_bar(ax, "AB pre->post distance", ab, 2))
    _save_fig("or_distance_bar.png", lambda ax: _distance_bar(ax, "OR reference distance", orr, 5))
    _save_fig("ref_distance_combined_bar.png",
              lambda ax: _distance_bar(ax, "AB+OR reference distance (combined)", comb, 6))

    # single-sheet dashboard (4x2)
    fig, axes = plt.subplots(4, 2, figsize=(13, 16))
    fig.suptitle(title, fontsize=15, fontweight="bold")
    _pie(axes[0, 0], "Instruction style (role) distribution", role_items)
    _length_bar(axes[0, 1], stats)
    _pie(axes[1, 0], "goal-absent vs normal", [("normal", n - ga), ("goal-absent", ga)])
    _pie(axes[1, 1], "goal-absent kind distribution", akd)
    _distance_bar(axes[2, 0], "AB pre->post distance", ab, 2)
    _distance_bar(axes[2, 1], "OR reference distance", orr, 5)
    _distance_bar(axes[3, 0], "AB+OR reference distance (combined)", comb, 6)
    axes[3, 1].axis("off")  # spare slot
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = os.path.join(out_dir, "dashboard.png")
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    written.append(p)

    return written
