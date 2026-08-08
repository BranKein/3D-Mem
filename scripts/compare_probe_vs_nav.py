#!/usr/bin/env python
"""Referent probe vs destination probe, paired subgoal by subgoal.

    python scripts/compare_probe_vs_nav.py --plot cmp_probe_vs_nav.png

Both probes run the same episodes, so every subgoal appears in both and the two can be
compared as a paired test rather than as two independent averages. That is what says
whether producing a destination costs anything *beyond* working out the referent: if nav
failures were simply the referent failures plus noise, "nav only" would be near zero.

The nav sweep's extra GA_absent_object/not_found turn has no counterpart in the referent
probe and is dropped from the pairing.
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.compare_feasibility import (  # noqa: E402
    GRID, INK, INK_MUTED, SERIES_COLORS, sort_roles,
)

PROBE_GLOB = "results/exp_feasibility_refonbench_nothink_*"
NAV_GLOB = "results/exp_feasibility_nav_refonbench_nothink_*"


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def model_of(summary_path):
    with open(summary_path) as f:
        return json.load(f).get("model")


def collect(probe_glob, nav_glob):
    """model -> {(episode, order): (probe_correct, nav_correct)}"""
    probe = {}
    for d in sorted(glob.glob(probe_glob)):
        rec = os.path.join(d, "feasibility_records_incremental.jsonl")
        res = os.path.join(d, "feasibility_results_incremental.json")
        if os.path.exists(rec) and os.path.exists(res):
            probe[model_of(res)] = load(rec)

    paired = {}
    for d in sorted(glob.glob(nav_glob)):
        rec = os.path.join(d, "feasibility_nav_records.jsonl")
        res = os.path.join(d, "feasibility_nav_results.json")
        if not (os.path.exists(rec) and os.path.exists(res)):
            continue
        model = model_of(res)
        if model not in probe:
            continue
        by_key = {(r["episode_id"], r["order"]): r for r in probe[model]}
        rows = {}
        for r in load(rec):
            if r.get("after_not_found"):
                continue  # no counterpart in the referent probe
            key = (r["episode_id"], r["order"])
            other = by_key.get(key)
            if other:
                rows[key] = (other["correct"], r["correct"], r["role"])
        paired[model] = rows
    return paired


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--probe-glob", default=PROBE_GLOB)
    p.add_argument("--nav-glob", default=NAV_GLOB)
    p.add_argument("--plot", default=None)
    args = p.parse_args(argv)

    paired = collect(args.probe_glob, args.nav_glob)
    if not paired:
        print("no model has both a referent run and a nav run", file=sys.stderr)
        return 1

    order = sorted(paired, key=lambda m: len(paired[m]))
    print(f"{'model':<16}{'n':>7}{'referent':>10}{'nav':>8}{'both ok':>10}"
          f"{'probe only':>12}{'nav only':>10}{'both bad':>10}")
    print("-" * 83)
    for model in order:
        rows = paired[model]
        n = len(rows)
        pc = sum(1 for a, b, _ in rows.values() if a)
        nc = sum(1 for a, b, _ in rows.values() if b)
        both = sum(1 for a, b, _ in rows.values() if a and b)
        ponly = sum(1 for a, b, _ in rows.values() if a and not b)
        nonly = sum(1 for a, b, _ in rows.values() if b and not a)
        neither = sum(1 for a, b, _ in rows.values() if not a and not b)
        print(f"{model:<16}{n:>7}{pc/n:>9.1%}{nc/n:>8.1%}{both/n:>10.1%}"
              f"{ponly/n:>12.1%}{nonly/n:>10.1%}{neither/n:>10.1%}")

    # per style, per model
    roles = sort_roles({r for rows in paired.values() for _, _, r in rows.values()})
    print()
    header = f"{'instruction style':<20}{'n':>6}"
    for model in order:
        header += f"{model + ' probe':>20}{model + ' nav':>18}"
    print(header)
    print("-" * len(header))
    per_style = {}
    for role in roles:
        row = None
        cells = []
        for model in order:
            vals = [(a, b) for a, b, r in paired[model].values() if r == role]
            if not vals:
                cells.append(None)
                continue
            pr = sum(1 for a, _ in vals if a) / len(vals)
            nv = sum(1 for _, b in vals if b) / len(vals)
            cells.append((pr, nv))
            row = len(vals)
        per_style[role] = cells
        line = f"{role:<20}{row or 0:>6}"
        for c in cells:
            line += f"{c[0]:>19.1%} {c[1]:>17.1%}" if c else f"{'-':>20}{'-':>18}"
        print(line)

    if args.plot:
        plot(order, roles, per_style, args.plot)
    return 0


def plot(models, roles, per_style, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 5.2),
                             sharey=True)
    axes = [axes] if len(models) == 1 else list(axes)
    fig.patch.set_facecolor("#fcfcfb")

    for ax, model, i in zip(axes, models, range(len(models))):
        xs = range(len(roles))
        probe = [100 * per_style[r][i][0] if per_style[r][i] else 0 for r in roles]
        nav = [100 * per_style[r][i][1] if per_style[r][i] else 0 for r in roles]
        ax.bar([x - 0.21 for x in xs], probe, width=0.38, color=SERIES_COLORS[0],
               label="referent probe", linewidth=0)
        ax.bar([x + 0.21 for x in xs], nav, width=0.38, color=SERIES_COLORS[1],
               label="destination probe", linewidth=0)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(roles, rotation=45, ha="right", fontsize=8, color=INK_MUTED)
        ax.set_title(model, fontsize=11, color=INK, loc="left")
        ax.set_ylim(0, 105)
        ax.set_facecolor("#fcfcfb")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=8)

    axes[0].set_ylabel("joint SR (%)", fontsize=9, color=INK_MUTED)
    axes[0].legend(fontsize=8, frameon=False, loc="upper center",
                   bbox_to_anchor=(0.5, -0.42), ncol=2, labelcolor=INK_MUTED)
    fig.suptitle("Which object is it? vs Where does the robot go?",
                 fontsize=13, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    print(f"\nwrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
