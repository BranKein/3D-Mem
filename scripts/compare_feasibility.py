#!/usr/bin/env python
"""Compare feasibility runs side by side, per instruction style.

    python scripts/compare_feasibility.py results/exp_feasibility_refonbench_*
    python scripts/compare_feasibility.py --mode all_at_once results/a results/b
    python scripts/compare_feasibility.py --metric referent_sr results/a results/b
    python scripts/compare_feasibility.py --first-episodes 50 results/a results/b
    python scripts/compare_feasibility.py --plot cmp.png results/exp_feasibility_*

Each argument is a results directory written by run_refonbench_feasibility.py. The
column label is taken from the summary json's `model` field, falling back to the
directory name. With exactly two runs a delta column is added.

`--first-episodes N` recomputes every run from its per-subgoal records over the first N
episodes, which is what makes a full run comparable with one that was cut short by
`--episodes-per-scene N`: both truncate to the same stable prefix. It is a prefix of the
shard, not of the episode ids -- the builder drops episodes it cannot place, so the ids
have holes -- so the order is taken from the records file, which is written in shard
order.

Metrics (see run_refonbench_feasibility.py):
  sr           joint -- right referent AND right category (default)
  referent_sr  pointed at the right object
  category_sr  named the right category
"""

import argparse
import json
import os
import re
import sys

# same presentation order as scripts/summarize_refonbench.py
ROLE_ORDER = [
    "S",
    "AB_pre",
    "AB_post",
    "AR_pre",
    "AR_post",
    "OR_post",
    "AB_pre+OR_post",
]

# The split the results actually turn on: an instruction that names its own target
# versus one that has to be resolved against an earlier subgoal.
NEW_OBJECT_ROLES = ("S", "AB_pre", "AR_pre")
BACK_REF_ROLES = ("AB_post", "AR_post", "OR_post", "AB_pre+OR_post")

# Validated categorical palette, light surface #fcfcfb, assigned in this fixed order and
# never cycled (scripts/validate_palette.js of the dataviz skill: all checks pass on the
# adjacent pairlist; three slots sit under 3:1 contrast, so the printed table is the
# required relief -- --plot always prints it).
SERIES_COLORS = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]
METRIC_LABEL = {
    "sr": "joint SR",
    "referent_sr": "referent SR",
    "category_sr": "category SR",
}
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d9d8d4"


def load(results_dir, mode, first_episodes=None):
    path = os.path.join(results_dir, f"feasibility_results_{mode}.json")
    if not os.path.exists(path):
        return None, f"no {os.path.basename(path)} in {results_dir}"
    with open(path) as f:
        summary = json.load(f)
    if first_episodes:
        recomputed, err = _from_records(results_dir, mode, first_episodes)
        if err:
            return None, err
        summary = dict(summary, **recomputed)
    return summary, None


def _from_records(results_dir, mode, first_episodes):
    """Re-aggregate the first N episodes from the per-subgoal records."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run_refonbench_feasibility import aggregate

    path = os.path.join(results_dir, f"feasibility_records_{mode}.jsonl")
    if not os.path.exists(path):
        return None, f"--first-episodes needs {os.path.basename(path)}, missing in {results_dir}"

    seen, keep, records = [], set(), []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            ep = r["episode_id"]
            if ep not in keep:
                if len(seen) >= first_episodes:
                    continue
                seen.append(ep)
                keep.add(ep)
            records.append(r)
    if len(seen) < first_episodes:
        print(f"note: {results_dir} holds only {len(seen)} episodes", file=sys.stderr)
    return dict(aggregate(records), num_records=len(records)), None


def label_for(summary, results_dir):
    model = summary.get("model")
    return model or os.path.basename(os.path.normpath(results_dir))


def sort_roles(roles):
    known = [r for r in ROLE_ORDER if r in roles]
    return known + sorted(r for r in roles if r not in ROLE_ORDER)


def parse_size(model):
    """`qwen3.5:0.8b` -> 0.8, `gemma4:26b-a4b-it-qat` -> 26.0, unknown -> None.

    Only the tag is searched, so the 2.5 in `qwen2.5vl` is not mistaken for a size, and
    the first match wins, so an MoE tag gives total rather than active parameters.
    """
    _, _, tag = str(model).partition(":")
    m = re.search(r"(\d+(?:\.\d+)?)\s*b\b", tag)
    return float(m.group(1)) if m else None


def group_sr(summary, roles, metric):
    """Sample-weighted metric over a set of styles, or None if none are present."""
    n = tot = 0
    for role in roles:
        cell = summary["per_style"].get(role)
        if cell:
            n += cell["count"]
            tot += cell["count"] * cell[metric]
    return tot / n if n else None


def plot(runs, metric, mode, path, first_episodes=None):
    """Two panels: per-style bars, and the metric against model size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    roles = sort_roles({r for _, s in runs for r in s["per_style"]})
    colors = {label: SERIES_COLORS[i % len(SERIES_COLORS)]
              for i, (label, _) in enumerate(runs)}

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1.75, 1]}
    )
    fig.patch.set_facecolor("#fcfcfb")

    # --- per-style grouped bars -------------------------------------------------
    span = 0.82
    width = span / len(runs)
    for i, (label, summary) in enumerate(runs):
        xs, ys = [], []
        for j, role in enumerate(roles):
            cell = summary["per_style"].get(role)
            if cell:
                xs.append(j - span / 2 + width * (i + 0.5))
                ys.append(100 * cell[metric])
        # the 2px surface gap between adjacent bars comes out of the bar width
        ax.bar(xs, ys, width=width * 0.88, color=colors[label], label=label,
               linewidth=0)
    ax.set_xticks(range(len(roles)))
    ax.set_xticklabels(roles, rotation=30, ha="right", fontsize=9, color=INK_MUTED)
    ax.set_ylabel(METRIC_LABEL.get(metric, metric) + " (%)", fontsize=9, color=INK_MUTED)
    ax.set_title("by instruction style", fontsize=11, color=INK, loc="left")

    # --- metric against model size ---------------------------------------------
    families = {}
    for label, summary in runs:
        size = parse_size(label)
        if size:
            families.setdefault(str(label).partition(":")[0], []).append(
                (size, 100 * summary["overall"][metric], label)
            )
    for points in families.values():
        points.sort()
        # only join points inside one family -- a line across families would draw a
        # scaling curve out of models that share nothing but a parameter count
        if len(points) > 1:
            ax2.plot([p[0] for p in points], [p[1] for p in points],
                     color=GRID, linewidth=2, zorder=1)
        for size, value, label in points:
            ax2.scatter([size], [value], s=80, color=colors[label], zorder=2,
                        edgecolors="#fcfcfb", linewidths=2)
            # drop the label below the point when it would otherwise run into the top
            above = value <= 88
            ax2.annotate(f"{label} {value:.0f}%", (size, value),
                         textcoords="offset points",
                         xytext=(0, 10 if above else -11),
                         ha="center", va="bottom" if above else "top",
                         fontsize=8, color=INK_MUTED)
    sizes = sorted({p[0] for pts in families.values() for p in pts})
    if sizes:
        ax2.set_xscale("log")
        # named sizes beat decade ticks: the models are the x values, and "2 x 10^1"
        # is not a model
        ax2.set_xticks(sizes)
        ax2.set_xticklabels([f"{s:g}B" for s in sizes])
        ax2.minorticks_off()
        ax2.set_xlim(min(sizes) / 1.9, max(sizes) * 1.9)
    ax2.set_xlabel("parameters (B, total)", fontsize=9, color=INK_MUTED)
    ax2.set_title("by model size", fontsize=11, color=INK, loc="left")

    for a in (ax, ax2):
        a.set_ylim(0, 105)
        a.set_facecolor("#fcfcfb")
        a.grid(axis="y", color=GRID, linewidth=0.8)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(GRID)
        a.tick_params(colors=INK_MUTED, labelsize=9)

    ax.legend(fontsize=8, frameon=False, ncol=min(len(runs), 4),
              loc="upper center", bbox_to_anchor=(0.5, -0.22), labelcolor=INK_MUTED)
    scope = f", first {first_episodes} episodes" if first_episodes else ""
    fig.suptitle(f"RefON instruction feasibility - {mode}{scope}",
                 fontsize=13, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    print(f"\nwrote {path}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dirs", nargs="+", help="results/<exp_name>_<model> dirs")
    parser.add_argument("--mode", default="incremental",
                        choices=["incremental", "all_at_once"])
    parser.add_argument("--metric", default="sr",
                        choices=["sr", "referent_sr", "category_sr"])
    parser.add_argument("--plot", metavar="PATH", default=None,
                        help="also write a PNG: per-style bars + metric vs model size")
    parser.add_argument("--first-episodes", type=int, default=None,
                        help="re-aggregate each run over the first N episodes of the "
                             "shard, so runs of different length compare like for like")
    args = parser.parse_args(argv)

    runs = []
    for d in args.results_dirs:
        summary, err = load(d, args.mode, args.first_episodes)
        if err:
            print(f"skipping: {err}", file=sys.stderr)
            continue
        runs.append((label_for(summary, d), summary))
    if not runs:
        print("no runs to compare", file=sys.stderr)
        return 1

    roles = sort_roles({r for _, s in runs for r in s["per_style"]})
    width = max(20, max(len(l) for l, _ in runs) + 2)

    header = f"{'instruction style':<20}{'n':>6}" + "".join(f"{l:>{width}}" for l, _ in runs)
    if len(runs) == 2:
        header += f"{'delta':>10}"
    scope = f"  first {args.first_episodes} episodes" if args.first_episodes else ""
    print(f"mode={args.mode}  metric={args.metric}{scope}")
    print(header)
    print("-" * len(header))

    for role in roles + ["__overall__"]:
        if role == "__overall__":
            print("-" * len(header))
            cells = [s["overall"] for _, s in runs]
            name, n = "ALL", cells[0]["count"]
        else:
            cells = [s["per_style"].get(role) for _, s in runs]
            name = role
            n = next((c["count"] for c in cells if c), 0)
        row = f"{name:<20}{n:>6}"
        for c in cells:
            row += f"{c[args.metric]:>{width}.1%}" if c else f"{'-':>{width}}"
        if len(runs) == 2 and all(cells):
            row += f"{cells[1][args.metric] - cells[0][args.metric]:>+10.1%}"
        print(row)

    # the split the numbers actually turn on, which no single style row shows
    print("-" * len(header))
    for name, group in (("names its target", NEW_OBJECT_ROLES),
                        ("back-reference", BACK_REF_ROLES)):
        values = [group_sr(s, group, args.metric) for _, s in runs]
        row = f"{name:<20}{'':>6}"
        for v in values:
            row += f"{v:>{width}.1%}" if v is not None else f"{'-':>{width}}"
        if len(runs) == 2 and all(v is not None for v in values):
            row += f"{values[1] - values[0]:>+10.1%}"
        print(row)

    if args.plot:
        plot(runs, args.metric, args.mode, args.plot, args.first_episodes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
