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
# Reference distance = how many instructions back the referent sits (order - ref_order).
# Binned rather than plotted per value: the distribution is heavily front-loaded (on the
# long-episode set, 262 of 452 back references point 1-2 instructions back and the tail
# past 17 has single-digit counts), so per-value points would be noise dressed as a curve.
DISTANCE_BINS = [
    (1, 1, "1"),
    (2, 2, "2"),
    (3, 4, "3-4"),
    (5, 8, "5-8"),
    (9, 16, "9-16"),
    (17, 10**6, "17+"),
]
MIN_BIN_SAMPLES = 5

METRIC_LABEL = {
    "sr": "joint SR",
    "referent_sr": "referent SR",
    "category_sr": "category SR",
    "action_sr": "action SR",
    "coordinate_sr": "coord SR",
}

# metric -> the per-record boolean it averages, for the reference-distance curve
METRIC_FIELD = {
    "sr": "correct",
    "referent_sr": "referent_correct",
    "category_sr": "category_correct",
    "action_sr": "action_correct",
    "coordinate_sr": "coordinate_correct",
}


def filenames(mode, nav):
    """(summary json, records jsonl) for a run directory."""
    if nav:
        return "feasibility_nav_results.json", "feasibility_nav_records.jsonl"
    return (f"feasibility_results_{mode}.json", f"feasibility_records_{mode}.jsonl")
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d9d8d4"


def load(results_dir, mode, first_episodes=None, nav=False):
    name, _ = filenames(mode, nav)
    path = os.path.join(results_dir, name)
    if not os.path.exists(path):
        return None, f"no {name} in {results_dir}"
    with open(path) as f:
        summary = json.load(f)
    if first_episodes:
        recomputed, err = _from_records(results_dir, mode, first_episodes, nav)
        if err:
            return None, err
        summary = dict(summary, **recomputed)
    return summary, None


def _repo_root():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def _read_records(results_dir, mode, nav=False):
    _, name = filenames(mode, nav)
    path = os.path.join(results_dir, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return [json.loads(line) for line in f]


def ref_distances(dataset_dir):
    """(episode_id, order) -> how many instructions back that subgoal's referent sits.

    Read from the dataset rather than the records: `ref_order` is what the instruction
    literally points at ("the 2nd one" -> 2), which is the distance the agent has to
    reach across. The records' own gt_refers_to is the *first* mention of the object,
    which can be nearer when an episode visits one object repeatedly.
    """
    _repo_root()
    from src.refonbench_utils import list_shard_files, load_shard

    distances = {}
    for name in list_shard_files(dataset_dir):
        shard = load_shard(os.path.join(dataset_dir, name))
        for episode in shard["episodes"]:
            for subtask in episode["subtasks"]:
                ref_order = subtask.get("ref_order")
                if ref_order:
                    key = (episode.get("episode_id"), subtask.get("order"))
                    distances[key] = subtask["order"] - ref_order
    return distances


def distance_curve(records, distances, metric):
    """{bin label: (n, rate)} for one run, over back-reference subgoals only."""
    field = METRIC_FIELD[metric]
    buckets = {label: [0, 0] for _, _, label in DISTANCE_BINS}
    for r in records:
        d = distances.get((r["episode_id"], r["order"]))
        if d is None:
            continue
        for lo, hi, label in DISTANCE_BINS:
            if lo <= d <= hi:
                buckets[label][0] += 1
                buckets[label][1] += int(r.get(field, False))
                break
    return {label: (n, hits / n) for label, (n, hits) in buckets.items() if n}


def _from_records(results_dir, mode, first_episodes, nav=False):
    """Re-aggregate the first N episodes from the per-subgoal records."""
    _repo_root()
    from run_refonbench_feasibility import aggregate

    all_records = _read_records(results_dir, mode, nav)
    if all_records is None:
        return None, f"--first-episodes needs the records jsonl, missing in {results_dir}"

    seen, keep, records = [], set(), []
    for r in all_records:
        ep = r["episode_id"]
        if ep not in keep:
            if len(seen) >= first_episodes:
                continue
            seen.append(ep)
            keep.add(ep)
        records.append(r)
    if len(seen) < first_episodes:
        print(f"note: {results_dir} holds only {len(seen)} episodes", file=sys.stderr)
    part_scores = (
        (("action_correct", "action_sr"), ("coordinate_correct", "coordinate_sr"))
        if nav
        else (("referent_correct", "referent_sr"), ("category_correct", "category_sr"))
    )
    return (
        dict(aggregate(records, part_scores=part_scores), num_records=len(records)),
        None,
    )


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


def plot(runs, metric, mode, path, first_episodes=None, curves=None, nav=False):
    """Per-style bars, the metric against model size, and against reference distance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    roles = sort_roles({r for _, s in runs for r in s["per_style"]})
    colors = {label: SERIES_COLORS[i % len(SERIES_COLORS)]
              for i, (label, _) in enumerate(runs)}

    panels = 3 if curves else 2
    fig, axes = plt.subplots(
        1, panels, figsize=(7.5 * panels / 1.35, 5.6),
        gridspec_kw={"width_ratios": [1.75, 1, 1.15][:panels]},
    )
    ax, ax2 = axes[0], axes[1]
    ax3 = axes[2] if curves else None
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
    # the nav sweep adds GA_absent_object/not_found, long enough to reach the legend at
    # 30 degrees -- steepen the labels rather than truncate a style name
    ax.set_xticklabels(roles, rotation=45, ha="right", fontsize=9, color=INK_MUTED)
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

    # --- metric against how far back the referent sits -------------------------
    if curves:
        labels = [lab for _, _, lab in DISTANCE_BINS
                  if any(lab in c for c in curves.values())]
        for label, curve in curves.items():
            xs = [i for i, lab in enumerate(labels)
                  if lab in curve and curve[lab][0] >= MIN_BIN_SAMPLES]
            ys = [100 * curve[labels[i]][1] for i in xs]
            ax3.plot(xs, ys, color=colors[label], linewidth=2, marker="o",
                     markersize=7, markeredgecolor="#fcfcfb", markeredgewidth=1.5)
        ax3.set_xticks(range(len(labels)))
        ax3.set_xticklabels(labels)
        ax3.set_xlabel("reference distance (instructions back)", fontsize=9,
                       color=INK_MUTED)
        ax3.set_title("by reference distance", fontsize=11, color=INK, loc="left")
        # sample counts, so a thin bin is not read as a trend
        any_curve = next(iter(curves.values()))
        for i, lab in enumerate(labels):
            n = any_curve.get(lab, (0, 0))[0]
            if n:
                ax3.annotate(f"n={n}", (i, 0), textcoords="offset points",
                             xytext=(0, 4), ha="center", fontsize=7, color=INK_MUTED)

    for a in [a for a in (ax, ax2, ax3) if a is not None]:
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
              loc="upper center", bbox_to_anchor=(0.5, -0.46), labelcolor=INK_MUTED)
    scope = f", first {first_episodes} episodes" if first_episodes else ""
    what = "destination (explore / coordinate / infeasible)" if nav else mode
    fig.suptitle(f"RefON instruction feasibility - {what}{scope}",
                 fontsize=13, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    print(f"\nwrote {path}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dirs", nargs="+", help="results/<exp_name>_<model> dirs")
    parser.add_argument("--mode", default="incremental",
                        choices=["incremental", "all_at_once"])
    parser.add_argument("--nav", action="store_true",
                        help="compare run_refonbench_feasibility_nav.py runs "
                             "(explore / coordinate / infeasible) instead")
    parser.add_argument("--metric", default="sr",
                        choices=["sr", "referent_sr", "category_sr",
                                 "action_sr", "coordinate_sr"])
    parser.add_argument("--plot", metavar="PATH", default=None,
                        help="also write a PNG: per-style bars, metric vs model size, "
                             "and metric vs reference distance")
    parser.add_argument("--dataset", default=None,
                        help="shard dir used for reference distances (default: the "
                             "test_data_dir recorded in the run's summary json)")
    parser.add_argument("--first-episodes", type=int, default=None,
                        help="re-aggregate each run over the first N episodes of the "
                             "shard, so runs of different length compare like for like")
    args = parser.parse_args(argv)

    runs = []
    for d in args.results_dirs:
        summary, err = load(d, args.mode, args.first_episodes, args.nav)
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
    what = "feasibility-nav" if args.nav else f"mode={args.mode}"
    print(f"{what}  metric={args.metric}{scope}")
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
        # reference distances come from the dataset the runs were scored against
        dataset = args.dataset or runs[0][1].get("test_data_dir")
        curves = None
        if dataset and os.path.isdir(dataset):
            distances = ref_distances(dataset)
            curves = {}
            for (label, _), d in zip(runs, args.results_dirs):
                records = _read_records(d, args.mode, args.nav)
                if records:
                    curves[label] = distance_curve(records, distances, args.metric)
            curves = curves or None
        elif dataset:
            print(f"note: no reference-distance panel, {dataset} is not a directory "
                  f"(pass --dataset)", file=sys.stderr)
        plot(runs, args.metric, args.mode, args.plot, args.first_episodes, curves,
             nav=args.nav)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
