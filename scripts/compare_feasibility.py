#!/usr/bin/env python
"""Compare feasibility runs side by side, per instruction style.

    python scripts/compare_feasibility.py results/exp_feasibility_refonbench_*
    python scripts/compare_feasibility.py --mode all_at_once results/a results/b
    python scripts/compare_feasibility.py --metric referent_sr results/a results/b
    python scripts/compare_feasibility.py --first-episodes 50 results/a results/b

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


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dirs", nargs="+", help="results/<exp_name>_<model> dirs")
    parser.add_argument("--mode", default="incremental",
                        choices=["incremental", "all_at_once"])
    parser.add_argument("--metric", default="sr",
                        choices=["sr", "referent_sr", "category_sr"])
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
