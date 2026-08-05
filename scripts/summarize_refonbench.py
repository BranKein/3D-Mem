#!/usr/bin/env python
"""Summarise a RefON-Bench run as a per-role SR / SPL table.

    python scripts/summarize_refonbench.py                       # results/exp_eval_refonbench
    python scripts/summarize_refonbench.py path/to/results_dir
    python scripts/summarize_refonbench.py --csv out.csv

Two sources, in order of preference:

  pkl  `success_by_task_*.pkl` / `spl_by_task_*.pkl`, written by Logger.save_results()
       once an episode completes. Authoritative, and carries per-subtask sample counts.

  log  the run's `log_*.log`. Used when no pkl exists yet, so a run that is still going
       (or was interrupted mid-episode) can still be summarised. The logger reprints the
       running averages after every subtask, so the last occurrence of each role is its
       current value, and `Role: X` lines give the counts.

Per-role numbers are distance-based: logger_goatbench.py accumulates success_by_task
from success_by_distance. The overall block reports both snapshot and distance.
"""

import argparse
import glob
import os
import pickle
import re
import sys
from collections import defaultdict

# The order roles are presented in; anything unknown is appended alphabetically.
ROLE_ORDER = [
    "S",
    "AB_pre",
    "AB_post",
    "AR_pre",
    "AR_post",
    "OR_post",
    "AB_pre+OR_post",
]

# Roles folded into "S" before aggregating, because the *_pre subgoals name their target
# outright ("Find the clothes. Let's call it A1.") -- the alias they bind only matters to
# the *_post subgoal that refers back to it, so as a navigation task they are the same
# thing as a plain S. Pass --no-merge-roles to report them separately.
#
# AB_pre+OR_post deliberately stays out: it binds an alias *and* refers back
# ("Find the 1st one again. Let's call it A1."), so it cannot be resolved without the
# history and folding it in would inflate S with anaphoric cases.
ROLE_MERGE = {
    "AB_pre": "S",
    "AR_pre": "S",
}


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _merge_role(role, merge=True):
    return ROLE_MERGE.get(role, role) if merge else role


def _sort_roles(roles):
    known = [r for r in ROLE_ORDER if r in roles]
    rest = sorted(r for r in roles if r not in ROLE_ORDER)
    return known + rest


# --------------------------------------------------------------------------- #
# sources
# --------------------------------------------------------------------------- #
def from_pickles(results_dir, merge=True):
    """Merge every split's *_by_task_*.pkl. Returns (per_role, overall) or None."""
    succ_files = sorted(glob.glob(os.path.join(results_dir, "success_by_task_*.pkl")))
    spl_files = sorted(glob.glob(os.path.join(results_dir, "spl_by_task_*.pkl")))
    if not succ_files or not spl_files:
        return None

    # The pkls hold the raw per-subtask values, so folding roles together is just
    # concatenation -- no re-weighting needed and no precision lost.
    success, spl = defaultdict(list), defaultdict(list)
    for path in succ_files:
        with open(path, "rb") as f:
            for role, values in pickle.load(f).items():
                success[_merge_role(role, merge)].extend(values)
    for path in spl_files:
        with open(path, "rb") as f:
            for role, values in pickle.load(f).items():
                spl[_merge_role(role, merge)].extend(values)

    per_role = {}
    for role in set(success) | set(spl):
        per_role[role] = {
            "n": len(success.get(role, [])),
            "sr": _mean(success.get(role, [])),
            "spl": _mean(spl.get(role, [])),
        }

    overall = {}
    for key, pattern in [
        ("sr_snapshot", "success_by_snapshot_*.pkl"),
        ("sr_distance", "success_by_distance_*.pkl"),
        ("spl_snapshot", "spl_by_snapshot_*.pkl"),
        ("spl_distance", "spl_by_distance_*.pkl"),
    ]:
        values = []
        for path in sorted(glob.glob(os.path.join(results_dir, pattern))):
            with open(path, "rb") as f:
                values.extend(pickle.load(f).values())
        if values:
            overall[key] = _mean(values)
    return per_role, overall


# A subtask logs its role twice: "Role: X | Instruction: ..." when it starts, and a
# bare "Role: X" right after log_subtask_result() when it finishes. Only the second
# form is a completed subtask, so anchor to end-of-line or the counts come out doubled.
_ROLE_LINE = re.compile(r"Role:\s*(\S+)\s*$")
_SR_ROLE = re.compile(r"Success rate for (\S+):\s*([\d.]+)")
_SPL_ROLE = re.compile(r"SPL for (\S+):\s*([\d.]+)")
_OVERALL = re.compile(r"(Success rate|SPL) by (snapshot|distance):\s*([\d.]+)")


def from_logs(results_dir, merge=True):
    """Recover the running averages from the run log. Returns (per_role, overall)."""
    logs = sorted(glob.glob(os.path.join(results_dir, "log_*.log")))
    if not logs:
        return None

    counts = defaultdict(int)
    sr, spl, overall = {}, {}, {}
    for path in logs:
        with open(path, errors="replace") as f:
            for line in f:
                m = _ROLE_LINE.search(line)
                if m:
                    counts[m.group(1)] += 1
                    continue
                m = _SR_ROLE.search(line)
                if m:
                    sr[m.group(1)] = float(m.group(2))  # last write wins
                    continue
                m = _SPL_ROLE.search(line)
                if m:
                    spl[m.group(1)] = float(m.group(2))
                    continue
                m = _OVERALL.search(line)
                if m:
                    metric = "sr" if m.group(1) == "Success rate" else "spl"
                    overall[f"{metric}_{m.group(2)}"] = float(m.group(3)) / 100.0

    if not sr and not spl:
        return None

    # Only the final running mean survives in the log, not the individual values, so
    # folding roles has to re-weight by each role's sample count.
    merged = defaultdict(lambda: {"n": 0, "sr_sum": 0.0, "spl_sum": 0.0})
    for role in set(sr) | set(spl):
        n = counts.get(role, 0)
        target = merged[_merge_role(role, merge)]
        target["n"] += n
        # the log prints percentages; normalise to the pkl's 0-1 scale
        target["sr_sum"] += sr.get(role, 0.0) / 100.0 * n
        target["spl_sum"] += spl.get(role, 0.0) / 100.0 * n

    per_role = {
        role: {
            "n": acc["n"],
            "sr": acc["sr_sum"] / acc["n"] if acc["n"] else float("nan"),
            "spl": acc["spl_sum"] / acc["n"] if acc["n"] else float("nan"),
        }
        for role, acc in merged.items()
    }
    return per_role, overall


# --------------------------------------------------------------------------- #
def render(per_role, overall, source, results_dir, merge=True):
    lines = []
    lines.append(f"RefON-Bench summary  ({results_dir})")
    lines.append(f"source: {source}")
    if merge:
        folded = ", ".join(sorted(ROLE_MERGE))
        lines.append(f"roles folded into S: {folded}   (--no-merge-roles to separate)")
    lines.append("")
    lines.append(f"{'role':<16}{'N':>6}{'SR %':>10}{'SPL %':>10}")
    lines.append("-" * 42)

    total_n = 0
    for role in _sort_roles(per_role):
        r = per_role[role]
        total_n += r["n"]
        lines.append(f"{role:<16}{r['n']:>6}{100 * r['sr']:>10.2f}{100 * r['spl']:>10.2f}")

    lines.append("-" * 42)
    if "sr_distance" in overall:
        lines.append(
            f"{'ALL (distance)':<16}{total_n:>6}"
            f"{100 * overall['sr_distance']:>10.2f}"
            f"{100 * overall.get('spl_distance', float('nan')):>10.2f}"
        )
    if "sr_snapshot" in overall:
        lines.append(
            f"{'ALL (snapshot)':<16}{total_n:>6}"
            f"{100 * overall['sr_snapshot']:>10.2f}"
            f"{100 * overall.get('spl_snapshot', float('nan')):>10.2f}"
        )
    lines.append("")
    lines.append("Per-role numbers are distance-based (success_by_task is accumulated")
    lines.append("from success_by_distance in logger_goatbench.py).")
    if source.startswith("log"):
        lines.append("")
        lines.append("NOTE: read from the log, not from pickles -- the run has not")
        lines.append("      completed an episode yet, so these are running averages.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "results_dir",
        nargs="?",
        default="results/exp_eval_refonbench",
        help="run output directory (default: results/exp_eval_refonbench)",
    )
    ap.add_argument("--csv", help="also write the table to this CSV file")
    ap.add_argument(
        "--from-log",
        action="store_true",
        help="ignore pickles and read the log (running averages of an in-flight run)",
    )
    ap.add_argument(
        "--no-merge-roles",
        action="store_true",
        help=f"report every role separately instead of folding {sorted(ROLE_MERGE)} into S",
    )
    args = ap.parse_args()
    merge = not args.no_merge_roles

    if not os.path.isdir(args.results_dir):
        sys.exit(f"no such directory: {args.results_dir}")

    result, source = None, None
    if not args.from_log:
        result = from_pickles(args.results_dir, merge=merge)
        source = "pickles (success_by_task / spl_by_task)"
    if result is None:
        result = from_logs(args.results_dir, merge=merge)
        source = "log (no completed episode yet)"
    if result is None:
        sys.exit(
            f"no results found in {args.results_dir} -- "
            "expected success_by_task_*.pkl or log_*.log with subtask results"
        )

    per_role, overall = result
    print(render(per_role, overall, source, args.results_dir, merge=merge))

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("role,n,sr,spl\n")
            for role in _sort_roles(per_role):
                r = per_role[role]
                f.write(f"{role},{r['n']},{r['sr']:.6f},{r['spl']:.6f}\n")
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
