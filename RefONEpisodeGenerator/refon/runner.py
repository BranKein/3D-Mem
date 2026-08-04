"""Resumable, per-scene runner for command 1 (style list generation).

Each scene is balanced and written on its own, then recorded as done in the progress
file (<out_dir>/_progress.json). Only one scene's lists live in memory at a time, and
statistics are accumulated incrementally (checkpointed in the progress file). If the
run is interrupted, re-running with the same out_dir resumes from the next unfinished
scene — finished scene files are kept and skipped.

Per-scene RNG is derived deterministically from (cfg.seed, slot), so a scene generated
after a resume is identical to one generated in a single pass — and so parallel workers
produce exactly the same output as a sequential run.

Generation is pure-Python CPU work, so parallelism uses multiprocessing (threads would
not help under the GIL). workers>1 builds scenes in a process pool; the main process
owns the progress file and merges each worker's per-scene statistics accumulator.
"""
from __future__ import annotations

import os
import random
from typing import Dict, Optional

from .balancer import attach_targets, balance_lists
from .config import GeneratorConfig
from .progress import ProgressTracker
from .statistics import (
    accumulate,
    finalize_accumulator,
    format_statistics,
    merge_accumulators,
    new_accumulator,
    normalize_accumulator,
)
from .storage import (
    PROGRESS_FILE,
    write_config,
    write_scene_file,
    write_statistics,
)


def _scene_rng(seed: int, slot: int) -> random.Random:
    return random.Random((seed * 1_000_003) ^ (slot * 2_654_435_761))


def _generate_scene(payload):
    """Generate + write one scene, returning its statistics accumulator.

    Top-level (picklable) so it can run in a worker process. Deterministic via
    _scene_rng(seed, slot), independent of ordering/parallelism.
    """
    out_dir, slot, cfg, per, seed = payload
    lists = balance_lists(cfg, per, _scene_rng(seed, slot))
    write_scene_file(out_dir, slot, lists)
    acc = accumulate(new_accumulator(), lists)
    return {"slot": slot, "acc": acc, "n": len(lists)}


def run_generate(
    cfg: GeneratorConfig,
    out_dir: str,
    num_scenes: Optional[int] = None,
    progress: bool = False,
    make_plots: bool = True,
    workers: int = 1,
) -> Dict:
    """Generate style lists scene-by-scene with checkpointing. Returns final stats.

    workers>1 generates scenes in parallel with a process pool (each scene is
    independent and deterministic per (seed, slot)). The main process owns the progress
    file and merges each worker's per-scene statistics accumulator.

    When make_plots is set, statistics charts are written at the end (best-effort: if
    matplotlib is unavailable the plots are skipped, the generation still succeeds).
    """
    n_scenes = num_scenes if num_scenes is not None else cfg.num_scenes
    per = cfg.episodes_per_scene
    total = n_scenes * per

    os.makedirs(os.path.join(out_dir, "scenes"), exist_ok=True)
    write_config(out_dir, cfg)

    tracker, resuming = ProgressTracker.load_or_create(
        os.path.join(out_dir, PROGRESS_FILE),
        command="generate",
        meta={"num_scenes": n_scenes, "episodes_per_scene": per, "workers": workers},
    )
    acc = normalize_accumulator(tracker.agg) if (resuming and tracker.agg) else new_accumulator()
    done = tracker.completed
    tracker.save()  # write the progress file right away, before the first scene
    if resuming:
        print(f"[generate] resuming: {len(done)}/{n_scenes} scenes already done")

    todo = [slot for slot in range(n_scenes) if slot not in done]
    done_n = len(done)

    def _record(res: Dict) -> None:
        nonlocal done_n
        merge_accumulators(acc, res["acc"])
        tracker.mark_done(res["slot"], acc)
        done_n += 1
        if progress or done_n % 10 == 0 or done_n == n_scenes:
            print(f"[generate] {done_n}/{n_scenes} scenes done "
                  f"({done_n * per} episodes)")

    if workers <= 1:
        for slot in todo:
            _record(_generate_scene((out_dir, slot, cfg, per, cfg.seed)))
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        payloads = [(out_dir, slot, cfg, per, cfg.seed) for slot in todo]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_generate_scene, p) for p in payloads]
            for fut in as_completed(futures):
                _record(fut.result())

    stats = finalize_accumulator(acc)
    stats = attach_targets(stats, cfg, total)
    stats["num_scenes"] = n_scenes
    stats["episodes_per_scene"] = per
    write_statistics(out_dir, stats)
    tracker.mark_finished()
    print(format_statistics(stats, "STYLE LIST GENERATION — STATISTICS"))

    if make_plots:
        try:
            from .plots_mpl import write_plots
            plots_dir = os.path.join(out_dir, "plots")
            write_plots(stats, plots_dir, title=f"RefON statistics — {out_dir}")
            print(f"[generate] plots saved to: {plots_dir}")
        except ImportError:
            print("[generate] (matplotlib not available — skipped plots; "
                  "run 'plot' separately)")

    print(f"\n[generate] saved to: {out_dir}")
    return stats
