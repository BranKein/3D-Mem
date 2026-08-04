"""RefONEpisodeGenerator CLI.

Two commands:
  generate : cycle-generate instruction style lists -> save to a folder (with statistics)
  build    : a generated style-list folder -> real HM3D episode json (shards)

Auxiliary commands:
  validate : re-check every list in a style-list folder against the automaton grammar
  plot     : save statistics pie/bar charts as PNG (matplotlib)
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional

from .config import BuilderConfig, GeneratorConfig
from .statistics import compute_statistics, format_statistics
from .storage import load_style_lists
from .validate import validate


# ---------------------------------------------------------------------- #
# generate (command 1)
# ---------------------------------------------------------------------- #
def cmd_generate(args: argparse.Namespace) -> int:
    from .runner import run_generate
    from .storage import load_config_from_dir

    # config resolution: --config wins; else <out>/config.json (put-config-in-folder
    # flow); else defaults.
    if args.config:
        with open(args.config) as f:
            cfg = GeneratorConfig.from_dict(json.load(f))
    else:
        cfg = load_config_from_dir(args.out)
        if cfg is not None:
            print(f"[generate] using config from {os.path.join(args.out, 'config.json')}")
        else:
            cfg = GeneratorConfig()
    if args.num_scenes is not None:
        cfg.num_scenes = args.num_scenes
    if args.episodes_per_scene is not None:
        cfg.episodes_per_scene = args.episodes_per_scene
    if args.goal_absent_ratio is not None:
        cfg.goal_absent_ratio = args.goal_absent_ratio
    if args.seed is not None:
        cfg.seed = args.seed

    print(
        f"[generate] scenes={cfg.num_scenes} episodes/scene={cfg.episodes_per_scene} "
        f"(= {cfg.num_scenes * cfg.episodes_per_scene} episodes), "
        f"goal_absent={cfg.goal_absent_ratio}"
    )
    run_generate(cfg, args.out, progress=args.progress,
                 make_plots=not args.no_plot, workers=args.workers)
    return 0


# ---------------------------------------------------------------------- #
# build (command 2)
# ---------------------------------------------------------------------- #
def _load_scene_ids(args: argparse.Namespace) -> List[str]:
    if args.scenes:
        return [s.strip() for s in args.scenes.split(",") if s.strip()]
    if args.scenes_file:
        with open(args.scenes_file) as f:
            text = f.read()
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [str(x) for x in data]
            if isinstance(data, dict):  # {slot: scene_id} or {"scenes":[...]}
                if "scenes" in data:
                    return [str(x) for x in data["scenes"]]
                return [str(data[k]) for k in sorted(data, key=lambda x: int(x))]
        except json.JSONDecodeError:
            return [ln.strip() for ln in text.splitlines() if ln.strip()]
    raise SystemExit("build: provide scene ids via --scenes or --scenes-file")


def cmd_build(args: argparse.Namespace) -> int:
    from .episode_builder import build_dataset  # lazy import (depends on habitat)
    from .storage import list_scene_slots

    n_slots = len(list_scene_slots(args.input))
    scene_ids = _load_scene_ids(args)

    bcfg = BuilderConfig()
    if args.builder_config:
        with open(args.builder_config) as f:
            bcfg = BuilderConfig.from_dict(json.load(f))
    if args.split is not None:
        bcfg.split = args.split
    if args.dataset_root is not None:
        bcfg.dataset_root = args.dataset_root
    if args.max_objects is not None:
        bcfg.max_objects = args.max_objects
    if args.hm3d_root is not None:
        bcfg.hm3d_root = args.hm3d_root
    if args.no_compress:
        bcfg.compress = False
    if args.seed is not None:
        bcfg.seed = args.seed

    if bcfg.hm3d_root:
        from .scene_paths import default_scene_dataset_config

        print(f"[build] hm3d_root = {bcfg.hm3d_root}")
        cfg_path = bcfg.scene_dataset_config or default_scene_dataset_config(bcfg.hm3d_root)
        if cfg_path is None:
            print(f"[build] WARNING no hm3d_annotated_basis.scene_dataset_config.json found "
                  f"in or next to {bcfg.hm3d_root}; falling back to the loader default")
        else:
            print(f"[build] scene_dataset_config = {cfg_path}")

    print(f"[build] {n_slots} scene slots, {len(scene_ids)} scene ids provided")
    written = build_dataset(args.input, scene_ids, bcfg, progress=True, workers=args.workers)
    print(f"\n[build] wrote {len(written)} shards under "
          f"{os.path.join(bcfg.dataset_root, bcfg.split, 'content')}")
    return 0


# ---------------------------------------------------------------------- #
# scenes (auxiliary) — list the scenes available in an HM3D root
# ---------------------------------------------------------------------- #
def cmd_scenes(args: argparse.Namespace) -> int:
    """List scenes under 3D-Mem's HM3D directory, for feeding to `build --scenes-file`.

    Only inspects directory entries -- no scene is opened.
    """
    from .scene_paths import default_scene_dataset_config, list_scenes

    names = list_scenes(args.hm3d_root, args.split)
    if not names:
        print(f"[scenes] no annotated scenes found under {args.hm3d_root}"
              + (f"/{args.split}" if args.split else "")
              + "\n         expected <root>/<split>/<index>-<hash>/<hash>.basis.glb")
        return 1
    if args.limit:
        names = names[: args.limit]

    cfg_path = default_scene_dataset_config(args.hm3d_root)
    print(f"[scenes] {len(names)} scene(s) under {args.hm3d_root}"
          + (f"/{args.split}" if args.split else ""))
    print(f"[scenes] scene_dataset_config: {cfg_path or '(not found)'}")

    if args.out:
        with open(args.out, "w") as f:
            f.write(f"# scenes from {args.hm3d_root}"
                    + (f" split={args.split}" if args.split else "") + "\n")
            for n in names:
                f.write(n + "\n")
        print(f"[scenes] wrote {args.out}")
    else:
        for n in names:
            print(f"  {n}")
    return 0


# ---------------------------------------------------------------------- #
# validate (auxiliary)
# ---------------------------------------------------------------------- #
def cmd_validate(args: argparse.Namespace) -> int:
    groups, _cfg, _stats = load_style_lists(args.input)
    all_lists = [sl for g in groups for sl in g]
    bad = 0
    for i, sl in enumerate(all_lists):
        ok, errs = validate(sl)
        if not ok:
            bad += 1
            print(f"  INVALID #{i}: {sl.role_sequence()} -> {errs}")
    stats = compute_statistics(all_lists)
    print(format_statistics(stats, "VALIDATION — RECOMPUTED STATISTICS"))
    print(f"\n[validate] {len(all_lists)} lists, {bad} invalid")
    return 1 if bad else 0


# ---------------------------------------------------------------------- #
# sample (auxiliary) — print one random episode per length to the console
# ---------------------------------------------------------------------- #
def cmd_sample(args: argparse.Namespace) -> int:
    import random as _random

    from .preview import print_samples
    rng = _random.Random(args.seed) if args.seed is not None else _random.Random()
    return print_samples(args.input, rng, color=not args.no_color, max_scenes=args.max_scenes)


# ---------------------------------------------------------------------- #
# plot (auxiliary) — save statistics pie/bar charts as PNG (matplotlib)
# ---------------------------------------------------------------------- #
def cmd_plot(args: argparse.Namespace) -> int:
    import json as _json

    try:
        from .plots_mpl import write_plots
    except ImportError as exc:
        print(f"[plot] could not import matplotlib: {exc}\n"
              f"       install it with 'pip install matplotlib' and retry.")
        return 1

    stats: dict = {}
    stats_path = os.path.join(args.input, "statistics.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = _json.load(f)
    # if statistics.json is missing or recomputation is requested, recompute from the style
    # lists present so far (works mid-generation on a partial folder too)
    if not stats or args.recompute:
        groups, _cfg, _ = load_style_lists(args.input)
        all_lists = [sl for g in groups for sl in g]
        stats = compute_statistics(all_lists)
        stats.setdefault("num_scenes", len(groups))

    # attach target distributions from the folder's config.json so the target overlays
    # show even when statistics.json isn't written yet (partial / recompute)
    from .storage import load_config_from_dir
    gcfg = load_config_from_dir(args.input)
    if gcfg is not None and stats.get("num_lists"):
        from .balancer import attach_targets
        attach_targets(stats, gcfg, stats["num_lists"])

    out_dir = args.out or os.path.join(args.input, "plots")
    written = write_plots(stats, out_dir, title=f"RefON statistics — {args.input}")
    print(f"[plot] wrote {len(written)} PNG files to {out_dir}/")
    for w in written:
        print(f"  {os.path.basename(w)}")
    print(f"\ndashboard: {os.path.join(out_dir, 'dashboard.png')}")
    return 0


# ---------------------------------------------------------------------- #
# parser
# ---------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="refon",
        description="HM3D Referential Object-goal Navigation episode generator",
    )
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="generate instruction style lists (command 1)")
    g.add_argument("-o", "--out", required=True, help="output folder")
    g.add_argument("--config", help="GeneratorConfig JSON file")
    g.add_argument("--num-scenes", type=int, help="number of scene slots")
    g.add_argument("--episodes-per-scene", type=int)
    g.add_argument("--goal-absent-ratio", type=float)
    g.add_argument("--seed", type=int)
    g.add_argument("--progress", action="store_true", help="print balancing progress logs")
    g.add_argument("--no-plot", action="store_true",
                   help="disable the automatic plot after generation (default: auto-plot)")
    g.add_argument("--workers", type=int, default=1,
                   help="parallel process count (per-scene multiprocessing; default 1=sequential)")
    g.set_defaults(func=cmd_generate)

    b = sub.add_parser("build", help="style list -> HM3D episode json (command 2)")
    b.add_argument("-i", "--input", required=True, help="style-list folder made by generate")
    b.add_argument("--scenes", help="comma-separated scene_ids (in slot order)")
    b.add_argument("--scenes-file", help="scene_id list file (txt one-per-line / json list / json dict)")
    b.add_argument("--builder-config", help="BuilderConfig JSON file")
    b.add_argument("--split", help="dataset split (default train)")
    b.add_argument("--dataset-root", help="output dataset root")
    b.add_argument("--max-objects", type=int, help="distinct-object budget per episode")
    b.add_argument("--hm3d-root",
                   help="HM3D dir holding train/ val/ (3D-Mem's scene_data_path, e.g. ../data/hm3d). "
                        "Lets --scenes/--scenes-file name scenes by bare hash.")
    b.add_argument("--no-compress", action="store_true",
                   help="write plain .json shards instead of .json.gz")
    b.add_argument("--seed", type=int)
    b.add_argument("--workers", type=int, default=1,
                   help="parallel process count (per-scene multiprocessing; default 1=sequential)")
    b.set_defaults(func=cmd_build)

    sc = sub.add_parser("scenes", help="list scenes in an HM3D root (for --scenes-file)")
    sc.add_argument("--hm3d-root", required=True,
                    help="HM3D dir holding train/ val/ (3D-Mem's scene_data_path)")
    sc.add_argument("--split", help="restrict to one split (val / train / ...)")
    sc.add_argument("-o", "--out", help="write the list to this file instead of stdout")
    sc.add_argument("--limit", type=int, help="keep only the first N scenes")
    sc.set_defaults(func=cmd_scenes)

    v = sub.add_parser("validate", help="re-validate a style-list folder")
    v.add_argument("-i", "--input", required=True)
    v.set_defaults(func=cmd_validate)

    sp = sub.add_parser("sample", help="print one random episode per length (shows reference structure)")
    sp.add_argument("-i", "--input", required=True, help="folder made by generate")
    sp.add_argument("--seed", type=int, help="random seed (for reproducibility; omit for a fresh pick each time)")
    sp.add_argument("--max-scenes", type=int, default=12,
                    help="number of scene files to sample from (default 12)")
    sp.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    sp.set_defaults(func=cmd_sample)

    pl = sub.add_parser("plot", help="save statistics pie/bar charts as PNG (matplotlib)")
    pl.add_argument("-i", "--input", required=True, help="folder made by generate")
    pl.add_argument("-o", "--out", help="chart output folder (default: <input>/plots)")
    pl.add_argument("--recompute", action="store_true",
                    help="recompute statistics from the style lists instead of statistics.json")
    pl.set_defaults(func=cmd_plot)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
