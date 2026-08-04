"""Saving/loading the command 1 output folder.

Folder layout:
  <out_dir>/
    config.json            # the GeneratorConfig used for generation
    statistics.json        # statistics (machine-readable)
    statistics.txt         # statistics (human-readable)
    scenes/
      scene_000.json       # {"scene_slot":0, "style_lists":[ {episode_slot, ...StyleList}, ...]}
      scene_001.json
      ...

Command 2 takes this folder and maps each scene slot's style_lists onto a real scene.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

from .config import GeneratorConfig
from .statistics import format_statistics
from .style_token import StyleList

PROGRESS_FILE = "_progress.json"


def scene_file_path(out_dir: str, slot: int) -> str:
    return os.path.join(out_dir, "scenes", f"scene_{slot:03d}.json")


def write_scene_file(out_dir: str, slot: int, lists: List[StyleList]) -> str:
    """Write one scene's style lists immediately (atomic)."""
    scenes_dir = os.path.join(out_dir, "scenes")
    os.makedirs(scenes_dir, exist_ok=True)
    payload = {
        "scene_slot": slot,
        "num_episodes": len(lists),
        "style_lists": [{"episode_slot": i, **sl.to_dict()} for i, sl in enumerate(lists)],
    }
    path = scene_file_path(out_dir, slot)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)
    return path


def write_config(out_dir: str, cfg: GeneratorConfig) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)


def load_config_from_dir(out_dir: str) -> Optional[GeneratorConfig]:
    """Load <out_dir>/config.json if present (the 'put config in the folder' flow)."""
    path = os.path.join(out_dir, "config.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return GeneratorConfig.from_dict(json.load(f))


def write_statistics(out_dir: str, stats: Dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "statistics.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(out_dir, "statistics.txt"), "w") as f:
        f.write(format_statistics(stats, "STYLE LIST GENERATION — STATISTICS"))
        f.write("\n")


def save_style_lists(
    out_dir: str,
    groups: List[List[StyleList]],
    cfg: GeneratorConfig,
    stats: Dict,
) -> None:
    scenes_dir = os.path.join(out_dir, "scenes")
    os.makedirs(scenes_dir, exist_ok=True)

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    with open(os.path.join(out_dir, "statistics.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(out_dir, "statistics.txt"), "w") as f:
        f.write(format_statistics(stats, "STYLE LIST GENERATION — STATISTICS"))
        f.write("\n")

    for slot, group in enumerate(groups):
        payload = {
            "scene_slot": slot,
            "num_episodes": len(group),
            "style_lists": [
                {"episode_slot": i, **sl.to_dict()} for i, sl in enumerate(group)
            ],
        }
        path = os.path.join(scenes_dir, f"scene_{slot:03d}.json")
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)


def list_scene_slots(in_dir: str) -> List[int]:
    """Sorted list of scene slot indices present under <in_dir>/scenes/."""
    scenes_dir = os.path.join(in_dir, "scenes")
    if not os.path.isdir(scenes_dir):
        raise FileNotFoundError(f"no scenes/ under {in_dir}")
    slots = []
    for fn in os.listdir(scenes_dir):
        if fn.startswith("scene_") and fn.endswith(".json"):
            slots.append(int(fn[len("scene_"):-len(".json")]))
    return sorted(slots)


def load_scene_file(in_dir: str, slot: int) -> List[StyleList]:
    """Load a single scene's style lists (for per-scene / parallel building)."""
    with open(scene_file_path(in_dir, slot)) as f:
        payload = json.load(f)
    return [StyleList.from_dict(d) for d in payload["style_lists"]]


def load_style_lists(in_dir: str) -> Tuple[List[List[StyleList]], Dict, Dict]:
    """Return (groups, config_dict, stats)."""
    scenes_dir = os.path.join(in_dir, "scenes")
    if not os.path.isdir(scenes_dir):
        raise FileNotFoundError(f"no scenes/ under {in_dir}")

    cfg_dict: Dict = {}
    cfg_path = os.path.join(in_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg_dict = json.load(f)
    stats: Dict = {}
    stats_path = os.path.join(in_dir, "statistics.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)

    files = sorted(
        fn for fn in os.listdir(scenes_dir)
        if fn.startswith("scene_") and fn.endswith(".json")
    )
    groups: List[List[StyleList]] = []
    for fn in files:
        with open(os.path.join(scenes_dir, fn)) as f:
            payload = json.load(f)
        group = [StyleList.from_dict(d) for d in payload["style_lists"]]
        groups.append(group)
    return groups, cfg_dict, stats
