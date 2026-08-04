"""Shard loading and subtask preparation for the RefON referential-navigation benchmark.

RefONEpisodeGenerator writes GOAT-Bench-shaped shards, with two deliberate differences:

  * shards are gzipped (``<scene>.json.gz``) rather than plain json, and
  * an episode carries an ordered ``subtasks`` list of dicts (order / role / category /
    object_id / instruction / alias / ref_order / ordinal_k) instead of GOAT's
    ``tasks`` triples ``[category, goal_type, object_id]``.

The ``goals`` catalog is already GOAT-compatible, so the only thing needed here is to
turn ``subtasks`` into the per-subtask navigation goals the evaluation loop consumes.
Unlike GOAT's "object" goal type -- which accepts *any* instance of the category -- a
referential subtask always names one specific instance, so every subtask resolves to
exactly one goal instance.
"""

import gzip
import json
import logging
import os
from typing import Dict, List, Optional

# files that live in the shard directory but are not shards
_NON_SHARD_NAMES = {"_build_progress.json"}


def list_shard_files(test_data_dir: str) -> List[str]:
    """Shard file names in `test_data_dir`, sorted, with generator bookkeeping removed."""
    files = []
    for name in os.listdir(test_data_dir):
        if name.startswith(".") or name in _NON_SHARD_NAMES:
            continue
        if not (name.endswith(".json") or name.endswith(".json.gz")):
            continue
        files.append(name)
    return sorted(files)


def load_shard(path: str) -> Dict:
    """Load a shard, transparently gunzipping ``.json.gz``."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def scene_name_from_shard(shard_file_name: str) -> str:
    """``yX5efd48dLf.json.gz`` / ``yX5efd48dLf.json`` -> ``yX5efd48dLf``."""
    return os.path.basename(shard_file_name).split(".")[0]


def _index_goals_by_object_id(all_navigation_goals: Dict) -> Dict[str, Dict]:
    """object_id -> goal instance, across every category key of the goals catalog."""
    index = {}
    for goal_instances in all_navigation_goals.values():
        for goal in goal_instances:
            index[goal["object_id"]] = goal
    return index


def prepare_refon_navigation_subtasks(
    episode: Dict,
    all_navigation_goals: Dict,
    skip_goal_absent: bool = True,
) -> List[Dict]:
    """Resolve an episode's ``subtasks`` against the shard's ``goals`` catalog.

    Returns one entry per runnable subtask::

        {"order", "role", "instruction", "category", "goal": [goal_instance]}

    ``goal`` is a single-element list to match the shape the GOAT-derived logger and
    evaluation loop expect.

    Goal-absent subtasks (``GA_*``: an unbound alias, an out-of-range ordinal, or a
    category that is not in the scene) name no object at all. 3D-Mem has no notion of
    "the right answer is to stop", so by default they are dropped rather than scored as
    a guaranteed failure. Pass ``skip_goal_absent=False`` to keep them out of the run
    but visible in the returned list -- they are still skipped, only reported.
    """
    goals_by_object_id = _index_goals_by_object_id(all_navigation_goals)

    resolved: List[Dict] = []
    for subtask in episode["subtasks"]:
        object_id = subtask.get("object_id")
        if subtask.get("goal_absent") or object_id is None:
            if not skip_goal_absent:
                logging.info(
                    f"Subtask order {subtask.get('order')} role {subtask.get('role')} is "
                    f"goal-absent and cannot be scored by this evaluator; skipping"
                )
            continue

        goal = goals_by_object_id.get(object_id)
        if goal is None:
            logging.warning(
                f"Subtask order {subtask.get('order')} references object_id "
                f"'{object_id}', which is missing from the shard's goals catalog; skipping"
            )
            continue
        if not goal.get("view_points"):
            logging.warning(
                f"Goal '{object_id}' has no view points, so it has no reachable target; "
                f"skipping subtask order {subtask.get('order')}"
            )
            continue

        resolved.append(
            {
                "order": subtask.get("order"),
                "role": subtask.get("role"),
                "instruction": subtask.get("instruction"),
                "category": goal["object_category"],
                "goal": [goal],
            }
        )
    return resolved


def select_episodes(
    episodes: List[Dict],
    episodes_per_scene: Optional[int],
    split: int,
) -> List[Dict]:
    """Pick which of a shard's episodes to run.

    RefON shards can hold thousands of episodes per scene (GOAT's val_unseen holds ten),
    so the shard is first truncated to the first ``episodes_per_scene`` episodes -- a
    stable prefix, so the same subset is evaluated on every re-run and by every split.

    ``split`` then keeps GOAT's parallelisation contract: ``--split k`` runs the k-th
    episode (1-indexed) of that subset, so k can be fanned out across processes.
    ``--split 0`` runs the whole subset in one process.
    """
    if episodes_per_scene is not None and episodes_per_scene > 0:
        episodes = episodes[:episodes_per_scene]
    if split <= 0:
        return episodes
    return episodes[split - 1 : split]
