"""command 2 — style list -> real HM3D episode json (GOAT-compatible referential shard).

Maps the reference structure of a style list (ref_order/ordinal_k/alias) onto the
scene's real objects. The reference resolution semantics live in each
InstructionStyle.resolve(), so here we only implement the BuildContext that provides
scene resources and resolve/render tokens in order.

Output schema (chosen by the user: array of subtask dicts):
  shard = {
    "episodes": [ {episode_id, scene_id, start_position, start_rotation,
                   info, goal_absent, subtasks:[ {order, role, category,
                   object_id, instruction, alias?, ref_order?, ordinal_k?,
                   goal_absent?, reason?}, ... ]}, ... ],
    "goals": { "<scene_key>_<category>": [ <goal_instance dict>, ... ] }  # only objects actually used
  }

HM3DSceneLoader depends on habitat-sim, so it is imported lazily.
"""
from __future__ import annotations

import gzip
import json
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

from .config import BuilderConfig
from .scene_paths import default_scene_dataset_config, resolve_scene, scene_hash, to_goat_scene_id
from .style_token import StyleList
from .styles import STYLE_BY_NAME
from .styles.base import Resolved


class NoUsableObject(RuntimeError):
    pass


def _import_loader():
    """Import hm3d_scene_loader (with a friendly error if missing)."""
    candidates = [
        os.environ.get("HM3D_SCENE_LOADER_PATH"),
        os.path.expanduser("~/PycharmProjects/hm3d-scene-loader"),
    ]
    for p in candidates:
        if p and os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    try:
        from hm3d_scene_loader import HM3DSceneLoader, LoaderConfig  # type: ignore
        return HM3DSceneLoader, LoaderConfig
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Could not import hm3d_scene_loader. "
            "Set the HM3D_SCENE_LOADER_PATH environment variable to its path, "
            "or run inside a habitat-sim environment. Cause: %s" % exc
        )


class _EpisodeContext:
    """BuildContext implementation used while building one episode."""

    def __init__(self, builder: "SceneEpisodeBuilder", rng: random.Random):
        self._b = builder
        self.rng = rng
        self.visit: List[Tuple[Optional[str], Optional[str]]] = []  # order-1 -> (object_id, category)
        self.aliases: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
        self.used_ids: set = set()

    # --- BuildContext protocol ---
    def choose_new_object(self) -> Tuple[str, str]:
        b = self._b
        distinct = len(self.used_ids)
        pool = [
            o for o in b.objects
            if o.object_id not in self.used_ids and o.object_id not in b.failed_ids
        ]
        # over budget or no new candidate -> reuse an already-visited object
        if (distinct >= b.cfg.max_objects or not pool) and self._reusable():
            return self.rng.choice(self._reusable())

        self.rng.shuffle(pool)
        for o in pool:
            if b.ensure_goal_instance(o) is not None:  # only objects that have viewpoints
                self.used_ids.add(o.object_id)
                b.used_object_ids.add(o.object_id)
                return o.object_id, o.category

        if self._reusable():
            return self.rng.choice(self._reusable())
        raise NoUsableObject("no goal object with viewpoints exists in this scene")

    def object_at(self, order: int) -> Tuple[Optional[str], Optional[str]]:
        return self.visit[order - 1]

    def bind_alias(self, alias: str, object_id: Optional[str], category: Optional[str]) -> None:
        self.aliases[alias] = (object_id, category)

    def sample_absent_category(self) -> str:
        b = self._b
        pool = [c for c in b.cfg.absent_category_pool if c not in b.present_categories]
        if not pool:
            pool = b.cfg.absent_category_pool or ["dragon"]
        return self.rng.choice(pool)

    # --- internal ---
    def _reusable(self) -> List[Tuple[str, str]]:
        return [(oid, cat) for (oid, cat) in self.visit if oid is not None]


class SceneEpisodeBuilder:
    """Shard builder for a single scene."""

    def __init__(self, loader, cfg: BuilderConfig, rng: random.Random):
        self.loader = loader
        self.cfg = cfg
        self.rng = rng
        self.scene_key = loader.scene_key
        self.objects = loader.get_semantic_objects(cfg.category_whitelist)
        self.present_categories = {o.category for o in self.objects}
        self._by_id = {o.object_id: o for o in self.objects}
        self._goal_cache: Dict[str, object] = {}   # object_id -> GoalInstance
        self.failed_ids: set = set()
        self.used_object_ids: set = set()

    def ensure_goal_instance(self, obj):
        """Build/cache obj's GoalInstance (with viewpoints). None if it has no viewpoints."""
        oid = obj.object_id
        if oid in self._goal_cache:
            return self._goal_cache[oid]
        if oid in self.failed_ids:
            return None
        gi = self.loader.build_goal_instance(
            obj,
            with_view_points=True,
            with_image_goals=self.cfg.with_image_goals,
        )
        if not gi.view_points:
            self.failed_ids.add(oid)
            return None
        self._goal_cache[oid] = gi
        return gi

    def _viewpoint_positions(self, object_id: Optional[str]) -> List[List[float]]:
        if object_id is None:
            return []
        gi = self._goal_cache.get(object_id)
        if gi is None:
            return []
        return [vp.agent_state.position for vp in gi.view_points]

    def build_episode(
        self, style_list: StyleList, episode_id: str, scene_id: str
    ) -> Optional[Dict]:
        ctx = _EpisodeContext(self, self.rng)
        subtasks: List[Dict] = []
        try:
            for tok in style_list.tokens:
                style = STYLE_BY_NAME[tok.role]
                r: Resolved = style.resolve_and_render(tok, ctx)
                ctx.visit.append((r.object_id, r.category))
                subtasks.append(self._subtask_dict(r))
        except NoUsableObject:
            return None

        # start state: based on the first subtask object's viewpoints
        first_oid = ctx.visit[0][0] if ctx.visit else None
        vps = self._viewpoint_positions(first_oid)
        if vps:
            start = self.loader.sample_start_state(vps)
            if start is None:
                return None
            pos, rot, geo = start
        else:
            # no viewpoints (e.g. a length-1 goal-absent-only episode) -> random navigable start
            p = self.loader.sample_navigable_point()
            yaw = self.rng.uniform(0, 6.283185)
            import math
            pos = [float(x) for x in p]
            rot = [0.0, float(math.sin(yaw / 2)), 0.0, float(math.cos(yaw / 2))]
            geo = None

        episode = {
            "episode_id": episode_id,
            "scene_id": scene_id,
            "scene_dataset_config": self.loader.config.scene_dataset_config,
            "start_position": pos,
            "start_rotation": rot,
            "info": {"geodesic_distance": geo} if geo is not None else {},
            "goal_absent": style_list.goal_absent,
            "subtasks": subtasks,
        }
        return episode

    def _subtask_dict(self, r: Resolved) -> Dict:
        d: Dict = {
            "order": r.order,
            "role": r.role,
            "category": r.category,
            "object_id": r.object_id,
            "instruction": r.instruction,
        }
        if r.alias is not None:
            d["alias"] = r.alias
        if r.ref_order is not None:
            d["ref_order"] = r.ref_order
        if r.ordinal_k is not None:
            d["ordinal_k"] = r.ordinal_k
        if r.goal_absent:
            d["goal_absent"] = True
            d["expected_behavior"] = "stop"
            if r.reason:
                d["reason"] = r.reason
        return d

    def build_goals_catalog(self) -> Dict[str, List[Dict]]:
        """Assemble the goals dict from only the objects actually used."""
        catalog: Dict[str, List[Dict]] = {}
        seen: set = set()
        for oid in self.used_object_ids:
            gi = self._goal_cache.get(oid)
            if gi is None:
                continue
            key = f"{self.scene_key}_{gi.object_category}"
            if oid in seen:
                continue
            seen.add(oid)
            catalog.setdefault(key, []).append(
                gi.to_dict(include_lang=False, include_image=self.cfg.with_image_goals)
            )
        return catalog

    def build_shard(self, style_lists: List[StyleList], scene_id: str) -> Dict:
        episodes: List[Dict] = []
        for i, sl in enumerate(style_lists):
            ep = self.build_episode(sl, episode_id=str(i), scene_id=scene_id)
            if ep is not None:
                episodes.append(ep)
        goals = self.build_goals_catalog()
        return {"episodes": episodes, "goals": goals}


def _make_loader_cfg(LoaderConfig, cfg: BuilderConfig, loader_overrides: Optional[Dict]):
    loader_cfg = LoaderConfig()
    if cfg.category_whitelist is not None:
        loader_cfg.category_whitelist = cfg.category_whitelist
    if cfg.scene_root is not None:
        loader_cfg.scene_root = cfg.scene_root
    if cfg.scene_dataset_config is not None:
        loader_cfg.scene_dataset_config = cfg.scene_dataset_config
    elif cfg.hm3d_root:
        # keep the loader on the same scene_dataset_config the evaluator uses
        auto = default_scene_dataset_config(cfg.hm3d_root)
        if auto is not None:
            loader_cfg.scene_dataset_config = auto
    if cfg.viewpoint_min_iou is not None:
        loader_cfg.viewpoint.min_iou = cfg.viewpoint_min_iou
    if cfg.viewpoint_num_candidates is not None:
        loader_cfg.viewpoint.num_candidates = cfg.viewpoint_num_candidates
    if cfg.viewpoint_max_view_points is not None:
        loader_cfg.viewpoint.max_view_points = cfg.viewpoint_max_view_points
    if cfg.min_geodesic is not None:
        loader_cfg.start.min_geodesic = cfg.min_geodesic
    if cfg.max_geodesic is not None:
        loader_cfg.start.max_geodesic = cfg.max_geodesic
    if loader_overrides:
        for k, v in loader_overrides.items():
            setattr(loader_cfg, k, v)
    return loader_cfg


def _shard_out_path(cfg: BuilderConfig, scene_id: str) -> str:
    # The stem must be the bare scene hash: 3D-Mem identifies a shard's scene by its
    # file name, and the goal-catalog keys are prefixed with the same hash.
    out_root = os.path.join(cfg.dataset_root, cfg.split, "content")
    ext = "json.gz" if cfg.compress else "json"
    return os.path.join(out_root, f"{scene_hash(scene_id)}.{ext}")


def build_one_scene(
    in_dir: str, slot: int, scene_id: str,
    cfg: BuilderConfig, loader_overrides: Optional[Dict] = None,
) -> Dict:
    """Build (and atomically write) the shard for one scene slot.

    Top-level (picklable) so it can run in a worker process. Skips if the shard file
    already exists. Loads only this scene's style lists. Returns a summary dict.
    """
    from .storage import load_scene_file

    out = _shard_out_path(cfg, scene_id)
    if os.path.exists(out):
        return {"slot": slot, "scene_id": scene_id, "out": out, "skipped": True}

    os.makedirs(os.path.dirname(out), exist_ok=True)
    HM3DSceneLoader, LoaderConfig = _import_loader()
    lists = load_scene_file(in_dir, slot)
    loader_cfg = _make_loader_cfg(LoaderConfig, cfg, loader_overrides)

    # Resolve to the actual mesh so the generator and the evaluator agree on the scene
    # even when the scene list only names a hash. The episode records the GOAT-style
    # relative id, which stays portable across machines.
    glb_path = resolve_scene(scene_id, cfg.hm3d_root)
    episode_scene_id = to_goat_scene_id(glb_path)

    with HM3DSceneLoader(glb_path, loader_cfg) as loader:
        builder = SceneEpisodeBuilder(loader, cfg, random.Random(cfg.seed + slot))
        shard = builder.build_shard(lists, episode_scene_id)
        tmp = out + ".tmp"
        opener = gzip.open if cfg.compress else open
        with opener(tmp, "wt") as f:
            json.dump(shard, f, indent=2)
        os.replace(tmp, out)
        n_ep, n_goals = len(shard["episodes"]), len(shard["goals"])
    return {"slot": slot, "scene_id": scene_id, "out": out,
            "n_ep": n_ep, "n_goals": n_goals, "empty": n_ep == 0}


def _worker(payload):
    """ProcessPool worker: unpack and build one scene. Errors are returned, not raised,
    so one bad scene doesn't kill the pool."""
    in_dir, slot, scene_id, cfg, loader_overrides = payload
    try:
        return build_one_scene(in_dir, slot, scene_id, cfg, loader_overrides)
    except Exception as exc:  # pragma: no cover
        return {"slot": slot, "scene_id": scene_id, "error": repr(exc)}


def build_dataset(
    in_dir: str,
    scene_ids: List[str],
    cfg: BuilderConfig,
    loader_overrides: Optional[Dict] = None,
    progress: bool = True,
    workers: int = 1,
) -> Dict[str, str]:
    """Map each scene slot onto a scene_id and build/save the shards.

    Resumable: each shard is written atomically (temp + rename) right after its scene is
    built; a shard file's existence means "done", and a progress file
    (_build_progress.json) records completed slots. Re-running skips finished scenes.
    Only one scene's loader/lists are in memory per worker.

    workers > 1 builds scenes in parallel with a process pool (scenes are independent;
    each worker runs its own habitat-sim). The main process owns the progress file, so
    there is no concurrent-write race.

    Return: {scene_id: saved shard path}
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from .progress import ProgressTracker
    from .storage import list_scene_slots

    slots = list_scene_slots(in_dir)
    if len(scene_ids) < len(slots):
        raise ValueError(
            f"not enough scene_ids: {len(slots)} scene slots, {len(scene_ids)} scene_ids"
        )

    # Resolve every scene reference before any work starts: a full build takes hours, and
    # a typo in the scene list should not surface once most of it is already done. This
    # only stats directories -- no scene is opened.
    unresolved = []
    for slot in range(len(slots)):
        try:
            resolve_scene(scene_ids[slot], cfg.hm3d_root)
        except (FileNotFoundError, ValueError) as exc:
            unresolved.append(f"  slot {slot}: {exc}")
    if unresolved:
        raise FileNotFoundError(
            f"{len(unresolved)} of {len(slots)} scene(s) could not be resolved:\n"
            + "\n".join(unresolved)
        )

    out_root = os.path.join(cfg.dataset_root, cfg.split, "content")
    os.makedirs(out_root, exist_ok=True)
    tracker, resuming = ProgressTracker.load_or_create(
        os.path.join(out_root, "_build_progress.json"),
        command="build", meta={"split": cfg.split, "workers": workers},
    )
    tracker.save()  # write the progress file right away, before the first scene
    if resuming:
        print(f"[build] resuming: {len(tracker.completed)}/{len(slots)} scenes already done")

    todo = [s for s in slots
            if not tracker.is_done(s) and not os.path.exists(_shard_out_path(cfg, scene_ids[s]))]
    print(f"[build] {len(todo)} scenes to build (workers={workers})")
    written: Dict[str, str] = {}
    empty_scenes: List[str] = list(tracker.data.get("empty_scenes", []))
    done_n = len(slots) - len(todo)

    def _record(res: Dict) -> None:
        nonlocal done_n
        done_n += 1
        if res.get("error"):
            print(f"[build] scene slot {res['slot']} ERROR: {res['error']}")
            return
        if res.get("empty"):
            # broken/degenerate scene: no viewable objects -> 0 episodes. Flag it (do
            # not silently pretend it's a good shard) but still mark done so resume
            # doesn't loop on it. The user can replace these with spare scenes.
            if res["scene_id"] not in empty_scenes:
                empty_scenes.append(res["scene_id"])
            tracker.data["empty_scenes"] = empty_scenes
            print(f"[build] WARNING slot {res['slot']} -> {os.path.basename(res['out'])}: "
                  f"0 episodes (no viewable objects; scene likely has broken semantics)")
        tracker.mark_done(res["slot"])
        if not res.get("skipped"):
            written[res["scene_id"]] = res["out"]
        if progress and not res.get("empty"):
            tag = "skipped" if res.get("skipped") else f"{res.get('n_ep')} episodes, {res.get('n_goals')} goal-keys"
            print(f"[build] {done_n}/{len(slots)} slot {res['slot']} -> {os.path.basename(res['out'])}: {tag}")

    if workers <= 1:
        for slot in todo:
            _record(build_one_scene(in_dir, slot, scene_ids[slot], cfg, loader_overrides))
    else:
        payloads = [(in_dir, slot, scene_ids[slot], cfg, loader_overrides) for slot in todo]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_worker, p) for p in payloads]
            for fut in as_completed(futures):
                _record(fut.result())

    tracker.mark_finished()
    if empty_scenes:
        print(f"\n[build] {len(empty_scenes)} EMPTY scene(s) (0 episodes, broken semantics) — "
              f"consider replacing with spare annotated scenes:")
        for s in empty_scenes:
            print(f"          {s}")
    return written
