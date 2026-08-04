"""Pipeline configuration.

GeneratorConfig : policy for command 1 (style list generation/balancing)
BuilderConfig   : policy for command 2 (building real episode json)

Configs can be passed as JSON files (CLI --config). length_ratios and the
scene/episode counts are the key inputs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional


@dataclass
class GeneratorConfig:
    # Target ratio for each instruction-style-list 'length'. key=length, value=relative ratio.
    # Exact matching is hard, so the balancer approximates it while tracking statistics.
    length_ratios: Dict[int, float] = field(default_factory=lambda: {
        2: 0.10, 3: 0.20, 4: 0.25, 5: 0.20, 6: 0.13, 7: 0.07, 8: 0.05,
    })
    # Target ratio for the AB pre->post distance (= AB_post.order - matching AB_pre.order).
    # key=distance (>=1), value=relative ratio. Empty {} = no targeting for this dim.
    ab_distance_ratios: Dict[int, float] = field(default_factory=dict)
    # Target ratio for the OR reference distance (= order - ordinal_k of an OR back-ref).
    # key=distance (>=1), value=relative ratio. Empty {} = no targeting for this dim.
    or_distance_ratios: Dict[int, float] = field(default_factory=dict)

    # episodes per scene, number of scenes (num_scenes can also be overridden by a CLI arg)
    episodes_per_scene: int = 10
    num_scenes: int = 5

    # fraction of all episodes that are goal-absent (end with a nonsensical instruction)
    goal_absent_ratio: float = 0.15

    # per-style base sampling weight overrides (falls back to the class base_weight)
    style_weights: Dict[str, float] = field(default_factory=dict)
    # whether style evenness is one of the balanced dimensions (uniform target over styles)
    even_styles: bool = True

    max_length: int = 12        # automaton hard cap (prevents runaway length)
    seed: int = 0

    # --- balancer (generate -> measure -> prune -> refill -> repeat) ---
    # All ratio balancing (length / ab_distance / or_distance / styles) is done by
    # freely generating lists, measuring the distributions, deleting the lists that
    # hurt the fit the most, regenerating, and keeping the ones that fit — repeated.
    refine_iterations: int = 60       # number of prune+refill rounds
    prune_fraction: float = 0.2       # fraction of the pool pruned each round
    refill_oversample: int = 4        # candidates generated per needed slot when refilling
    # per-dimension weights in the fit error (length is exactly achievable so it leads)
    balance_weights: Dict[str, float] = field(default_factory=lambda: {
        "length": 2.0, "ab_distance": 1.0, "or_distance": 1.0, "styles": 0.3,
    })

    def to_dict(self) -> Dict:
        d = asdict(self)
        # JSON turns int keys into str, so stringify explicitly
        d["length_ratios"] = {str(k): v for k, v in self.length_ratios.items()}
        d["ab_distance_ratios"] = {str(k): v for k, v in self.ab_distance_ratios.items()}
        d["or_distance_ratios"] = {str(k): v for k, v in self.or_distance_ratios.items()}
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "GeneratorConfig":
        d = dict(d)
        for key in ("length_ratios", "ab_distance_ratios", "or_distance_ratios"):
            if key in d:
                d[key] = {int(k): float(v) for k, v in d[key].items()}
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class BuilderConfig:
    """command 2: policy for converting style lists -> hm3d episode json."""
    split: str = "train"
    # output dataset root. final path: {dataset_root}/{split}/content/{scene}.json[.gz]
    dataset_root: str = "data/datasets/refon/v1"
    # gzip the shards. 3D-Mem's run_refonbench_evaluation.py reads either form, so this
    # is only about disk space vs. being able to open a shard in an editor.
    compress: bool = True

    # distinct-object budget per episode (cap on new objects chosen by S/AB_pre/AR_pre).
    # once exceeded, existing objects are reused.
    max_objects: int = 8

    # category whitelist allowed as goal candidates (None = all scene categories)
    category_whitelist: Optional[list] = None
    # pool of 'plausible but absent' categories used by the absent_object goal-absent style
    absent_category_pool: list = field(default_factory=lambda: [
        "dragon", "spaceship", "dinosaur", "piano", "elephant",
        "fireplace", "fountain", "treadmill", "aquarium", "chandelier",
    ])
    with_image_goals: bool = False
    seed: int = 0

    # --- HM3D dataset location ---
    # Directory holding the HM3D split folders (train/, val/, ...), i.e. exactly what
    # 3D-Mem calls `scene_data_path` in cfg/eval_*.yaml. Point this at 3D-Mem's copy
    # (e.g. "data/hm3d") and the generator builds episodes against the same meshes the
    # evaluator will load. With it set, scenes may be named by bare hash or split
    # folder; see refon/scene_paths.py.
    hm3d_root: Optional[str] = None

    # --- HM3DSceneLoader settings (applied to LoaderConfig in command 2) ---
    # None means keep the loader's own default.
    # scene_root is only consulted for scene refs that hm3d_root could not resolve.
    scene_root: Optional[str] = None                 # e.g. "data/scene_datasets/"
    # None + hm3d_root set -> auto-detected next to / inside hm3d_root.
    scene_dataset_config: Optional[str] = None        # *.scene_dataset_config.json path
    viewpoint_min_iou: Optional[float] = None         # frame-coverage threshold to accept a viewpoint
    viewpoint_num_candidates: Optional[int] = None
    viewpoint_max_view_points: Optional[int] = None
    min_geodesic: Optional[float] = None              # start-point geodesic distance bounds
    max_geodesic: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "BuilderConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})
