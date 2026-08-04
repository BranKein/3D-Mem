"""HM3DSceneLoader.

scene_id 하나를 받아서, 그 scene의 에피소드를 생성하는 데 필요한 모든 정보를
habitat-sim(lab)에서 끌어오는 단일 진입점.

GOAT-Bench 레포(goat_bench/)에는 episode '생성' 코드가 없다. 이 로더는 그
빠진 seam을 메우기 위한 것으로, 다음 책임만 진다:

  (1) scene를 sim에 올리고 런타임과 동일한 navmesh를 재계산한다.
  (2) semantic object들을 goal 후보로 enumerate 한다.
  (3) 각 goal 후보의 viewpoint / image-goal 카메라 파라미터를 샘플링한다.
  (4) pathfinder 질의(geodesic distance, navigable sampling)를 노출한다.
  (5) 위를 모아 GOAT 호환 goal-catalog dict를 만든다.

이 로더는 "어떤 카테고리를 고를지 / subtask를 몇 개로 엮을지 / 거리 분포를
어떻게 강제할지" 같은 '정책'은 결정하지 않는다. 그건 이 로더를 호출하는
상위 episode 빌더의 몫이다. (그래서 정책 파라미터는 모두 LoaderConfig로 노출.)

참고한 호출 패턴:
  - sim 구성/navmesh: goat_bench/utils/visualize/ovon_goals.py:_config_sim
  - semantic 접근:     goat_bench/utils/visualize/ovon_goals.py:42,56
  - AABB 접근:         goat_bench/utils/utils.py:86-132
  - geodesic 질의:     goat_bench/measurements/nav.py:85-93,195-208
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import habitat_sim
    from habitat_sim import bindings as hsim
    from habitat_sim.agent.agent import AgentConfiguration, AgentState as SimAgentState
except Exception as exc:  # pragma: no cover - habitat-sim은 런타임 의존성
    habitat_sim = None
    _IMPORT_ERROR = exc

from .config import LoaderConfig
from .types import (
    AgentState,
    GoalInstance,
    ImageGoalParam,
    SemanticObject,
    ViewPoint,
)


def _quat_to_list(q) -> List[float]:
    """habitat quaternion -> [qx, qy, qz, qw]."""
    return [float(q.x), float(q.y), float(q.z), float(q.w)]


def _yaw_quat_facing(src: np.ndarray, tgt: np.ndarray) -> List[float]:
    """src에서 tgt(객체 중심)를 바라보는 yaw-only quaternion [qx,qy,qz,qw].

    habitat: -z가 정면, y가 up. 평면(xz)에서의 heading만 정렬한다.
    """
    direction = np.asarray(tgt, dtype=np.float64) - np.asarray(src, dtype=np.float64)
    yaw = np.arctan2(-direction[0], -direction[2])  # heading about +y
    half = yaw / 2.0
    return [0.0, float(np.sin(half)), 0.0, float(np.cos(half))]


class HM3DSceneLoader:
    """단일 HM3D scene 로더. with 문(context manager)로 쓰는 것을 권장."""

    def __init__(self, scene_id: str, config: Optional[LoaderConfig] = None):
        if habitat_sim is None:  # pragma: no cover
            raise ImportError(
                f"habitat_sim import 실패: {_IMPORT_ERROR}. habitat-sim 환경에서 실행하세요."
            )
        self.config = config or LoaderConfig()
        self.scene_id = scene_id
        self.scene_path = self._resolve_scene_path(scene_id)
        # goal-catalog 키 prefix. GOAT 규약: glb basename (예: "4ok3usBNeis.basis.glb")
        self.scene_key = os.path.basename(self.scene_path)
        self._rng = np.random.default_rng(self.config.random_seed)
        self._sim = None
        self._agent = None
        self._open()

    # ------------------------------------------------------------------ #
    # 생애주기
    # ------------------------------------------------------------------ #
    def __enter__(self) -> "HM3DSceneLoader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._sim is not None:
            self._sim.close()
            self._sim = None

    def _resolve_scene_path(self, scene_id: str) -> str:
        if os.path.isabs(scene_id) or os.path.exists(scene_id):
            return scene_id
        return os.path.join(self.config.scene_root, scene_id)

    def _open(self) -> None:
        cfg = self.config
        sim_cfg = hsim.SimulatorConfiguration()
        sim_cfg.enable_physics = cfg.enable_physics
        sim_cfg.gpu_device_id = cfg.gpu_device_id
        sim_cfg.scene_dataset_config_file = cfg.scene_dataset_config
        sim_cfg.scene_id = self.scene_path

        sensor_specs = []
        for name, stype in (
            ("color", habitat_sim.SensorType.COLOR),
            ("depth", habitat_sim.SensorType.DEPTH),
            ("semantic", habitat_sim.SensorType.SEMANTIC),
        ):
            spec = habitat_sim.CameraSensorSpec()
            spec.uuid = f"{name}_sensor"
            spec.sensor_type = stype
            spec.resolution = [cfg.sensor.height, cfg.sensor.width]
            spec.position = [0.0, cfg.sensor.sensor_height, 0.0]
            spec.hfov = cfg.sensor.hfov
            spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            sensor_specs.append(spec)

        agent_cfg = AgentConfiguration(
            height=cfg.navmesh.agent_height,
            radius=cfg.navmesh.agent_radius,
            sensor_specifications=sensor_specs,
            action_space={
                "look_up": habitat_sim.ActionSpec(
                    "look_up", habitat_sim.ActuationSpec(amount=30)
                ),
                "look_down": habitat_sim.ActionSpec(
                    "look_down", habitat_sim.ActuationSpec(amount=30)
                ),
            },
        )
        self._sim = habitat_sim.Simulator(
            habitat_sim.Configuration(sim_cfg, [agent_cfg])
        )
        self._agent = self._sim.agents[0]
        self._sim.seed(cfg.random_seed)
        self._recompute_navmesh()

    def _recompute_navmesh(self) -> None:
        assert self._sim.pathfinder.is_loaded, "pathfinder not loaded!"
        ns = habitat_sim.NavMeshSettings()
        ns.set_defaults()
        ns.agent_height = self.config.navmesh.agent_height
        ns.agent_radius = self.config.navmesh.agent_radius
        ns.agent_max_climb = self.config.navmesh.agent_max_climb
        ns.cell_height = self.config.navmesh.cell_height
        # habitat-sim >=0.2.4: include_static_objects moved onto NavMeshSettings; the
        # recompute_navmesh kwarg was removed. Set it on the settings and call with 2 args.
        ns.include_static_objects = self.config.navmesh.include_static_objects
        ok = self._sim.recompute_navmesh(self._sim.pathfinder, ns)
        assert ok, "Failed to build navmesh!"

    # ------------------------------------------------------------------ #
    # (2) Semantic objects  — goal 후보 enumeration
    # ------------------------------------------------------------------ #
    def get_semantic_objects(
        self, categories: Optional[Sequence[str]] = None
    ) -> List[SemanticObject]:
        """이 scene의 semantic object들을 goal 후보로 반환.

        categories 가 주어지면 그 카테고리만, 아니면 config.category_whitelist,
        둘 다 없으면 전체를 반환한다.
        """
        whitelist = None
        if categories is not None:
            whitelist = set(categories)
        elif self.config.category_whitelist is not None:
            whitelist = set(self.config.category_whitelist)

        out: List[SemanticObject] = []
        for o in self._sim.semantic_scene.objects:
            if o is None or o.category is None:
                continue
            cat = o.category.name()
            if whitelist is not None and cat not in whitelist:
                continue
            aabb = o.aabb
            # habitat-sim >=0.3: aabb is a Magnum Range3D whose center()/size() are
            # methods (older versions exposed .center / .sizes as attributes).
            center = aabb.center() if callable(aabb.center) else aabb.center
            size_attr = getattr(aabb, "size", None)
            if size_attr is not None:
                sizes = size_attr() if callable(size_attr) else size_attr
            else:
                sizes = aabb.sizes
            out.append(
                SemanticObject(
                    object_id=str(o.id),
                    semantic_id=int(o.semantic_id),
                    category=cat,
                    center=[float(c) for c in center],
                    sizes=[float(s) for s in sizes],
                    raw_handle=o,
                )
            )
        return out

    def get_categories(self) -> Dict[str, int]:
        """scene 내 카테고리 -> instance 개수."""
        counts: Dict[str, int] = {}
        for o in self._sim.semantic_scene.objects:
            if o is None or o.category is None:
                continue
            counts[o.category.name()] = counts.get(o.category.name(), 0) + 1
        return counts

    # ------------------------------------------------------------------ #
    # (4) Pathfinder 질의
    # ------------------------------------------------------------------ #
    def is_navigable(self, point: Sequence[float]) -> bool:
        return bool(self._sim.pathfinder.is_navigable(np.asarray(point, np.float32)))

    def snap_point(self, point: Sequence[float]) -> Optional[np.ndarray]:
        snapped = self._sim.pathfinder.snap_point(np.asarray(point, np.float32))
        snapped = np.asarray(snapped)
        if np.any(np.isnan(snapped)):
            return None
        return snapped

    def sample_navigable_point(self) -> np.ndarray:
        return np.asarray(self._sim.pathfinder.get_random_navigable_point())

    def island_radius(self, point: Sequence[float]) -> float:
        return float(self._sim.pathfinder.island_radius(np.asarray(point, np.float32)))

    def geodesic_distance(
        self,
        source: Sequence[float],
        targets: Sequence[Sequence[float]] | Sequence[float],
    ) -> float:
        """source에서 target(들) 중 가장 가까운 지점까지의 geodesic 거리.

        GOAT 런타임의 도달 판정과 동일하게, 여러 viewpoint를 한 번에 넘겨
        '가장 가까운 viewpoint까지의 거리'를 얻는 용도. 도달 불가면 inf.

        habitat-sim >=0.2.x: Simulator.geodesic_distance 가 제거되어
        PathFinder + (Multi)GoalShortestPath 로 직접 질의한다.
        """
        src = np.asarray(source, np.float32)
        arr = np.asarray(targets, dtype=np.float32)
        ends = arr.reshape(1, -1) if arr.ndim == 1 else arr
        ends = [np.asarray(e, np.float32) for e in ends]
        pf = self._sim.pathfinder
        try:
            path = habitat_sim.MultiGoalShortestPath()
            path.requested_start = src
            path.requested_ends = ends
            if pf.find_path(path):
                return float(path.geodesic_distance)
            return float("inf")
        except Exception:
            best = float("inf")
            for e in ends:
                sp = habitat_sim.ShortestPath()
                sp.requested_start = src
                sp.requested_end = e
                if pf.find_path(sp):
                    best = min(best, float(sp.geodesic_distance))
            return best

    # ------------------------------------------------------------------ #
    # 렌더 / 가시성
    # ------------------------------------------------------------------ #
    def render_at(
        self, position: Sequence[float], rotation: Sequence[float]
    ) -> Dict[str, np.ndarray]:
        st = SimAgentState()
        st.position = np.asarray(position, np.float32)
        st.rotation = np.quaternion(rotation[3], rotation[0], rotation[1], rotation[2])
        self._agent.set_state(st)
        return self._sim.get_sensor_observations()

    def _frame_coverage(self, obs: Dict[str, np.ndarray], semantic_id: int) -> float:
        mask = obs["semantic_sensor"] == semantic_id
        return float(mask.sum()) / float(mask.size)

    # ------------------------------------------------------------------ #
    # (3a) Viewpoint 샘플링
    # ------------------------------------------------------------------ #
    def sample_view_points(self, obj: SemanticObject) -> List[ViewPoint]:
        """obj가 보이는 navigable viewpoint 리스트를 샘플링.

        알고리즘(단순/대체 가능):
          1. 객체 중심 주변 반경 sample_radius 안에서 navigable 후보를 샘플.
          2. 각 후보에서 객체 중심을 바라보는 yaw로 정렬.
          3. semantic mask 픽셀 비율(frame_coverage)을 IoU 대용으로 측정.
          4. min_iou 이상인 것만 채택, max_view_points까지.
        """
        vcfg = self.config.viewpoint
        center = np.asarray(obj.center, dtype=np.float64)
        results: List[ViewPoint] = []

        for _ in range(vcfg.num_candidates):
            if len(results) >= vcfg.max_view_points:
                break
            # 반경 내 무작위 평면 오프셋 -> snap_point로 navmesh에 투영
            ang = self._rng.uniform(0, 2 * np.pi)
            rad = self._rng.uniform(0.5, vcfg.sample_radius)
            cand = center + np.array([np.cos(ang) * rad, 0.0, np.sin(ang) * rad])
            snapped = self.snap_point(cand)
            if snapped is None:
                continue

            rot = (
                _yaw_quat_facing(snapped, center)
                if vcfg.turn_to_face
                else [0.0, 0.0, 0.0, 1.0]
            )
            best_cov = 0.0
            obs = self.render_at(snapped, rot)
            best_cov = max(best_cov, self._frame_coverage(obs, obj.semantic_id))
            # head tilt 후보를 돌며 최대 coverage 탐색 (ovon_goals.py 패턴)
            for _act in ("look_down", "look_up", "look_up"):
                obs = self._sim.step(_act)
                best_cov = max(best_cov, self._frame_coverage(obs, obj.semantic_id))

            if best_cov >= vcfg.min_iou:
                results.append(
                    ViewPoint(
                        agent_state=AgentState(
                            position=[float(x) for x in snapped], rotation=rot
                        ),
                        iou=best_cov,
                        frame_coverage=best_cov,
                    )
                )
        return results

    # ------------------------------------------------------------------ #
    # (3b) Image goal 카메라 파라미터 샘플링
    # ------------------------------------------------------------------ #
    def sample_image_goals(self, obj: SemanticObject) -> List[ImageGoalParam]:
        """image-modality goal용 카메라 파라미터 리스트 샘플링.

        viewpoint와 유사하나, 저장 포맷이 (position, rotation, frame/object
        coverage, hfov, image_dimensions)이고 navmesh 위 제약이 없다(자유 카메라).
        """
        icfg = self.config.image_goal
        center = np.asarray(obj.center, dtype=np.float64)
        out: List[ImageGoalParam] = []

        attempts = 0
        while len(out) < icfg.num_goals and attempts < icfg.num_goals * 8:
            attempts += 1
            ang = self._rng.uniform(0, 2 * np.pi)
            rad = self._rng.uniform(0.5, self.config.viewpoint.sample_radius)
            cam = center + np.array([np.cos(ang) * rad, 0.0, np.sin(ang) * rad])
            cam[1] = center[1] + self._rng.uniform(0.0, 0.5)
            rot = _yaw_quat_facing(cam, center)
            obs = self.render_at(cam, rot)
            fcov = self._frame_coverage(obs, obj.semantic_id)
            if fcov < icfg.min_frame_coverage:
                continue
            # object_coverage = (객체 픽셀) / (객체 전체가 차지할 수 있는 픽셀)의 근사.
            # 정확한 값은 별도 OBB 투영이 필요하나, 여기선 frame_coverage 기반 근사.
            ocov = min(1.0, fcov * 4.0)
            if ocov < icfg.min_object_coverage:
                continue
            out.append(
                ImageGoalParam(
                    position=[float(x) for x in cam],
                    rotation=rot,
                    frame_coverage=fcov,
                    object_coverage=ocov,
                    hfov=icfg.hfov,
                    image_dimensions=list(icfg.image_dimensions),
                )
            )
        return out

    # ------------------------------------------------------------------ #
    # (5) Goal-catalog 조립  — GOAT 호환 dict
    # ------------------------------------------------------------------ #
    def build_goal_instance(
        self,
        obj: SemanticObject,
        with_view_points: bool = True,
        with_image_goals: bool = False,
        lang_desc: Optional[str] = None,
        children_object_categories: Optional[List[str]] = None,
    ) -> GoalInstance:
        return GoalInstance(
            object_category=obj.category,
            object_id=obj.object_id,
            position=list(obj.center),
            view_points=self.sample_view_points(obj) if with_view_points else [],
            children_object_categories=children_object_categories or [],
            lang_desc=lang_desc,
            image_goals=self.sample_image_goals(obj) if with_image_goals else [],
        )

    def build_goal_catalog(
        self,
        categories: Optional[Sequence[str]] = None,
        with_view_points: bool = True,
        with_image_goals: bool = False,
    ) -> Dict[str, List[Dict]]:
        """GOAT shard의 top-level "goals" 딕셔너리(직렬화 dict)를 만든다.

        키: f"{scene_key}_{category}"  (예: "4ok3usBNeis.basis.glb_bed")
        값: 해당 카테고리의 모든 instance(GoalInstance.to_dict()) 리스트.
        """
        catalog: Dict[str, List[Dict]] = {}
        for obj in self.get_semantic_objects(categories):
            inst = self.build_goal_instance(
                obj,
                with_view_points=with_view_points,
                with_image_goals=with_image_goals,
            )
            if with_view_points and not inst.view_points:
                continue  # 도달 가능한 viewpoint가 없으면 goal 후보에서 제외
            key = f"{self.scene_key}_{obj.category}"
            catalog.setdefault(key, []).append(inst.to_dict())
        return catalog

    # ------------------------------------------------------------------ #
    # (6) start position 샘플링
    # ------------------------------------------------------------------ #
    def sample_start_state(
        self, goal_view_points: Sequence[Sequence[float]]
    ) -> Optional[Tuple[List[float], List[float], float]]:
        """첫 goal의 viewpoint들로부터 거리 제약을 만족하는 navigable 시작점 샘플.

        반환: (start_position, start_rotation, geodesic_distance) 또는 None.
        rotation은 무작위 yaw.
        """
        scfg = self.config.start
        targets = [np.asarray(v, np.float32) for v in goal_view_points]
        if not targets:
            return None
        ref_height = float(targets[0][1])

        for _ in range(scfg.max_attempts):
            p = self.sample_navigable_point()
            if scfg.same_floor_only and abs(float(p[1]) - ref_height) > 0.5:
                continue
            geo = self.geodesic_distance(p, targets)
            if not np.isfinite(geo):
                continue
            if scfg.min_geodesic <= geo <= scfg.max_geodesic:
                yaw = self._rng.uniform(0, 2 * np.pi)
                rot = [0.0, float(np.sin(yaw / 2)), 0.0, float(np.cos(yaw / 2))]
                return [float(x) for x in p], rot, float(geo)
        return None
