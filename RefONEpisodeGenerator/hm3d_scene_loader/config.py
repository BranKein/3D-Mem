"""HM3DSceneLoader 설정값.

기본값은 GOAT-Bench 런타임과 동일하게 맞춰져 있다.
- navmesh: goat_bench/task/simulator.py + ver_goat_monolithic.yaml 오버라이드
  (agent_height=1.41, agent_radius=0.17, agent_max_climb=0.10, cell_height=0.05)
- sensor 위치/해상도: goat_bench/utils/visualize/ovon_goals.py:_config_sim

⚠️ 빌더(이 로더)와 런타임 평가기의 navmesh 파라미터가 다르면
   생성한 viewpoint가 런타임에서 navigable하지 않을 수 있다. 반드시 일치시킬 것.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class NavmeshConfig:
    agent_height: float = 1.41
    agent_radius: float = 0.17
    agent_max_climb: float = 0.10
    cell_height: float = 0.05
    include_static_objects: bool = False


@dataclass
class SensorConfig:
    width: int = 512
    height: int = 512
    hfov: int = 100              # 가시성 판정용 센서 fov (image goal hfov와 별개)
    sensor_height: float = 1.41  # 카메라의 agent 기준 높이
    tilt_steps_deg: List[int] = field(default_factory=lambda: [0, -30, 30])
    # 가시성 판정 시 head tilt 각도 후보 (0=수평, 음수=look_down 관례에 맞게 조정)


@dataclass
class ViewpointConfig:
    """객체 1개당 viewpoint 샘플링 정책."""
    sample_radius: float = 2.0        # 객체 중심 기준 viewpoint 후보 반경 (m)
    num_candidates: int = 64          # navigable 후보 샘플 수
    min_iou: float = 0.05             # 채택 IoU 하한
    max_view_points: int = 50         # instance당 viewpoint 상한
    turn_to_face: bool = True         # 객체 중심을 바라보도록 yaw 정렬


@dataclass
class ImageGoalConfig:
    """image-modality goal 카메라 파라미터 샘플링 정책."""
    num_goals: int = 16
    hfov: int = 74
    image_dimensions: List[int] = field(default_factory=lambda: [512, 512])
    min_frame_coverage: float = 0.05
    min_object_coverage: float = 0.10


@dataclass
class StartSamplingConfig:
    """episode start point / subtask 간 거리 제약."""
    min_geodesic: float = 1.0
    max_geodesic: float = 30.0
    max_attempts: int = 200
    same_floor_only: bool = True


@dataclass
class LoaderConfig:
    # HM3D scene_datasets 루트. scene_id가 이 경로 기준 상대경로일 때 join된다.
    scene_root: str = "data/scene_datasets/"
    scene_dataset_config: str = (
        "data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
    )
    gpu_device_id: int = 0
    enable_physics: bool = False
    random_seed: int = 1

    navmesh: NavmeshConfig = field(default_factory=NavmeshConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    viewpoint: ViewpointConfig = field(default_factory=ViewpointConfig)
    image_goal: ImageGoalConfig = field(default_factory=ImageGoalConfig)
    start: StartSamplingConfig = field(default_factory=StartSamplingConfig)

    # goal 후보로 허용할 카테고리 화이트리스트(None이면 전체 허용).
    category_whitelist: List[str] = None
