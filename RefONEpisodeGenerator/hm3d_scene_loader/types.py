"""HM3DSceneLoader가 주고받는 데이터 타입 정의.

모든 좌표/회전은 habitat 관례를 따른다.
- position: [x, y, z]  (y는 위쪽(up), 단위 meter)
- rotation: [qx, qy, qz, qw]  (quaternion, 직렬화 시 4-list)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

Vec3 = List[float]   # [x, y, z]
Quat = List[float]   # [qx, qy, qz, qw]


@dataclass
class SemanticObject:
    """sim.semantic_scene.objects 한 원소를 래핑한 goal 후보.

    object_id 는 HM3DSem instance 핸들 문자열(예: "bench_1", "bed_199")로,
    GOAT goal catalog의 instance 식별자와 동일한 규약이다.
    """
    object_id: str          # instance handle, 예: "bench_1"
    semantic_id: int        # semantic sensor에 찍히는 정수 id (가시성 판정용)
    category: str           # category.name(), 예: "bench"
    center: Vec3            # aabb.center  (객체 중심, goal["position"]으로 직렬화)
    sizes: Vec3             # aabb.sizes   (x,y,z 전체 변의 길이)
    raw_handle: Optional[object] = None  # 원본 habitat_sim semantic object (디버깅용)

    @property
    def half_extents(self) -> np.ndarray:
        return np.asarray(self.sizes, dtype=np.float64) / 2.0


@dataclass
class AgentState:
    position: Vec3
    rotation: Quat


@dataclass
class ViewPoint:
    """goal 도달 판정에 쓰이는 viewpoint. GOAT goals[*]["view_points"][i] 와 동일 포맷."""
    agent_state: AgentState
    iou: float = 0.0              # 객체 OBB 대비 가시 영역 비율 (선택, 디버깅/품질용)
    frame_coverage: float = 0.0  # semantic mask 픽셀 비율 (선택)

    def to_dict(self) -> Dict:
        return {
            "agent_state": {
                "position": list(self.agent_state.position),
                "rotation": list(self.agent_state.rotation),
            },
            "iou": round(float(self.iou), 5),
        }


@dataclass
class ImageGoalParam:
    """image-modality goal 한 장의 카메라 파라미터.

    GOAT goals[*]["image_goals"][i] 와 동일 포맷. 픽셀은 저장하지 않고
    이 파라미터로 오프라인에서 렌더링/임베딩한다.
    """
    position: Vec3
    rotation: Quat
    frame_coverage: float
    object_coverage: float
    hfov: int = 74
    image_dimensions: List[int] = field(default_factory=lambda: [512, 512])

    def to_dict(self) -> Dict:
        return {
            "position": list(self.position),
            "rotation": list(self.rotation),
            "frame_coverage": self.frame_coverage,
            "object_coverage": self.object_coverage,
            "hfov": self.hfov,
            "image_dimensions": list(self.image_dimensions),
        }


@dataclass
class GoalInstance:
    """goals[key] 리스트의 한 instance. GOAT goal catalog 1:1 포맷."""
    object_category: str
    object_id: str
    position: Vec3
    view_points: List[ViewPoint] = field(default_factory=list)
    children_object_categories: List[str] = field(default_factory=list)
    lang_desc: Optional[str] = None
    image_goals: List[ImageGoalParam] = field(default_factory=list)

    def to_dict(self, include_lang: bool = True, include_image: bool = True) -> Dict:
        d: Dict = {
            "object_category": self.object_category,
            "object_id": self.object_id,
            "position": list(self.position),
            "view_points": [vp.to_dict() for vp in self.view_points],
            "children_object_categories": list(self.children_object_categories),
        }
        # GOAT 실측: lang_desc/image_goals 는 일부 instance에만 존재.
        if include_lang and self.lang_desc is not None:
            d["lang_desc"] = self.lang_desc
        if include_image and self.image_goals:
            d["image_goals"] = [ig.to_dict() for ig in self.image_goals]
        return d
