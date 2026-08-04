"""HM3D scene path resolution against 3D-Mem's dataset layout.

3D-Mem points at HM3D with a single directory that holds the split folders::

    <hm3d_root>/train/00745-yX5efd48dLf/yX5efd48dLf.basis.glb
    <hm3d_root>/val/00877-4ok3usBNeis/4ok3usBNeis.basis.glb

(that is ``scene_data_path`` in ``cfg/eval_*.yaml``; see ``src/scene_goatbench.py``,
which builds the mesh path exactly this way). This module resolves a scene reference
against that root so the generator and the evaluator load the very same mesh, and so
scenes can be named in whichever short form is convenient:

    ``4ok3usBNeis``                                     scene hash
    ``00877-4ok3usBNeis``                               split folder name
    ``val/00877-4ok3usBNeis``                           split-qualified
    ``hm3d/val/00877-4ok3usBNeis/4ok3usBNeis.basis.glb`` GOAT-style relative id
    ``/abs/path/to/4ok3usBNeis.basis.glb``              absolute mesh path

Nothing here opens a scene -- it is directory lookups only.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

# split subdirectories searched under an hm3d root, in priority order
SPLITS = ("val", "train", "minival", "test", "example")

_GLB_SUFFIX = ".basis.glb"


def scene_hash(scene_ref: str) -> str:
    """The bare HM3D scene hash of any supported scene reference.

    ``hm3d/train/00745-yX5efd48dLf/yX5efd48dLf.basis.glb`` -> ``yX5efd48dLf``
    ``00877-4ok3usBNeis``                                  -> ``4ok3usBNeis``
    ``4ok3usBNeis``                                        -> ``4ok3usBNeis``

    This is also the shard file stem 3D-Mem expects, and the prefix of the goal-catalog
    keys (``<hash>.basis.glb_<category>``).
    """
    ref = scene_ref.rstrip("/")
    base = os.path.basename(ref)
    if base.endswith(".glb"):
        base = base.split(".")[0]
    # split-folder form is "<5-digit index>-<hash>"
    if "-" in base:
        base = base.split("-")[-1]
    return base


def _scene_folders(hm3d_root: str, split: Optional[str] = None) -> Dict[str, str]:
    """{scene folder name: absolute folder path} under `hm3d_root`."""
    folders: Dict[str, str] = {}
    splits = (split,) if split else SPLITS
    for sp in splits:
        split_dir = os.path.join(hm3d_root, sp)
        if not os.path.isdir(split_dir):
            continue
        for name in sorted(os.listdir(split_dir)):
            path = os.path.join(split_dir, name)
            if name.startswith(".") or not os.path.isdir(path):
                continue
            folders.setdefault(name, path)
    return folders


def list_scenes(hm3d_root: str, split: Optional[str] = None) -> List[str]:
    """Scene folder names (``00877-4ok3usBNeis``) available under `hm3d_root`.

    Only folders that actually carry the annotated mesh are returned, so scenes from a
    partially downloaded or non-semantic HM3D copy are filtered out rather than failing
    later inside habitat.
    """
    out = []
    for name, path in _scene_folders(hm3d_root, split).items():
        if os.path.exists(os.path.join(path, scene_hash(name) + _GLB_SUFFIX)):
            out.append(name)
    return sorted(out)


def resolve_scene(scene_ref: str, hm3d_root: Optional[str]) -> str:
    """Absolute path to a scene's ``*.basis.glb``.

    An absolute path or an existing relative path is taken as-is, so scene lists written
    for the standalone generator keep working. Anything else is looked up by hash under
    `hm3d_root`.
    """
    if os.path.isabs(scene_ref) or os.path.exists(scene_ref):
        return os.path.abspath(scene_ref)
    if not hm3d_root:
        raise ValueError(
            f"cannot resolve scene '{scene_ref}': it is not an existing path and no "
            f"hm3d_root is configured (set BuilderConfig.hm3d_root or pass --hm3d-root)"
        )

    wanted = scene_hash(scene_ref)
    for name, folder in _scene_folders(hm3d_root).items():
        if scene_hash(name) != wanted:
            continue
        glb = os.path.join(folder, wanted + _GLB_SUFFIX)
        if os.path.exists(glb):
            return os.path.abspath(glb)
    raise FileNotFoundError(
        f"scene '{scene_ref}' (hash '{wanted}') not found under {hm3d_root}; "
        f"expected {hm3d_root}/<split>/<index>-{wanted}/{wanted}{_GLB_SUFFIX}"
    )


def to_goat_scene_id(glb_path: str) -> str:
    """GOAT-style relative scene id, e.g. ``hm3d/val/00877-4ok3usBNeis/4ok3usBNeis.basis.glb``.

    Recorded in the episode for provenance. 3D-Mem identifies the scene from the shard
    file name rather than this field, so it only has to be readable, not resolvable.
    """
    parts = os.path.normpath(glb_path).split(os.sep)
    if len(parts) >= 3:
        return "hm3d/" + "/".join(parts[-3:])
    return glb_path


def default_scene_dataset_config(hm3d_root: str) -> Optional[str]:
    """The ``*.scene_dataset_config.json`` that goes with `hm3d_root`, if present.

    3D-Mem keeps it one level above the split folders
    (``data/hm3d_annotated_basis.scene_dataset_config.json`` next to ``data/hm3d/``),
    while a stock HM3D download keeps it inside. Check both.
    """
    name = "hm3d_annotated_basis.scene_dataset_config.json"
    for candidate in (
        os.path.join(hm3d_root, name),
        os.path.join(os.path.dirname(os.path.normpath(hm3d_root)), name),
    ):
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return None
