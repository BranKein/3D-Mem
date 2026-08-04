"""Resumable progress tracking — a small checkpoint file in the output folder.

Long runs (command 1 at scale, command 2 which loads habitat per scene and is slower)
must survive interruption. Each command processes one scene at a time, writes that
scene's output immediately, and records the scene as done in a progress file. On the
next run, if the progress file exists, completed scenes are skipped and work resumes.

The file is written atomically (temp + os.replace) so an interruption never corrupts it.
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Dict, List, Optional


class ProgressTracker:
    def __init__(self, path: str, data: Dict):
        self.path = path
        self.data = data

    @classmethod
    def load_or_create(
        cls, path: str, command: str, meta: Optional[Dict] = None
    ) -> "tuple[ProgressTracker, bool]":
        """Return (tracker, resuming). resuming=True if a matching progress file existed."""
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                data = None
            if data and data.get("command") == command:
                return cls(path, data), True
        data = {"command": command, "completed": [], "meta": meta or {}, "agg": None}
        return cls(path, data), False

    @property
    def completed(self) -> set:
        return set(self.data.get("completed", []))

    def is_done(self, key) -> bool:
        return key in self.completed

    @property
    def agg(self) -> Optional[Dict]:
        return self.data.get("agg")

    @property
    def finished(self) -> bool:
        return bool(self.data.get("finished", False))

    def save(self) -> None:
        """Write the progress file now (called at command start so it exists before
        the first scene, and after each scene completes)."""
        self._save()

    def mark_done(self, key, agg: Optional[Dict] = None) -> None:
        comp: List = self.data.setdefault("completed", [])
        if key not in comp:
            comp.append(key)
        if agg is not None:
            self.data["agg"] = agg
        self._save()

    def mark_finished(self) -> None:
        self.data["finished"] = True
        self._save()

    def _save(self) -> None:
        d = os.path.dirname(self.path) or "."
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.data, f, indent=2)
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
