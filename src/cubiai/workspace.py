"""Workspace management utilities."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Workspace:
    """Represents a structured workspace for a pipeline run."""

    input_path: Path
    root: Path
    keep_intermediate: bool = False
    resume: bool = False

    @classmethod
    def create(
        cls,
        *,
        input_path: Path,
        root: Path,
        keep_intermediate: bool = False,
        resume: bool = False,
    ) -> "Workspace":
        root.mkdir(parents=True, exist_ok=True)
        (root / "layers").mkdir(parents=True, exist_ok=True)
        (root / "masks").mkdir(parents=True, exist_ok=True)
        (root / "Live2D").mkdir(parents=True, exist_ok=True)
        (root / "logs").mkdir(parents=True, exist_ok=True)
        return cls(input_path=input_path, root=root, keep_intermediate=keep_intermediate, resume=resume)

    @property
    def layers_dir(self) -> Path:
        return self.root / "layers"

    @property
    def masks_dir(self) -> Path:
        return self.root / "masks"

    @property
    def live2d_dir(self) -> Path:
        return self.root / "Live2D"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    def write_json(self, path: Path, data: Any) -> None:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def save_metadata(self, filename: str, payload: Any) -> Path:
        path = self.root / filename
        self.write_json(path, payload)
        return path
