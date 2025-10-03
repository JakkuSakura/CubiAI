"""Shared runtime context for pipeline stages."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..config.models import AppConfig
from ..workspace import Workspace


@dataclass
class PipelineContext:
    config: AppConfig
    workspace: Workspace
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("cubiai.pipeline"))
    data: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def record_output(self, name: str, value: Any) -> None:
        self.outputs[name] = value

    def record_diagnostic(self, stage: str, payload: Any) -> None:
        self.diagnostics.setdefault("stages", {})[stage] = payload
