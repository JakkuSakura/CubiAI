"""Pipeline runner orchestrating stage execution."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterable, List

from pydantic import BaseModel, Field

from ..config.models import AppConfig
from ..errors import CubiAIError, MissingDependencyError, PipelineStageError
from ..workspace import Workspace
from .context import PipelineContext
from .stages import PipelineStage, build_default_stages


class StageResult(BaseModel):
    name: str
    status: str
    duration_seconds: float
    detail: str | None = None


class PipelineResult(BaseModel):
    stage_results: List[StageResult] = Field(default_factory=list)
    outputs: dict[str, Path] = Field(default_factory=dict)
    diagnostics_path: Path | None = None


class PipelineRunner:
    """High-level orchestrator that executes configured pipeline stages."""

    def __init__(self, config: AppConfig, workspace: Workspace, stages: Iterable[PipelineStage] | None = None) -> None:
        self.config = config
        self.workspace = workspace
        self.logger = logging.getLogger("cubiai.pipeline")
        self.stages = list(stages or build_default_stages(config=config, workspace=workspace))

    def run(self) -> PipelineResult:
        ctx = PipelineContext(config=self.config, workspace=self.workspace)
        stage_results: list[StageResult] = []

        for stage in self.stages:
            start = time.perf_counter()
            try:
                self.logger.debug("Running stage %s", stage.name)
                stage.run(ctx)
                duration = time.perf_counter() - start
                stage_results.append(
                    StageResult(name=stage.name, status="completed", duration_seconds=duration)
                )
            except MissingDependencyError as exc:
                duration = time.perf_counter() - start
                detail = str(exc)
                stage_results.append(
                    StageResult(
                        name=stage.name,
                        status="failed",
                        duration_seconds=duration,
                        detail=detail,
                    )
                )
                self.logger.error("Stage %s failed: %s", stage.name, detail)
                raise
            except PipelineStageError as exc:
                duration = time.perf_counter() - start
                stage_results.append(
                    StageResult(
                        name=stage.name,
                        status="failed",
                        duration_seconds=duration,
                        detail=exc.message,
                    )
                )
                self.logger.exception("Stage %s failed", stage.name)
                raise
            except CubiAIError as exc:
                duration = time.perf_counter() - start
                stage_results.append(
                    StageResult(
                        name=stage.name,
                        status="failed",
                        duration_seconds=duration,
                        detail=str(exc),
                    )
                )
                self.logger.exception("Stage %s failed", stage.name)
                raise
            except Exception as exc:  # noqa: BLE001
                duration = time.perf_counter() - start
                stage_results.append(
                    StageResult(
                        name=stage.name,
                        status="failed",
                        duration_seconds=duration,
                        detail=str(exc),
                    )
                )
                self.logger.exception("Unexpected failure in stage %s", stage.name)
                raise PipelineStageError(stage=stage.name, message="Unexpected error", cause=exc) from exc

        diagnostics_path = None
        if ctx.diagnostics:
            diagnostics_path = self.workspace.save_metadata("diagnostics.json", ctx.diagnostics)

        return PipelineResult(
            stage_results=stage_results,
            outputs={key: Path(value) if not isinstance(value, Path) else value for key, value in ctx.outputs.items()},
            diagnostics_path=diagnostics_path,
        )
