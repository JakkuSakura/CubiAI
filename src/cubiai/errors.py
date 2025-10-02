"""Custom exception types used across the CubiAI pipeline."""
from __future__ import annotations


class CubiAIError(Exception):
    """Base exception for CubiAI-specific errors."""


class MissingDependencyError(CubiAIError):
    """Raised when an optional dependency required for a stage is unavailable."""


class PipelineStageError(CubiAIError):
    """Raised when a pipeline stage fails irrecoverably."""

    def __init__(self, stage: str, message: str, *, cause: Exception | None = None) -> None:
        super().__init__(f"Stage '{stage}' failed: {message}")
        self.stage = stage
        self.cause = cause
        self.message = message
