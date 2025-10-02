"""Utilities for loading CubiAI profile configurations."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from .models import ProfileConfig

DEFAULT_PROFILE_DIR = Path(__file__).resolve().parents[2] / "profiles"


@dataclass(frozen=True)
class InlineOverride:
    key_path: tuple[str, ...]
    value: Any

    @classmethod
    def parse(cls, expression: str) -> "InlineOverride":
        if "=" not in expression:
            msg = "Override must be of the form key=value"
            raise ValueError(msg)
        key, raw_value = expression.split("=", 1)
        key_path = tuple(part.strip() for part in key.split("."))
        value = _parse_value(raw_value)
        return cls(key_path=key_path, value=value)


def _parse_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def discover_profiles() -> list[Path]:
    if DEFAULT_PROFILE_DIR.exists():
        return sorted(DEFAULT_PROFILE_DIR.glob("*.yaml"))
    return []


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if (
            isinstance(value, dict)
            and key in base
            and isinstance(base[key], dict)
        ):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _apply_override(payload: dict[str, Any], override: InlineOverride) -> None:
    cursor = payload
    for key in override.key_path[:-1]:
        cursor = cursor.setdefault(key, {})  # type: ignore[assignment]
    cursor[override.key_path[-1]] = override.value  # type: ignore[index]


def load_profile(
    profile_name: str,
    overrides_path: Path | None = None,
    inline_overrides: Iterable[InlineOverride] = (),
) -> ProfileConfig:
    profile_path = _resolve_profile_path(profile_name)
    if profile_path is None:
        msg = f"Profile '{profile_name}' not found"
        raise FileNotFoundError(msg)

    payload: dict[str, Any] = {}
    if profile_path.exists():
        payload = yaml.safe_load(profile_path.read_text()) or {}

    if overrides_path is not None:
        override_payload = yaml.safe_load(overrides_path.read_text()) or {}
        payload = _deep_update(payload, override_payload)

    for inline in inline_overrides:
        _apply_override(payload, inline)

    return ProfileConfig.model_validate(payload)


def _resolve_profile_path(profile_name: str) -> Path | None:
    candidate = DEFAULT_PROFILE_DIR / f"{profile_name}.yaml"
    if candidate.exists():
        return candidate
    alternative = Path(profile_name)
    if alternative.exists():
        return alternative
    return None
