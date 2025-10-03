"""Utilities for loading CubiAI configuration."""
from __future__ import annotations

from pathlib import Path

import yaml

from .models import AppConfig

DEFAULT_CONFIG_PATH =  "config/cubiai.yaml"


def load_config(config_path: Path | None = None) -> AppConfig:
    path = config_path or DEFAULT_CONFIG_PATH
    if not path.exists():
        msg = f"Configuration file not found: {path}"
        raise FileNotFoundError(msg)

    payload = yaml.safe_load(path.read_text()) or {}
    return AppConfig.model_validate(payload)
