"""Configuration loading with paths anchored to the project, not the shell cwd."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = project_path(path)
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    config = copy.deepcopy(config)
    for section in ("model", "data", "train", "eval"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing configuration section: {section}")
    config.setdefault("seed", 20261023)
    config.setdefault("output_root", "outputs")
    config.setdefault("device", "auto")
    config.setdefault("cpu_threads", 4)
    return config
