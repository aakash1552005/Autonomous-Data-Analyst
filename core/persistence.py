"""
core/persistence.py
===================
Run persistence, safe path resolution, and filesystem artifact management.
"""

from __future__ import annotations

import datetime
import json
import re
from pathlib import Path
from typing import Any

from core.dio import DIO


def sanitize_filename(name: str) -> str:
    """
    Sanitize a file or dataset name to prevent path traversal or filesystem collisions.
    """
    # Remove any directory path components
    base_name = Path(name).name
    # Keep only alphanumeric, hyphen, underscore, and dot
    clean = re.sub(r"[^\w\-\.]", "_", base_name)
    # Strip leading/trailing dots or underscores
    clean = clean.strip("._")
    return clean or "dataset"


def create_run_directory(base_runs_dir: str | Path, dataset_name: str) -> Path:
    """
    Create a new run directory with format: `runs/{timestamp}_{dataset_name}/`.
    Also initializes subdirectories like `charts/`.
    """
    base_dir = Path(base_runs_dir).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    clean_name = sanitize_filename(dataset_name)
    run_dir_name = f"{timestamp}_{clean_name}"

    run_dir = resolve_run_path(base_dir, run_dir_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "charts").mkdir(exist_ok=True)
    return run_dir


def resolve_run_path(base_dir: str | Path, subpath: str | Path) -> Path:
    """
    Resolve a subpath within a base directory, guarding against path traversal attacks.
    """
    base = Path(base_dir).resolve()
    target = (base / subpath).resolve()
    if not (target == base or str(target).startswith(str(base) + "\\") or str(target).startswith(str(base) + "/")):
        raise ValueError(f"Path traversal detected: {subpath} attempts to escape {base}")
    return target


def save_dio_json(dio: DIO | dict[str, Any], file_path: str | Path) -> Path:
    """
    Save the Dataset Intelligence Object (DIO) to a JSON file.
    """
    path = Path(file_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(dio, DIO):
        json_content = dio.to_json(indent=2)
    else:
        json_content = json.dumps(dio, indent=2, default=str)

    # Temporary file write + rename for atomic persistence
    temp_path = path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        f.write(json_content)
    temp_path.replace(path)
    return path


def load_dio_json(file_path: str | Path) -> DIO:
    """
    Load and parse a DIO instance from a JSON file.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"DIO JSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return DIO.from_json(content)


def save_metrics_json(metrics: dict[str, Any], file_path: str | Path) -> Path:
    """
    Save summary metrics to `metrics.json`.
    """
    path = Path(file_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    return path


def load_metrics_json(file_path: str | Path) -> dict[str, Any]:
    """
    Load summary metrics from `metrics.json`.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Metrics file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
