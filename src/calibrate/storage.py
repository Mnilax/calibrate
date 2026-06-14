"""JSON file storage for predictions."""

from __future__ import annotations

import json
from pathlib import Path

from calibrate.models import PredictionStore

DEFAULT_DIR = Path.home() / ".calibrate"
DEFAULT_FILE = DEFAULT_DIR / "predictions.json"


def get_data_path(override: str | None = None) -> Path:
    """Return the path to the data file, creating parent dirs if needed."""
    if override:
        path = Path(override)
    else:
        path = DEFAULT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load(data_path: str | None = None) -> PredictionStore:
    """Load predictions from JSON file."""
    path = get_data_path(data_path)
    if not path.exists():
        return PredictionStore()
    with open(path, "r", encoding="utf-8") as f:
        return PredictionStore.from_dict(json.load(f))


def save(store: PredictionStore, data_path: str | None = None) -> None:
    """Save predictions to JSON file."""
    path = get_data_path(data_path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(store.to_dict(), f, indent=2, ensure_ascii=False)
