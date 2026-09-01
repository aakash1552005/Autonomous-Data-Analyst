"""
agents/ml/model_persister.py
============================
Model artifact persistence, SHA-256 integrity hashing, and reload verification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import joblib
import pandas as pd
from core.hashing import compute_file_hash


def save_and_verify_model_artifact(
    pipeline: Any,
    output_dir: Path,
    sample_X: pd.DataFrame,
) -> tuple[str, str]:
    """
    Save trained pipeline to model.pkl, compute SHA-256 hash, and verify reload.
    Returns (model_pkl_path, model_sha256_hash).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pkl"

    # 1. Save artifact with joblib
    joblib.dump(pipeline, str(model_path))

    # 2. Compute SHA-256 hash
    model_hash = compute_file_hash(model_path)

    # 3. Verify Reload and Prediction Inference
    reloaded_pipeline = joblib.load(str(model_path))
    test_preds = reloaded_pipeline.predict(sample_X.head(2))
    if len(test_preds) != min(2, len(sample_X)):
        raise RuntimeError("Reloaded model failed prediction verification")

    return str(model_path.resolve()), model_hash
