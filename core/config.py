"""
core/config.py
==============
Configuration loading, validation, and typed representations.
Loads settings from `config.yaml` and environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LLMConfig:
    provider: str = "ollama"
    model: str = "llama3.1:8b"
    host: str = "http://localhost:11434"
    max_tokens_per_call: int = 500
    temperature: float = 0.2
    timeout_seconds: int = 60
    api_key: str = ""

    def validate(self) -> None:
        if self.provider not in ("ollama", "openai"):
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
        if self.max_tokens_per_call <= 0:
            raise ValueError(f"max_tokens_per_call must be positive, got {self.max_tokens_per_call}")
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError(f"temperature must be between 0.0 and 2.0, got {self.temperature}")
        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {self.timeout_seconds}")


@dataclass
class SecurityConfig:
    max_llm_tokens_per_run: int = 20000

    def validate(self) -> None:
        if self.max_llm_tokens_per_run <= 0:
            raise ValueError(f"max_llm_tokens_per_run must be positive, got {self.max_llm_tokens_per_run}")


@dataclass
class EDAConfig:
    max_charts: int = 6

    def validate(self) -> None:
        if self.max_charts <= 0:
            raise ValueError(f"max_charts must be positive, got {self.max_charts}")


@dataclass
class MLConfig:
    min_rows_for_ml: int = 30
    min_rows_per_class: int = 5
    leakage_threshold: float = 0.95
    train_test_split: float = 0.8
    random_state: int = 42
    min_rows_xgboost: int = 500
    class_imbalance_threshold: float = 0.90
    metric_leakage_f1_threshold: float = 0.98
    metric_leakage_r2_threshold: float = 0.98

    def validate(self) -> None:
        if self.min_rows_for_ml <= 0:
            raise ValueError(f"min_rows_for_ml must be positive, got {self.min_rows_for_ml}")
        if self.min_rows_per_class <= 0:
            raise ValueError(f"min_rows_per_class must be positive, got {self.min_rows_per_class}")
        if not (0.0 < self.leakage_threshold <= 1.0):
            raise ValueError(f"leakage_threshold must be between 0.0 and 1.0, got {self.leakage_threshold}")
        if not (0.0 < self.train_test_split < 1.0):
            raise ValueError(f"train_test_split must be between 0.0 and 1.0, got {self.train_test_split}")
        if not (0.5 <= self.class_imbalance_threshold < 1.0):
            raise ValueError(f"class_imbalance_threshold must be between 0.5 and 1.0, got {self.class_imbalance_threshold}")


@dataclass
class AppConfig:
    max_upload_size_mb: int = 200
    llm: LLMConfig = field(default_factory=LLMConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    eda: EDAConfig = field(default_factory=EDAConfig)
    ml: MLConfig = field(default_factory=MLConfig)

    def validate(self) -> None:
        if self.max_upload_size_mb <= 0:
            raise ValueError(f"max_upload_size_mb must be positive, got {self.max_upload_size_mb}")
        self.llm.validate()
        self.security.validate()
        self.eda.validate()
        self.ml.validate()


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """
    Load configuration from YAML file and apply environment variable overrides.
    """
    if config_path is None:
        # Default to config.yaml in project root
        base_dir = Path(__file__).resolve().parent.parent
        config_path = base_dir / "config.yaml"
    else:
        config_path = Path(config_path)

    raw_data: dict[str, Any] = {}
    if config_path.is_file():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    raw_data = loaded
        except Exception as e:
            raise ValueError(f"Failed to parse config file at {config_path}: {e}") from e

    # Extract sections with defaults
    max_upload_size_mb = int(os.getenv("MAX_UPLOAD_SIZE_MB", raw_data.get("max_upload_size_mb", 200)))

    raw_llm = raw_data.get("llm", {})
    llm_config = LLMConfig(
        provider=os.getenv("LLM_PROVIDER", raw_llm.get("provider", "ollama")),
        model=os.getenv("OLLAMA_MODEL", raw_llm.get("model", "llama3.1:8b")),
        host=os.getenv("OLLAMA_HOST", raw_llm.get("host", "http://localhost:11434")),
        max_tokens_per_call=int(raw_llm.get("max_tokens_per_call", 500)),
        temperature=float(raw_llm.get("temperature", 0.2)),
        timeout_seconds=int(raw_llm.get("timeout_seconds", 60)),
        api_key=os.getenv("OPENAI_API_KEY", raw_llm.get("api_key", "")),
    )

    raw_sec = raw_data.get("security", {})
    security_config = SecurityConfig(
        max_llm_tokens_per_run=int(os.getenv("MAX_LLM_TOKENS_PER_RUN", raw_sec.get("max_llm_tokens_per_run", 20000)))
    )

    raw_eda = raw_data.get("eda", {})
    eda_config = EDAConfig(
        max_charts=int(raw_eda.get("max_charts", 6))
    )

    raw_ml = raw_data.get("ml", {})
    ml_config = MLConfig(
        min_rows_for_ml=int(os.getenv("MIN_ROWS_FOR_ML", raw_ml.get("min_rows_for_ml", 30))),
        min_rows_per_class=int(os.getenv("MIN_ROWS_PER_CLASS", raw_ml.get("min_rows_per_class", 5))),
        leakage_threshold=float(os.getenv("LEAKAGE_THRESHOLD", raw_ml.get("leakage_threshold", 0.95))),
        train_test_split=float(raw_ml.get("train_test_split", 0.8)),
        random_state=int(raw_ml.get("random_state", 42)),
        min_rows_xgboost=int(raw_ml.get("min_rows_xgboost", 500)),
        class_imbalance_threshold=float(raw_ml.get("class_imbalance_threshold", 0.90)),
        metric_leakage_f1_threshold=float(raw_ml.get("metric_leakage_f1_threshold", 0.98)),
        metric_leakage_r2_threshold=float(raw_ml.get("metric_leakage_r2_threshold", 0.98)),
    )

    app_config = AppConfig(
        max_upload_size_mb=max_upload_size_mb,
        llm=llm_config,
        security=security_config,
        eda=eda_config,
        ml=ml_config,
    )
    app_config.validate()
    return app_config
