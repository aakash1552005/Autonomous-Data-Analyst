"""
core/dio.py
===========
Dataset Intelligence Object (DIO) — the shared data contract across all agents.
Supports creation, section mutation, JSON serialization/deserialization, and validation.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator, MutableMapping


@dataclass
class IngestionMetadata:
    n_rows: int = 0
    n_columns: int = 0
    file_type: str = "csv"  # csv | xlsx
    encoding: str = "utf-8"


@dataclass
class ArtifactPaths:
    cleaned_csv: str | None = None
    removed_rows_csv: str | None = None
    model_pkl: str | None = None
    pdf_report: str | None = None
    pptx_report: str | None = None
    chart_paths: list[str] = field(default_factory=list)


@dataclass
class DomainGuess:
    domain: str = "generic"
    confidence: float = 0.0


@dataclass
class QualityAssessment:
    score: int = 0
    issues: list[str] = field(default_factory=list)


@dataclass
class EDAResults:
    summary_stats: dict[str, Any] = field(default_factory=dict)
    correlations: dict[str, Any] = field(default_factory=dict)
    charts: list[str] = field(default_factory=list)


@dataclass
class MLResults:
    problem_type: str = "none"  # classification | regression | none
    target_column: str | None = None
    target_score: float | None = None
    models_tried: list[dict[str, Any]] = field(default_factory=list)
    best_model: str | None = None
    feature_importance: dict[str, float] = field(default_factory=dict)
    feature_explanations: list[str] = field(default_factory=list)
    class_imbalance_warning: bool = False
    verification_warnings: list[str] = field(default_factory=list)


@dataclass
class ReportPaths:
    pdf_path: str | None = None
    pptx_path: str | None = None


@dataclass
class LLMUsageRecord:
    total_tokens: int = 0
    total_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DIO(MutableMapping[str, Any]):
    """
    Dataset Intelligence Object (DIO).
    Centralized structured schema connecting all analysis agents.
    """
    schema_version: str = "1.0.0"
    dataset_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    dataset_hash: str = ""
    file_name: str = ""
    ingestion: dict[str, Any] = field(default_factory=lambda: asdict(IngestionMetadata()))
    artifacts: dict[str, Any] = field(default_factory=lambda: asdict(ArtifactPaths()))
    columns: list[dict[str, Any]] = field(default_factory=list)
    date_columns: list[dict[str, Any]] = field(default_factory=list)
    domain_guess: dict[str, Any] = field(default_factory=lambda: asdict(DomainGuess()))
    quality: dict[str, Any] = field(default_factory=lambda: asdict(QualityAssessment()))
    cleaning_log: list[dict[str, Any]] = field(default_factory=list)
    eda: dict[str, Any] = field(default_factory=lambda: asdict(EDAResults()))
    ml: dict[str, Any] = field(default_factory=lambda: asdict(MLResults()))
    insights: list[dict[str, Any] | str] = field(default_factory=list)
    reports: dict[str, Any] = field(default_factory=lambda: asdict(ReportPaths()))
    progress: dict[str, str] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)
    agent_metrics: list[dict[str, Any]] = field(default_factory=list)
    decision_log: list[dict[str, Any]] = field(default_factory=list)
    llm_usage: dict[str, Any] = field(default_factory=lambda: asdict(LLMUsageRecord()))

    # --- MutableMapping Interface for Dict-Like Access ---
    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"Invalid DIO attribute: {key}")

    def __delitem__(self, key: str) -> None:
        raise NotImplementedError("Deleting DIO top-level sections is not permitted.")

    def __iter__(self) -> Iterator[str]:
        return iter(self.__dataclass_fields__)

    def __len__(self) -> int:
        return len(self.__dataclass_fields__)

    # --- Serialization & Deserialization ---
    def to_dict(self) -> dict[str, Any]:
        """Convert DIO to a pure Python dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize DIO to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DIO:
        """Construct a DIO instance from a dictionary."""
        known_fields = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    @classmethod
    def from_json(cls, json_str: str) -> DIO:
        """Construct a DIO instance from a JSON string."""
        data = json.loads(json_str)
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object for DIO deserialization.")
        return cls.from_dict(data)

    @classmethod
    def create_empty(cls, file_name: str, dataset_hash: str = "", dataset_id: str | None = None) -> DIO:
        """Factory method to initialize a new, empty DIO for a dataset."""
        return cls(
            dataset_id=dataset_id or str(uuid.uuid4()),
            dataset_hash=dataset_hash,
            file_name=file_name,
        )

    def validate(self) -> list[str]:
        """
        Validate the DIO schema integrity.
        Returns a list of validation issue strings (empty if valid).
        """
        issues: list[str] = []
        if not self.schema_version:
            issues.append("Missing 'schema_version'")
        if not self.dataset_id:
            issues.append("Missing 'dataset_id'")
        if not isinstance(self.columns, list):
            issues.append("'columns' must be a list")
        if not isinstance(self.progress, dict):
            issues.append("'progress' must be a dictionary")
        if not isinstance(self.errors, list):
            issues.append("'errors' must be a list")
        if not isinstance(self.agent_metrics, list):
            issues.append("'agent_metrics' must be a list")
        if not isinstance(self.decision_log, list):
            issues.append("'decision_log' must be a list")
        return issues
