"""
sources/base_adapter.py
=======================
Abstract SourceAdapter seam interface.
Enables pluggable data source ingestion (CSV, XLSX in V1; JSON, Parquet, SQL in V2)
without altering downstream analysis agents.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import pandas as pd


class SourceAdapter(ABC):
    """
    Abstract interface for all tabular data source adapters.
    """

    @abstractmethod
    def validate(self, path_or_conn: str | Path | Any) -> bool:
        """Verify that the source is valid and readable by this adapter."""
        raise NotImplementedError

    @abstractmethod
    def load(self, path_or_conn: str | Path | Any, **kwargs: Any) -> pd.DataFrame:
        """Load and return a normalized pandas DataFrame."""
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Return adapter metadata (source_type, supported_extensions, version)."""
        raise NotImplementedError
