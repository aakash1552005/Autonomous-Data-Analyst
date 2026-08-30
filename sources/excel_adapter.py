"""
sources/excel_adapter.py
========================
Excel data source adapter.
Handles .xlsx and legacy .xls spreadsheets using openpyxl.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd

from sources.base_adapter import SourceAdapter


class ExcelAdapter(SourceAdapter):
    """
    Adapter for reading Excel (.xlsx, .xls) files into pandas DataFrames.
    """

    def validate(self, path_or_conn: str | Path | Any) -> bool:
        path = Path(path_or_conn).resolve()
        if not path.is_file():
            return False
        if path.suffix.lower() not in (".xlsx", ".xls"):
            return False
        return path.stat().st_size > 0

    def load(self, path_or_conn: str | Path | Any, **kwargs: Any) -> pd.DataFrame:
        path = Path(path_or_conn).resolve()
        if not self.validate(path):
            raise ValueError(f"Invalid or unreadable Excel file: {path}")

        sheet_name = kwargs.get("sheet_name", 0)
        try:
            df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
        except Exception as e:
            raise RuntimeError(f"Failed to read Excel file at {path}: {e}") from e

        return df

    def describe(self) -> dict[str, Any]:
        return {
            "source_type": "excel",
            "supported_extensions": [".xlsx", ".xls"],
            "version": "1.0.0",
        }
