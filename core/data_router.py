"""
core/data_router.py
===================
Dataset Size Router and Ingestion Coordinator.
Validates input datasets, routes to appropriate SourceAdapters,
and creates the initial Dataset Intelligence Object (DIO).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd

from core.dio import DIO
from core.persistence import create_run_directory, save_dio_json
from security.file_validator import FileValidator, FileValidationError, ValidationResult
from sources.base_adapter import SourceAdapter
from sources.csv_adapter import CSVAdapter
from sources.excel_adapter import ExcelAdapter


class DataRouter:
    """
    Coordinates security validation, adapter selection, dataset loading, and initial DIO initialization.
    """

    def __init__(self, max_upload_size_mb: int = 200) -> None:
        self.max_upload_size_mb = max_upload_size_mb
        self.validator = FileValidator(max_upload_size_mb=max_upload_size_mb)
        self.adapters: dict[str, SourceAdapter] = {
            "csv": CSVAdapter(),
            "xlsx": ExcelAdapter(),
            "xls": ExcelAdapter(),
        }

    def ingest(
        self,
        file_path: str | Path,
        base_runs_dir: str | Path | None = None,
    ) -> tuple[pd.DataFrame, DIO, Path | None]:
        """
        Execute security validation, load dataset into a DataFrame, initialize DIO, and optionally create run directory.

        Parameters:
            file_path: Path to raw dataset file.
            base_runs_dir: Base directory for run artifacts (optional).

        Returns:
            tuple[pd.DataFrame, DIO, Path | None]: (loaded_dataframe, initialized_dio, run_directory)
        """
        path = Path(file_path).resolve()

        # 1. Security & structural validation
        val_result: ValidationResult = self.validator.validate(path)
        if not val_result.is_valid:
            error_msg = "; ".join(val_result.errors)
            raise FileValidationError(f"File validation failed for '{path.name}': {error_msg}")

        # 2. Select Source Adapter
        adapter = self.adapters.get(val_result.file_type)
        if not adapter:
            raise FileValidationError(f"No source adapter available for file type '{val_result.file_type}'")

        # 3. Load DataFrame
        df = adapter.load(
            path,
            encoding=val_result.encoding,
            delimiter=val_result.delimiter,
        )

        if df.empty or len(df) == 0:
            raise FileValidationError(f"Dataset '{path.name}' contains zero data rows after parsing.")

        # 4. Initialize Dataset Intelligence Object (DIO)
        dio = DIO.create_empty(
            file_name=path.name,
            dataset_hash=val_result.dataset_hash,
        )
        dio.ingestion = {
            "n_rows": int(len(df)),
            "n_columns": int(len(df.columns)),
            "file_type": val_result.file_type,
            "encoding": val_result.encoding,
        }

        # 5. Create run directory and persist initial DIO if base_runs_dir is provided
        run_dir: Path | None = None
        if base_runs_dir is not None:
            run_dir = create_run_directory(base_runs_dir, dataset_name=path.name)
            save_dio_json(dio, run_dir / "dio.json")

        return df, dio, run_dir
