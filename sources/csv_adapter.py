"""
sources/csv_adapter.py
======================
CSV data source adapter.
Handles comma, semicolon, tab, and pipe-delimited tabular files with automatic encoding detection.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any
import pandas as pd

from sources.base_adapter import SourceAdapter


class CSVAdapter(SourceAdapter):
    """
    Adapter for reading CSV and delimited text files into pandas DataFrames.
    """

    def validate(self, path_or_conn: str | Path | Any) -> bool:
        path = Path(path_or_conn).resolve()
        if not path.is_file():
            return False
        if path.suffix.lower() not in (".csv", ".txt", ".tsv"):
            return False
        return path.stat().st_size > 0

    def load(self, path_or_conn: str | Path | Any, **kwargs: Any) -> pd.DataFrame:
        path = Path(path_or_conn).resolve()
        if not self.validate(path):
            raise ValueError(f"Invalid or unreadable CSV file: {path}")

        encoding = kwargs.get("encoding", self._detect_encoding(path))
        delimiter = kwargs.get("delimiter", self._detect_delimiter(path, encoding))

        try:
            df = pd.read_csv(
                path,
                sep=delimiter,
                encoding=encoding,
                on_bad_lines="skip",
                low_memory=False,
            )
        except Exception as e:
            # Fallback with python engine if C engine encounters parser errors
            try:
                df = pd.read_csv(
                    path,
                    sep=delimiter,
                    encoding=encoding,
                    engine="python",
                    on_bad_lines="skip",
                )
            except Exception as e2:
                raise RuntimeError(f"Failed to parse CSV file at {path}: {e2}") from e2

        return df

    def describe(self) -> dict[str, Any]:
        return {
            "source_type": "csv",
            "supported_extensions": [".csv", ".tsv", ".txt"],
            "version": "1.0.0",
        }

    def _detect_encoding(self, path: Path) -> str:
        with open(path, "rb") as f:
            sample = f.read(8192)
        try:
            sample.decode("utf-8")
            return "utf-8"
        except UnicodeDecodeError:
            try:
                sample.decode("latin-1")
                return "latin-1"
            except Exception:
                return "utf-8"

    def _detect_delimiter(self, path: Path, encoding: str) -> str:
        try:
            with open(path, "r", encoding=encoding, errors="replace") as f:
                sample = "".join(f.readline() for _ in range(25))
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample, delimiters=",\t;|")
            return dialect.delimiter
        except Exception:
            return ","
