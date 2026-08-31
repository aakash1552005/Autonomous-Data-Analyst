"""
agents/cleaning/cleaning_agent.py
=================================
Agent 2: Cleaning Agent.
Orchestrates deterministic data cleaning, missing value imputation, duplicate removal,
type coercion, date normalization/preservation, and IQR outlier detection.
Generates reversible cleaning logs and artifact sidecars (cleaned_data.csv, removed_rows.csv).
"""

from __future__ import annotations

import datetime
from pathlib import Path
import time
from typing import Any
import pandas as pd

from core.base_agent import BaseAgent, ProgressState
from core.dio import DIO
from core.hashing import compute_file_hash
from core.persistence import resolve_run_path
from agents.cleaning.duplicate_handler import handle_duplicate_rows
from agents.cleaning.type_coercer import coerce_dataframe_types
from agents.cleaning.date_handler import handle_all_dates
from agents.cleaning.imputer import impute_dataframe
from agents.cleaning.outlier_detector import detect_dataframe_outliers


class CleaningAgent(BaseAgent):
    """
    Agent 2: Deterministic, transparent, and reversible tabular data cleaning.
    """
    name = "cleaning"

    def run(
        self,
        df: pd.DataFrame,
        dio: DIO | dict[str, Any],
        run_dir: Path | str | None = None,
    ) -> tuple[pd.DataFrame, DIO | dict[str, Any]]:
        """
        Execute Cleaning Agent pipeline.
        Consumes raw DataFrame + Phase 3 DIO.
        Produces cleaned DataFrame + structured cleaning_log + persistent artifacts.
        """
        start_time = time.time()
        dio["progress"][self.name] = ProgressState.RUNNING.value
        warnings_count = 0
        errors_count = 0

        rows_before = len(df)
        cols_before = len(df.columns)

        try:
            # 1. Duplicate Row Detection & Sidecar Separation
            df_no_dups, removed_rows_df, dup_log = handle_duplicate_rows(df)

            # 2. Inferred Type Coercion
            coerced_df, type_logs = coerce_dataframe_types(
                df=df_no_dups,
                columns_info=dio.get("columns", []),
            )

            # 3. Date Normalization / Ambiguous Preservation
            dated_df, date_logs = handle_all_dates(
                df=coerced_df,
                date_columns_info=dio.get("date_columns", []),
            )
            for d_log in date_logs:
                if d_log.get("warning"):
                    warnings_count += 1

            # 4. Missing Value Imputation (Numeric Skewness & Categorical Thresholds)
            cleaned_df, impute_logs = impute_dataframe(
                df=dated_df,
                columns_info=dio.get("columns", []),
            )

            # 5. Outlier Detection (Flag-only, zero deletions/modifications)
            outlier_logs = detect_dataframe_outliers(
                df=cleaned_df,
                columns_info=dio.get("columns", []),
            )

            # 6. Consolidate Cleaning Log
            all_logs: list[dict[str, Any]] = []
            if dup_log is not None:
                all_logs.append(dup_log)
            all_logs.extend(type_logs)
            all_logs.extend(date_logs)
            all_logs.extend(impute_logs)
            all_logs.extend(outlier_logs)

            dio["cleaning_log"] = all_logs

            # 7. Determine Run Directory & Save Artifacts
            if run_dir is not None:
                target_dir = Path(run_dir).resolve()
            else:
                base_runs = Path("runs").resolve()
                ds_id = str(dio.get("dataset_id", "default_run"))
                target_dir = resolve_run_path(base_runs, ds_id)

            target_dir.mkdir(parents=True, exist_ok=True)
            artifacts_dir = target_dir / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            cleaned_csv_path = artifacts_dir / "cleaned_data.csv"
            removed_rows_path = artifacts_dir / "removed_rows.csv"

            # Save cleaned dataset and removed rows sidecar
            cleaned_df.to_csv(cleaned_csv_path, index=False, encoding="utf-8")
            removed_rows_df.to_csv(removed_rows_path, index=False, encoding="utf-8")

            cleaned_hash = compute_file_hash(cleaned_csv_path)

            dio["artifacts"]["cleaned_csv"] = str(cleaned_csv_path)
            dio["artifacts"]["removed_rows_csv"] = str(removed_rows_path)

            # 8. Record Provenance & Decision Log
            dio["decision_log"].append({
                "agent": self.name,
                "action": "dataset_cleaned",
                "rows_before": rows_before,
                "rows_after": len(cleaned_df),
                "cols_before": cols_before,
                "cols_after": len(cleaned_df.columns),
                "duplicates_removed": len(removed_rows_df),
                "cleaned_dataset_hash": cleaned_hash,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            })

            dio["progress"][self.name] = ProgressState.FINISHED.value

        except Exception as e:
            errors_count += 1
            dio["progress"][self.name] = ProgressState.FAILED.value
            dio["errors"].append({
                "agent": self.name,
                "code": "CLN_001",
                "message": f"Cleaning agent failed: {str(e)}",
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            })
            raise

        finally:
            runtime_seconds = round(time.time() - start_time, 3)
            dio["agent_metrics"].append({
                "agent": self.name,
                "runtime_seconds": runtime_seconds,
                "warnings": warnings_count,
                "errors": errors_count,
            })

        return cleaned_df, dio
