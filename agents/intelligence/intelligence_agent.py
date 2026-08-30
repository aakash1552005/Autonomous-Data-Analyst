"""
agents/intelligence/intelligence_agent.py
=========================================
Agent 1: Intelligence Agent.
Orchestrates schema profiling, date resolution, semantic column labeling,
PII detection, domain classification, and data quality scoring.
Mutates only the designated Intelligence sections of the Dataset Intelligence Object (DIO).
"""

from __future__ import annotations

import datetime
import time
from typing import Any
import pandas as pd

from core.base_agent import BaseAgent, ProgressState
from core.dio import DIO
from llm.base import LLMProvider
from agents.intelligence.schema_profiler import profile_schema
from agents.intelligence.date_resolver import resolve_all_dates
from agents.intelligence.pii_detector import detect_column_pii
from agents.intelligence.semantic_labeler import SemanticLabeler
from agents.intelligence.domain_classifier import classify_domain
from agents.intelligence.quality_scorer import calculate_quality_score


class IntelligenceAgent(BaseAgent):
    """
    Agent 1: Deep tabular intelligence, profiling, and metadata extraction.
    """
    name = "intelligence"

    def __init__(self, llm_provider: LLMProvider | None = None) -> None:
        super().__init__()
        self.llm_provider = llm_provider
        self.semantic_labeler = SemanticLabeler(llm_provider=llm_provider)

    def run(self, df: pd.DataFrame, dio: DIO | dict[str, Any]) -> tuple[pd.DataFrame, DIO | dict[str, Any]]:
        """
        Execute Agent 1 pipeline and populate the Intelligence sections of DIO.
        """
        start_time = time.time()
        dio["progress"][self.name] = ProgressState.RUNNING.value
        warnings_count = 0
        errors_count = 0

        try:
            # 1. Structural Schema Profiling
            schema = profile_schema(df)
            raw_columns = schema["columns"]
            duplicate_row_pct = schema["duplicate_row_pct"]

            # Update ingestion record if not already set
            if not dio.get("ingestion") or dio["ingestion"].get("n_rows", 0) == 0:
                dio["ingestion"] = {
                    "n_rows": schema["n_rows"],
                    "n_columns": schema["n_columns"],
                    "file_type": dio.get("ingestion", {}).get("file_type", "csv"),
                    "encoding": dio.get("ingestion", {}).get("encoding", "utf-8"),
                }

            # 2. Date Format Resolution
            date_columns = resolve_all_dates(df)
            dio["date_columns"] = date_columns

            for d_info in date_columns:
                dio["decision_log"].append({
                    "agent": self.name,
                    "action": "date_resolved",
                    "column": d_info["column"],
                    "detected_format": d_info["detected_format"],
                    "confidence": d_info["confidence"],
                    "evidence": d_info["evidence"],
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                })
                if d_info.get("needs_user_confirmation"):
                    warnings_count += 1

            # 3. PII Detection & Semantic Labeling (Per Column)
            final_columns_info: list[dict[str, Any]] = []
            for col_meta in raw_columns:
                col_name = col_meta["name"]
                series = df[col_name]

                # Step A: PII Detection (Must happen before any LLM call)
                is_pii, pii_type = detect_column_pii(col_name, series)
                if is_pii:
                    dio["decision_log"].append({
                        "agent": self.name,
                        "action": "pii_detected",
                        "column": col_name,
                        "pii_type": pii_type,
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    })

                # Step B: Semantic Labeling (Rule -> Pattern -> Masked LLM)
                label, conf, method = self.semantic_labeler.label_column(
                    col_name=col_name,
                    series=series,
                    is_pii=is_pii,
                    pii_type=pii_type,
                )

                if conf < 0.70:
                    warnings_count += 1

                # Step C: Target Candidate Heuristic
                is_target_candidate = label in ("target_label", "churn", "default", "status", "outcome") or (
                    col_meta["dtype_inferred"] in ("category", "int", "float")
                    and col_meta["unique_count"] >= 2
                    and col_meta["unique_count"] <= 10
                    and not is_pii
                    and col_meta["null_pct"] < 0.20
                )

                col_record = {
                    "name": col_name,
                    "dtype_raw": col_meta["dtype_raw"],
                    "dtype_inferred": col_meta["dtype_inferred"],
                    "semantic_label": label,
                    "confidence": conf,
                    "method_used": method,
                    "is_pii": is_pii,
                    "pii_type": pii_type,
                    "null_pct": col_meta["null_pct"],
                    "unique_count": col_meta["unique_count"],
                    "is_target_candidate": is_target_candidate,
                }
                final_columns_info.append(col_record)

                dio["decision_log"].append({
                    "agent": self.name,
                    "action": "semantic_labeled",
                    "column": col_name,
                    "label": label,
                    "confidence": conf,
                    "method": method,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                })

            dio["columns"] = final_columns_info

            # 4. Domain Classification
            domain_info = classify_domain(final_columns_info)
            dio["domain_guess"] = domain_info

            dio["decision_log"].append({
                "agent": self.name,
                "action": "domain_classified",
                "domain": domain_info["domain"],
                "confidence": domain_info["confidence"],
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            })

            # 5. Quality Assessment
            quality_info = calculate_quality_score(
                columns_info=final_columns_info,
                date_columns_info=date_columns,
                duplicate_row_pct=duplicate_row_pct,
            )
            dio["quality"] = {
                "score": quality_info["score"],
                "issues": quality_info["issues"],
            }

            dio["progress"][self.name] = ProgressState.FINISHED.value

        except Exception as e:
            errors_count += 1
            dio["progress"][self.name] = ProgressState.FAILED.value
            dio["errors"].append({
                "agent": self.name,
                "code": "INT_001",
                "message": f"Intelligence agent failed: {str(e)}",
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

        return df, dio
