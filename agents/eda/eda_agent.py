"""
agents/eda/eda_agent.py
=======================
Agent 3: Exploratory Data Analysis (EDA) Agent.
Performs summary statistics computation, Pearson and Spearman correlation analysis,
and deterministic Plotly + Kaleido chart generation.
Zero LLM calls, zero network dependencies, and strict DIO boundary isolation.
"""

from __future__ import annotations

import datetime
from pathlib import Path
import time
from typing import Any
import pandas as pd

from core.base_agent import BaseAgent, ProgressState
from core.config import AppConfig, load_config
from core.dio import DIO
from core.persistence import resolve_run_path
from agents.eda.summary_stats import compute_dataset_summary_stats
from agents.eda.correlations import compute_correlations
from agents.eda.chart_generator import generate_all_eda_charts


class EDAAgent(BaseAgent):
    """
    Agent 3: Deterministic Exploratory Data Analysis and static chart visualization.
    """
    name = "eda"

    def __init__(self, config: AppConfig | None = None) -> None:
        super().__init__()
        self.config = config or load_config()

    def run(
        self,
        df: pd.DataFrame,
        dio: DIO | dict[str, Any],
        run_dir: Path | str | None = None,
    ) -> tuple[pd.DataFrame, DIO | dict[str, Any]]:
        """
        Execute EDA pipeline.
        Consumes cleaned DataFrame + Phase 4 DIO.
        Populates ONLY dio['eda'] and dio['artifacts']['chart_paths'].
        """
        start_time = time.time()
        dio["progress"][self.name] = ProgressState.RUNNING.value
        warnings_count = 0
        errors_count = 0

        try:
            # 1. Determine Chart Artifact Directory
            if run_dir is not None:
                target_dir = Path(run_dir).resolve()
            else:
                base_runs = Path("runs").resolve()
                ds_id = str(dio.get("dataset_id", "default_run"))
                target_dir = resolve_run_path(base_runs, ds_id)

            target_dir.mkdir(parents=True, exist_ok=True)
            charts_dir = target_dir / "artifacts" / "charts"
            charts_dir.mkdir(parents=True, exist_ok=True)

            # 2. Compute Summary Statistics
            columns_info = dio.get("columns", [])
            summary_stats = compute_dataset_summary_stats(df, columns_info=columns_info)

            # 3. Compute Pearson and Spearman Correlations
            correlations = compute_correlations(df, columns_info=columns_info)

            # 4. Generate Deterministic Plotly / Kaleido Charts (Zero LLM)
            max_charts = self.config.eda.max_charts
            chart_paths = generate_all_eda_charts(
                df=df,
                columns_info=columns_info,
                date_columns_info=dio.get("date_columns", []),
                correlations_info=correlations,
                output_dir=charts_dir,
                max_charts=max_charts,
            )

            # 5. Populate DIO EDA & Artifacts Sections
            dio["eda"] = {
                "summary_stats": summary_stats,
                "correlations": correlations,
                "charts": chart_paths,
            }
            dio["artifacts"]["chart_paths"] = chart_paths

            # 6. Record Provenance
            dio["decision_log"].append({
                "agent": self.name,
                "action": "eda_completed",
                "charts_generated": len(chart_paths),
                "max_charts_configured": max_charts,
                "numeric_columns_analyzed": len(correlations.get("numeric_columns_analyzed", [])),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            })

            dio["progress"][self.name] = ProgressState.FINISHED.value

        except Exception as e:
            errors_count += 1
            dio["progress"][self.name] = ProgressState.FAILED.value
            dio["errors"].append({
                "agent": self.name,
                "code": "EDA_001",
                "message": f"EDA agent failed: {str(e)}",
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
