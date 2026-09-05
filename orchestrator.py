"""
orchestrator.py
===============
Phase 9: Pipeline Orchestrator for Autonomous Data Analyst.

Sequences the end-to-end analytical workflow:
  Stage 1: Validation & Ingestion (DataRouter) [FATAL]
  Stage 2: Intelligence Profiling (IntelligenceAgent) [RECOVERABLE]
  Stage 3: Data Cleaning (CleaningAgent) [FATAL]
  Stage 4: Exploratory Data Analysis & Viz (EDAAgent) [RECOVERABLE]
  Stage 5: Machine Learning Modeling (MLAgent) [RECOVERABLE]
  Stage 6: Executive Insights & Narrative (InsightAgent) [RECOVERABLE]
  Stage 7: Executive Report Generation (ReportAgent) [RECOVERABLE]

Enforces DIO data contract integrity, stage isolation, structured progress callbacks,
failure classification (fatal vs recoverable), and filesystem artifact persistence.
"""

from __future__ import annotations

import datetime
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from core.base_agent import ProgressState
from core.config import AppConfig, load_config
from core.data_router import DataRouter
from core.dio import DIO
from core.logger import setup_run_logger
from core.persistence import create_run_directory, resolve_run_path, save_dio_json
from llm.base import LLMProvider
from security.file_validator import FileValidationError

# Agents
from agents.intelligence.intelligence_agent import IntelligenceAgent
from agents.cleaning.cleaning_agent import CleaningAgent
from agents.eda.eda_agent import EDAAgent
from agents.ml.ml_agent import MLAgent
from agents.insight.insight_agent import InsightAgent
from agents.report.report_agent import ReportAgent

module_logger = logging.getLogger("autonomous_data_analyst.orchestrator")


@dataclass
class StageInfo:
    """Metadata describing a single pipeline stage."""
    name: str
    label: str
    target_progress: float
    is_fatal: bool


PIPELINE_STAGES: list[StageInfo] = [
    StageInfo("validation", "File Ingestion & Validation", 0.15, is_fatal=True),
    StageInfo("intelligence", "Dataset Intelligence", 0.30, is_fatal=False),
    StageInfo("cleaning", "Data Cleaning", 0.45, is_fatal=True),
    StageInfo("eda", "Exploratory Data Analysis", 0.60, is_fatal=False),
    StageInfo("ml", "Machine Learning Modeling", 0.75, is_fatal=False),
    StageInfo("insight", "Executive Insights", 0.90, is_fatal=False),
    StageInfo("report", "Report & Deliverable Generation", 1.00, is_fatal=False),
]


@dataclass
class OrchestratorResult:
    """
    Structured outcome of an orchestrator pipeline execution.
    """
    status: str  # "completed" | "partial" | "failed"
    run_dir: Path | None = None
    dio: DIO | None = None
    df: pd.DataFrame | None = None
    stage_timings: dict[str, float] = field(default_factory=dict)
    stage_statuses: dict[str, str] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.status in ("completed", "partial")


ProgressCallbackType = Callable[[str, float, str], None]


class Orchestrator:
    """
    Pipeline orchestrator that coordinates all 7 analytical agents.
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        llm_provider: LLMProvider | None = None,
        progress_callback: ProgressCallbackType | None = None,
        agents: dict[str, Any] | None = None,
    ) -> None:
        self.config = config or load_config()
        self.llm_provider = llm_provider
        self.progress_callback = progress_callback
        self.custom_agents = agents or {}

    def _notify(
        self,
        callback: ProgressCallbackType | None,
        stage: str,
        pct: float,
        message: str,
    ) -> None:
        """Invoke progress callback if registered."""
        cb = callback or self.progress_callback
        if cb is not None:
            try:
                cb(stage, pct, message)
            except Exception as e:
                module_logger.warning(f"Progress callback raised exception: {e}")

    def run(
        self,
        file_path: str | Path,
        run_dir: str | Path | None = None,
        preferred_target: str | None = None,
        progress_callback: ProgressCallbackType | None = None,
    ) -> OrchestratorResult:
        """
        Execute the complete autonomous analysis pipeline.

        Parameters:
            file_path: Path to dataset (.csv or .xlsx)
            run_dir: Optional explicit run directory. If None, created under config.pipeline.runs_dir.
            preferred_target: Optional user-specified ML target column name.
            progress_callback: Optional callable(stage_name, progress_pct, message).

        Returns:
            OrchestratorResult detailing execution status, DIO, timings, and artifacts.
        """
        input_path = Path(file_path).resolve()
        start_wall_time = time.time()

        stage_timings: dict[str, float] = {}
        stage_statuses: dict[str, str] = {stage.name: "pending" for stage in PIPELINE_STAGES}
        collected_errors: list[dict[str, Any]] = []

        overall_status = "completed"
        dio: DIO | None = None
        df: pd.DataFrame | None = None
        actual_run_dir: Path | None = None
        run_logger: Any = logging.LoggerAdapter(module_logger, {"run_id": "pre-run"})

        # Resolve or prepare run directory
        if run_dir is not None:
            actual_run_dir = Path(run_dir).resolve()
            actual_run_dir.mkdir(parents=True, exist_ok=True)
            (actual_run_dir / "charts").mkdir(exist_ok=True)

        # -------------------------------------------------------------
        # STAGE 1: Validation & Ingestion (FATAL)
        # -------------------------------------------------------------
        self._notify(progress_callback, "validation", 0.05, f"Validating and ingesting {input_path.name}...")
        t0 = time.time()
        stage_statuses["validation"] = "running"

        try:
            router = DataRouter(max_upload_size_mb=self.config.max_upload_size_mb)
            base_runs = None if actual_run_dir is not None else self.config.pipeline.runs_dir
            df, dio, created_run_dir = router.ingest(input_path, base_runs_dir=base_runs)

            if actual_run_dir is None and created_run_dir is not None:
                actual_run_dir = created_run_dir

            # Initialize per-run logger
            dataset_id = str(dio.get("dataset_id", "run"))
            run_logger = setup_run_logger(
                run_id=dataset_id,
                run_dir=actual_run_dir,
                log_level=getattr(logging, self.config.pipeline.log_level, logging.INFO),
            )
            run_logger.info(f"Pipeline execution started for dataset '{input_path.name}' (run_id={dataset_id})")

            stage_timings["validation"] = round(time.time() - t0, 3)
            stage_statuses["validation"] = "completed"
            run_logger.info(f"Stage 1 [validation] completed in {stage_timings['validation']}s: {len(df)} rows, {len(df.columns)} columns.")
            self._notify(progress_callback, "validation", 0.15, "File validation and ingestion successful.")

        except Exception as exc:
            duration = round(time.time() - t0, 3)
            stage_timings["validation"] = duration
            stage_statuses["validation"] = "failed"
            err_dict = {
                "stage": "validation",
                "agent": "data_router",
                "error": str(exc),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            collected_errors.append(err_dict)
            run_logger.error(f"Fatal validation error on '{input_path.name}': {exc}", exc_info=True)
            self._notify(progress_callback, "validation", 0.15, f"Validation failed: {exc}")

            return OrchestratorResult(
                status="failed",
                run_dir=actual_run_dir,
                dio=dio,
                df=None,
                stage_timings=stage_timings,
                stage_statuses=stage_statuses,
                errors=collected_errors,
                artifacts={},
            )

        # -------------------------------------------------------------
        # STAGE 2: Intelligence Agent (RECOVERABLE)
        # -------------------------------------------------------------
        self._notify(progress_callback, "intelligence", 0.18, "Analyzing dataset schema, semantics, and domain...")
        t0 = time.time()
        stage_statuses["intelligence"] = "running"
        run_logger.info("Stage 2 [intelligence] starting...")

        try:
            agent_intel = self.custom_agents.get(
                "intelligence",
                IntelligenceAgent(llm_provider=self.llm_provider),
            )
            df, dio = agent_intel.run(df, dio)
            stage_timings["intelligence"] = round(time.time() - t0, 3)
            stage_statuses["intelligence"] = "completed"
            run_logger.info(f"Stage 2 [intelligence] completed in {stage_timings['intelligence']}s: domain={dio.get('domain_guess', {}).get('domain')}, quality={dio.get('quality', {}).get('score')}/100.")
            self._notify(progress_callback, "intelligence", 0.30, "Intelligence profiling completed.")
        except Exception as exc:
            duration = round(time.time() - t0, 3)
            stage_timings["intelligence"] = duration
            stage_statuses["intelligence"] = "failed"
            overall_status = "partial"

            err_dict = {
                "stage": "intelligence",
                "agent": "intelligence",
                "error": str(exc),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            collected_errors.append(err_dict)
            dio["errors"].append(err_dict)
            dio["progress"]["intelligence"] = ProgressState.FAILED.value
            run_logger.error(f"Stage 2 [intelligence] failed: {exc}", exc_info=True)
            self._notify(progress_callback, "intelligence", 0.30, f"Intelligence stage failed (continuing): {exc}")

        # -------------------------------------------------------------
        # STAGE 3: Cleaning Agent (FATAL)
        # -------------------------------------------------------------
        self._notify(progress_callback, "cleaning", 0.33, "Executing deterministic data cleaning...")
        t0 = time.time()
        stage_statuses["cleaning"] = "running"
        run_logger.info("Stage 3 [cleaning] starting...")

        try:
            agent_cleaning = self.custom_agents.get("cleaning", CleaningAgent())
            df, dio = agent_cleaning.run(df, dio, run_dir=actual_run_dir)
            stage_timings["cleaning"] = round(time.time() - t0, 3)
            stage_statuses["cleaning"] = "completed"
            run_logger.info(f"Stage 3 [cleaning] completed in {stage_timings['cleaning']}s: {len(dio.get('cleaning_log', []))} transformations recorded.")
            self._notify(progress_callback, "cleaning", 0.45, "Data cleaning completed successfully.")
        except Exception as exc:
            duration = round(time.time() - t0, 3)
            stage_timings["cleaning"] = duration
            stage_statuses["cleaning"] = "failed"
            err_dict = {
                "stage": "cleaning",
                "agent": "cleaning",
                "error": str(exc),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            collected_errors.append(err_dict)
            dio["errors"].append(err_dict)
            dio["progress"]["cleaning"] = ProgressState.FAILED.value
            run_logger.error(f"Fatal cleaning error: {exc}", exc_info=True)
            self._notify(progress_callback, "cleaning", 0.45, f"Cleaning failed: {exc}")

            if actual_run_dir is not None and self.config.pipeline.save_dio_json:
                save_dio_json(dio, actual_run_dir / "dio.json")

            return OrchestratorResult(
                status="failed",
                run_dir=actual_run_dir,
                dio=dio,
                df=df,
                stage_timings=stage_timings,
                stage_statuses=stage_statuses,
                errors=collected_errors,
                artifacts=dio.get("artifacts", {}),
            )

        # -------------------------------------------------------------
        # STAGE 4: Exploratory Data Analysis & Viz (RECOVERABLE)
        # -------------------------------------------------------------
        self._notify(progress_callback, "eda", 0.48, "Generating statistical profiles and visualizations...")
        t0 = time.time()
        stage_statuses["eda"] = "running"
        run_logger.info("Stage 4 [eda] starting...")

        try:
            agent_eda = self.custom_agents.get("eda", EDAAgent(config=self.config))
            df, dio = agent_eda.run(df, dio, run_dir=actual_run_dir)
            stage_timings["eda"] = round(time.time() - t0, 3)
            stage_statuses["eda"] = "completed"
            n_charts = len(dio.get("artifacts", {}).get("chart_paths", []))
            run_logger.info(f"Stage 4 [eda] completed in {stage_timings['eda']}s: {n_charts} chart artifacts generated.")
            self._notify(progress_callback, "eda", 0.60, "EDA charts and statistics generated.")
        except Exception as exc:
            duration = round(time.time() - t0, 3)
            stage_timings["eda"] = duration
            stage_statuses["eda"] = "failed"
            overall_status = "partial"

            err_dict = {
                "stage": "eda",
                "agent": "eda",
                "error": str(exc),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            collected_errors.append(err_dict)
            dio["errors"].append(err_dict)
            dio["progress"]["eda"] = ProgressState.FAILED.value
            run_logger.error(f"Stage 4 [eda] failed: {exc}", exc_info=True)
            self._notify(progress_callback, "eda", 0.60, f"EDA stage failed (continuing): {exc}")

        # -------------------------------------------------------------
        # STAGE 5: Machine Learning Agent (RECOVERABLE)
        # -------------------------------------------------------------
        self._notify(progress_callback, "ml", 0.63, "Evaluating candidate machine learning models...")
        t0 = time.time()
        stage_statuses["ml"] = "running"
        run_logger.info("Stage 5 [ml] starting...")

        try:
            agent_ml = self.custom_agents.get("ml", MLAgent(config=self.config))
            df, dio = agent_ml.run(df, dio, run_dir=actual_run_dir, preferred_target=preferred_target)
            stage_timings["ml"] = round(time.time() - t0, 3)

            ml_prog = dio.get("progress", {}).get("ml")
            if ml_prog == ProgressState.SKIPPED.value:
                stage_statuses["ml"] = "skipped"
                run_logger.info(f"Stage 5 [ml] completed (skipped) in {stage_timings['ml']}s: no eligible target identified.")
                self._notify(progress_callback, "ml", 0.75, "Machine learning skipped (no eligible target).")
            else:
                stage_statuses["ml"] = "completed"
                run_logger.info(f"Stage 5 [ml] completed in {stage_timings['ml']}s: best_model={dio.get('ml', {}).get('best_model')}.")
                self._notify(progress_callback, "ml", 0.75, "Machine learning modeling completed.")
        except Exception as exc:
            duration = round(time.time() - t0, 3)
            stage_timings["ml"] = duration
            stage_statuses["ml"] = "failed"
            overall_status = "partial"

            err_dict = {
                "stage": "ml",
                "agent": "ml",
                "error": str(exc),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            collected_errors.append(err_dict)
            dio["errors"].append(err_dict)
            dio["progress"]["ml"] = ProgressState.FAILED.value
            run_logger.error(f"Stage 5 [ml] failed: {exc}", exc_info=True)
            self._notify(progress_callback, "ml", 0.75, f"ML stage failed (continuing): {exc}")

        # -------------------------------------------------------------
        # STAGE 6: Insight Agent (RECOVERABLE)
        # -------------------------------------------------------------
        self._notify(progress_callback, "insight", 0.78, "Synthesizing executive insights and narratives...")
        t0 = time.time()
        stage_statuses["insight"] = "running"
        run_logger.info("Stage 6 [insight] starting...")

        try:
            agent_insight = self.custom_agents.get(
                "insight",
                InsightAgent(config=self.config, llm_provider=self.llm_provider),
            )
            df, dio = agent_insight.run(df, dio, run_dir=actual_run_dir)
            stage_timings["insight"] = round(time.time() - t0, 3)
            stage_statuses["insight"] = "completed"
            n_insights = len(dio.get("insights", []))
            run_logger.info(f"Stage 6 [insight] completed in {stage_timings['insight']}s: {n_insights} verified insights synthesized.")
            self._notify(progress_callback, "insight", 0.90, "Executive insights synthesized.")
        except Exception as exc:
            duration = round(time.time() - t0, 3)
            stage_timings["insight"] = duration
            stage_statuses["insight"] = "failed"
            overall_status = "partial"

            err_dict = {
                "stage": "insight",
                "agent": "insight",
                "error": str(exc),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            collected_errors.append(err_dict)
            dio["errors"].append(err_dict)
            dio["progress"]["insight"] = ProgressState.FAILED.value
            run_logger.error(f"Stage 6 [insight] failed: {exc}", exc_info=True)
            self._notify(progress_callback, "insight", 0.90, f"Insight stage failed (continuing): {exc}")

        # -------------------------------------------------------------
        # STAGE 7: Report Agent (RECOVERABLE)
        # -------------------------------------------------------------
        self._notify(progress_callback, "report", 0.92, "Compiling PDF report and executive presentation...")
        t0 = time.time()
        stage_statuses["report"] = "running"
        run_logger.info("Stage 7 [report] starting...")

        try:
            agent_report = self.custom_agents.get("report", ReportAgent(config=self.config))
            df, dio = agent_report.run(df, dio, run_dir=actual_run_dir)
            stage_timings["report"] = round(time.time() - t0, 3)
            stage_statuses["report"] = "completed"
            run_logger.info(f"Stage 7 [report] completed in {stage_timings['report']}s: PDF & PPTX deliverables compiled.")
            self._notify(progress_callback, "report", 1.00, "Executive report and presentation generated.")
        except Exception as exc:
            duration = round(time.time() - t0, 3)
            stage_timings["report"] = duration
            stage_statuses["report"] = "failed"
            overall_status = "partial"

            err_dict = {
                "stage": "report",
                "agent": "report",
                "error": str(exc),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            collected_errors.append(err_dict)
            dio["errors"].append(err_dict)
            dio["progress"]["report"] = ProgressState.FAILED.value
            run_logger.error(f"Stage 7 [report] failed: {exc}", exc_info=True)
            self._notify(progress_callback, "report", 1.00, f"Report stage failed: {exc}")

        # -------------------------------------------------------------
        # Finalization & Persistence
        # -------------------------------------------------------------
        if actual_run_dir is not None and self.config.pipeline.save_dio_json:
            save_dio_json(dio, actual_run_dir / "dio.json")

        total_duration = round(time.time() - start_wall_time, 3)
        stage_timings["total_pipeline"] = total_duration

        final_artifacts = dio.get("artifacts", {}) if dio else {}

        run_logger.info(f"Pipeline execution completed with status '{overall_status}' in {total_duration}s.")

        # Flush all handlers to guarantee file persistence
        if hasattr(run_logger, "logger") and hasattr(run_logger.logger, "handlers"):
            for handler in run_logger.logger.handlers:
                try:
                    handler.flush()
                except Exception:
                    pass

        self._notify(
            progress_callback,
            "pipeline",
            1.00,
            f"Pipeline finished with status '{overall_status}' in {total_duration}s.",
        )

        return OrchestratorResult(
            status=overall_status,
            run_dir=actual_run_dir,
            dio=dio,
            df=df,
            stage_timings=stage_timings,
            stage_statuses=stage_statuses,
            errors=collected_errors,
            artifacts=final_artifacts,
        )
