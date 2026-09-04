"""
agents/report/report_agent.py
=============================
Agent 6: Report Agent.
Transforms the verified Dataset Intelligence Object (DIO) produced by upstream
Agents 1–5 into professional executive PDF and PowerPoint (PPTX) deliverables.
Strict downstream consumer: zero recalculation, zero raw row access, zero LLM calls.
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
from core.logger import get_logger
from core.persistence import resolve_run_path
from agents.report.pdf_generator import generate_pdf_report
from agents.report.pptx_generator import generate_pptx_report

logger = get_logger("agents.report")


class ReportAgent(BaseAgent):
    """
    Agent 6: Executive Report & Presentation Generator.
    Produces high-fidelity PDF and PPTX deliverables directly from DIO facts.
    """
    name: str = "report"

    def __init__(self, config: AppConfig | None = None) -> None:
        super().__init__()
        self.config = config or load_config()

    def run(
        self,
        df: pd.DataFrame | None,
        dio: DIO | dict[str, Any],
        run_dir: Path | str | None = None,
    ) -> tuple[pd.DataFrame | None, DIO | dict[str, Any]]:
        """
        Execute the Report Agent pipeline.
        Consumes structured DIO metadata and produces PDF and PPTX artifacts.
        
        Mutates ONLY:
        - dio["reports"]
        - dio["artifacts"]["pdf_report"]
        - dio["artifacts"]["pptx_report"]
        - dio["progress"]["report"]
        - dio["agent_metrics"]
        - dio["decision_log"]
        - dio["errors"] (if an error occurs)
        """
        start_time = time.time()
        logger.info(f"ReportAgent: Starting report compilation for {dio.get('file_name', 'dataset')}")
        dio["progress"][self.name] = ProgressState.RUNNING.value

        warnings: list[str] = []
        errors_count = 0
        pdf_path_str: str | None = None
        pptx_path_str: str | None = None

        # 1. Determine Artifact Target Directory
        try:
            if run_dir is not None:
                target_dir = Path(run_dir).resolve()
            else:
                base_runs = Path("runs").resolve()
                ds_id = str(dio.get("dataset_id", "default_run"))
                target_dir = resolve_run_path(base_runs, ds_id)

            target_dir.mkdir(parents=True, exist_ok=True)
            artifacts_dir = target_dir / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            pdf_file = artifacts_dir / "report.pdf"
            pptx_file = artifacts_dir / "presentation.pptx"

        except Exception as e:
            logger.error(f"ReportAgent: Failed to initialize artifact directory: {e}")
            dio["errors"].append({
                "agent": self.name,
                "code": "RPT_001",
                "message": f"Artifact directory initialization failed: {str(e)}",
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            })
            dio["progress"][self.name] = ProgressState.FAILED.value
            return df, dio

        # 2. Generate PDF Report (with error boundary)
        try:
            logger.info("ReportAgent: Generating PDF report...")
            generated_pdf = generate_pdf_report(dio, pdf_file)
            if generated_pdf.is_file() and generated_pdf.stat().st_size > 0:
                pdf_path_str = str(generated_pdf)
                logger.info(f"ReportAgent: PDF generated successfully ({generated_pdf.stat().st_size} bytes)")
            else:
                raise RuntimeError("PDF file was created but appears empty or invalid")
        except Exception as e:
            errors_count += 1
            err_msg = f"PDF compilation failed: {str(e)}"
            logger.warning(f"ReportAgent: {err_msg}")
            warnings.append(err_msg)
            dio["errors"].append({
                "agent": self.name,
                "code": "RPT_001",
                "message": err_msg,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            })

        # 3. Generate PPTX Presentation (with independent error boundary)
        try:
            logger.info("ReportAgent: Generating PowerPoint presentation...")
            generated_pptx = generate_pptx_report(dio, pptx_file)
            if generated_pptx.is_file() and generated_pptx.stat().st_size > 0:
                pptx_path_str = str(generated_pptx)
                logger.info(f"ReportAgent: PPTX generated successfully ({generated_pptx.stat().st_size} bytes)")
            else:
                raise RuntimeError("PPTX file was created but appears empty or invalid")
        except Exception as e:
            errors_count += 1
            err_msg = f"PowerPoint presentation generation failed: {str(e)}"
            logger.warning(f"ReportAgent: {err_msg}")
            warnings.append(err_msg)
            dio["errors"].append({
                "agent": self.name,
                "code": "RPT_002",
                "message": err_msg,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            })

        # 4. Determine Composite Delivery Status
        if pdf_path_str and pptx_path_str:
            status = "completed"
        elif pdf_path_str or pptx_path_str:
            status = "partial"
        else:
            status = "failed"

        # 5. Populate DIO Reports & Artifacts
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        dio["reports"] = {
            "status": status,
            "pdf_path": pdf_path_str,
            "pptx_path": pptx_path_str,
            "generated_at": now_iso,
            "warnings": warnings,
        }

        # Update DIO artifacts section
        if "artifacts" in dio and isinstance(dio["artifacts"], dict):
            dio["artifacts"]["pdf_report"] = pdf_path_str
            dio["artifacts"]["pptx_report"] = pptx_path_str

        # 6. Record Provenance Decision Log
        dio["decision_log"].append({
            "agent": self.name,
            "action": "reports_generated",
            "status": status,
            "pdf_generated": bool(pdf_path_str),
            "pptx_generated": bool(pptx_path_str),
            "timestamp": now_iso,
        })

        # 7. Record Agent Metrics
        elapsed = time.time() - start_time
        dio["agent_metrics"].append({
            "agent": self.name,
            "runtime_seconds": round(elapsed, 4),
            "warnings": len(warnings),
            "errors": errors_count,
            "status": status,
        })

        # Set final progress state
        if status == "failed":
            dio["progress"][self.name] = ProgressState.FAILED.value
        else:
            dio["progress"][self.name] = ProgressState.FINISHED.value

        logger.info(f"ReportAgent: Execution finished with status '{status}' in {elapsed:.3f}s")
        return df, dio
