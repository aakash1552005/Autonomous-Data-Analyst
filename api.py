"""
api.py
======
Lightweight Flask REST API for Autonomous Data Analyst.
Enables workflow automation (e.g., n8n, Zapier, Webhooks, CI/CD).

Endpoints:
  GET  /health                   - Service status and health check
  POST /analyze                  - Trigger analysis run (sync or async)
  GET  /runs                     - List recent analysis runs
  GET  /runs/<run_id>            - Get status and safe summary of a run
  GET  /runs/<run_id>/artifacts  - List generated artifacts for a run
  GET  /runs/<run_id>/artifacts/<filename> - Download an artifact file
  POST /chat                     - Interactive natural-language Q&A on run results
"""

from __future__ import annotations

import datetime
from datetime import timezone
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any
import uuid

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename
import pandas as pd

from core.config import AppConfig, load_config
from core.dio import DIO
from core.persistence import resolve_run_path
from orchestrator import Orchestrator, OrchestratorResult
from agents.chat.chat_agent import ChatAgent

logger = logging.getLogger("autonomous_data_analyst.api")

# Prohibited keys in request payloads to prevent injection attacks
PROHIBITED_PAYLOAD_KEYS = {
    "python_code",
    "shell_command",
    "sql_query",
    "eval_expression",
    "script",
    "command",
    "exec",
    "eval",
}

ALLOWED_EXTENSIONS = {".csv", ".xlsx"}


class RunRecord:
    """Thread-safe representation of an analysis run."""

    def __init__(
        self,
        run_id: str,
        file_path: str,
        status: str = "queued",
        preferred_target: str | None = None,
    ) -> None:
        self.run_id = run_id
        self.file_path = file_path
        self.preferred_target = preferred_target
        self.status = status  # queued | running | completed | partial | failed
        self.created_at = datetime.datetime.now(timezone.utc).isoformat()
        self.completed_at: str | None = None
        self.duration_seconds: float = 0.0
        self.run_dir: Path | None = None
        self.stage_statuses: dict[str, str] = {}
        self.stage_timings: dict[str, float] = {}
        self.errors: list[dict[str, Any]] = []
        self.artifacts: list[dict[str, Any]] = []
        self.dio: DIO | None = None
        self.df: pd.DataFrame | None = None

    def to_summary_dict(self) -> dict[str, Any]:
        """Safe summary representation with NO raw PII."""
        summary = {
            "run_id": self.run_id,
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "duration_seconds": round(self.duration_seconds, 2),
            "stage_statuses": self.stage_statuses,
            "error_count": len(self.errors),
            "artifact_count": len(self.artifacts),
        }
        if self.dio:
            domain_info = self.dio.get("domain_guess", {})
            quality_info = self.dio.get("quality_report", {})
            summary["domain"] = domain_info.get("domain", "unknown")
            summary["quality_score"] = quality_info.get("overall_score")
        return summary


class RunStore:
    """In-memory and filesystem-backed storage for pipeline runs."""

    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = runs_dir
        self.runs: dict[str, RunRecord] = {}
        self._lock = threading.Lock()

    def get(self, run_id: str) -> RunRecord | None:
        with self._lock:
            if run_id in self.runs:
                return self.runs[run_id]

        # Check if run exists on disk under runs_dir
        candidate_dir = self.runs_dir / run_id
        if not candidate_dir.exists():
            # Try searching directory with timestamp prefix
            matches = list(self.runs_dir.glob(f"*{run_id}*"))
            if matches:
                candidate_dir = matches[0]

        if candidate_dir.exists() and candidate_dir.is_dir():
            rec = RunRecord(run_id=run_id, file_path=str(candidate_dir))
            rec.run_dir = candidate_dir
            dio_file = candidate_dir / "dio.json"
            if dio_file.exists():
                try:
                    dio = DIO.load_json(dio_file)
                    rec.dio = dio
                    rec.status = dio.get("pipeline_status", "completed")
                    cleaned_csv = candidate_dir / "cleaned_data.csv"
                    if cleaned_csv.exists():
                        rec.df = pd.read_csv(cleaned_csv)
                except Exception as e:
                    logger.warning(f"Error loading dio from {dio_file}: {e}")
            rec.artifacts = self._scan_artifacts(candidate_dir)
            with self._lock:
                self.runs[run_id] = rec
            return rec

        return None

    def put(self, record: RunRecord) -> None:
        with self._lock:
            self.runs[record.run_id] = record

    def list_all(self) -> list[dict[str, Any]]:
        with self._lock:
            return [rec.to_summary_dict() for rec in reversed(list(self.runs.values()))]

    def _scan_artifacts(self, run_dir: Path) -> list[dict[str, Any]]:
        artifacts = []
        if not run_dir.exists():
            return artifacts
        for p in run_dir.rglob("*"):
            if p.is_file() and not p.name.startswith("."):
                rel = p.relative_to(run_dir)
                artifacts.append({
                    "filename": p.name,
                    "relative_path": str(rel).replace("\\", "/"),
                    "size_bytes": p.stat().st_size,
                })
        return artifacts


def create_app(config: AppConfig | None = None) -> Flask:
    app = Flask(__name__)
    cfg = config or load_config()

    project_root = Path(__file__).resolve().parent
    runs_dir = project_root / cfg.pipeline.runs_dir
    runs_dir.mkdir(parents=True, exist_ok=True)
    uploads_dir = runs_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    store = RunStore(runs_dir=runs_dir)
    chat_agent = ChatAgent(config=cfg)

    # ─── Endpoints ────────────────────────────────────────────────────────────

    @app.route("/health", methods=["GET"])
    def health() -> Any:
        return jsonify({
            "status": "ok",
            "version": "1.1.0",
            "service": "Autonomous Data Analyst API",
            "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
            "endpoints": [
                "GET /health",
                "POST /analyze",
                "GET /runs",
                "GET /runs/<run_id>",
                "GET /runs/<run_id>/artifacts",
                "GET /runs/<run_id>/artifacts/<filename>",
                "POST /chat",
            ],
        }), 200

    @app.route("/analyze", methods=["POST"])
    def analyze() -> Any:
        # 1. Payload Security Validation
        content_type = request.content_type or ""

        if "multipart/form-data" in content_type:
            # File upload
            if "file" not in request.files:
                return jsonify({"error": "No file uploaded in multipart form-data"}), 400
            uploaded = request.files["file"]
            if not uploaded.filename:
                return jsonify({"error": "Empty filename"}), 400

            ext = Path(uploaded.filename).suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                return jsonify({
                    "error": f"Unsupported file extension '{ext}'. Only CSV and XLSX are supported."
                }), 400

            safe_name = f"{uuid.uuid4().hex}_{secure_filename(uploaded.filename)}"
            target_path = uploads_dir / safe_name
            uploaded.save(target_path)
            file_path = target_path
            preferred_target = request.form.get("preferred_target")
            is_async = request.form.get("async", "false").lower() in ("true", "1", "yes")

        elif request.is_json:
            data = request.get_json(silent=True) or {}
            # Check for prohibited command injection keys
            for key in data.keys():
                if key.lower() in PROHIBITED_PAYLOAD_KEYS:
                    return jsonify({
                        "error": "Refusal: Untrusted code or command execution parameters are strictly prohibited."
                    }), 400

            raw_path = data.get("file_path")
            if not raw_path:
                return jsonify({"error": "Missing 'file_path' in JSON request"}), 400

            # Path Traversal Guard
            try:
                candidate = Path(raw_path).resolve()
            except Exception as e:
                return jsonify({"error": f"Invalid path syntax: {e}"}), 400

            # Must exist and be CSV/XLSX
            if not candidate.exists() or not candidate.is_file():
                return jsonify({"error": f"File not found: {raw_path}"}), 404

            ext = candidate.suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                return jsonify({
                    "error": f"Unsupported file extension '{ext}'. Only CSV and XLSX are supported."
                }), 400

            file_path = candidate
            preferred_target = data.get("preferred_target")
            is_async = bool(data.get("async", False))
        else:
            return jsonify({
                "error": "Unsupported Content-Type. Please use application/json or multipart/form-data."
            }), 415

        # 2. Setup Run Record
        run_id = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        record = RunRecord(
            run_id=run_id,
            file_path=str(file_path),
            status="queued",
            preferred_target=preferred_target,
        )
        store.put(record)

        def _execute_pipeline(rec: RunRecord) -> None:
            rec.status = "running"
            start_t = time.time()
            try:
                orchestrator = Orchestrator(config=cfg)
                res: OrchestratorResult = orchestrator.run(
                    file_path=rec.file_path,
                    run_dir=runs_dir / rec.run_id,
                    preferred_target=rec.preferred_target,
                )
                rec.status = res.status
                rec.run_dir = res.run_dir
                rec.dio = res.dio
                rec.df = res.df
                rec.stage_statuses = res.stage_statuses
                rec.stage_timings = res.stage_timings
                rec.errors = res.errors
                if res.run_dir:
                    rec.artifacts = store._scan_artifacts(res.run_dir)
            except Exception as e:
                logger.exception(f"Pipeline execution error for {rec.run_id}: {e}")
                rec.status = "failed"
                rec.errors.append({"stage": "orchestrator", "error": str(e)})
            finally:
                rec.duration_seconds = time.time() - start_t
                rec.completed_at = datetime.datetime.now(timezone.utc).isoformat()

        if is_async:
            thread = threading.Thread(target=_execute_pipeline, args=(record,), daemon=True)
            thread.start()
            return jsonify({
                "run_id": run_id,
                "status": "queued",
                "message": "Analysis initiated in background.",
                "poll_url": f"/runs/{run_id}",
            }), 202
        else:
            _execute_pipeline(record)
            response_data = record.to_summary_dict()
            response_data["artifacts"] = record.artifacts
            return jsonify(response_data), 200

    @app.route("/runs", methods=["GET"])
    def list_runs() -> Any:
        return jsonify({"runs": store.list_all()}), 200

    @app.route("/runs/<run_id>", methods=["GET"])
    def get_run(run_id: str) -> Any:
        record = store.get(run_id)
        if not record:
            return jsonify({"error": f"Run '{run_id}' not found"}), 404
        data = record.to_summary_dict()
        data["stage_timings"] = record.stage_timings
        data["errors"] = record.errors
        return jsonify(data), 200

    @app.route("/runs/<run_id>/artifacts", methods=["GET"])
    def get_run_artifacts(run_id: str) -> Any:
        record = store.get(run_id)
        if not record:
            return jsonify({"error": f"Run '{run_id}' not found"}), 404
        return jsonify({
            "run_id": run_id,
            "artifact_count": len(record.artifacts),
            "artifacts": record.artifacts,
        }), 200

    @app.route("/runs/<run_id>/artifacts/<path:filename>", methods=["GET"])
    def download_artifact(run_id: str, filename: str) -> Any:
        record = store.get(run_id)
        if not record or not record.run_dir:
            return jsonify({"error": f"Run '{run_id}' or its artifact directory not found"}), 404

        # Security check on filename
        safe_path = Path(record.run_dir) / filename
        try:
            resolved = safe_path.resolve()
            if not resolved.is_relative_to(Path(record.run_dir).resolve()):
                return jsonify({"error": "Path traversal prohibited"}), 403
        except Exception:
            return jsonify({"error": "Invalid path"}), 400

        if not resolved.exists() or not resolved.is_file():
            return jsonify({"error": f"Artifact '{filename}' not found"}), 404

        return send_from_directory(
            str(resolved.parent),
            resolved.name,
            as_attachment=True,
        )

    @app.route("/chat", methods=["POST"])
    def chat_endpoint() -> Any:
        if not request.is_json:
            return jsonify({"error": "Expected application/json"}), 415

        data = request.get_json(silent=True) or {}
        # Injection check
        for key in data.keys():
            if key.lower() in PROHIBITED_PAYLOAD_KEYS:
                return jsonify({
                    "response": "Refusal: Untrusted code or command execution parameters are strictly prohibited.",
                    "status": "refusal",
                    "tier": 1,
                }), 400

        run_id = data.get("run_id")
        query = data.get("query", "")

        if not run_id:
            return jsonify({"error": "Missing 'run_id'"}), 400
        if not query:
            return jsonify({"error": "Missing 'query'"}), 400

        record = store.get(run_id)
        if not record:
            return jsonify({"error": f"Run '{run_id}' not found"}), 404

        dio = record.dio or {}
        df = record.df

        result = chat_agent.run_query(query=query, df=df, dio=dio)
        result["run_id"] = run_id
        return jsonify(result), 200

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
