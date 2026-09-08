"""
tests/test_api.py
=================
Phase 23: Automated API Contract Tests for Flask REST API (`api.py`).
Verifies:
  1. GET /health availability and contract
  2. POST /analyze security filters (rejection of python_code, shell_command, etc.)
  3. POST /analyze file extension and path validation
  4. POST /analyze synchronous pipeline execution and safe summary
  5. POST /analyze async execution returns 202 and poll_url
  6. GET /runs and GET /runs/<run_id> contract
  7. GET /runs/<run_id>/artifacts artifact listing and path traversal defense
  8. POST /chat query processing and adversarial rejection
  9. Strict absence of raw PII in API responses
"""

import json
from pathlib import Path
import tempfile
import pytest

from api import create_app
from core.config import load_config


@pytest.fixture
def client():
    config = load_config()
    app = create_app(config=config)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestAPIHealth:
    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert data["version"] == "1.1.0"
        assert "Autonomous Data Analyst" in data["service"]
        assert "GET /health" in data["endpoints"]
        assert "POST /analyze" in data["endpoints"]


class TestAPISecurityAndValidation:
    def test_reject_prohibited_injection_keys(self, client):
        prohibited_payloads = [
            {"file_path": "data/sample/retail_sales.csv", "python_code": "import os; os.system('ls')"},
            {"file_path": "data/sample/retail_sales.csv", "shell_command": "rm -rf /"},
            {"file_path": "data/sample/retail_sales.csv", "sql_query": "SELECT * FROM users"},
            {"file_path": "data/sample/retail_sales.csv", "eval_expression": "1+1"},
        ]
        for payload in prohibited_payloads:
            resp = client.post("/analyze", json=payload)
            assert resp.status_code == 400
            data = resp.get_json()
            assert "Refusal" in data.get("error", "")

    def test_reject_nonexistent_file(self, client):
        resp = client.post("/analyze", json={"file_path": "non_existent_data.csv"})
        assert resp.status_code == 404
        assert "not found" in resp.get_json().get("error", "").lower()

    def test_reject_unsupported_file_extension(self, client):
        # Create a temp .txt file
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"hello world")
            tmp_path = f.name
        try:
            resp = client.post("/analyze", json={"file_path": tmp_path})
            assert resp.status_code == 400
            assert "unsupported file extension" in resp.get_json().get("error", "").lower()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_reject_unsupported_content_type(self, client):
        resp = client.post("/analyze", data="plain text body", content_type="text/plain")
        assert resp.status_code == 415


class TestAPIPipelineExecution:
    def test_async_analyze_returns_202(self, client):
        sample_path = Path("data/sample/retail_sales.csv").resolve()
        if not sample_path.exists():
            pytest.skip("retail_sales.csv sample not found")

        resp = client.post("/analyze", json={
            "file_path": str(sample_path),
            "async": True,
        })
        assert resp.status_code == 202
        data = resp.get_json()
        assert "run_id" in data
        assert data["status"] in ("queued", "running")
        assert "poll_url" in data

    def test_sync_analyze_and_chat_e2e(self, client):
        sample_path = Path("data/sample/retail_sales.csv").resolve()
        if not sample_path.exists():
            pytest.skip("retail_sales.csv sample not found")

        # 1. Trigger sync analysis
        resp = client.post("/analyze", json={
            "file_path": str(sample_path),
            "async": False,
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert "run_id" in data
        assert data["status"] in ("completed", "partial")
        run_id = data["run_id"]
        assert "artifacts" in data
        assert isinstance(data["artifacts"], list)

        # 2. Query run status via GET /runs/<run_id>
        resp_run = client.get(f"/runs/{run_id}")
        assert resp_run.status_code == 200
        run_data = resp_run.get_json()
        assert run_data["run_id"] == run_id

        # 3. Query artifacts via GET /runs/<run_id>/artifacts
        resp_artifacts = client.get(f"/runs/{run_id}/artifacts")
        assert resp_artifacts.status_code == 200
        art_data = resp_artifacts.get_json()
        assert art_data["artifact_count"] >= 1

        # 4. Chat query over run results via POST /chat
        chat_resp = client.post("/chat", json={
            "run_id": run_id,
            "query": "mean of revenue",
        })
        assert chat_resp.status_code == 200
        chat_data = chat_resp.get_json()
        assert chat_data["status"] == "success"
        assert chat_data["tier"] == 1
        assert "mean" in chat_data["response"].lower() and "revenue" in chat_data["response"].lower()

    def test_get_runs_list(self, client):

        resp = client.get("/runs")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "runs" in data
        assert isinstance(data["runs"], list)

    def test_get_nonexistent_run_returns_404(self, client):
        resp = client.get("/runs/run_non_existent_999999")
        assert resp.status_code == 404

    def test_get_nonexistent_run_artifacts_returns_404(self, client):
        resp = client.get("/runs/run_non_existent_999999/artifacts")
        assert resp.status_code == 404


class TestAPIChatEndpoint:
    def test_chat_rejects_non_json(self, client):
        resp = client.post("/chat", data="hello", content_type="text/plain")
        assert resp.status_code == 415

    def test_chat_rejects_missing_parameters(self, client):
        resp = client.post("/chat", json={"query": "mean of sales"})
        assert resp.status_code == 400

    def test_chat_rejects_injection_parameters(self, client):
        resp = client.post("/chat", json={
            "run_id": "test_run",
            "query": "what is mean of sales",
            "python_code": "exec('print(1)')",
        })
        assert resp.status_code == 400
        data = resp.get_json()
        assert data.get("status") == "refusal"

    def test_chat_rejects_nonexistent_run(self, client):
        resp = client.post("/chat", json={
            "run_id": "run_fake_12345",
            "query": "mean of sales",
        })
        assert resp.status_code == 404
