# n8n Workflow Automation for Autonomous Data Analyst

This directory contains pre-built, tested n8n workflows that integrate with the Autonomous Data Analyst Flask REST API (`api.py`).

---

## Workflows Included

| File | Name | Description |
|------|------|-------------|
| `workflows/analysis_trigger.json` | **Analysis Pipeline Trigger** | Webhook or scheduled endpoint to safely invoke `POST /analyze` with a CSV/XLSX dataset. |
| `workflows/analysis_completion.json` | **Run Completion Poller** | Polls `GET /runs/{run_id}` until completion and triggers downstream notifications. |
| `workflows/failure_notification.json` | **Pipeline Failure Alert** | Intercepts partial or failed runs and dispatches alerts with stage diagnostics. |
| `workflows/scheduled_benchmark.json` | **Periodic Benchmark Trigger** | Cron-triggered runner executing regression benchmarks against sample datasets. |
| `workflows/report_delivery.json` | **Executive Report Delivery** | Downloads `report.pdf` from `GET /runs/{run_id}/artifacts/report.pdf` and delivers to recipients. |
| `workflows/monitoring.json` | **Health & Uptime Monitor** | Periodically checks `GET /health` and reports uptime status. |

---

## Quick Setup

### 1. Start the Flask REST API
In your Autonomous Data Analyst environment:
```powershell
.\.venv\Scripts\python.exe api.py
```
By default, the API starts at `http://127.0.0.1:5000`.

### 2. Import into n8n
1. Open your n8n workspace (self-hosted or desktop).
2. Go to **Workflows** -> **Import from File...**
3. Select any workflow file from `n8n/workflows/`.
4. Update the `baseUrl` variable (defaults to `http://127.0.0.1:5000` or `http://host.docker.internal:5000` if n8n is running in Docker).
5. Activate the workflow!
