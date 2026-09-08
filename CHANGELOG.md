# Changelog

All notable changes to the Autonomous Data Analyst project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1.0] - 2026-09-08

### Added
- **Agent 7 — Interactive Q&A Engine**:
  - Whitelist-only deterministic execution architecture supporting 11 mathematical operations: `mean`, `sum`, `median`, `std`, `variance`, `min`, `max`, `count`, `value_counts`, `groupby_mean`, `groupby_sum`.
  - Sample standard deviation and variance enforced with degrees of freedom `ddof=1`.
  - Structural PII shield intercepting and blocking queries targeting sensitive columns (emails, phone numbers, national IDs, credit cards) and identifier columns.
  - Bounded DIO factual retrieval and LLM context synthesis with offline heuristic fallback.
- **Flask REST API (`api.py`)**:
  - Lightweight REST interface providing endpoints: `GET /health`, `POST /analyze`, `GET /runs`, `GET /runs/<run_id>`, `GET /runs/<run_id>/artifacts`, `GET /runs/<run_id>/artifacts/<filename>`, and `POST /chat`.
  - Multi-tier security gate blocking prohibited payload keys (`python_code`, `shell_command`, `sql_query`, `eval_expression`) and path traversal attempts.
  - Synchronous and asynchronous execution options for pipeline runs.
- **n8n Workflow Automation**:
  - 6 production workflow templates in `n8n/workflows/`:
    - `analysis_trigger.json`: Webhook-triggered analysis initiation.
    - `analysis_completion.json`: Polling and event formatting for completed runs.
    - `failure_notification.json`: Diagnostic alerting on pipeline failures.
    - `scheduled_benchmark.json`: Automated daily benchmark execution.
    - `report_delivery.json`: Downloader and dispatcher for executive PDF reports.
    - `monitoring.json`: Uptime and health checking every 5 minutes.
  - Integration guide and example payloads in `n8n/README.md` and `n8n/examples/example_payload.json`.
- **System Documentation**:
  - `ARCHITECTURE.md`: Detailed architecture diagrams, agent catalog, DIO contracts, chat dual-tier flow, and REST API design.
  - `SECURITY.md`: Security policies, threat model, PII-before-LLM rule, whitelist-only execution, and input sanitization.
- **Reliability & Contract Hardening**:
  - `freeze_analytical_sections()` and `is_analytical_frozen()` for immutable analytical DIO enforcement.
  - Defensive recovery for None/empty DataFrames, NaN-only columns, infinity values, and mixed types across stage boundaries.
  - Adversarial prompt injection defense covering jailbreaks, DAN prompts, and exfiltration attacks.

### Changed
- **Streamlit Tab 8 (Interactive Chat)**:
  - Enhanced user guidance for all 11 supported operations.
  - Clear, user-friendly refusal responses when queries target sensitive or identifier columns.
  - Formatted numerical values with thousand separators and precision controls.
- **Package Distribution & Packaging**:
  - Synchronized `pyproject.toml` and `requirements.txt` with Flask.
  - Validated PEP 517 editable installations (`pip install -e .`).

### Security
- Zero arbitrary code execution (`eval()`, `exec()`, `os.system()`, `subprocess`) verified via automated static audit.
- Zero raw PII leakage across LLM prompt templates, chat responses, log streams, and benchmark reports.
- Zero hardcoded secrets, credentials, or API keys in the codebase and n8n workflows.

---

## [1.0.0] - 2026-09-08

### Added
- Core multi-agent autonomous data analysis pipeline:
  - Agent 1: Schema profiling, date resolution, semantic labeling, PII detection, domain classification, and dataset quality scoring.
  - Agent 2: Reversible data cleaning, type coercion, whitespace stripping, missing value imputation, and duplicate extraction.
  - Agent 3: Exploratory data analysis, univariate and bivariate analysis, correlation matrices, and automated chart generation (Plotly/Kaleido).
  - Agent 4: Machine learning target variable detection, model training (Linear/Logistic, Decision Tree, Random Forest, XGBoost), cross-validation, and feature importance.
  - Agent 5: Executive business insights with deterministic 100% numerical grounding verification against DIO facts.
  - Agent 6: Executive ReportLab PDF reports and python-pptx presentation slide decks.
- Streamlit interactive multi-tab dashboard (`app.py`).
- Benchmark and evaluation layer with 11 evaluators across 5 canonical datasets.
