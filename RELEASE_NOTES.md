# Release Notes — Autonomous Data Analyst v1.1.0

**Release Date:** September 8, 2026  
**Release Tag:** `v1.1.0`  
**Previous Tag:** `v1.0.0`

---

## Executive Summary

Autonomous Data Analyst v1.1.0 represents a major expansion of system capabilities, introducing conversational dataset intelligence (Agent 7), external workflow automation via n8n and a REST API (`api.py`), hardened privacy and reliability defenses, and comprehensive architectural documentation.

All 24 phases of the product roadmap are now complete and verified.

---

## What's New in v1.1.0

### 1. Conversational Dataset Intelligence (Agent 7)
- **11 Whitelisted Operations**: Real-time aggregation over analyzed datasets without manual scripting:
  - Central tendency: `mean`, `median`
  - Dispersion: `std` (sample, `ddof=1`), `variance` (sample, `ddof=1`)
  - Bounds & Extrema: `min`, `max`, `count`
  - Grouping: `value_counts`, `groupby_mean`, `groupby_sum`
- **Zero Arbitrary Execution**: No `eval()`, `exec()`, or dynamic code execution. Queries are parsed and executed using deterministic pandas operations.
- **Structural PII Shield**: Queries targeting personal information or identifier columns are automatically intercepted and refused.
- **Enhanced Streamlit Interface**: Tab 8 now features full conversation history, guided suggestions, and thousand-separated numerical formatting.

### 2. Workflow Automation (n8n & REST API)
- **Flask REST API (`api.py`)**:
  - `GET /health` — Service healthcheck and capabilities.
  - `POST /analyze` — Synchronous and asynchronous pipeline triggers with strict input validation.
  - `GET /runs` & `GET /runs/<run_id>` — Run status and PII-free analytical summaries.
  - `GET /runs/<run_id>/artifacts/<filename>` — Secure artifact downloads with path traversal defenses.
  - `POST /chat` — Conversational Q&A over completed runs.
- **6 Production n8n Workflows**:
  - Webhook pipeline trigger, completion poller, failure alerting, scheduled benchmark runner, PDF report delivery, and uptime monitor.

### 3. Reliability & Security Hardening
- **DIO Contract Immutability**: Analytical sections are frozen upon pipeline completion (`freeze_analytical_sections()`), preventing accidental mutation by downstream or interactive processes.
- **Edge-Case Tolerance**: Defensively handles empty DataFrames, None objects, all-NaN columns, infinities, and missing DIO blocks.
- **Prompt Injection Gate**: Blocks adversarial jailbreaks, instruction overrides, system prompt extraction, and code execution vectors.

### 4. Comprehensive Documentation
- Added [ARCHITECTURE.md](ARCHITECTURE.md) and [SECURITY.md](SECURITY.md).
- Updated [README.md](README.md) with full v1.0 vs v1.1 vs v2.0 roadmap.
- Validated one-click launcher scripts (`run.bat`, `run.ps1`) and standard editable packaging (`pip install -e .`).

---

## Upgrade Guide

1. Pull the latest commits from `master`.
2. Activate your virtual environment and install updated dependencies:
   ```powershell
   .\.venv\Scripts\pip.exe install -e .
   ```
3. To launch the Streamlit UI:
   ```powershell
   .\run.bat
   # or
   .\run.ps1
   ```
4. To launch the Flask REST API for n8n:
   ```powershell
   .\.venv\Scripts\python.exe api.py
   ```
