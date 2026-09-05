# Phase 10 Pre-Flight Audit & Baseline Report

**Project:** Autonomous Data Analyst
**Phase:** Phase 10 — Benchmark & Evaluation Layer
**Timestamp:** 2026-09-05T11:07:00+05:30

---

## 1. Git Environment & Repository State

- **Branch:** `master`
- **HEAD Commit:** `2e4f3dc feat(integration): complete Phase 9 orchestrator and Streamlit pipeline`
- **Working Tree Status:** `Clean` (`nothing to commit, working tree clean`)
- **Remote:** `origin -> https://github.com/aakash1552005/Autonomous-Data-Analyst.git` (synchronized)

---

## 2. Runtime Environment & Dependencies

- **Python Version:** `3.14.7` (tags/v3.14.7:823f032, MSC v.1944 64 bit AMD64)
- **Key Installed Packages:**
  - `pandas` 3.0.5
  - `numpy` 2.5.2
  - `scikit-learn` 1.9.0
  - `xgboost` 3.4.1
  - `plotly` 7.0.0
  - `kaleido` 1.3.0
  - `reportlab` 5.0.1
  - `python-pptx` 1.0.2
  - `streamlit` 1.62.0
  - `pytest` 9.1.1
  - `pyyaml` 6.0.3
- **Baseline Profiler Status:**
  - `ydata-profiling`: **NOT INSTALLED** (`ModuleNotFoundError: No module named 'ydata_profiling'`).
  - Evaluation record will mark baseline comparison as `YDATA_PROFILING_UNAVAILABLE` per specification.

---

## 3. Verified Phase 9 Test Regression Baseline

- **Full Regression Command:** `.venv\Scripts\python.exe -m pytest tests/ -q`
- **Total Tests Run:** `298`
- **Passed:** `298`
- **Failed:** `0`
- **Skipped:** `0`
- **Errors:** `0`
- **Warnings:** `2139` (NumPy 2.5 `joblib` array shape deprecation warnings only)
- **Duration:** `1473.30s` (`24m 33s`)

---

## 4. Existing Benchmark Assets & Instrumentation

### Benchmark Datasets (in `data/sample/`):
1. **`retail_sales.csv`** (6 rows, 9 columns):
   - Domain: Retail/Sales
   - Features: Numeric prices, quantities, discounts, dates, customer identifiers
   - ML Applicability: Regression (`Revenue`)
2. **`healthcare_patients.csv`** (7 rows, 9 columns):
   - Domain: Healthcare
   - Features: Patient identifiers, ages, blood pressure, diagnosis labels
   - ML Applicability: Classification (`diagnosis`)
   - Security: Sensitive health metrics and identifiers
3. **`financial_loans.csv`** (7 rows, 9 columns):
   - Domain: Finance/Banking
   - Features: Loan amounts, interest rates, credit scores, approval status
   - ML Applicability: Classification (`loan_approved`)
4. **`mixed_messy_data.csv`** (8 rows, 8 columns):
   - Domain: Mixed / Messy E-commerce
   - Features: Duplicate rows, null values, corrupted currency strings, messy dates
   - Cleaning Target: Duplicate extraction, type coercion, missing value imputation
5. **`ambiguous_dates_pii.csv`** (4 rows, 7 columns):
   - Domain: Subscription / User Accounts
   - Features: Ambiguous format dates (`DD/MM` vs `MM/DD`), synthetic PII (names, emails, phones, credit cards)
   - Evaluation Target: Safe ambiguous date preservation, PII isolation boundary, ML safe skipping

### Existing Instrumentation in Phase 0–9 Pipeline:
- **`DIO` Contract**: `ingestion`, `columns`, `date_columns`, `domain_guess`, `quality`, `cleaning_log`, `eda`, `ml`, `insights`, `reports`, `progress`, `errors`, `agent_metrics`, `decision_log`, `llm_usage`, `artifacts`.
- **Timing Instrumentation**: `OrchestratorResult.stage_timings` records per-stage runtimes (`validation`, `intelligence`, `cleaning`, `eda`, `ml`, `insight`, `report`, `total_pipeline`).
- **Status Instrumentation**: `OrchestratorResult.stage_statuses` records stage outcomes (`completed`, `skipped`, `failed`, `pending`).
- **Token Usage Instrumentation**: `TokenGovernor` and `dio["llm_usage"]` tracks prompt, completion, total tokens, and call counts.
- **Persistence Instrumentation**: `save_dio_json()` to `run_dir/dio.json` and `setup_run_logger()` to `run_dir/pipeline.log`.

---

## 5. Phase 10 Implementation Plan

### Architecture: `tests/benchmark/` Module Structure
```text
tests/benchmark/
├── __init__.py
├── ground_truth.py        # Strict, deterministic ground-truth schema & benchmark dataset registry
├── evaluators/
│   ├── __init__.py
│   ├── semantic_evaluator.py     # Metric 1: Semantic label precision/recall/accuracy
│   ├── domain_evaluator.py       # Metric 2: Domain classification accuracy & confidence
│   ├── date_evaluator.py         # Metric 3: Date resolution & ambiguous handling accuracy
│   ├── pii_evaluator.py          # Metric 4: PII Recall, Precision & Zero-Leakage audit
│   ├── cleaning_evaluator.py     # Metric 5: Duplicate removal, imputation & type coercion
│   ├── eda_evaluator.py          # Metric 6: Statistical fact & chart artifact verification
│   ├── ml_evaluator.py           # Metric 7: Task type, target selection & metric integrity
│   ├── insight_evaluator.py      # Metric 8: Numerical grounding & hallucination detection
│   ├── report_evaluator.py       # Metric 9: Document text extraction & factual fidelity
│   ├── runtime_evaluator.py      # Metric 10: Total & per-stage runtime tracking
│   └── baseline_comparator.py    # Metric 11: ydata-profiling baseline comparator (handles unavailable)
├── benchmark_runner.py    # Orchestrates execution of evaluation layer across datasets
├── report_generator.py    # Generates machine-readable evaluation_results.json & human-readable evaluation.html
└── tests/                 # Dedicated unit, adversarial, and reproducibility tests for the benchmark layer
```

### Key Deliverables:
1. **Machine-readable Results**: `evaluation_results.json` adhering to a strict schema.
2. **Executive HTML Report**: `evaluation.html` with modern styling, executive summary, dataset cards, aggregate tables, security scorecards, and baseline comparisons.
3. **Adversarial & Reproducibility Suite**: Tests for ground truth validation, edge cases, zero division, graceful degradations, and identical cross-run determinism.
