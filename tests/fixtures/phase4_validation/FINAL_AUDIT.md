# Autonomous Data Analyst — Phase 4 Final Architectural Audit

This document formally records the Phase 4 architectural hardening and test quality audit prior to proceeding to Phase 5.

---

## 1. Platform Accuracy Policy

The platform does **not** claim universal "100% accuracy" for arbitrary user datasets. Outputs are strictly classified into three tiers of certainty:

### Tier A: EXACT (Deterministic Operations)
* **Components**: File validation, SHA-256 hashing, row/column counts, missing-value counts, duplicate detection, deterministic type coercion, unambiguous date resolution, mathematical calculations, aggregations, and derived dashboard metrics.
* **Guarantee**: 100% deterministic, exact, unit-tested, and verifiable.

### Tier B: STATISTICAL / INFERRED (Bounded Confidence)
* **Components**: Outlier fences (IQR), correlation matrices, distribution skewness, domain classification, target candidate identification, and time-series seasonality.
* **Requirement**: Must include confidence score, evidence list, source columns, calculation method, and warnings. Never presented as infallible truth.

### Tier C: AI / LLM INTERPRETATION (Natural-Language Explanations)
* **Components**: Narrative business summaries, automated chart explanations, report recommendations, and interactive chat responses.
* **Requirement**: Bounded token usage, mandatory PII masking, provenance linking to underlying Tier A metrics, and uncertainty flags. When confidence is insufficient, the system returns `UNKNOWN` or `NEEDS_REVIEW` rather than hallucinating.

---

## 2. DIO Section Ownership Matrix

The Dataset Intelligence Object (DIO) enforces strict module boundaries. Agents may only mutate their assigned sections:

| DIO Field / Section | Responsible Agent / Phase | Status in Phase 4 | Future Agent Access |
| :--- | :--- | :--- | :--- |
| `schema_version`, `dataset_id` | Core Foundation / Ingestion | Initialized | Read-only |
| `file_name`, `dataset_hash` | Security & Ingestion (Phase 2) | Populated & Verified | Read-only |
| `ingestion` | Security & Ingestion (Phase 2) | Populated (`n_rows`, `n_cols`, `file_type`, `encoding`) | Read-only |
| `columns` | Intelligence Agent (Phase 3) | Populated (dtypes, labels, PII, null counts) | Read-only |
| `date_columns` | Intelligence Agent (Phase 3) | Populated (formats, confidence, confirmation flags) | Read-only |
| `domain_guess` | Intelligence Agent (Phase 3) | Populated (`domain`, `confidence`) | Read-only |
| `quality` | Intelligence Agent (Phase 3) | Populated (`score`, `issues`) | Read-only |
| `cleaning_log` | Cleaning Agent (Phase 4) | Populated (imputations, drops, outlier flags) | Read-only |
| `artifacts.cleaned_csv` | Cleaning Agent (Phase 4) | Populated (path to `cleaned_data.csv`) | Read-only |
| `artifacts.removed_rows_csv`| Cleaning Agent (Phase 4) | Populated (path to `removed_rows.csv`) | Read-only |
| `eda` | EDA Agent (Phase 5) | **UNTOUCHED (Empty)** | Owned by Phase 5 |
| `ml` | ML Agent (Phase 6) | **UNTOUCHED (Empty)** | Owned by Phase 6 |
| `insights` | Insight Agent (Phase 7) | **UNTOUCHED (Empty)** | Owned by Phase 7 |
| `reports` | Report Agent (Phase 9) | **UNTOUCHED (Empty)** | Owned by Phase 9 |
| `progress` | Pipeline Orchestrator / Agents | Updated per agent state transition | Shared state |
| `decision_log`, `agent_metrics` | All Agents | Appended chronologically per agent | Provenance logging |

---

## 3. Original Data Immutability

* Ingested raw files and their SHA-256 hashes are immutable.
* Cleaning Agent writes transformed datasets to isolated run directories (`runs/{run_id}/artifacts/cleaned_data.csv`).
* Downstream agents (EDA, ML, Insights, Reports) consume the validated cleaned artifact or explicit raw reference, never modifying original source files.

---

## 4. Artifact Lineage & Traceability

Every pipeline artifact maintains full cryptographic and structural provenance:
```text
Original Raw File
  └── Ingestion (SHA-256 Raw Hash: H_raw, IngestionMetadata)
        └── Intelligence (Schema, PII Masking, Date Resolution, Quality Score)
              └── Cleaning (CleaningAgent, Duplicate Removal, Skewness Imputation, IQR Flags)
                    ├── Removed Rows Sidecar (artifacts/removed_rows.csv)
                    └── Cleaned Data (artifacts/cleaned_data.csv, SHA-256 Cleaned Hash: H_clean)
                          └── Downstream Agents (EDA -> ML -> Insights -> Reports)
```

---

## 5. Reproducibility & LLM Independence

* **Deterministic Execution**: Given identical data and configuration, Cleaning Agent transformations produce identical outputs on every run.
* **Zero LLM Dependency**: Cleaning Agent makes zero LLM calls, contains zero network dependencies, and executes entirely offline without Ollama or external APIs.
* **Structural PII Gate**: Columns flagged `is_pii = True` are structurally blocked from reaching any LLM fallback.

---

## 6. Lossless Round-Trip Reconstruction

Reversibility was mathematically and programmatically verified across all 5 benchmark datasets:
1. `retail_sales.csv`: PASS
2. `healthcare_patients.csv`: PASS
3. `financial_loans.csv`: PASS
4. `mixed_messy_data.csv`: PASS
5. `ambiguous_dates_pii.csv`: PASS

Reconstruction recovers 100% of rows, columns, null masks, non-null values, and original row ordering.

---

## 7. Known Architectural Boundaries

* **Legacy `.xls`**: Unsupported in V1; rejected with clear migration error to `.xlsx`.
* **Date Imputation**: Missing dates are preserved as `NaN` (never synthetically imputed).
* **Ambiguous Dates**: Preserved raw with explicit warnings requesting user confirmation.
* **Outliers**: Identified and recorded in cleaning log under a strict flag-only policy (zero deletions).
