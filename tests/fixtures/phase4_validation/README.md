# Phase 4 Cleaning Agent Real Data Validation & Hardening Review

This directory contains quantitative validation records, human review evaluations, and formal lossless round-trip reconstruction proofs for the Cleaning Agent across the 5 benchmark datasets.

---

## Reversibility & Accuracy Principle

### Reversibility Guarantee
`reversible = true` guarantees exact **logical data reconstruction**:
1. All imputed cells can be restored back to `NaN` using recorded `original_null_indices`.
2. All date normalizations can be mapped back to their raw string representations.
3. All dropped duplicate rows are recovered from `removed_rows.csv` and re-inserted at their exact original row indices (`_orig_row_index`).
4. Original row ordering and column counts are restored identically.

*Note on Data Types*: Reversible data reconstruction guarantees exact equality of cell values, null masks, strings, and numeric values. When reloading raw CSV files into pandas, physical implementation dtypes (e.g. nullable integer `Int64` vs `float64` for null columns) are subject to standard CSV type inference, but logical data content and values are 100% losslessly preserved.

### Engineering & Accuracy Standard
The autonomous platform does **not** claim universal "100% analytical accuracy" for arbitrary user datasets. Instead, it adheres to the core engineering principle:
> *"Never silently produce an incorrect result. Prefer UNKNOWN, preserve ambiguity, and request human confirmation whenever deterministic confidence is insufficient."*

---

## Benchmark Datasets Evaluation & 5/5 Reconstruction

### 1. Retail Sales Dataset (`retail_sales.csv`)
* **Original Rows**: 6 | **Cleaned Rows**: 6 | **Rows Removed**: 0
* **Duplicate Rows Removed**: 0
* **Missing Values Before**: 0 | **Missing Values After**: 0
* **Columns Affected by Imputation**: None
* **Outliers Detected**: 1 (`revenue` = 408.00 flagged by IQR [4.94, 155.03])
* **Outliers Modified**: 0 (FLAG-ONLY policy strictly respected)
* **Ambiguous Dates Preserved**: 0 (`25/01/2024` was unambiguously resolved as `DD/MM/YYYY` and normalized to `2024-01-25`)
* **Reconstruction Proof**: PASS (`assert_lossless_round_trip_reconstruction` verified 100% match against raw DataFrame)

### 2. Healthcare Patients Dataset (`healthcare_patients.csv`)
* **Original Rows**: 6 | **Cleaned Rows**: 6 | **Rows Removed**: 0
* **Duplicate Rows Removed**: 0
* **Missing Values Before**: 2 (`glucose` row 4, `discharge_date` row 5) | **Missing Values After**: 1 (`discharge_date` row 5 preserved as NaN)
* **Date Imputation Policy**: CRITICAL SAFETY RULE ENFORCED — Missing dates are NEVER mode-imputed or synthetically invented. `discharge_date` missing value was preserved as missing (`NaN`).
* **Numeric Imputation**: `glucose` skewness = 0.61 (|skew| <= 1.0) -> mean imputation of 122.0 mg/dL with `generated_synthetic: True`.
* **Outliers Detected**: 0
* **Reconstruction Proof**: PASS (`assert_lossless_round_trip_reconstruction` verified 100% match against raw DataFrame)

### 3. Financial Loans Dataset (`financial_loans.csv`)
* **Original Rows**: 6 | **Cleaned Rows**: 6 | **Rows Removed**: 0
* **Duplicate Rows Removed**: 0
* **Missing Values Before**: 0 | **Missing Values After**: 0
* **Outliers Detected**: 1 (`account_balance` = 15000.75 flagged by IQR)
* **Outliers Modified**: 0 (Preserved intact)
* **Reconstruction Proof**: PASS (`assert_lossless_round_trip_reconstruction` verified 100% match against raw DataFrame)

### 4. Mixed Messy Data (`mixed_messy_data.csv`)
* **Original Rows**: 7 | **Cleaned Rows**: 6 | **Rows Removed**: 1
* **Duplicate Rows Removed**: 1 (Row `REC-05`, identical across all fields)
* **Missing Values Before**: 2 (`user_category` row 6, `metric_score` row 2) | **Missing Values After**: 0
* **Categorical Imputation**: `user_category` missing percentage = 16.7% (<= 30.0%) -> mode `"Standard"` with `generated_synthetic: True`.
* **Numeric Imputation**: `metric_score` skewness = -0.28 (|skew| <= 1.0) -> mean `71.5`.
* **Artifacts Generated**: `cleaned_data.csv`, `removed_rows.csv`
* **Reconstruction Proof**: PASS (`assert_lossless_round_trip_reconstruction` restored all 7 rows, including `REC-05` duplicate, with 100% exact equality)

### 5. Ambiguous Dates & Sensitive PII Dataset (`ambiguous_dates_pii.csv`)
* **Original Rows**: 4 | **Cleaned Rows**: 4 | **Rows Removed**: 0
* **Duplicate Rows Removed**: 0
* **Missing Values Before**: 0 | **Missing Values After**: 0
* **Ambiguous Dates Preserved**: 1 (`subscription_date` preserved raw with warning)
* **PII Columns**: `full_name`, `email`, `phone_number`, `credit_card` (Kept unmutated, zero LLM calls)
* **Reconstruction Proof**: PASS (`assert_lossless_round_trip_reconstruction` verified 100% match against raw DataFrame)
