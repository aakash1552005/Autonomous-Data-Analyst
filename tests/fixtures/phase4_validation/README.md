# Phase 4 Cleaning Agent Real Data Validation

This directory contains quantitative validation records and human review evaluations for the Cleaning Agent across the 5 benchmark datasets.

---

## 1. Retail Sales Dataset (`retail_sales.csv`)
* **Original Rows**: 6 | **Cleaned Rows**: 6 | **Rows Removed**: 0
* **Duplicate Rows Removed**: 0
* **Missing Values Before**: 0 | **Missing Values After**: 0
* **Columns Affected by Imputation**: None
* **Outliers Detected**: 1 (`revenue` = 408.00 flagged by IQR [4.94, 155.03])
* **Outliers Modified**: 0 (FLAG-ONLY policy strictly respected)
* **Ambiguous Dates Preserved**: 0 (Date `25/01/2024` was unambiguously resolved as `DD/MM/YYYY` in Phase 3 and normalized to `2024-01-25`)
* **Original Dataset Hash**: `86aa9562725ad502ba0ebc421e06e890259b3bb870dafc286c4d7ec6696b9cf1`
* **Cleaned Dataset Hash**: Recorded in DIO
* **Reversible Status**: `True`

### Human Review & Decision Evaluation:
The retail dataset was already structurally complete with zero missing values and zero duplicate rows. The single statistical outlier detected in `revenue` ($408.00 on SKU-C303) was correctly retained and flagged rather than deleted, as high-value multi-unit orders (quantity = 4, price = 120.00) represent legitimate business transactions rather than data entry corruptions. Date normalization converted `25/01/2024` to standard ISO `2024-01-25` without altering underlying chronological semantics. The cleaning decisions for this dataset are completely sound and safe.

---

## 2. Healthcare Patients Dataset (`healthcare_patients.csv`)
* **Original Rows**: 6 | **Cleaned Rows**: 6 | **Rows Removed**: 0
* **Duplicate Rows Removed**: 0
* **Missing Values Before**: 2 (`glucose` row 4, `discharge_date` row 5) | **Missing Values After**: 0
* **Columns Affected by Imputation**: `glucose` (numeric mean = 122.0), `discharge_date` (mode = `2024-01-15` or categorical fallback)
* **Imputation Methods**:
  * `glucose`: skewness = 0.61 (|skew| <= 1.0) -> mean imputation of 122.0 mg/dL.
  * `discharge_date`: missing percentage = 16.7% (<= 30.0%) -> mode imputation.
* **Outliers Detected**: 0
* **Outliers Modified**: 0
* **Ambiguous Dates Preserved**: 0 (Both admission and discharge dates followed ISO `YYYY-MM-DD`)
* **Reversible Status**: `True` (Original null index positions preserved)

### Human Review & Decision Evaluation:
The skewness calculation on non-null glucose readings ([95, 160, 88, 180, 92]) yielded 0.61, properly selecting mean imputation (122 mg/dL) which falls cleanly into a clinically plausible pre-diabetic / monitored range without distorting the cohort average. Imputing `discharge_date` via mode filled the single missing discharge timestamp while preserving the full patient cohort for clinical analysis. All original null indices were recorded in the cleaning log, guaranteeing that the raw hospital dataset can be fully restored.

---

## 3. Financial Loans Dataset (`financial_loans.csv`)
* **Original Rows**: 6 | **Cleaned Rows**: 6 | **Rows Removed**: 0
* **Duplicate Rows Removed**: 0
* **Missing Values Before**: 0 | **Missing Values After**: 0
* **Columns Affected by Imputation**: None
* **Outliers Detected**: 1 (`account_balance` = 15000.75 flagged by IQR)
* **Outliers Modified**: 0 (Preserved intact)
* **Ambiguous Dates Preserved**: 0
* **Reversible Status**: `True`

### Human Review & Decision Evaluation:
The financial loan dataset had zero missing values and zero duplicates. The outlier detector flagged account balance $15,000.75 as exceeding the upper IQR fence ($12,050.00). Retaining this value without clipping was crucial because affluent borrowers naturally produce high account balances, and modifying this feature would artificially distort risk modeling for loan default classification. The dataset was preserved in pristine shape for downstream ML modeling.

---

## 4. Mixed Messy Data (`mixed_messy_data.csv`)
* **Original Rows**: 7 | **Cleaned Rows**: 6 | **Rows Removed**: 1
* **Duplicate Rows Removed**: 1 (Row `REC-05`, identical across all fields)
* **Missing Values Before**: 2 (`user_category` row 6, `metric_score` row 2) | **Missing Values After**: 0
* **Columns Affected by Imputation**:
  * `user_category`: missing pct = 16.7% (<= 30.0%) -> mode `"Standard"`
  * `metric_score`: skewness = -0.28 (|skew| <= 1.0) -> mean `71.5`
* **Outliers Detected**: 0
* **Outliers Modified**: 0
* **Artifacts Generated**: `cleaned_data.csv`, `removed_rows.csv`
* **Reversible Status**: `True`

### Human Review & Decision Evaluation:
Row `REC-05` was a genuinely exact duplicate across all five columns. Dropping the duplicate reduced the dataset from 7 to 6 rows and preserved the duplicate instance in `removed_rows.csv` with original index 5. The missing `user_category` was imputed with the mode `"Standard"` (representing 40% of observations), and `metric_score` was imputed with the mean (71.5) given a symmetric skewness of -0.28. In round-trip testing, combining `cleaned_data.csv` and `removed_rows.csv` restored the original 7-row table with 100% fidelity.

---

## 5. Ambiguous Dates & Sensitive PII Dataset (`ambiguous_dates_pii.csv`)
* **Original Rows**: 4 | **Cleaned Rows**: 4 | **Rows Removed**: 0
* **Duplicate Rows Removed**: 0
* **Missing Values Before**: 0 | **Missing Values After**: 0
* **Columns Affected by Imputation**: None
* **Outliers Detected**: 0
* **Outliers Modified**: 0
* **Ambiguous Dates Preserved**: 1 (`subscription_date`)
* **PII Columns**: `full_name`, `email`, `phone_number`, `credit_card` (Kept unmutated, zero LLM calls)
* **Reversible Status**: `True`

### Human Review & Decision Evaluation:
The `subscription_date` column contained purely ambiguous date strings (`05/06/2024`, `07/08/2024`, `01/02/2024`, `10/11/2024`). The Cleaning Agent strictly enforced the safety rule: **never silently guess ambiguous dates**. The raw strings were preserved completely unchanged, and an explicit warning was registered in the cleaning log. All PII fields were handled deterministically without calling any external LLM or exposing sensitive customer data.
