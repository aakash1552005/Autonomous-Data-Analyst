# Phase 6 Machine Learning Agent Real Data Validation & Human Review

This directory contains real data validation records, human review assessments, target selection justifications, leakage prevention evidence, and data sufficiency decisions for the Machine Learning (ML) Agent across all 5 benchmark datasets.

---

## 1. Machine Learning Strategy & Safety Principles

1. **Deterministic Target Selection**: Targets are selected using Intelligence DIO metadata (`is_target_candidate == True` or `semantic_label in ('target_label', 'target_value')`), prioritizing explicit outcome keywords (`default`, `churn`, `status`, `target`, `outcome`).
2. **Configurable Data Sufficiency Thresholds**:
   - `min_rows_for_ml: 30` (default)
   - `min_rows_per_class: 5` (default)
   - Datasets below these thresholds are safely flagged with `ML_003_INSUFFICIENT_DATA` rather than fitting untrustworthy, overfitted models.
3. **Multi-Layer Data Leakage Prevention**:
   - Direct target exclusion: $X$ never contains $y$.
   - PII exclusion: columns with `is_pii == True` are strictly stripped from $X$.
   - Identifier exclusion: columns with `semantic_label == 'identifier'` are strictly stripped from $X$.
   - Post-outcome leakage: columns with $|r| \ge 0.95$, $NMI \ge 0.95$, or matching post-outcome naming patterns (e.g. `cancellation_date`, `discharge_date`, `outcome_notes`) are strictly excluded.
   - Preprocessing isolation: all scaling, imputation, and encoding transformers are fitted **strictly on the training partition** after train/test split.
4. **Zero-LLM Operation**: All ML decisions (task detection, target selection, splitting, preprocessing, model selection, and metric evaluation) are executed 100% deterministically without LLM reliance.

---

## 2. Benchmark Datasets Qualitative Human Review

### 1. Retail Sales (`retail_sales.csv`)
* **Observed Data**: 6 rows, 7 columns (`order_id`, `product_id`, `order_date`, `price`, `quantity`, `discount_pct`, `revenue`).
* **Target Candidate Detected**: `revenue` (continuous metric).
* **ML Task Evaluation**: **INSUFFICIENT_DATA (Safely Rejected)**
* **Human Review Answers**:
  1. *Task Type*: Regression (`revenue` numeric metric).
  2. *Target Selection*: `revenue` was correctly identified as the dependent financial variable.
  3. *Exclusions*: `order_id` (identifier), `product_id` (identifier), `order_date` (temporal), and `revenue` (target).
  4. *Feature Set*: If trained with sufficient data, `price`, `quantity`, and `discount_pct` form a semantically sound predictor set.
  5. *Model Performance*: N/A — Model was not trained because total sample size ($N = 6$) is below `min_rows_for_ml = 30`.
  6. *Baseline Comparison*: N/A.
  7. *Leakage Check*: `revenue` has an exact mathematical relationship with `price * quantity * (1 - discount_pct)`. Fitting a statistical model on 6 rows would be deceptive.
  8. *Appropriateness of Decision*: **CORRECT REJECTION** — Returning `ML_003_INSUFFICIENT_DATA` protects users from misleading predictions.
  9. *Action Required*: User should provide a larger transactional dataset ($\ge 30$ records).

---

### 2. Healthcare Patients (`healthcare_patients.csv`)
* **Observed Data**: 6 rows, 7 columns (`patient_id`, `full_name`, `admission_date`, `discharge_date`, `age`, `systolic_bp`, `diastolic_bp`, `glucose`).
* **Target Candidate Detected**: None / Continuous vitals.
* **ML Task Evaluation**: **INSUFFICIENT_DATA (Safely Rejected)**
* **Human Review Answers**:
  1. *Task Type*: Unsupported / Insufficient data.
  2. *Target Selection*: No unambiguous clinical outcome target exists (e.g., mortality, readmission).
  3. *Exclusions*: `patient_id` (identifier), `full_name` (PII), `admission_date` (date), `discharge_date` (post-outcome date).
  4. *Feature Set*: Vitals (`age`, `bp`, `glucose`).
  5. *Model Performance*: N/A.
  6. *Baseline Comparison*: N/A.
  7. *Leakage Check*: `discharge_date` was correctly flagged as a post-outcome temporal feature.
  8. *Appropriateness of Decision*: **CORRECT REJECTION** — The dataset is a descriptive clinical cohort without a labeled target and has only 6 rows.
  9. *Action Required*: Clinical outcome target column must be designated by user.

---

### 3. Financial Loans (`financial_loans.csv`)
* **Observed Data**: 6 rows, 6 columns (`transaction_id`, `account_balance`, `loan_amount`, `interest_rate`, `credit_score`, `default_status`).
* **Target Candidate Detected**: `default_status` (Binary classification target).
* **ML Task Evaluation**: **INSUFFICIENT_DATA (Target Correctly Identified, Safely Rejected)**
* **Human Review Answers**:
  1. *Task Type*: Binary Classification (`default_status`: 0 vs 1).
  2. *Target Selection*: `default_status` was accurately selected as the primary credit risk outcome.
  3. *Exclusions*: `transaction_id` (identifier), `default_status` (target).
  4. *Feature Set*: `account_balance`, `loan_amount`, `interest_rate`, `credit_score` make complete domain sense for loan underwriting.
  5. *Model Performance*: N/A — Model was not trained because $N = 6$ is below `min_rows_for_ml = 30`, and minority class has only 2 samples (< `min_rows_per_class = 5`).
  6. *Baseline Comparison*: N/A.
  7. *Leakage Check*: Financial metrics showed no artificial synthetic leakage with default status.
  8. *Appropriateness of Decision*: **CORRECT REJECTION** — Training a loan default classifier on 6 samples with 2 defaults would have severe overconfidence and 0% generalizability.
  9. *Action Required*: Upload $\ge 30$ historical loan records for automated baseline modeling.

---

### 4. Mixed Messy Data (`mixed_messy_data.csv`)
* **Observed Data**: 6 rows, 4 columns (`record_id`, `metric_score`, `user_category`, `flag_active`).
* **Target Candidate Detected**: `metric_score` / `flag_active`.
* **ML Task Evaluation**: **INSUFFICIENT_DATA (Safely Rejected)**
* **Human Review Answers**:
  1. *Task Type*: Regression / Classification.
  2. *Target Selection*: Ambiguous target candidates between score and activity flag.
  3. *Exclusions*: `record_id` (identifier).
  4. *Feature Set*: Low-dimensionality messy features.
  5. *Model Performance*: N/A.
  6. *Baseline Comparison*: N/A.
  7. *Leakage Check*: No post-outcome leakage detected.
  8. *Appropriateness of Decision*: **CORRECT REJECTION** — Insufficient sample size ($N = 6$).
  9. *Action Required*: Ingest larger dataset with designated business objective.

---

### 5. Ambiguous Dates & PII (`ambiguous_dates_pii.csv`)
* **Observed Data**: 4 rows, 6 columns (`user_id`, `full_name`, `email`, `phone_number`, `credit_card`, `subscription_date`, `amount`).
* **Target Candidate Detected**: `amount` (or None).
* **ML Task Evaluation**: **INSUFFICIENT_DATA / UNSUPPORTED (Safely Rejected)**
* **Human Review Answers**:
  1. *Task Type*: Unsupported.
  2. *Target Selection*: `amount` is a transaction field, not a predictive target.
  3. *Exclusions*: `user_id` (identifier), `full_name`, `email`, `phone_number`, `credit_card` (all PII), `subscription_date` (ambiguous date).
  4. *Feature Set*: Zero generalizable non-PII features exist.
  5. *Model Performance*: N/A.
  6. *Baseline Comparison*: N/A.
  7. *Leakage Check*: Stripping all PII and identifiers leaves zero usable predictive features, triggering safe rejection.
  8. *Appropriateness of Decision*: **EXEMPLARY PRIVACY & SAFETY REJECTION** — Proves the system will never build models on PII or force ML when no real predictive problem exists.
  9. *Action Required*: None — dataset is purely contact information and billing records.
