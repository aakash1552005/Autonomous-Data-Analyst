# Phase 5 EDA Agent Real Data Validation & Human Review

This directory contains quantitative validation records, qualitative human evaluations, PII boundary verifications, and target candidate awareness documentation for the Exploratory Data Analysis (EDA) Agent across all 5 benchmark datasets.

---

## 1. Deterministic Chart Selection Rule Table

The EDA Agent selects visualization charts without an LLM using the following deterministic rule table:

| Precedence | Dataset Condition | Generated Chart Type | Visual / Statistical Purpose |
| :--- | :--- | :--- | :--- |
| **1** | $\ge 3$ numeric columns with non-zero variance | **Correlation Heatmap** | Global feature association overview |
| **2** | Verified Date column + primary Numeric column | **Time-Series Line Trend** | Chronological metric progression |
| **3** | $\ge 2$ numeric columns | **Scatter Plot** | Top correlated numeric pair interaction |
| **4** | Numeric columns | **Histograms (Distributions)** | Unimodal/multimodal spread & outliers |
| **5** | Categorical columns ($\le 15$ unique categories) | **Categorical Bar Charts** | Category frequency breakdowns |
| **6** | Numeric column (+ optional categorical split) | **Box Plots** | Quartile spread and IQR dispersion |

### Boundary & Isolation Filters
* **Ambiguous Dates**: Date columns with `needs_user_confirmation=True` are **strictly excluded** from time-series line trend charts.
* **PII Protection**: Columns flagged with `is_pii=True` (e.g. `full_name`, `email`, `phone_number`, `credit_card`) and `semantic_label="identifier"` are **strictly excluded** from chart generation to prevent sensitive leakage into chart titles, axes, or filenames.
* **Target Candidate Awareness**: Intelligence-identified targets (`is_target_candidate=True` or `semantic_label="target_label"`) are indexed in `dio["eda"]["target_candidates"]` to ensure clear separation between predictive targets and input features for Phase 6 ML without data leakage.

---

## 2. Benchmark Datasets Qualitative & Quantitative Review

### 1. Retail Sales Dataset (`retail_sales.csv`)
* **Features Analyzed**: `price`, `quantity`, `discount_pct`, `revenue` (Identifier `order_id` excluded from charts)
* **Summary Statistics Evaluation**:
  * `revenue`: Mean = $104.34, Median = $45.78, Min = $15.50, Max = $408.00 (IQR = 48.42, Skewness = 1.34)
  * `quantity`: Mean = 3.00, Median = 3.00, Min = 1.00, Max = 5.00
  * `discount_pct`: Mean = 0.04, Min = 0.00, Max = 0.15
* **Correlation Findings**: Strong positive correlation between `quantity` and `revenue` ($r = 0.85$), and `price` and `revenue` ($r = 0.84$).
* **Selected Charts (6/6)**:
  1. `01_correlation_heatmap.png` — Multi-feature correlation matrix
  2. `02_trend_price_over_time.png` — Price trend along verified `order_date`
  3. `03_scatter_price_vs_revenue.png` — Price-revenue interaction
  4. `04_distribution_price.png` — Unit price histogram
  5. `04_distribution_quantity.png` — Quantity histogram
  6. `04_distribution_discount_pct.png` — Discount distribution
* **Plausibility & Anomaly Review**: **PASS** — Statistics are mathematically exact. The high-value transaction ($408.00) is visually transparent on distributions without causing calculation distortion.

---

### 2. Healthcare Patients Dataset (`healthcare_patients.csv`)
* **Features Analyzed**: `age`, `systolic_bp`, `diastolic_bp`, `glucose` (Identifier `patient_id` excluded from charts)
* **Summary Statistics Evaluation**:
  * `glucose`: Count = 6 (5 original + 1 imputed), Mean = 122.00 mg/dL, Min = 95.00, Max = 150.00
  * `systolic_bp`: Mean = 127.33 mmHg, Min = 118.00, Max = 142.00
  * `diastolic_bp`: Mean = 82.50 mmHg, Min = 76.00, Max = 90.00
  * `age`: Mean = 48.83 years, Min = 28.00, Max = 67.00
* **Correlation Findings**: Blood pressure coupling ($r = 0.93$ between `systolic_bp` and `diastolic_bp`) accurately reflects cardiovascular physiology.
* **Selected Charts (6/6)**:
  1. `01_correlation_heatmap.png` — Vitals correlation matrix
  2. `02_trend_age_over_time.png` — Admission date trend
  3. `03_scatter_systolic_bp_vs_diastolic_bp.png` — Blood pressure association
  4. `04_distribution_age.png`
  5. `04_distribution_systolic_bp.png`
  6. `04_distribution_diastolic_bp.png`
* **Plausibility & Anomaly Review**: **PASS** — The preserved missing `discharge_date` remained missing, while the imputed glucose (122.0 mg/dL) fits clinical distributions naturally.

---

### 3. Financial Loans Dataset (`financial_loans.csv`)
* **Features Analyzed**: `applicant_age`, `annual_income`, `loan_amount`, `credit_score`, `account_balance`
* **Target Candidate**: `default_status` (Categorical target: `"Paid"`, `"Default"`) correctly identified and isolated in `dio["eda"]["target_candidates"]`.
* **Summary Statistics Evaluation**:
  * `loan_amount`: Mean = $24,666.67, Median = $25,000.00, Min = $5,000.00, Max = $45,000.00
  * `account_balance`: Mean = $5,988.59, Median = $4,750.00, Max = $15,000.75 (Outlier flagged in Phase 4)
  * `credit_score`: Mean = 718.33, Min = 640.00, Max = 790.00
* **Selected Charts (6/6)**:
  1. `01_correlation_heatmap.png` — Financial risk feature correlation
  2. `02_trend_applicant_age_over_time.png` — Application date trend
  3. `03_scatter_annual_income_vs_loan_amount.png` — Debt-to-income distribution
  4. `04_distribution_applicant_age.png`
  5. `04_distribution_annual_income.png`
  6. `04_distribution_loan_amount.png`
* **Plausibility & Anomaly Review**: **PASS** — High-balance account outlier ($15,000.75) is visible in scatter plots. Feature definitions were not contaminated by target statistics.

---

### 4. Mixed Messy Data (`mixed_messy_data.csv`)
* **Features Analyzed**: `metric_score`, `user_category`, `flag_active` (Identifier `record_id` excluded from charts)
* **Summary Statistics Evaluation**:
  * `metric_score`: Count = 6, Mean = 71.53, Median = 74.20, Min = 45.00, Max = 92.00, Skewness = -0.28
  * `user_category`: Unique = 3, Top Category = `"Standard"` (Freq = 3, including 1 synthetic mode imputation)
  * `flag_active`: Unique = 2 (`True`: 4, `False`: 2)
* **Selected Charts (4/4 — Graceful Degradation)**:
  1. `04_distribution_metric_score.png` — Metric score distribution
  2. `05_bar__user_category__counts.png` — Category breakdown
  3. `05_bar_flag_active_counts.png` — Status frequency
  4. `06_boxplot_metric_score.png` — Score dispersion
* **Plausibility & Anomaly Review**: **PASS** — Because only 1 numeric column existed, correlation heatmap and scatter plots were skipped gracefully without error.

---

### 5. Ambiguous Dates & PII Dataset (`ambiguous_dates_pii.csv`)
* **Features Analyzed**: `amount`
* **PII & Privacy Protection**:
  * `full_name`, `email`, `phone_number`, `credit_card`, and `user_id` were **strictly excluded** from all chart artifacts, titles, and metadata.
  * Ambiguous date `subscription_date` was **strictly excluded** from time-series line trends.
* **Summary Statistics Evaluation**:
  * `amount`: Mean = $49.99, Median = $39.99, Min = $19.99, Max = $99.99, Std = 35.59
* **Selected Charts (2/2 — Graceful Degradation)**:
  1. `04_distribution_amount.png` — Subscription amount distribution
  2. `06_boxplot_amount.png` — Amount spread boxplot
* **Plausibility & Anomaly Review**: **PASS** — Zero sensitive PII tokens entered chart artifacts or filenames. Ambiguous date was not forced into a time series.
