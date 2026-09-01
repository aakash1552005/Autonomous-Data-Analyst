# Phase 5 EDA Agent Real Data Validation & Human Review

This directory contains quantitative validation records, human review evaluations, and artifact references for the Exploratory Data Analysis (EDA) Agent across all 5 benchmark datasets.

---

## Deterministic Chart Selection Rule Table

The EDA Agent selects visualization charts without an LLM using the following deterministic rule table:

| Precedence | Dataset Condition | Generated Chart Type | Config / Precedence Rationale |
| :--- | :--- | :--- | :--- |
| **1** | $\ge 3$ numeric columns with non-zero variance | **Correlation Heatmap** | Global feature association overview |
| **2** | Verified Date column + primary Numeric column | **Time-Series Line Trend** | Chronological metric progression |
| **3** | $\ge 2$ numeric columns | **Scatter Plot** | Top correlated numeric pair interaction |
| **4** | Numeric columns | **Histograms (Distributions)** | Unimodal/multimodal spread & outliers |
| **5** | Categorical columns ($\le 15$ unique categories) | **Categorical Bar Charts** | Category frequency breakdowns |
| **6** | Numeric column (+ optional categorical split) | **Box Plots** | Quartile spread and IQR dispersion |

*Configuration*: The maximum chart limit is configured via `config.yaml` (`eda.max_charts: 6`). When fewer columns are available, the agent gracefully degrades without raising exceptions.

---

## Benchmark Datasets Human Review & Evaluation

### 1. Retail Sales Dataset (`retail_sales.csv`)
* **Numeric Columns Analyzed**: `order_id`, `price`, `quantity`, `discount_pct`, `revenue`
* **Summary Statistics Verification**:
  * `revenue`: Mean = $104.34, Median = $45.78, Min = $15.50, Max = $408.00 (IQR = 48.42)
  * `quantity`: Mean = 3.00, Median = 3.00, Min = 1.00, Max = 5.00
  * `discount_pct`: Mean = 0.04, Max = 0.15
* **Correlation Findings**: Strong positive correlation between `quantity` and `revenue` ($r = 0.85$), and `price` and `revenue` ($r = 0.84$).
* **Generated PNG Charts (6/6)**:
  1. `01_correlation_heatmap.png` — Multi-feature correlation matrix
  2. `02_trend_order_id_over_time.png` — Order progression timeline
  3. `03_scatter_price_vs_revenue.png` — Price-revenue relationship
  4. `04_distribution_order_id.png` — Order ID distribution
  5. `04_distribution_price.png` — Unit price distribution
  6. `04_distribution_quantity.png` — Quantity histogram
* **Human Review Assessment**: **PASS** — Statistics match the raw and cleaned transactions exactly. High-ticket order SKU-C303 ($408.00) is visually apparent on scatter and distribution plots.

---

### 2. Healthcare Patients Dataset (`healthcare_patients.csv`)
* **Numeric Columns Analyzed**: `patient_id`, `age`, `systolic_bp`, `diastolic_bp`, `glucose`
* **Summary Statistics Verification**:
  * `glucose`: Count = 6 (5 original + 1 imputed), Mean = 122.00 mg/dL, Min = 95.00, Max = 150.00
  * `systolic_bp`: Mean = 127.33 mmHg, Min = 118.00, Max = 142.00
  * `age`: Mean = 48.83 years, Min = 28.00, Max = 67.00
* **Generated PNG Charts (6/6)**:
  1. `01_correlation_heatmap.png` — Vitals correlation matrix
  2. `02_trend_patient_id_over_time.png` — Admission timeline
  3. `03_scatter_systolic_bp_vs_diastolic_bp.png` — Blood pressure association ($r = 0.93$)
  4. `04_distribution_patient_id.png`
  5. `04_distribution_age.png`
  6. `04_distribution_systolic_bp.png`
* **Human Review Assessment**: **PASS** — Blood pressure correlation ($r = 0.93$) correctly reflects physiological coupling. Imputed glucose preserved sensible glycemic range.

---

### 3. Financial Loans Dataset (`financial_loans.csv`)
* **Numeric Columns Analyzed**: `loan_id`, `applicant_age`, `annual_income`, `loan_amount`, `credit_score`, `account_balance`
* **Summary Statistics Verification**:
  * `loan_amount`: Mean = $24,666.67, Median = $25,000.00, Min = $5,000.00, Max = $45,000.00
  * `account_balance`: Mean = $5,988.59, Max = $15,000.75 (Outlier flagged in cleaning)
  * `credit_score`: Mean = 718.33, Min = 640.00, Max = 790.00
* **Generated PNG Charts (6/6)**:
  1. `01_correlation_heatmap.png` — Financial risk correlation matrix
  2. `02_trend_loan_id_over_time.png` — Application date trend
  3. `03_scatter_annual_income_vs_loan_amount.png` — Debt-to-income distribution
  4. `04_distribution_loan_id.png`
  5. `04_distribution_applicant_age.png`
  6. `04_distribution_annual_income.png`
* **Human Review Assessment**: **PASS** — Numerical outputs are verified. Outlier account balance ($15,000.75) is cleanly isolated without skewing standard statistics.

---

### 4. Mixed Messy Data (`mixed_messy_data.csv`)
* **Numeric Columns Analyzed**: `metric_score`
* **Summary Statistics Verification**:
  * `metric_score`: Count = 6, Mean = 71.53, Median = 74.20, Min = 45.00, Max = 92.00
  * `user_category`: Unique = 3, Top Category = `"Standard"` (Freq = 3, including 1 imputed)
* **Generated PNG Charts (4/4 — Graceful Degradation)**:
  1. `04_distribution_metric_score.png` — Metric score distribution
  2. `05_bar_record_id_counts.png` — Record ID breakdown
  3. `05_bar__user_category__counts.png` — User category frequencies
  4. `05_bar_flag_active_counts.png` — Active/Inactive status counts
* **Human Review Assessment**: **PASS** — Fewer than 3 numeric columns were present, so the correlation heatmap was skipped gracefully. All categorical columns with $\le 15$ values generated frequency charts.

---

### 5. Ambiguous Dates & PII Dataset (`ambiguous_dates_pii.csv`)
* **Numeric Columns Analyzed**: `amount`
* **Summary Statistics Verification**:
  * `amount`: Mean = $49.99, Median = $39.99, Min = $19.99, Max = $99.99
  * PII Columns (`full_name`, `email`, `phone_number`, `credit_card`): Preserved intact; zero LLM leakage
* **Generated PNG Charts (2/2 — Graceful Degradation)**:
  1. `04_distribution_amount.png` — Subscription amount distribution
  2. `06_boxplot_amount.png` — Amount dispersion boxplot
* **Human Review Assessment**: **PASS** — Ambiguous date was not forced into a time-series line trend. PII fields were not exposed.
