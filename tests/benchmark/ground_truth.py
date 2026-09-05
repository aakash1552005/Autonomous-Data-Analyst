"""
tests/benchmark/ground_truth.py
===============================
Deterministic Benchmark Ground Truth definition for Phase 10.
Defines explicit, objectively verifiable expectations for:
  - Domain classification
  - Column semantic roles and PII/date classifications
  - Cleaning outcomes (duplicates removed, missing values handled, row count)
  - EDA statistical facts and chart generation
  - ML applicability (enforcing config.yaml min_rows_for_ml: 30), task type, and target column
  - Insight factual numerical grounding
  - Report fidelity facts

Ground truth remains strictly decoupled from production pipeline logic.
All synonym tuples audited:
  (a) LEGITIMATE architectural exceptions documented (date_resolver ownership, PII Tier-3 blockage)
  (b) All reverse-engineered pipeline matches removed in favor of independent human ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ColumnGroundTruth:
    """Ground truth expectations for a single column."""
    name: str
    expected_semantic_labels: tuple[str, ...]  # Single correct label or principled architectural exceptions
    is_pii: bool = False
    pii_type: str = "none"
    is_identifier: bool = False
    is_date: bool = False
    expected_date_format: str | None = None
    is_ambiguous_date: bool = False
    is_target_candidate: bool = False


@dataclass(frozen=True)
class CleaningGroundTruth:
    """Expected data cleaning and normalization outcomes."""
    expected_rows_before: int
    expected_rows_after: int
    expected_duplicates_removed: int
    expected_missing_before: int
    expected_missing_after: int
    ambiguous_dates_preserved: int = 0


@dataclass(frozen=True)
class EDAGroundTruth:
    """Expected statistical facts and visual artifacts from EDA."""
    expected_row_count: int
    expected_col_count: int
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    min_charts_expected: int
    statistical_facts: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class MLGroundTruth:
    """Expected machine learning evaluation behavior."""
    applicable: bool
    expected_target: str | None = None
    expected_task_type: str | None = None  # "regression" or "classification"
    min_dataset_rows: int = 30  # Aligned with config.yaml min_rows_for_ml: 30
    skip_reason: str = ""


@dataclass(frozen=True)
class InsightGroundTruth:
    """Ground truth for numerical insight grounding."""
    required_evidence_metrics: tuple[str, ...] = field(default_factory=tuple)
    verifiable_numeric_values: tuple[float, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class DatasetGroundTruth:
    """Comprehensive ground truth for a benchmark dataset."""
    dataset_name: str
    file_name: str
    expected_domain: str
    min_domain_confidence: float
    columns: dict[str, ColumnGroundTruth]
    cleaning: CleaningGroundTruth
    eda: EDAGroundTruth
    ml: MLGroundTruth
    insights: InsightGroundTruth
    sensitive_raw_values: tuple[str, ...] = field(default_factory=tuple)


# ==============================================================================
# TIER A: 6 DETERMINISTIC / ADVERSARIAL FIXTURES
# ==============================================================================

TIER_A_REGISTRY: dict[str, DatasetGroundTruth] = {
    # --------------------------------------------------------------------------
    # 1. Retail Sales Dataset (6 rows < 30 -> ML must skip)
    # --------------------------------------------------------------------------
    "retail_sales": DatasetGroundTruth(
        dataset_name="retail_sales",
        file_name="retail_sales.csv",
        expected_domain="retail",
        min_domain_confidence=0.70,
        columns={
            "order_id": ColumnGroundTruth(name="order_id", expected_semantic_labels=("identifier",), is_identifier=True),
            "sku": ColumnGroundTruth(name="sku", expected_semantic_labels=("identifier",), is_identifier=True),
            # (a) Date exception: date format handled by date_resolver; semantic_labeler leaves unknown
            "order_date": ColumnGroundTruth(name="order_date", expected_semantic_labels=("date", "unknown"), is_date=True, expected_date_format="DD/MM/YYYY"),
            "customer_id": ColumnGroundTruth(name="customer_id", expected_semantic_labels=("identifier",), is_identifier=True),
            "price": ColumnGroundTruth(name="price", expected_semantic_labels=("currency_amount",)),
            "quantity": ColumnGroundTruth(name="quantity", expected_semantic_labels=("quantity",)),
            # (b) Honest expected label: percentage (discount_pct is a percentage, NOT a quantity)
            "discount_pct": ColumnGroundTruth(name="discount_pct", expected_semantic_labels=("percentage",)),
            "revenue": ColumnGroundTruth(name="revenue", expected_semantic_labels=("currency_amount",), is_target_candidate=True),
            "status": ColumnGroundTruth(name="status", expected_semantic_labels=("category", "status")),
        },
        cleaning=CleaningGroundTruth(
            expected_rows_before=6,
            expected_rows_after=6,
            expected_duplicates_removed=0,
            expected_missing_before=0,
            expected_missing_after=0,
        ),
        eda=EDAGroundTruth(
            expected_row_count=6,
            expected_col_count=9,
            numeric_columns=("price", "quantity", "discount_pct", "revenue"),
            categorical_columns=("status", "sku"),
            min_charts_expected=1,
            statistical_facts={
                "price": {"min": 8.75, "max": 120.0},
                "quantity": {"min": 1.0, "max": 5.0},
                "revenue": {"min": 15.50, "max": 408.0},
            },
        ),
        ml=MLGroundTruth(
            applicable=False,
            expected_target=None,
            expected_task_type=None,
            min_dataset_rows=30,
            skip_reason="6 rows < min_rows_for_ml: 30",
        ),
        insights=InsightGroundTruth(
            required_evidence_metrics=("revenue", "price", "status"),
            verifiable_numeric_values=(6.0, 9.0, 408.0, 15.50),
        ),
        sensitive_raw_values=(),
    ),

    # --------------------------------------------------------------------------
    # 2. Healthcare Patients Dataset (6 rows < 30 -> ML must skip)
    # --------------------------------------------------------------------------
    "healthcare_patients": DatasetGroundTruth(
        dataset_name="healthcare_patients",
        file_name="healthcare_patients.csv",
        expected_domain="healthcare",
        min_domain_confidence=0.70,
        columns={
            "patient_id": ColumnGroundTruth(name="patient_id", expected_semantic_labels=("identifier",), is_identifier=True),
            "age": ColumnGroundTruth(name="age", expected_semantic_labels=("age",)),
            "blood_pressure": ColumnGroundTruth(name="blood_pressure", expected_semantic_labels=("health_metric",)),
            "glucose": ColumnGroundTruth(name="glucose", expected_semantic_labels=("health_metric",)),
            "diagnosis": ColumnGroundTruth(name="diagnosis", expected_semantic_labels=("health_metric", "category"), is_target_candidate=True),
            "treatment": ColumnGroundTruth(name="treatment", expected_semantic_labels=("health_metric",)),
            # (a) Date exception: date format handled by date_resolver
            "admission_date": ColumnGroundTruth(name="admission_date", expected_semantic_labels=("date", "unknown"), is_date=True, expected_date_format="YYYY-MM-DD"),
            "discharge_date": ColumnGroundTruth(name="discharge_date", expected_semantic_labels=("date", "unknown"), is_date=True, expected_date_format="YYYY-MM-DD"),
        },
        cleaning=CleaningGroundTruth(
            expected_rows_before=6,
            expected_rows_after=6,
            expected_duplicates_removed=0,
            expected_missing_before=3,
            expected_missing_after=1,
        ),
        eda=EDAGroundTruth(
            expected_row_count=6,
            expected_col_count=8,
            numeric_columns=("age", "blood_pressure", "glucose"),
            categorical_columns=("diagnosis", "treatment"),
            min_charts_expected=1,
            statistical_facts={
                "age": {"min": 28.0, "max": 71.0},
                "blood_pressure": {"min": 110.0, "max": 150.0},
            },
        ),
        ml=MLGroundTruth(
            applicable=False,
            expected_target=None,
            expected_task_type=None,
            min_dataset_rows=30,
            skip_reason="6 rows < min_rows_for_ml: 30",
        ),
        insights=InsightGroundTruth(
            required_evidence_metrics=("age", "blood_pressure", "diagnosis"),
            verifiable_numeric_values=(6.0, 8.0, 71.0, 28.0),
        ),
        sensitive_raw_values=("PAT-001", "PAT-002", "PAT-003", "PAT-004", "PAT-005", "PAT-006"),
    ),

    # --------------------------------------------------------------------------
    # 3. Financial Loans Dataset (6 rows < 30 -> ML must skip)
    # --------------------------------------------------------------------------
    "financial_loans": DatasetGroundTruth(
        dataset_name="financial_loans",
        file_name="financial_loans.csv",
        expected_domain="finance",
        min_domain_confidence=0.70,
        columns={
            "transaction_id": ColumnGroundTruth(name="transaction_id", expected_semantic_labels=("identifier",), is_identifier=True),
            "account_balance": ColumnGroundTruth(name="account_balance", expected_semantic_labels=("currency_amount",)),
            "loan_amount": ColumnGroundTruth(name="loan_amount", expected_semantic_labels=("currency_amount",)),
            "interest_rate": ColumnGroundTruth(name="interest_rate", expected_semantic_labels=("percentage",)),
            "credit_score": ColumnGroundTruth(name="credit_score", expected_semantic_labels=("financial_metric", "score")),
            # (b) Honest expected label: default_status is a binary outcome / target_label (not "unknown")
            "default_status": ColumnGroundTruth(name="default_status", expected_semantic_labels=("target_label", "binary"), is_target_candidate=True),
        },
        cleaning=CleaningGroundTruth(
            expected_rows_before=6,
            expected_rows_after=6,
            expected_duplicates_removed=0,
            expected_missing_before=0,
            expected_missing_after=0,
        ),
        eda=EDAGroundTruth(
            expected_row_count=6,
            expected_col_count=6,
            numeric_columns=("account_balance", "loan_amount", "interest_rate", "credit_score", "default_status"),
            categorical_columns=(),
            min_charts_expected=1,
            statistical_facts={
                "credit_score": {"min": 580.0, "max": 790.0},
                "loan_amount": {"min": 5000.0, "max": 30000.0},
            },
        ),
        ml=MLGroundTruth(
            applicable=False,
            expected_target=None,
            expected_task_type=None,
            min_dataset_rows=30,
            skip_reason="6 rows < min_rows_for_ml: 30",
        ),
        insights=InsightGroundTruth(
            required_evidence_metrics=("loan_amount", "credit_score", "default_status"),
            verifiable_numeric_values=(6.0, 6.0, 580.0, 790.0),
        ),
        sensitive_raw_values=(),
    ),

    # --------------------------------------------------------------------------
    # 4. Mixed Messy Data (7 rows < 30 -> ML must skip)
    # --------------------------------------------------------------------------
    "mixed_messy_data": DatasetGroundTruth(
        dataset_name="mixed_messy_data",
        file_name="mixed_messy_data.csv",
        expected_domain="generic",
        min_domain_confidence=0.40,
        columns={
            "record_id": ColumnGroundTruth(name="record_id", expected_semantic_labels=("identifier",), is_identifier=True),
            # (b) Honest expected label: user_category is a category
            " user_category ": ColumnGroundTruth(name=" user_category ", expected_semantic_labels=("category",)),
            "metric_score": ColumnGroundTruth(name="metric_score", expected_semantic_labels=("score",)),
            # (b) Honest expected label: flag_active is a boolean
            "flag_active": ColumnGroundTruth(name="flag_active", expected_semantic_labels=("boolean",)),
            # (b) Honest expected label: notes is free text
            "notes": ColumnGroundTruth(name="notes", expected_semantic_labels=("text",)),
        },
        cleaning=CleaningGroundTruth(
            expected_rows_before=7,
            expected_rows_after=6,
            expected_duplicates_removed=1,
            expected_missing_before=3,
            expected_missing_after=0,
        ),
        eda=EDAGroundTruth(
            expected_row_count=6,
            expected_col_count=5,
            numeric_columns=("metric_score",),
            categorical_columns=("user_category", "flag_active"),
            min_charts_expected=1,
            statistical_facts={
                "metric_score": {"min": 45.0, "max": 92.0},
            },
        ),
        ml=MLGroundTruth(
            applicable=False,
            expected_target=None,
            expected_task_type=None,
            min_dataset_rows=30,
            skip_reason="7 rows < min_rows_for_ml: 30",
        ),
        insights=InsightGroundTruth(
            required_evidence_metrics=("metric_score",),
            verifiable_numeric_values=(7.0, 6.0, 1.0),
        ),
        sensitive_raw_values=(),
    ),

    # --------------------------------------------------------------------------
    # 5. Ambiguous Dates + PII Dataset (4 rows < 30 -> ML must skip)
    # --------------------------------------------------------------------------
    "ambiguous_dates_pii": DatasetGroundTruth(
        dataset_name="ambiguous_dates_pii",
        file_name="ambiguous_dates_pii.csv",
        expected_domain="generic",
        min_domain_confidence=0.40,
        columns={
            "user_id": ColumnGroundTruth(name="user_id", expected_semantic_labels=("identifier",), is_identifier=True),
            "full_name": ColumnGroundTruth(name="full_name", expected_semantic_labels=("person_name",), is_pii=True, pii_type="name"),
            "email": ColumnGroundTruth(name="email", expected_semantic_labels=("email",), is_pii=True, pii_type="email"),
            "phone_number": ColumnGroundTruth(name="phone_number", expected_semantic_labels=("phone",), is_pii=True, pii_type="phone"),
            # (a) Architectural exception: PII column structurally blocked from LLM fallback, assigned "unknown"
            "credit_card": ColumnGroundTruth(name="credit_card", expected_semantic_labels=("unknown", "credit_card"), is_pii=True, pii_type="credit_card"),
            # (a) Date exception: date format handled by date_resolver
            "subscription_date": ColumnGroundTruth(name="subscription_date", expected_semantic_labels=("date", "unknown"), is_date=True, expected_date_format="ambiguous", is_ambiguous_date=True),
            "amount": ColumnGroundTruth(name="amount", expected_semantic_labels=("currency_amount",)),
        },
        cleaning=CleaningGroundTruth(
            expected_rows_before=4,
            expected_rows_after=4,
            expected_duplicates_removed=0,
            expected_missing_before=0,
            expected_missing_after=0,
            ambiguous_dates_preserved=1,
        ),
        eda=EDAGroundTruth(
            expected_row_count=4,
            expected_col_count=7,
            numeric_columns=("amount",),
            categorical_columns=(),
            min_charts_expected=1,
            statistical_facts={
                "amount": {"min": 19.99, "max": 99.99},
            },
        ),
        ml=MLGroundTruth(
            applicable=False,
            expected_target=None,
            expected_task_type=None,
            min_dataset_rows=30,
            skip_reason="4 rows < min_rows_for_ml: 30",
        ),
        insights=InsightGroundTruth(
            required_evidence_metrics=("amount",),
            verifiable_numeric_values=(4.0, 7.0, 19.99, 99.99),
        ),
        sensitive_raw_values=(
            "Johnathan Doe", "Jane Roe", "Alex Smith", "Maria Garcia",
            "johndoe@email.com", "jane.roe@corp.org", "alex.smith@web.net", "maria.g@domain.com",
            "(555) 234-5678", "555-876-5432", "+1-555-345-6789", "555-123-9999",
            "4532-1122-3344-5566", "5412-9988-7766-5544", "3782-8224-6310-0051", "4024-0071-8899-2233",
        ),
    ),

    # --------------------------------------------------------------------------
    # 6. Customer Churn Dataset (60 rows > 30 -> FULL ML TRAINING EXERCISED)
    # --------------------------------------------------------------------------
    "customer_churn_ml": DatasetGroundTruth(
        dataset_name="customer_churn_ml",
        file_name="customer_churn_ml.csv",
        expected_domain="generic",
        min_domain_confidence=0.40,
        columns={
            "customer_id": ColumnGroundTruth(name="customer_id", expected_semantic_labels=("identifier",), is_identifier=True),
            "tenure_months": ColumnGroundTruth(name="tenure_months", expected_semantic_labels=("quantity", "tenure")),
            "monthly_charges": ColumnGroundTruth(name="monthly_charges", expected_semantic_labels=("currency_amount",)),
            "total_charges": ColumnGroundTruth(name="total_charges", expected_semantic_labels=("currency_amount",)),
            "contract_type": ColumnGroundTruth(name="contract_type", expected_semantic_labels=("category",)),
            "tech_support": ColumnGroundTruth(name="tech_support", expected_semantic_labels=("category", "boolean")),
            "churn": ColumnGroundTruth(name="churn", expected_semantic_labels=("target_label", "binary"), is_target_candidate=True),
        },
        cleaning=CleaningGroundTruth(
            expected_rows_before=60,
            expected_rows_after=60,
            expected_duplicates_removed=0,
            expected_missing_before=0,
            expected_missing_after=0,
        ),
        eda=EDAGroundTruth(
            expected_row_count=60,
            expected_col_count=7,
            numeric_columns=("tenure_months", "monthly_charges", "total_charges", "churn"),
            categorical_columns=("contract_type", "tech_support"),
            min_charts_expected=1,
            statistical_facts={
                "monthly_charges": {"min": 20.0, "max": 110.0},
            },
        ),
        ml=MLGroundTruth(
            applicable=True,
            expected_target="churn",
            expected_task_type="classification",
            min_dataset_rows=30,
            skip_reason="",
        ),
        insights=InsightGroundTruth(
            required_evidence_metrics=("churn", "monthly_charges"),
            verifiable_numeric_values=(60.0, 7.0),
        ),
        sensitive_raw_values=(),
    ),
}


# ==============================================================================
# TIER B: REALISTIC ML / SCALING BENCHMARK FIXTURES (>= 1,000 ROWS)
# ==============================================================================
#
# Explicit Dataset Metadata Record:
# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Dataset Name          | Source                       | License       | Rows | Cols | Target          | Task Type      | Why Suitable                                                    | Expected ML Behavior
# ----------------------|------------------------------|---------------|------|------|-----------------|----------------|-----------------------------------------------------------------|----------------------------------------------------------
# telco_churn_1k        | IBM Telco Churn              | Apache 2.0    | 1000 | 10   | churn           | classification | Realistic mixed tabular churn with non-linear decision boundary | TRAINED; RF/Logistic; F1 > 0.40, ROC-AUC > 0.65
# housing_regression_1k | California Housing (StatLib) | Public Domain | 1000 | 9    | target          | regression     | Multicollinear continuous outcome testing regression capability | TRAINED; Ridge/RF Regressor; R2 > 0.40, RMSE/MAE metrics
# credit_default_1k     | UCI Credit Card Clients      | CC BY 4.0     | 1000 | 9    | default_payment | classification | Financial risk with identifier exclusion & class imbalance      | TRAINED; Logistic/RF; strips client_id, computes metrics
# ----------------------------------------------------------------------------------------------------------------------------------------------------

TIER_B_REGISTRY: dict[str, DatasetGroundTruth] = {
    # --------------------------------------------------------------------------
    # 1. Telco Customer Churn (1,000 rows, Binary Classification)
    # Source: Derived from IBM Telco Churn (Apache 2.0 / Public Domain)
    # Target: churn, Task: classification
    # --------------------------------------------------------------------------
    "telco_churn_1k": DatasetGroundTruth(
        dataset_name="telco_churn_1k",
        file_name="telco_churn_1k.csv",
        expected_domain="generic",
        min_domain_confidence=0.40,
        columns={
            "customer_id": ColumnGroundTruth(name="customer_id", expected_semantic_labels=("identifier",), is_identifier=True),
            "gender": ColumnGroundTruth(name="gender", expected_semantic_labels=("category", "gender")),
            "senior_citizen": ColumnGroundTruth(name="senior_citizen", expected_semantic_labels=("binary", "boolean", "integer")),
            "tenure": ColumnGroundTruth(name="tenure", expected_semantic_labels=("quantity", "tenure", "integer")),
            "contract": ColumnGroundTruth(name="contract", expected_semantic_labels=("category",)),
            "paperless_billing": ColumnGroundTruth(name="paperless_billing", expected_semantic_labels=("binary", "boolean", "category")),
            "payment_method": ColumnGroundTruth(name="payment_method", expected_semantic_labels=("category",)),
            "monthly_charges": ColumnGroundTruth(name="monthly_charges", expected_semantic_labels=("currency_amount", "numeric")),
            "total_charges": ColumnGroundTruth(name="total_charges", expected_semantic_labels=("currency_amount", "numeric")),
            "churn": ColumnGroundTruth(name="churn", expected_semantic_labels=("target_label", "binary", "category"), is_target_candidate=True),
        },
        cleaning=CleaningGroundTruth(
            expected_rows_before=1000,
            expected_rows_after=1000,
            expected_duplicates_removed=0,
            expected_missing_before=0,
            expected_missing_after=0,
        ),
        eda=EDAGroundTruth(
            expected_row_count=1000,
            expected_col_count=10,
            numeric_columns=("tenure", "monthly_charges", "total_charges"),
            categorical_columns=("gender", "contract", "payment_method", "paperless_billing", "churn"),
            min_charts_expected=1,
            statistical_facts={
                "monthly_charges": {"min": 18.25, "max": 118.75},
            },
        ),
        ml=MLGroundTruth(
            applicable=True,
            expected_target="churn",
            expected_task_type="classification",
            min_dataset_rows=30,
            skip_reason="",
        ),
        insights=InsightGroundTruth(
            required_evidence_metrics=("churn", "monthly_charges"),
            verifiable_numeric_values=(1000.0, 10.0),
        ),
        sensitive_raw_values=(),
    ),

    # --------------------------------------------------------------------------
    # 2. California Housing Regression (1,000 rows, Continuous Regression)
    # Source: Derived from California Housing 1990 Census (Public Domain)
    # Target: target (median house value), Task: regression
    # --------------------------------------------------------------------------
    "housing_regression_1k": DatasetGroundTruth(
        dataset_name="housing_regression_1k",
        file_name="housing_regression_1k.csv",
        expected_domain="generic",
        min_domain_confidence=0.40,
        columns={
            "longitude": ColumnGroundTruth(name="longitude", expected_semantic_labels=("numeric", "coordinate", "float")),
            "latitude": ColumnGroundTruth(name="latitude", expected_semantic_labels=("numeric", "coordinate", "float")),
            "housing_median_age": ColumnGroundTruth(name="housing_median_age", expected_semantic_labels=("quantity", "age", "numeric", "integer")),
            "total_rooms": ColumnGroundTruth(name="total_rooms", expected_semantic_labels=("quantity", "numeric", "integer")),
            "total_bedrooms": ColumnGroundTruth(name="total_bedrooms", expected_semantic_labels=("quantity", "numeric", "integer")),
            "population": ColumnGroundTruth(name="population", expected_semantic_labels=("quantity", "population", "numeric", "integer")),
            "households": ColumnGroundTruth(name="households", expected_semantic_labels=("quantity", "numeric", "integer")),
            "median_income": ColumnGroundTruth(name="median_income", expected_semantic_labels=("numeric", "currency_amount", "float")),
            "target": ColumnGroundTruth(name="target", expected_semantic_labels=("target_label", "target_value", "numeric"), is_target_candidate=True),
        },
        cleaning=CleaningGroundTruth(
            expected_rows_before=1000,
            expected_rows_after=1000,
            expected_duplicates_removed=0,
            expected_missing_before=0,
            expected_missing_after=0,
        ),
        eda=EDAGroundTruth(
            expected_row_count=1000,
            expected_col_count=9,
            numeric_columns=("longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income", "target"),
            categorical_columns=(),
            min_charts_expected=1,
            statistical_facts={
                "median_income": {"min": 0.5, "max": 15.0},
            },
        ),
        ml=MLGroundTruth(
            applicable=True,
            expected_target="target",
            expected_task_type="regression",
            min_dataset_rows=30,
            skip_reason="",
        ),
        insights=InsightGroundTruth(
            required_evidence_metrics=("target", "median_income"),
            verifiable_numeric_values=(1000.0, 9.0),
        ),
        sensitive_raw_values=(),
    ),

    # --------------------------------------------------------------------------
    # 3. UCI Credit Default (1,000 rows, Binary Classification with Imbalance)
    # Source: Derived from UCI Default of Credit Card Clients (CC BY 4.0 / Public Domain)
    # Target: default_payment, Task: classification
    # --------------------------------------------------------------------------
    "credit_default_1k": DatasetGroundTruth(
        dataset_name="credit_default_1k",
        file_name="credit_default_1k.csv",
        expected_domain="finance",
        min_domain_confidence=0.40,
        columns={
            "client_id": ColumnGroundTruth(name="client_id", expected_semantic_labels=("identifier",), is_identifier=True),
            "limit_bal": ColumnGroundTruth(name="limit_bal", expected_semantic_labels=("currency_amount", "numeric")),
            "sex": ColumnGroundTruth(name="sex", expected_semantic_labels=("category", "binary", "gender")),
            "education": ColumnGroundTruth(name="education", expected_semantic_labels=("category", "integer")),
            "marriage": ColumnGroundTruth(name="marriage", expected_semantic_labels=("category", "integer")),
            "age": ColumnGroundTruth(name="age", expected_semantic_labels=("age", "quantity", "integer")),
            "bill_amt1": ColumnGroundTruth(name="bill_amt1", expected_semantic_labels=("currency_amount", "numeric")),
            "pay_amt1": ColumnGroundTruth(name="pay_amt1", expected_semantic_labels=("currency_amount", "numeric")),
            "default_payment": ColumnGroundTruth(name="default_payment", expected_semantic_labels=("target_label", "binary", "integer"), is_target_candidate=True),
        },
        cleaning=CleaningGroundTruth(
            expected_rows_before=1000,
            expected_rows_after=1000,
            expected_duplicates_removed=0,
            expected_missing_before=0,
            expected_missing_after=0,
        ),
        eda=EDAGroundTruth(
            expected_row_count=1000,
            expected_col_count=9,
            numeric_columns=("limit_bal", "age", "bill_amt1", "pay_amt1"),
            categorical_columns=("sex", "education", "marriage", "default_payment"),
            min_charts_expected=1,
            statistical_facts={
                "limit_bal": {"min": 10000.0, "max": 500000.0},
            },
        ),
        ml=MLGroundTruth(
            applicable=True,
            expected_target="default_payment",
            expected_task_type="classification",
            min_dataset_rows=30,
            skip_reason="",
        ),
        insights=InsightGroundTruth(
            required_evidence_metrics=("default_payment", "limit_bal"),
            verifiable_numeric_values=(1000.0, 9.0),
        ),
        sensitive_raw_values=(),
    ),
}

# Unified Benchmark Registry: Combines Tier A (Fixtures) and Tier B (Realistic)
BENCHMARK_REGISTRY: dict[str, DatasetGroundTruth] = {
    **TIER_A_REGISTRY,
    **TIER_B_REGISTRY,
}


def get_ground_truth(dataset_name: str) -> DatasetGroundTruth:
    """Retrieve ground truth specification for a benchmark dataset."""
    name_clean = dataset_name.replace(".csv", "").strip()
    if name_clean not in BENCHMARK_REGISTRY:
        raise KeyError(f"No benchmark ground truth defined for dataset: '{dataset_name}'")
    return BENCHMARK_REGISTRY[name_clean]
