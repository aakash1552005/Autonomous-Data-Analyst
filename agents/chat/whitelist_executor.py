"""
agents/chat/whitelist_executor.py
=================================
Whitelisted pandas execution engine for Agent 7 (Chat Agent).
Executes only explicit, hardcoded mathematical operations with an uncompromising
structural PII shield.

Zero arbitrary execution:
  - NO eval()
  - NO exec()
  - NO dynamic string code generation
  - NO arbitrary attribute or function invocation
  - NO SQL generation
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any
import numpy as np
import pandas as pd


@dataclass
class ExecutionResult:
    """Outcome of a whitelisted pandas calculation."""
    success: bool
    result_text: str
    operation: str
    column: str | None
    status: str  # "success", "refusal", "unavailable", "error"
    numeric_value: float | None = None
    data: dict[str, Any] | None = None


# Whitelist of permitted operations (V1.1 Increment 2: exactly 11 operations)
ALLOWED_OPERATIONS = {
    "mean",
    "sum",
    "count",
    "min",
    "max",
    "value_counts",
    "groupby_mean",
    "median",
    "std",
    "variance",
    "groupby_sum",
}

# Semantic labels that are classified as protected PII or identifiers
SENSITIVE_SEMANTIC_LABELS = {
    "identifier",
    "name",
    "person_name",
    "email",
    "phone",
    "ssn_national_id",
    "national_id",
    "credit_card",
    "address",
    "date_of_birth",
}

# Structural fallback patterns for PII and identifiers
IDENTIFIER_COL_REGEX = re.compile(
    r"(?i)(^(id|uuid|guid|_id|id_|key|code|sku|order_id|customer_id|patient_id|user_id|emp_id|employee_id|transaction_id)$|"
    r"(_id$|^id_|account_num|account_no|acc_num|acc_no|tracking_num|client_id))"
)
NAME_COL_REGEX = re.compile(
    r"(?i)^(full_?name|first_?name|last_?name|customer_?name|patient_?name|employee_?name|user_?name|client_?name|fname|lname|name)$"
)
EMAIL_COL_REGEX = re.compile(r"(?i)(email|e_mail|mail_address|mail)")
PHONE_COL_REGEX = re.compile(r"(?i)(phone|mobile|cell|telephone|contact_num|contact_no|fax)")
SSN_COL_REGEX = re.compile(r"(?i)(ssn|social_security|national_id|aadhaar|pan_card|tax_id|passport)")
CC_COL_REGEX = re.compile(r"(?i)(credit_card|card_num|cc_num|debit_card|card_no|account_number)")
ADDRESS_COL_REGEX = re.compile(r"(?i)(address|street|addr|street_address|home_address|postal_code|zip_code|zipcode)")
DOB_COL_REGEX = re.compile(r"(?i)(dob|birth|birthday|date_of_birth)")

SENSITIVE_NAME_PATTERNS = [
    IDENTIFIER_COL_REGEX,
    NAME_COL_REGEX,
    EMAIL_COL_REGEX,
    PHONE_COL_REGEX,
    SSN_COL_REGEX,
    CC_COL_REGEX,
    ADDRESS_COL_REGEX,
    DOB_COL_REGEX,
]


def is_column_sensitive(col_name: str, columns_info: list[dict[str, Any]]) -> bool:
    """
    Check if a column is flagged as PII, an identifier, or a sensitive entity in the trusted schema.
    Also applies a structural pattern check as a defense-in-depth shield.
    """
    clean_col = str(col_name).strip()

    # 1. Trusted schema metadata inspection
    for col_info in columns_info:
        if col_info.get("name") == clean_col:
            if col_info.get("is_pii") is True:
                return True
            if col_info.get("semantic_label") in SENSITIVE_SEMANTIC_LABELS:
                return True
            pii_type = col_info.get("pii_type")
            if pii_type and str(pii_type).lower() not in ("none", "", "null"):
                return True

    # 2. Structural pattern inspection on column name (defense-in-depth)
    for pat in SENSITIVE_NAME_PATTERNS:
        if pat.search(clean_col):
            return True

    return False


# Backward-compatible alias
_is_column_sensitive = is_column_sensitive


def execute_whitelisted_operation(
    df: pd.DataFrame,
    columns_info: list[dict[str, Any]],
    operation: str,
    column: str | None,
    group_column: str | None = None,
    safe_operations: set[str] | list[str] | None = None,
) -> ExecutionResult:
    """
    Safely execute a whitelisted pandas aggregation with structural PII shielding.

    Rules:
    1. Operation must be in the approved whitelist.
    2. Column existence is verified strictly against df.columns without fuzzy guessing.
    3. Structural PII shield runs BEFORE any DataFrame data access.
    4. Prohibited columns return explicit refusal.
    5. Nonexistent columns return unavailable-information response.
    """
    allowed_ops = set(safe_operations) if safe_operations else ALLOWED_OPERATIONS

    # 1. Operation Whitelist Validation
    if operation not in allowed_ops:
        return ExecutionResult(
            success=False,
            result_text="The requested analytical operation is not supported by the safe executor.",
            operation=operation,
            column=column,
            status="refusal",
        )

    # 2. Row count query (no column required)
    if operation == "count" and column is None:
        n_rows = len(df)
        return ExecutionResult(
            success=True,
            result_text=f"The dataset contains {n_rows} rows.",
            operation=operation,
            column=None,
            status="success",
            numeric_value=float(n_rows),
        )

    # 3. Column Existence Check
    if not column or column not in df.columns:
        return ExecutionResult(
            success=False,
            result_text="The requested information is not available in the dataset or analysis results.",
            operation=operation,
            column=column,
            status="unavailable",
        )

    # 4. Structural PII Shield (Target Column)
    if _is_column_sensitive(column, columns_info):
        return ExecutionResult(
            success=False,
            result_text=f"Refusal: Access to column '{column}' is restricted because it contains sensitive personal data or identifier records.",
            operation=operation,
            column=column,
            status="refusal",
        )

    # 5. Group Column Validation and PII Shield (for GroupBy)
    if operation in ("groupby_mean", "groupby_sum"):
        if not group_column or group_column not in df.columns:
            return ExecutionResult(
                success=False,
                result_text="The requested information is not available in the dataset or analysis results.",
                operation=operation,
                column=column,
                status="unavailable",
            )
        if _is_column_sensitive(group_column, columns_info):
            return ExecutionResult(
                success=False,
                result_text=f"Refusal: Grouping by '{group_column}' is restricted because it contains sensitive personal data or identifier records.",
                operation=operation,
                column=group_column,
                status="refusal",
            )

    # 6. Hardcoded Whitelisted Execution Paths ONLY
    series = df[column]

    if len(series) == 0 and operation in ("mean", "sum", "min", "max", "median", "std", "variance"):
        return ExecutionResult(
            success=True,
            result_text=f"Column '{column}' contains zero valid numeric entries.",
            operation=operation,
            column=column,
            status="success",
            numeric_value=None,
        )

    if operation == "mean":
        if not pd.api.types.is_numeric_dtype(series):
            return ExecutionResult(
                success=False,
                result_text=f"Column '{column}' is non-numeric; mean calculation cannot be performed.",
                operation=operation,
                column=column,
                status="error",
            )
        valid_series = pd.to_numeric(series, errors="coerce").dropna()
        if len(valid_series) == 0:
            return ExecutionResult(
                success=True,
                result_text=f"Column '{column}' contains zero valid numeric entries.",
                operation=operation,
                column=column,
                status="success",
                numeric_value=None,
            )
        res_val = round(float(valid_series.mean()), 4)
        return ExecutionResult(
            success=True,
            result_text=f"The mean of '{column}' is {res_val}.",
            operation=operation,
            column=column,
            status="success",
            numeric_value=res_val,
        )

    elif operation == "sum":
        if not pd.api.types.is_numeric_dtype(series):
            return ExecutionResult(
                success=False,
                result_text=f"Column '{column}' is non-numeric; sum calculation cannot be performed.",
                operation=operation,
                column=column,
                status="error",
            )
        valid_series = pd.to_numeric(series, errors="coerce").dropna()
        res_val = round(float(valid_series.sum()), 4)
        return ExecutionResult(
            success=True,
            result_text=f"The sum of '{column}' is {res_val}.",
            operation=operation,
            column=column,
            status="success",
            numeric_value=res_val,
        )

    elif operation == "count":
        valid_cnt = int(series.dropna().count())
        return ExecutionResult(
            success=True,
            result_text=f"The count of non-null values in '{column}' is {valid_cnt}.",
            operation=operation,
            column=column,
            status="success",
            numeric_value=float(valid_cnt),
        )

    elif operation == "min":
        if not pd.api.types.is_numeric_dtype(series):
            return ExecutionResult(
                success=False,
                result_text=f"Column '{column}' is non-numeric; minimum calculation cannot be performed.",
                operation=operation,
                column=column,
                status="error",
            )
        valid_series = pd.to_numeric(series, errors="coerce").dropna()
        if len(valid_series) == 0:
            return ExecutionResult(
                success=True,
                result_text=f"Column '{column}' contains zero valid numeric entries.",
                operation=operation,
                column=column,
                status="success",
                numeric_value=None,
            )
        res_val = round(float(valid_series.min()), 4)
        return ExecutionResult(
            success=True,
            result_text=f"The minimum value of '{column}' is {res_val}.",
            operation=operation,
            column=column,
            status="success",
            numeric_value=res_val,
        )

    elif operation == "max":
        if not pd.api.types.is_numeric_dtype(series):
            return ExecutionResult(
                success=False,
                result_text=f"Column '{column}' is non-numeric; maximum calculation cannot be performed.",
                operation=operation,
                column=column,
                status="error",
            )
        valid_series = pd.to_numeric(series, errors="coerce").dropna()
        if len(valid_series) == 0:
            return ExecutionResult(
                success=True,
                result_text=f"Column '{column}' contains zero valid numeric entries.",
                operation=operation,
                column=column,
                status="success",
                numeric_value=None,
            )
        res_val = round(float(valid_series.max()), 4)
        return ExecutionResult(
            success=True,
            result_text=f"The maximum value of '{column}' is {res_val}.",
            operation=operation,
            column=column,
            status="success",
            numeric_value=res_val,
        )

    elif operation == "value_counts":
        clean_s = series.dropna().astype(str).str.strip()
        clean_s = clean_s[clean_s != ""]
        top_cats = clean_s.value_counts().head(10).to_dict()
        if not top_cats:
            return ExecutionResult(
                success=True,
                result_text=f"Column '{column}' contains no distinct categories.",
                operation=operation,
                column=column,
                status="success",
                data={},
            )
        items_str = "\n".join([f"- **{cat}**: {count} occurrences" for cat, count in top_cats.items()])
        result_text = f"Top categories in '{column}':\n{items_str}"
        return ExecutionResult(
            success=True,
            result_text=result_text,
            operation=operation,
            column=column,
            status="success",
            data={str(k): int(v) for k, v in top_cats.items()},
        )

    elif operation == "groupby_mean":
        assert group_column is not None
        if not pd.api.types.is_numeric_dtype(series):
            return ExecutionResult(
                success=False,
                result_text=f"Column '{column}' is non-numeric; grouped mean cannot be calculated.",
                operation=operation,
                column=column,
                status="error",
            )
        # Safe groupby
        temp_df = pd.DataFrame({
            "grp": df[group_column].dropna().astype(str),
            "num": pd.to_numeric(df[column], errors="coerce"),
        }).dropna()

        if len(temp_df) == 0:
            return ExecutionResult(
                success=True,
                result_text=f"Zero valid records found for grouping '{column}' by '{group_column}'.",
                operation=operation,
                column=column,
                status="success",
            )

        gb_means = temp_df.groupby("grp")["num"].mean().round(4).head(10).to_dict()
        items_str = "\n".join([f"- **{cat}**: {val}" for cat, val in gb_means.items()])
        result_text = f"Average of '{column}' grouped by '{group_column}' (top categories):\n{items_str}"
        return ExecutionResult(
            success=True,
            result_text=result_text,
            operation=operation,
            column=column,
            status="success",
            data={str(k): float(v) for k, v in gb_means.items()},
        )

    elif operation == "median":
        if not pd.api.types.is_numeric_dtype(series):
            return ExecutionResult(
                success=False,
                result_text=f"Column '{column}' is non-numeric; median calculation cannot be performed.",
                operation=operation,
                column=column,
                status="error",
            )
        valid_series = pd.to_numeric(series, errors="coerce").dropna()
        if len(valid_series) == 0:
            return ExecutionResult(
                success=True,
                result_text=f"Column '{column}' contains zero valid numeric entries.",
                operation=operation,
                column=column,
                status="success",
                numeric_value=None,
            )
        res_val = round(float(valid_series.median()), 4)
        return ExecutionResult(
            success=True,
            result_text=f"The median of '{column}' is {res_val}.",
            operation=operation,
            column=column,
            status="success",
            numeric_value=res_val,
        )

    elif operation == "std":
        if not pd.api.types.is_numeric_dtype(series):
            return ExecutionResult(
                success=False,
                result_text=f"Column '{column}' is non-numeric; standard deviation cannot be performed.",
                operation=operation,
                column=column,
                status="error",
            )
        valid_series = pd.to_numeric(series, errors="coerce").dropna()
        if len(valid_series) == 0:
            return ExecutionResult(
                success=True,
                result_text=f"Column '{column}' contains zero valid numeric entries.",
                operation=operation,
                column=column,
                status="success",
                numeric_value=None,
            )
        res_val = round(float(valid_series.std()), 4) if len(valid_series) > 1 else 0.0
        return ExecutionResult(
            success=True,
            result_text=f"The standard deviation of '{column}' is {res_val}.",
            operation=operation,
            column=column,
            status="success",
            numeric_value=res_val,
        )

    elif operation == "variance":
        if not pd.api.types.is_numeric_dtype(series):
            return ExecutionResult(
                success=False,
                result_text=f"Column '{column}' is non-numeric; variance calculation cannot be performed.",
                operation=operation,
                column=column,
                status="error",
            )
        valid_series = pd.to_numeric(series, errors="coerce").dropna()
        if len(valid_series) == 0:
            return ExecutionResult(
                success=True,
                result_text=f"Column '{column}' contains zero valid numeric entries.",
                operation=operation,
                column=column,
                status="success",
                numeric_value=None,
            )
        res_val = round(float(valid_series.var()), 4) if len(valid_series) > 1 else 0.0
        return ExecutionResult(
            success=True,
            result_text=f"The variance of '{column}' is {res_val}.",
            operation=operation,
            column=column,
            status="success",
            numeric_value=res_val,
        )

    elif operation == "groupby_sum":
        assert group_column is not None
        if not pd.api.types.is_numeric_dtype(series):
            return ExecutionResult(
                success=False,
                result_text=f"Column '{column}' is non-numeric; grouped sum cannot be calculated.",
                operation=operation,
                column=column,
                status="error",
            )
        # Safe groupby
        temp_df = pd.DataFrame({
            "grp": df[group_column].dropna().astype(str),
            "num": pd.to_numeric(df[column], errors="coerce"),
        }).dropna()

        if len(temp_df) == 0:
            return ExecutionResult(
                success=True,
                result_text=f"Zero valid records found for grouping '{column}' by '{group_column}'.",
                operation=operation,
                column=column,
                status="success",
            )

        gb_sums = temp_df.groupby("grp")["num"].sum().round(4).head(10).to_dict()
        items_str = "\n".join([f"- **{cat}**: {val}" for cat, val in gb_sums.items()])
        result_text = f"Total of '{column}' grouped by '{group_column}' (top categories):\n{items_str}"
        return ExecutionResult(
            success=True,
            result_text=result_text,
            operation=operation,
            column=column,
            status="success",
            data={str(k): float(v) for k, v in gb_sums.items()},
        )

    return ExecutionResult(
        success=False,
        result_text="The requested analytical operation could not be resolved.",
        operation=operation,
        column=column,
        status="error",
    )
