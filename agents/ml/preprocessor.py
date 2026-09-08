"""
agents/ml/preprocessor.py
=========================
Leakage-safe preprocessing pipelines using scikit-learn ColumnTransformer.
Ensures transformers are fitted strictly on the training partition.
"""

from __future__ import annotations

from typing import Any
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessing_pipeline(
    df: pd.DataFrame,
    feature_cols: list[str],
    columns_info: list[dict[str, Any]] | None = None,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    """
    Construct ColumnTransformer for numeric scaling/imputation and categorical encoding.
    Returns (column_transformer, numeric_cols, categorical_cols).
    """
    col_dict = {c["name"]: c for c in (columns_info or [])}
    numeric_features: list[str] = []
    categorical_features: list[str] = []

    for col in feature_cols:
        meta = col_dict.get(col, {})
        dtype_inf = meta.get("dtype_inferred", "string")
        if (dtype_inf in ("int", "float") or pd.api.types.is_numeric_dtype(df[col])) and not pd.api.types.is_bool_dtype(df[col]):
            numeric_features.append(col)
        else:
            categorical_features.append(col)

    transformers = []

    if numeric_features:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("numeric", num_pipeline, numeric_features))

    if categorical_features:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=20)),
        ])
        transformers.append(("categorical", cat_pipeline, categorical_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return preprocessor, numeric_features, categorical_features
