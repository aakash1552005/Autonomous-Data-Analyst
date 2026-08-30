"""
tests/test_schema_profiler.py
=============================
Tests for schema profiling, data type inference, duplicate tracking, and anomaly detection.
"""

import pandas as pd
from agents.intelligence.schema_profiler import profile_schema, infer_column_dtype


def test_infer_column_dtype():
    assert infer_column_dtype(pd.Series([1, 2, 3, 4])) == "int"
    assert infer_column_dtype(pd.Series([1.5, 2.7, 3.1])) == "float"
    assert infer_column_dtype(pd.Series([True, False, True])) == "bool"
    assert infer_column_dtype(pd.Series(["yes", "no", "yes"])) == "bool"
    assert infer_column_dtype(pd.Series(["2024-01-15", "2024-02-20"])) == "date"
    assert infer_column_dtype(pd.Series(["Red", "Blue", "Green"] * 10)) == "category"


def test_profile_schema_dimensions_and_nulls():
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "name": ["Alpha", "Beta", None, "Delta"],
        "is_active": [True, False, True, True],
    })
    schema = profile_schema(df)

    assert schema["n_rows"] == 4
    assert schema["n_columns"] == 3
    assert schema["duplicate_rows_count"] == 0

    name_col = next(c for c in schema["columns"] if c["name"] == "name")
    assert name_col["null_pct"] == 0.25
    assert name_col["unique_count"] == 3
