"""
tests/test_source_adapters.py
=============================
Tests for CSVAdapter and ExcelAdapter implementations.
Verifies delimiter detection, encoding detection, and normalized DataFrame output.
"""

from pathlib import Path
import pytest
import pandas as pd
from sources.csv_adapter import CSVAdapter
from sources.excel_adapter import ExcelAdapter


def test_csv_adapter_comma_delimited(tmp_path: Path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("id,city,temp\n1,Boston,68.5\n2,Seattle,62.0\n", encoding="utf-8")

    adapter = CSVAdapter()
    assert adapter.validate(csv_path) is True

    df = adapter.load(csv_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ["id", "city", "temp"]
    assert df.iloc[0]["city"] == "Boston"


def test_csv_adapter_semicolon_delimited(tmp_path: Path):
    csv_path = tmp_path / "data_semi.csv"
    csv_path.write_text("id;product;revenue\n101;Widget;450.0\n102;Gadget;890.5\n", encoding="utf-8")

    adapter = CSVAdapter()
    df = adapter.load(csv_path)
    assert len(df) == 2
    assert list(df.columns) == ["id", "product", "revenue"]
    assert df.iloc[1]["revenue"] == 890.5


def test_csv_adapter_tab_delimited(tmp_path: Path):
    tsv_path = tmp_path / "data.tsv"
    tsv_path.write_text("col_1\tcol_2\nval_a\tval_b\nval_c\tval_d\n", encoding="utf-8")

    adapter = CSVAdapter()
    assert adapter.validate(tsv_path) is True
    df = adapter.load(tsv_path)
    assert len(df) == 2
    assert list(df.columns) == ["col_1", "col_2"]


def test_csv_adapter_latin1_encoding(tmp_path: Path):
    latin1_path = tmp_path / "latin1.csv"
    # Write Latin-1 characters (e.g. Café, résumé)
    content = "id,name\n1,Café\n2,Résumé\n"
    latin1_path.write_bytes(content.encode("latin-1"))

    adapter = CSVAdapter()
    df = adapter.load(latin1_path)
    assert len(df) == 2
    assert df.iloc[0]["name"] == "Café"


def test_excel_adapter_xlsx(tmp_path: Path):
    xlsx_path = tmp_path / "sheet.xlsx"
    original_df = pd.DataFrame({"sku": ["SKU1", "SKU2"], "stock": [100, 250], "active": [True, False]})
    original_df.to_excel(xlsx_path, index=False, engine="openpyxl")

    adapter = ExcelAdapter()
    assert adapter.validate(xlsx_path) is True

    df = adapter.load(xlsx_path)
    assert len(df) == 2
    assert list(df.columns) == ["sku", "stock", "active"]
    assert df.iloc[0]["stock"] == 100


def test_adapters_describe_metadata():
    csv_meta = CSVAdapter().describe()
    excel_meta = ExcelAdapter().describe()

    assert csv_meta["source_type"] == "csv"
    assert ".csv" in csv_meta["supported_extensions"]
    assert excel_meta["source_type"] == "excel"
    assert ".xlsx" in excel_meta["supported_extensions"]
