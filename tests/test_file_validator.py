"""
tests/test_file_validator.py
============================
Comprehensive tests for FileValidator and security boundary checks.
Verifies file signatures, size limits, zero-row guards, and extension sniffing.
"""

from pathlib import Path
import pytest
import pandas as pd
from security.file_validator import FileValidator, ValidationResult


def test_validator_accepts_valid_csv(tmp_path: Path):
    csv_file = tmp_path / "valid.csv"
    csv_file.write_text("id,name,amount\n1,Alice,100\n2,Bob,200\n", encoding="utf-8")

    validator = FileValidator(max_upload_size_mb=200)
    res: ValidationResult = validator.validate(csv_file)

    assert res.is_valid is True
    assert res.file_type == "csv"
    assert len(res.dataset_hash) == 64
    assert res.errors == []


def test_validator_accepts_valid_xlsx(tmp_path: Path):
    xlsx_file = tmp_path / "valid.xlsx"
    df = pd.DataFrame({"id": [1, 2, 3], "product": ["A", "B", "C"], "price": [10.5, 20.0, 30.25]})
    df.to_excel(xlsx_file, index=False, engine="openpyxl")

    validator = FileValidator(max_upload_size_mb=200)
    res: ValidationResult = validator.validate(xlsx_file)

    assert res.is_valid is True
    assert res.file_type == "xlsx"
    assert len(res.dataset_hash) == 64
    assert res.errors == []


def test_validator_rejects_unsupported_extensions(tmp_path: Path):
    json_file = tmp_path / "data.json"
    json_file.write_text('{"name": "test"}', encoding="utf-8")

    validator = FileValidator()
    res = validator.validate(json_file)

    assert res.is_valid is False
    assert any("Unsupported file extension" in err for err in res.errors)


def test_validator_rejects_empty_file(tmp_path: Path):
    empty_file = tmp_path / "empty.csv"
    empty_file.write_bytes(b"")

    validator = FileValidator()
    res = validator.validate(empty_file)

    assert res.is_valid is False
    assert "File is empty (0 bytes)." in res.errors


def test_validator_rejects_oversized_file(tmp_path: Path):
    # Set limit to 1 MB for testing
    validator = FileValidator(max_upload_size_mb=1)

    oversized_file = tmp_path / "large.csv"
    # Write 1.5 MB of data
    oversized_file.write_bytes(b"col1,col2\n" + b"123,456\n" * 150000)

    res = validator.validate(oversized_file)
    assert res.is_valid is False
    assert "Dataset too large for this version (maximum 1MB)." in res.errors


def test_validator_content_sniffing_catches_binary_disguised_as_csv(tmp_path: Path):
    fake_csv = tmp_path / "malicious.csv"
    # Write binary PE executable header (MZ)
    fake_csv.write_bytes(b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff\x00\x00")

    validator = FileValidator()
    res = validator.validate(fake_csv)

    assert res.is_valid is False
    assert any("Content sniffing failed" in err for err in res.errors)


def test_validator_content_sniffing_catches_text_disguised_as_xlsx(tmp_path: Path):
    fake_xlsx = tmp_path / "fake.xlsx"
    fake_xlsx.write_text("id,name,value\n1,Alpha,100\n", encoding="utf-8")

    validator = FileValidator()
    res = validator.validate(fake_xlsx)

    assert res.is_valid is False
    assert any("Content sniffing failed" in err for err in res.errors)


def test_validator_rejects_zero_rows_headers_only(tmp_path: Path):
    headers_only = tmp_path / "headers_only.csv"
    headers_only.write_text("col_a,col_b,col_c\n", encoding="utf-8")

    validator = FileValidator()
    res = validator.validate(headers_only)

    assert res.is_valid is False
    assert "Dataset contains no data rows (zero rows)." in res.errors


def test_validator_rejects_nonexistent_file():
    validator = FileValidator()
    res = validator.validate("non_existent_file_path.csv")
    assert res.is_valid is False
    assert any("File does not exist" in err for err in res.errors)
