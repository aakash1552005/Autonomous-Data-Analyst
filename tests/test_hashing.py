"""
tests/test_hashing.py
=====================
Tests for SHA-256 dataset hashing and deterministic reproducibility.
"""

from pathlib import Path
import pytest
from core.hashing import compute_file_hash, compute_bytes_hash, compute_string_hash


def test_compute_file_hash_deterministic(tmp_path: Path):
    sample_file = tmp_path / "sample.csv"
    sample_file.write_text("id,name,value\n1,Alpha,100\n2,Beta,200\n", encoding="utf-8")

    hash1 = compute_file_hash(sample_file)
    hash2 = compute_file_hash(sample_file)

    assert isinstance(hash1, str)
    assert len(hash1) == 64  # SHA-256 hex string length
    assert hash1 == hash2


def test_compute_file_hash_nonexistent_raises():
    with pytest.raises(FileNotFoundError):
        compute_file_hash("non_existent_file_path_123.csv")


def test_compute_bytes_and_string_hash():
    data_str = "hello autonomous data analyst"
    hash_str = compute_string_hash(data_str)
    hash_bytes = compute_bytes_hash(data_str.encode("utf-8"))

    assert hash_str == hash_bytes
    assert len(hash_str) == 64
