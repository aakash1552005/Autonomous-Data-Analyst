"""
core/hashing.py
===============
Reproducible dataset hashing utilities using SHA-256.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def compute_file_hash(file_path: str | Path, chunk_size: int = 65536) -> str:
    """
    Compute SHA-256 hash of a file using streaming chunks.
    Ensures low memory footprint even for files up to 200MB.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Cannot hash non-existent file: {path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_bytes_hash(data: bytes) -> str:
    """Compute SHA-256 hash of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def compute_string_hash(text: str) -> str:
    """Compute SHA-256 hash of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
