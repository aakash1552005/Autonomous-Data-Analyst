"""
security/file_validator.py
==========================
Security and input validation for uploaded datasets.
Enforces size limits, content signature sniffing, zero-row/column guards,
path traversal sanitization, and SHA-256 dataset hashing.
"""

from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.hashing import compute_file_hash
from core.persistence import sanitize_filename

# Disallowed binary magic byte prefixes (to catch binary executables/media renamed as .csv)
BINARY_SIGNATURES = {
    b"MZ": "Windows Executable (PE/EXE/DLL)",
    b"\x7fELF": "Linux Executable (ELF)",
    b"%PDF": "PDF Document",
    b"\x89PNG\r\n\x1a\n": "PNG Image",
    b"\xff\xd8\xff": "JPEG Image",
    b"GIF87a": "GIF Image",
    b"GIF89a": "GIF Image",
}

# Excel signatures
XLSX_ZIP_MAGIC = b"PK\x03\x04"
XLS_LEGACY_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


class FileValidationError(Exception):
    """Raised when file validation fails security or structural criteria."""


@dataclass
class ValidationResult:
    """Detailed summary of dataset security and structural validation."""
    is_valid: bool
    file_type: str = ""  # csv | xlsx | xls
    file_size_bytes: int = 0
    file_size_mb: float = 0.0
    dataset_hash: str = ""
    sanitized_name: str = ""
    encoding: str = "utf-8"
    delimiter: str = ","
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class FileValidator:
    """
    Validates uploaded tabular files before they reach any pipeline agent.
    """

    def __init__(self, max_upload_size_mb: int = 200) -> None:
        self.max_upload_size_mb = max_upload_size_mb
        self.max_upload_size_bytes = max_upload_size_mb * 1024 * 1024

    def validate(self, file_path: str | Path) -> ValidationResult:
        path = Path(file_path).resolve()
        if not path.is_file():
            return ValidationResult(
                is_valid=False,
                errors=[f"File does not exist: {path}"],
            )

        sanitized_name = sanitize_filename(path.name)
        file_size_bytes = path.stat().st_size
        file_size_mb = file_size_bytes / (1024 * 1024)

        result = ValidationResult(
            is_valid=True,
            file_size_bytes=file_size_bytes,
            file_size_mb=round(file_size_mb, 2),
            sanitized_name=sanitized_name,
        )

        # 1. Zero-byte check
        if file_size_bytes == 0:
            result.is_valid = False
            result.errors.append("File is empty (0 bytes).")
            return result

        # 2. Maximum upload size limit
        if file_size_bytes > self.max_upload_size_bytes:
            result.is_valid = False
            result.errors.append(
                f"Dataset too large for this version (maximum {self.max_upload_size_mb}MB)."
            )
            return result

        # 3. Extension check
        extension = path.suffix.lower()
        if extension not in (".csv", ".xlsx", ".xls"):
            result.is_valid = False
            result.errors.append(
                f"Unsupported file extension '{extension}'. Only .csv, .xlsx, and .xls are supported in V1."
            )
            return result

        # 4. SHA-256 Dataset Hash
        try:
            result.dataset_hash = compute_file_hash(path)
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Failed to compute file hash: {e}")
            return result

        # 5. Content Signature Sniffing & Structural Checks
        with open(path, "rb") as f:
            header_bytes = f.read(4096)

        if extension in (".xlsx", ".xls"):
            self._validate_excel(path, extension, header_bytes, result)
        else:
            self._validate_csv(path, header_bytes, result)

        return result

    def _validate_excel(
        self,
        path: Path,
        extension: str,
        header_bytes: bytes,
        result: ValidationResult,
    ) -> None:
        result.file_type = extension.lstrip(".")

        if extension == ".xlsx":
            if not header_bytes.startswith(XLSX_ZIP_MAGIC):
                result.is_valid = False
                result.errors.append(
                    "Content sniffing failed: .xlsx file is not a valid OpenXML Zip archive."
                )
                return

            try:
                with zipfile.ZipFile(path, "r") as zf:
                    namelist = zf.namelist()
                    if "[Content_Types].xml" not in namelist and not any(n.startswith("xl/") for n in namelist):
                        result.is_valid = False
                        result.errors.append(
                            "Invalid Excel structure: missing '[Content_Types].xml' or 'xl/' folder in .xlsx."
                        )
                        return
            except zipfile.BadZipFile:
                result.is_valid = False
                result.errors.append("Corrupted .xlsx file: unable to read zip structure.")
                return

        elif extension == ".xls":
            if not header_bytes.startswith(XLS_LEGACY_MAGIC):
                result.is_valid = False
                result.errors.append(
                    "Content sniffing failed: .xls file is not a valid legacy OLE2 compound document."
                )
                return

    def _validate_csv(
        self,
        path: Path,
        header_bytes: bytes,
        result: ValidationResult,
    ) -> None:
        result.file_type = "csv"

        # Check binary magic signatures
        for sig, desc in BINARY_SIGNATURES.items():
            if header_bytes.startswith(sig):
                result.is_valid = False
                result.errors.append(
                    f"Content sniffing failed: file has .csv extension but signature matches {desc}."
                )
                return

        # Detect encoding
        encoding = self._detect_encoding(header_bytes)
        result.encoding = encoding

        try:
            with open(path, "r", encoding=encoding, errors="replace") as f:
                sample_text = "".join(f.readline() for _ in range(50))
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Failed to read CSV text: {e}")
            return

        if not sample_text.strip():
            result.is_valid = False
            result.errors.append("Dataset contains no parseable text.")
            return

        # Detect delimiter & structure
        try:
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample_text, delimiters=",\t;|")
            result.delimiter = dialect.delimiter
        except Exception:
            # Fallback to comma if sniffer is inconclusive
            result.delimiter = ","

        # Inspect rows & columns
        reader = csv.reader(io.StringIO(sample_text), delimiter=result.delimiter)
        rows = [r for r in reader if any(cell.strip() for cell in r)]

        if not rows:
            result.is_valid = False
            result.errors.append("Dataset contains zero parseable rows.")
            return

        header = rows[0]
        if not header or len(header) == 0:
            result.is_valid = False
            result.errors.append("Dataset contains zero parseable columns.")
            return

        if len(rows) < 2:
            result.is_valid = False
            result.errors.append("Dataset contains no data rows (zero rows).")
            return

    def _detect_encoding(self, sample_bytes: bytes) -> str:
        """Detect encoding, preferring utf-8 then latin-1."""
        try:
            sample_bytes.decode("utf-8")
            return "utf-8"
        except UnicodeDecodeError:
            try:
                sample_bytes.decode("latin-1")
                return "latin-1"
            except Exception:
                return "utf-8"
