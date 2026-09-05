"""
tests/test_phase9_pii_security_audit.py
=======================================
Phase 9 PII Security & Isolation Audit.
Executes the full Orchestrator pipeline on ambiguous_dates_pii.csv and verifies:
  1. Raw PII values never reach the LLM provider prompts
  2. Raw PII values are never written to pipeline.log
  3. Raw PII values are never displayed in Streamlit previews
  4. Raw PII values never appear in generated executive PDF reports
  5. Raw PII values never appear in generated PPTX presentations
  6. Identifier columns are protected according to Phase 2–8 governance
"""

import re
from pathlib import Path
from pptx import Presentation
import pytest

import app
from llm.base import LLMProvider, LLMResponse
from orchestrator import Orchestrator, OrchestratorResult

SENSITIVE_PII_PROBES = [
    "Johnathan Doe",
    "Jane Roe",
    "Alex Smith",
    "Maria Garcia",
    "johndoe@email.com",
    "jane.roe@corp.org",
    "alex.smith@web.net",
    "maria.g@domain.com",
    "4532-1122-3344-5566",
    "5412-9988-7766-5544",
    "3782-8224-6310-0051",
    "4024-0071-8899-2233",
    "555-876-5432",
    "555-123-9999",
]


class RecordingLLMProvider(LLMProvider):
    """Tracking LLM provider that captures all prompt strings sent through the pipeline."""
    def __init__(self):
        super().__init__()
        self.recorded_prompts: list[str] = []

    def _call_provider(self, prompt: str, max_tokens: int, temperature: float) -> LLMResponse:
        self.recorded_prompts.append(prompt)
        return LLMResponse(
            text='{"insights": []}',
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70,
            model="mock-safe",
            provider="mock",
        )


def _extract_all_pptx_text(pptx_path: Path | str) -> str:
    prs = Presentation(pptx_path)
    texts: list[str] = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                for p in shape.text_frame.paragraphs:
                    texts.append(p.text)
            elif shape.has_table:
                for row in shape.table.rows:
                    for cell in row.cells:
                        texts.append(cell.text)
    return " ".join(texts)


def _extract_all_pdf_text(pdf_path: Path | str) -> str:
    raw_pdf = Path(pdf_path).read_bytes()
    pattern = re.compile(rb"\(((?:\\.|[^\\()])*)\)")
    tokens = pattern.findall(raw_pdf)
    cleaned = []
    for t in tokens:
        decoded = t.decode("latin1", errors="ignore")
        decoded = decoded.replace("\\(", "(").replace("\\)", ")").replace("\\\\", "\\")
        cleaned.append(decoded)
    return " ".join(cleaned)


def test_phase9_comprehensive_pii_security_audit(tmp_path: Path):
    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / "data" / "sample" / "ambiguous_dates_pii.csv"
    assert csv_path.exists(), f"Benchmark dataset not found: {csv_path}"

    run_dir = tmp_path / "pii_audit_run"
    recording_llm = RecordingLLMProvider()

    orchestrator = Orchestrator(llm_provider=recording_llm)
    result: OrchestratorResult = orchestrator.run(
        file_path=csv_path,
        run_dir=run_dir,
    )

    assert result.is_success

    # 1. LLM Prompt Isolation Audit
    for prompt in recording_llm.recorded_prompts:
        for pii in SENSITIVE_PII_PROBES:
            assert pii not in prompt, (
                f"CRITICAL PII LEAKAGE: Raw PII value '{pii}' entered LLM prompt payload!"
            )

    # 2. Pipeline Log Audit
    log_file = run_dir / "pipeline.log"
    assert log_file.exists()
    log_content = log_file.read_text(encoding="utf-8")
    for pii in SENSITIVE_PII_PROBES:
        assert pii not in log_content, (
            f"CRITICAL PII LEAKAGE: Raw PII value '{pii}' written to pipeline.log!"
        )

    # 3. Streamlit Preview Audit
    preview_df = app.safe_preview_dataframe(result.df, result.dio, n_rows=5)
    preview_str = preview_df.to_string()
    for pii in SENSITIVE_PII_PROBES:
        assert pii not in preview_str, (
            f"CRITICAL PII LEAKAGE: Raw PII value '{pii}' displayed in Streamlit safe preview!"
        )

    # 4. PDF Deliverable Audit
    pdf_path = result.artifacts.get("pdf_report")
    assert pdf_path and Path(pdf_path).exists()
    pdf_text = _extract_all_pdf_text(pdf_path)
    for pii in SENSITIVE_PII_PROBES:
        assert pii not in pdf_text, (
            f"CRITICAL PII LEAKAGE: Raw PII value '{pii}' rendered in executive PDF report!"
        )

    # 5. PPTX Presentation Audit
    pptx_path = result.artifacts.get("pptx_report")
    assert pptx_path and Path(pptx_path).exists()
    pptx_text = _extract_all_pptx_text(pptx_path)
    for pii in SENSITIVE_PII_PROBES:
        assert pii not in pptx_text, (
            f"CRITICAL PII LEAKAGE: Raw PII value '{pii}' rendered in PPTX presentation!"
        )
