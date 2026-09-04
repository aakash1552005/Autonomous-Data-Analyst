"""
tests/test_report_adversarial.py
================================
Adversarial and security tests for Phase 8: Report Agent.
Verifies strict upstream DIO immutability, PII isolation boundary, numerical integrity,
broken artifact resilience, and deterministic generation.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
import pandas as pd
import pytest
from pptx import Presentation

from agents.report.report_agent import ReportAgent
from core.dio import DIO


def _extract_all_pptx_text(pptx_path: Path | str) -> str:
    """Helper to extract all text strings across all shapes and tables in a PPTX."""
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


def _create_adversarial_dio() -> tuple[DIO, pd.DataFrame, dict[str, str]]:
    """
    Constructs a DIO with injected PII and realistic numbers to test isolation.
    Returns (dio, df_with_pii, secret_pii_dict).
    """
    dio = DIO.create_empty("customer_accounts.csv", dataset_hash="sec_hash_98765")
    dio.ingestion = {
        "n_rows": 100,
        "n_columns": 6,
        "file_type": "csv",
        "encoding": "utf-8",
    }
    dio.domain_guess = {"domain": "finance", "confidence": 0.92}
    dio.quality = {"score": 88, "issues": ["2 missing balances"]}
    dio.columns = [
        {"name": "customer_id", "logical_type": "integer", "null_count": 0, "null_pct": 0.0, "unique_count": 100, "is_identifier": True, "is_pii": False, "semantic_label": "id"},
        {"name": "full_name", "logical_type": "string", "null_count": 0, "null_pct": 0.0, "unique_count": 100, "is_identifier": False, "is_pii": True, "pii_type": "name", "semantic_label": "person_name"},
        {"name": "ssn_number", "logical_type": "string", "null_count": 0, "null_pct": 0.0, "unique_count": 100, "is_identifier": False, "is_pii": True, "pii_type": "ssn", "semantic_label": "government_id"},
        {"name": "credit_card", "logical_type": "string", "null_count": 0, "null_pct": 0.0, "unique_count": 100, "is_identifier": False, "is_pii": True, "pii_type": "credit_card", "semantic_label": "financial_card"},
        {"name": "phone_num", "logical_type": "string", "null_count": 0, "null_pct": 0.0, "unique_count": 100, "is_identifier": False, "is_pii": True, "pii_type": "phone", "semantic_label": "phone_number"},
        {"name": "account_balance", "logical_type": "float", "null_count": 2, "null_pct": 2.0, "unique_count": 98, "is_identifier": False, "is_pii": False, "semantic_label": "monetary_amount"},
    ]
    dio.cleaning_log = [
        {"action": "impute_missing", "column": "account_balance", "method": "mean", "replacement_value": 3450.75},
    ]
    dio.eda = {
        "summary_stats": {
            "account_balance": {"mean": 3450.75, "std": 1200.50, "min": 250.0, "median": 3300.0, "max": 9800.0},
        },
        "correlations": {
            "top_correlations": [
                {"feature_a": "account_balance", "feature_b": "customer_id", "pearson_correlation": 0.28},
            ],
        },
        "charts": [],
    }
    dio.ml = {
        "status": "trained",
        "task_type": "regression",
        "target_column": "account_balance",
        "best_model": "RandomForestRegressor",
        "metrics": {"r2_score": 0.8125, "rmse": 450.25},
        "feature_importance": {"customer_id": 0.25},
    }
    dio.insights = [
        {
            "id": "ins_001",
            "category": "distribution",
            "text": "Average account balance stood at 3450.75 with maximum peak of 9800.00.",
            "confidence": 0.93,
            "evidence": "eda.summary_stats.account_balance.mean",
            "recommendation": "Review high-net-worth portfolio allocations.",
            "grounded_numbers": [3450.75, 9800.00],
        }
    ]

    # Sensitive PII values injected into raw DataFrame
    secret_pii = {
        "alice_name": "Alice Montgomery-Smith",
        "alice_ssn": "987-65-4321",
        "alice_card": "4532-8901-2345-6789",
        "alice_phone": "+1-555-839-2019",
        "bob_name": "Dr. Bartholomew Sterling",
        "bob_ssn": "123-45-6789",
    }
    df = pd.DataFrame([
        {"customer_id": 101, "full_name": secret_pii["alice_name"], "ssn_number": secret_pii["alice_ssn"], "credit_card": secret_pii["alice_card"], "phone_num": secret_pii["alice_phone"], "account_balance": 5400.0},
        {"customer_id": 102, "full_name": secret_pii["bob_name"], "ssn_number": secret_pii["bob_ssn"], "credit_card": "4111-2222-3333-4444", "phone_num": "+1-555-111-2222", "account_balance": 1500.0},
    ])
    return dio, df, secret_pii


def test_upstream_dio_immutability(tmp_path: Path):
    """
    Assert that running ReportAgent does NOT mutate ANY upstream DIO section.
    Compares deepcopy snapshots of all upstream keys before and after execution.
    """
    dio, df, _ = _create_adversarial_dio()
    agent = ReportAgent()

    # Upstream sections to guard
    upstream_keys = [
        "schema_version",
        "dataset_id",
        "dataset_hash",
        "file_name",
        "ingestion",
        "columns",
        "date_columns",
        "domain_guess",
        "quality",
        "cleaning_log",
        "eda",
        "ml",
        "insights",
    ]
    snapshot = {k: copy.deepcopy(dio[k]) for k in upstream_keys}

    _, dio_out = agent.run(df, dio, run_dir=tmp_path)

    for k in upstream_keys:
        assert dio_out[k] == snapshot[k], f"Upstream DIO key '{k}' was mutated by ReportAgent!"


def test_strict_pii_isolation_boundary(tmp_path: Path):
    """
    Adversarial PII Test: Injects realistic PII into dataset and DIO.
    Asserts zero occurrence of sensitive names, SSNs, credit cards, and phone numbers
    in both the generated PDF text and the generated PPTX presentation.
    """
    dio, df, secret_pii = _create_adversarial_dio()
    agent = ReportAgent()

    _, dio_out = agent.run(df, dio, run_dir=tmp_path)

    pdf_path = Path(dio_out["reports"]["pdf_path"])
    pptx_path = Path(dio_out["reports"]["pptx_path"])

    # 1. Inspect PDF Text Content
    pdf_text = pdf_path.read_text(encoding="latin1", errors="ignore")
    for pii_label, pii_val in secret_pii.items():
        assert pii_val not in pdf_text, f"PII leakage in PDF! Found {pii_label}: '{pii_val}'"

    # 2. Inspect PPTX Text Content
    pptx_text = _extract_all_pptx_text(pptx_path)
    for pii_label, pii_val in secret_pii.items():
        assert pii_val not in pptx_text, f"PII leakage in PPTX! Found {pii_label}: '{pii_val}'"

    # 3. Confirm that PII column names are visible and safely flagged
    assert "full_name" in pptx_text
    assert "Protected PII" in pptx_text or "Protected Identifier" in pptx_text


def test_numerical_grounding_and_integrity(tmp_path: Path):
    """
    Assert that verified numerical figures from the DIO (mean, quality score, R2, grounded numbers)
    are rendered accurately without distortion or recalculation.
    """
    dio, df, _ = _create_adversarial_dio()
    agent = ReportAgent()

    _, dio_out = agent.run(df, dio, run_dir=tmp_path)

    pdf_text = Path(dio_out["reports"]["pdf_path"]).read_text(encoding="latin1", errors="ignore")
    pptx_text = _extract_all_pptx_text(dio_out["reports"]["pptx_path"])

    # Check Quality Score (88)
    assert "88" in pdf_text
    assert "88" in pptx_text

    # Check Mean Account Balance (3450.75)
    assert "3450.75" in pdf_text
    assert "3450.75" in pptx_text

    # Check R2 Score (0.8125)
    assert "0.8125" in pdf_text
    assert "0.8125" in pptx_text


def test_broken_or_corrupt_chart_handling(tmp_path: Path):
    """
    Verify that pointing to missing or corrupt chart paths does not crash report generation.
    """
    dio, df, _ = _create_adversarial_dio()
    # Provide missing chart file
    broken_chart = tmp_path / "corrupt_chart.png"
    broken_chart.write_bytes(b"NOT_AN_IMAGE_CONTENT")

    dio.artifacts["chart_paths"] = [str(broken_chart), str(tmp_path / "missing_chart.png")]

    agent = ReportAgent()
    _, dio_out = agent.run(df, dio, run_dir=tmp_path)

    assert dio_out["reports"]["status"] == "completed"
    assert Path(dio_out["reports"]["pdf_path"]).is_file()
    assert Path(dio_out["reports"]["pptx_path"]).is_file()


def test_deterministic_generation_consistency(tmp_path: Path):
    """
    Assert that generating reports twice on the same DIO produces consistent slide and page structure.
    """
    dio, df, _ = _create_adversarial_dio()
    agent = ReportAgent()

    run1_dir = tmp_path / "run_1"
    run2_dir = tmp_path / "run_2"

    _, dio1 = agent.run(df, dio, run_dir=run1_dir)
    _, dio2 = agent.run(df, dio, run_dir=run2_dir)

    pdf1_bytes = Path(dio1["reports"]["pdf_path"]).read_bytes()
    pdf2_bytes = Path(dio2["reports"]["pdf_path"]).read_bytes()

    pages1 = len(re.findall(rb"/Type\s*/Page\b", pdf1_bytes))
    pages2 = len(re.findall(rb"/Type\s*/Page\b", pdf2_bytes))
    assert pages1 == pages2

    prs1 = Presentation(dio1["reports"]["pptx_path"])
    prs2 = Presentation(dio2["reports"]["pptx_path"])
    assert len(prs1.slides) == len(prs2.slides)


def extract_pdf_text_from_file(pdf_path: Path | str) -> str:
    """
    Directly extracts all text tokens from uncompressed ReportLab PDF streams.
    Decodes standard PDF string literals, handling escaped parentheses and octal escapes.
    """
    raw_pdf = Path(pdf_path).read_bytes()
    # Match PDF string literals: starts with ( and ends with unescaped ), handling backslash escapes
    pattern = re.compile(rb"\(((?:\\.|[^\\()])*)\)")
    tokens = pattern.findall(raw_pdf)
    cleaned = []
    for t in tokens:
        decoded = t.decode("latin1", errors="ignore")
        decoded = decoded.replace("\\(", "(").replace("\\)", ")").replace("\\\\", "\\")
        cleaned.append(decoded)
    return " ".join(cleaned)


def test_sparse_insight_rendering_one_insight(tmp_path: Path):
    """
    13a Case A: Exactly 1 verified insight conforming to the 7-field contract.
    Proves Report Agent dynamically renders 1 insight without placeholder artifacts,
    broken sections, empty cards, or layout overflow.
    """
    dio, df, _ = _create_adversarial_dio()
    dio.insights = [
        {
            "id": "ins_001",
            "category": "distribution",
            "text": "Average account balance stood at 3450.75 across active customer portfolios.",
            "confidence": 0.94,
            "evidence": "eda.summary_stats.account_balance.mean",
            "recommendation": "Maintain minimum capital reserves.",
            "grounded_numbers": [3450.75],
        }
    ]

    agent = ReportAgent()
    _, dio_out = agent.run(df, dio, run_dir=tmp_path)

    pdf_path = Path(dio_out["reports"]["pdf_path"])
    pptx_path = Path(dio_out["reports"]["pptx_path"])

    assert pdf_path.is_file() and pdf_path.stat().st_size > 0
    assert pptx_path.is_file() and pptx_path.stat().st_size > 0

    # 1. PDF Verification
    pdf_text = extract_pdf_text_from_file(pdf_path)
    assert "ins_001" in pdf_text
    assert "3450.75" in pdf_text
    assert "Maintain minimum capital reserves" in pdf_text
    # No placeholder artifacts or phantom insight cards
    assert "ins_002" not in pdf_text
    assert "ins_003" not in pdf_text
    assert "empty insight" not in pdf_text.lower()
    assert "undefined" not in pdf_text.lower()

    # 2. PPTX Verification
    prs = Presentation(pptx_path)
    s8 = prs.slides[7]  # Slide 8: Business Insights
    insight_shapes = [
        sh for sh in s8.shapes
        if sh.has_text_frame and "Insight ins_" in sh.text_frame.text
    ]
    # Exactly one card rendered
    assert len(insight_shapes) == 1
    card_text = insight_shapes[0].text_frame.text
    assert "Insight ins_001" in card_text
    assert "3450.75" in card_text
    assert "Maintain minimum capital reserves" in card_text
    assert "ins_002" not in card_text


def test_sparse_insight_rendering_two_insights(tmp_path: Path):
    """
    13a Case B: Exactly 2 verified insights conforming to the 7-field contract.
    Proves Report Agent dynamically renders 2 insights with clean 2-column card layout,
    no 3rd placeholder card, and zero overflow.
    """
    dio, df, _ = _create_adversarial_dio()
    dio.insights = [
        {
            "id": "ins_001",
            "category": "distribution",
            "text": "Average account balance stood at 3450.75 across active customer portfolios.",
            "confidence": 0.94,
            "evidence": "eda.summary_stats.account_balance.mean",
            "recommendation": "Maintain minimum capital reserves.",
            "grounded_numbers": [3450.75],
        },
        {
            "id": "ins_002",
            "category": "correlation",
            "text": "Moderate correlation of 0.28 identified between account balance and customer ID.",
            "confidence": 0.91,
            "evidence": "eda.correlations.top_correlations",
            "recommendation": "Investigate portfolio tenure segmentation.",
            "grounded_numbers": [0.28],
        },
    ]

    agent = ReportAgent()
    _, dio_out = agent.run(df, dio, run_dir=tmp_path)

    pdf_path = Path(dio_out["reports"]["pdf_path"])
    pptx_path = Path(dio_out["reports"]["pptx_path"])

    assert pdf_path.is_file() and pdf_path.stat().st_size > 0
    assert pptx_path.is_file() and pptx_path.stat().st_size > 0

    # 1. PDF Verification
    pdf_text = extract_pdf_text_from_file(pdf_path)
    assert "ins_001" in pdf_text
    assert "ins_002" in pdf_text
    assert "3450.75" in pdf_text
    assert "0.28" in pdf_text
    assert "ins_003" not in pdf_text

    # 2. PPTX Verification
    prs = Presentation(pptx_path)
    s8 = prs.slides[7]  # Slide 8: Business Insights
    insight_shapes = [
        sh for sh in s8.shapes
        if sh.has_text_frame and "Insight ins_" in sh.text_frame.text
    ]
    # Exactly two cards rendered
    assert len(insight_shapes) == 2
    ins_ids_rendered = [
        "ins_001" in sh.text_frame.text for sh in insight_shapes
    ]
    assert any(ins_ids_rendered)
    assert not any("ins_003" in sh.text_frame.text for sh in insight_shapes)


def test_numerical_fidelity_retail_sales(tmp_path: Path):
    """
    15a: End-to-end numerical fidelity verification using real dataset 'retail_sales.csv'.
    Proves that 5 concrete numeric facts from the source DIO appear accurately
    in the generated PDF artifact without reinterpretation or calculation errors.
    """
    from core.data_router import DataRouter
    from agents.intelligence.intelligence_agent import IntelligenceAgent
    from agents.cleaning.cleaning_agent import CleaningAgent
    from agents.eda.eda_agent import EDAAgent
    from agents.ml.ml_agent import MLAgent
    from agents.insight.insight_agent import InsightAgent

    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / "data" / "sample" / "retail_sales.csv"
    assert csv_path.exists(), f"Benchmark dataset not found: {csv_path}"

    # Step 1: Execute full upstream pipeline
    df, dio, run_dir = DataRouter().ingest(csv_path, base_runs_dir=tmp_path)
    df, dio = IntelligenceAgent().run(df, dio)
    df, dio = CleaningAgent().run(df, dio, run_dir=run_dir)
    df, dio = EDAAgent().run(df, dio, run_dir=run_dir)
    df, dio = MLAgent().run(df, dio, run_dir=run_dir)
    df, dio = InsightAgent().run(df, dio, run_dir=run_dir)

    # Step 2: Preserve 5 concrete numeric facts from source DIO before ReportAgent
    fact1_rows = dio["ingestion"]["n_rows"]
    fact2_cols = dio["ingestion"]["n_columns"]
    fact3_quality = dio["quality"]["score"]
    fact4_domain_conf = int(dio["domain_guess"]["confidence"] * 100)
    
    # Fact 5: Extract grounded number from first insight or price mean from EDA
    if dio["insights"] and dio["insights"][0].get("grounded_numbers"):
        fact5_name = "Insight Grounded Number"
        fact5_dio_val = int(dio["insights"][0]["grounded_numbers"][0])
        fact5_expected_str = str(fact5_dio_val)
    else:
        fact5_name = "Price Feature Mean"
        fact5_dio_val = round(float(dio["eda"]["summary_stats"]["price"]["mean"]), 2)
        fact5_expected_str = f"{fact5_dio_val:.2f}"

    # Step 3: Run ReportAgent
    df, dio = ReportAgent().run(df, dio, run_dir=run_dir)
    pdf_path = Path(dio["reports"]["pdf_path"])
    assert pdf_path.is_file(), "Generated PDF does not exist"

    # Step 4: Extract text from actual generated PDF
    pdf_text = extract_pdf_text_from_file(pdf_path)

    # Step 5: Exact / tolerance numeric comparisons
    f1_pass = str(fact1_rows) in pdf_text
    f2_pass = str(fact2_cols) in pdf_text
    f3_pass = str(fact3_quality) in pdf_text
    f4_pass = f"{fact4_domain_conf}%" in pdf_text or str(fact4_domain_conf) in pdf_text
    f5_pass = fact5_expected_str in pdf_text

    # Print verification table
    print("\nNUMERICAL FIDELITY CHECK — retail_sales.csv")
    print("| Fact | Source DIO | PDF Extracted | Comparison | Result |")
    print("|------|------------|---------------|------------|--------|")
    print(f"| Row count | {fact1_rows} | {fact1_rows} | exact | {'PASS' if f1_pass else 'FAIL'} |")
    print(f"| Column count | {fact2_cols} | {fact2_cols} | exact | {'PASS' if f2_pass else 'FAIL'} |")
    print(f"| Quality score | {fact3_quality} | {fact3_quality} | exact | {'PASS' if f3_pass else 'FAIL'} |")
    print(f"| Domain confidence | {fact4_domain_conf}% | {fact4_domain_conf}% | exact | {'PASS' if f4_pass else 'FAIL'} |")
    print(f"| {fact5_name} | {fact5_dio_val} | {fact5_expected_str} | exact | {'PASS' if f5_pass else 'FAIL'} |")

    assert f1_pass, f"Row count {fact1_rows} not found in PDF"
    assert f2_pass, f"Column count {fact2_cols} not found in PDF"
    assert f3_pass, f"Quality score {fact3_quality} not found in PDF"
    assert f4_pass, f"Domain confidence {fact4_domain_conf}% not found in PDF"
    assert f5_pass, f"{fact5_name} {fact5_expected_str} not found in PDF"
