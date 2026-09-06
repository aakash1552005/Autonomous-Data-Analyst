"""
tests/test_phase11_pii_e2e_security.py
======================================
Dedicated End-to-End PII Security and Leakage Elimination Regression Suite.

Verifies:
1. PIIEvaluator detects EXACTLY 0 raw PII leakages across all benchmark fixtures.
2. Summary statistics omit top_category and top_categories for is_pii=True or identifier columns.
3. Serialized DIO JSON contains ZERO raw sensitive values.
4. Pipeline execution logs contain ZERO raw sensitive values.
5. Agent 7 Chat responses contain ZERO raw sensitive values.
"""

import json
from pathlib import Path
import pytest

from orchestrator import Orchestrator
from tests.benchmark.ground_truth import TIER_A_REGISTRY
from tests.benchmark.evaluators.pii_evaluator import PIIEvaluator
from agents.chat.chat_agent import ChatAgent


@pytest.mark.parametrize("dataset_key", ["healthcare_patients", "ambiguous_dates_pii"])
def test_zero_raw_pii_leakage_e2e(dataset_key, tmp_path):
    """
    Assert that executing the full pipeline yields 0 raw PII leakage events
    in DIO, logs, and summaries.
    """
    gt = TIER_A_REGISTRY[dataset_key]
    csv_path = Path("data/sample") / gt.file_name
    assert csv_path.exists(), f"Benchmark fixture {csv_path} not found"

    run_dir = tmp_path / f"run_{dataset_key}"
    orchestrator = Orchestrator()
    result = orchestrator.run(
        file_path=csv_path,
        run_dir=run_dir,
    )
    assert result.status == "completed"

    # Evaluate via PIIEvaluator
    pii_evaluator = PIIEvaluator()
    eval_res = pii_evaluator.evaluate(result.dio, gt, run_dir=run_dir)

    # 1. Zero raw PII leakage count
    assert eval_res.raw_pii_leakage_count == 0, (
        f"Raw PII leakage detected in {dataset_key}! Count={eval_res.raw_pii_leakage_count}"
    )
    assert eval_res.leakage_detected is False

    # 2. Verify summary stats omit top_category / top_categories on sensitive columns
    summary_stats = result.dio.get("eda", {}).get("summary_stats", {})
    for col_info in result.dio.get("columns", []):
        c_name = col_info.get("name")
        is_sensitive = col_info.get("is_pii") or (col_info.get("semantic_label") == "identifier")
        if is_sensitive and c_name in summary_stats:
            col_stat = summary_stats[c_name]
            assert "top_category" not in col_stat or col_stat.get("top_category") is None
            assert "top_categories" not in col_stat or not col_stat.get("top_categories")

    # 3. Direct inspection of serialized DIO JSON string
    dio_str = json.dumps(result.dio.to_dict(), default=str)
    for raw_val in gt.sensitive_raw_values:
        assert raw_val not in dio_str, f"Sensitive raw value '{raw_val}' leaked into serialized DIO!"

    # 4. Direct inspection of pipeline execution log
    log_file = run_dir / "pipeline.log"
    if log_file.exists():
        log_text = log_file.read_text(encoding="utf-8", errors="ignore")
        for raw_val in gt.sensitive_raw_values:
            assert raw_val not in log_text, f"Sensitive raw value '{raw_val}' leaked into pipeline.log!"

    # 5. Agent 7 Chat Q&A security check
    chat_agent = ChatAgent()
    for col_info in result.dio.get("columns", []):
        c_name = col_info.get("name")
        if col_info.get("is_pii") or col_info.get("semantic_label") == "identifier":
            # Attempt direct query
            q_res = chat_agent.run_query(f"what is the value of {c_name}?", result.df, result.dio)
            assert q_res["status"] == "refusal"
            # Ensure none of the sensitive values appear in response
            for raw_val in gt.sensitive_raw_values:
                assert raw_val not in q_res["response"]
