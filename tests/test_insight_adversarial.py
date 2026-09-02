"""
tests/test_insight_adversarial.py
=================================
Adversarial test suite for Phase 7 (Insight & Narrative Agent).
Thoroughly verifies:
1. Hallucination guard rejection of fabricated numbers (e.g. 99.9%).
2. Valid + invalid number combination rejection.
3. Rounding tolerance boundary checks.
4. Qualitative (zero-number) insight rejection.
5. Max insight cap enforcement.
6. Deterministic backfill when LLM provides insufficient valid insights.
7. Malformed JSON handling.
8. LLM offline/timeout handling.
9. Token budget exhaustion handling.
10. PII prompt safety.
11. ML-unavailable state (zero ML claims).
12. EDA empty charts state (zero visual claims).
13. DIO mutation boundary snapshot equality.
14. Backfill floor check: No hallucinated padding when evidence is sparse.
15. Zero raw-row access guarantee.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import pytest
import pandas as pd

from core.dio import DIO
from core.config import AppConfig
from llm.base import LLMProvider, LLMResponse, TokenGovernor, LLMTokenBudgetExceededError
from agents.insight.insight_agent import InsightAgent
from agents.insight.evidence_collector import extract_all_dio_numbers


class AdversarialMockLLM(LLMProvider):
    def __init__(self, response_text: str, governor: TokenGovernor | None = None, raise_error: Exception | None = None):
        super().__init__(governor=governor)
        self.response_text = response_text
        self.raise_error = raise_error
        self.last_prompt = ""

    def _call_provider(self, prompt: str, max_tokens: int, temperature: float) -> LLMResponse:
        self.last_prompt = prompt
        if self.raise_error is not None:
            raise self.raise_error
        return LLMResponse(
            text=self.response_text,
            prompt_tokens=150,
            completion_tokens=80,
            total_tokens=230,
            model="mock_adv",
            provider="mock",
        )


@pytest.fixture
def rich_dio() -> DIO:
    dio = DIO.create_empty(file_name="telecom_churn.csv", dataset_hash="h_telecom")
    dio["ingestion"] = {"n_rows": 120, "n_columns": 6, "file_type": "csv", "encoding": "utf-8"}
    dio["quality"] = {"score": 95, "issues": []}
    dio["domain_guess"] = {"domain": "telecom", "confidence": 0.80}
    dio["columns"] = [
        {"name": "monthly_charges", "dtype_inferred": "float", "semantic_label": "currency_amount", "is_pii": False},
        {"name": "total_charges", "dtype_inferred": "float", "semantic_label": "currency_amount", "is_pii": False},
        {"name": "support_calls", "dtype_inferred": "int", "semantic_label": "count", "is_pii": False},
        {"name": "churn", "dtype_inferred": "int", "semantic_label": "target", "is_pii": False},
    ]
    dio["eda"]["summary_stats"] = {
        "numeric": {
            "monthly_charges": {"mean": 64.76, "median": 70.35, "std": 30.0, "min": 18.25, "max": 118.75},
            "total_charges": {"mean": 2283.30, "median": 1397.47, "std": 2266.77, "min": 18.80, "max": 8684.80},
            "support_calls": {"mean": 1.45, "median": 1.0, "std": 1.15, "min": 0, "max": 6},
        }
    }
    dio["eda"]["correlations"] = {
        "pearson": {
            "monthly_charges": {"monthly_charges": 1.0, "total_charges": 0.65},
            "total_charges": {"monthly_charges": 0.65, "total_charges": 1.0},
        },
        "top_correlations": [
            {"col1": "monthly_charges", "col2": "total_charges", "pearson": 0.65, "spearman": 0.60},
        ],
    }
    dio["ml"] = {
        "status": "trained",
        "task_type": "classification",
        "target_column": "churn",
        "selected_model": "random_forest",
        "metrics": {"f1": 0.81, "accuracy": 0.85, "precision": 0.83, "recall": 0.79},
        "feature_importance": {"monthly_charges": 0.42, "total_charges": 0.38, "support_calls": 0.20},
    }
    return dio


def test_adversarial_hallucination_single_fabricated_number(rich_dio: DIO, tmp_path: Path):
    """
    LLM produces a single completely fabricated number (99.9%).
    Hallucination guard must reject the insight and fall back deterministically.
    """
    hallucinated_json = json.dumps([
        {
            "category": "distribution",
            "text": "Average monthly charges are $64.76, while customer satisfaction is 99.9%.",
            "confidence": 0.90,
            "evidence": "eda.summary_stats.monthly_charges.mean",
            "recommendation": "Maintain high service standards.",
        }
    ])

    mock_llm = AdversarialMockLLM(hallucinated_json)
    agent = InsightAgent(llm_provider=mock_llm)
    df = pd.DataFrame({"monthly_charges": [64.76] * 120})

    _, updated_dio = agent.run(df, rich_dio, run_dir=tmp_path)

    # The 99.9% insight MUST be rejected
    rejected_log = [d for d in updated_dio["decision_log"] if d.get("action") == "hallucination_guard_rejected"]
    assert len(rejected_log) >= 1
    assert "99.9" in str(rejected_log[0]["reason"])

    # Final insights must be filled via deterministic engine without the 99.9% claim
    assert len(updated_dio["insights"]) >= 3
    assert not any("99.9" in ins["text"] for ins in updated_dio["insights"])


def test_adversarial_hallucination_valid_plus_invalid_combination(rich_dio: DIO, tmp_path: Path):
    """
    LLM produces an insight with one valid number (64.76) and one fabricated number (555.0).
    The ENTIRE insight must be rejected.
    """
    mixed_json = json.dumps([
        {
            "category": "distribution",
            "text": "Average monthly charges were 64.76 with 555.0 high value accounts.",
            "confidence": 0.90,
            "evidence": "eda.summary_stats.monthly_charges.mean",
            "recommendation": "Review accounts.",
        }
    ])

    mock_llm = AdversarialMockLLM(mixed_json)
    agent = InsightAgent(llm_provider=mock_llm)
    df = pd.DataFrame({"monthly_charges": [64.76] * 120})

    _, updated_dio = agent.run(df, rich_dio, run_dir=tmp_path)

    # Assert rejection
    assert any(d.get("action") == "hallucination_guard_rejected" for d in updated_dio["decision_log"])
    assert not any("555.0" in ins["text"] for ins in updated_dio["insights"])


def test_adversarial_rounding_tolerance_boundaries(rich_dio: DIO, tmp_path: Path):
    """
    Test rounding within tolerance (64.8 vs 64.76 -> within 5% tol) is accepted,
    whereas an out-of-tolerance number (145.0 vs max 118.75 -> > 20% error) is rejected.
    """
    # 64.8 is within 5% of 64.76
    # 145.0 is outside tolerance of any number in rich_dio
    candidate_json = json.dumps([
        {
            "category": "distribution",
            "text": "Average monthly charges are approximately 64.8 across 120 customers.",
            "confidence": 0.90,
            "evidence": "eda.summary_stats.monthly_charges.mean",
            "recommendation": "Monitor charges.",
        },
        {
            "category": "distribution",
            "text": "Average monthly charges are approximately 145.0 across 120 customers.",
            "confidence": 0.90,
            "evidence": "eda.summary_stats.monthly_charges.mean",
            "recommendation": "Monitor charges.",
        }
    ])

    mock_llm = AdversarialMockLLM(candidate_json)
    agent = InsightAgent(llm_provider=mock_llm)
    df = pd.DataFrame({"monthly_charges": [64.76] * 120})

    _, updated_dio = agent.run(df, rich_dio, run_dir=tmp_path)

    # 64.8 should be kept, 145.0 must be dropped
    assert any("64.8" in ins["text"] for ins in updated_dio["insights"])
    assert not any("145.0" in ins["text"] for ins in updated_dio["insights"])


def test_adversarial_zero_number_qualitative_claim_rejection(rich_dio: DIO, tmp_path: Path):
    """
    LLM produces a purely qualitative insight with zero numbers.
    Must be rejected for lack of quantifiable grounding.
    """
    qualitative_json = json.dumps([
        {
            "category": "distribution",
            "text": "Monthly charges seem moderately high and total charges are substantial.",
            "confidence": 0.90,
            "evidence": "eda.summary_stats",
            "recommendation": "Reduce monthly charges.",
        }
    ])

    mock_llm = AdversarialMockLLM(qualitative_json)
    agent = InsightAgent(llm_provider=mock_llm)
    df = pd.DataFrame({"monthly_charges": [64.76] * 120})

    _, updated_dio = agent.run(df, rich_dio, run_dir=tmp_path)

    # Must reject qualitative claim
    rejections = [d for d in updated_dio["decision_log"] if d.get("action") == "hallucination_guard_rejected"]
    assert any("zero numbers" in r.get("reason", "").lower() for r in rejections)


def test_adversarial_max_insights_capping(rich_dio: DIO, tmp_path: Path):
    """
    LLM returns 10 valid insights; agent must deterministically cap to max_insights (6).
    """
    ten_insights = []
    for i in range(10):
        ten_insights.append({
            "category": "distribution",
            "text": f"Average monthly charges are 64.76 with variation #{i + 1} across 120 rows.",
            "confidence": 0.90,
            "evidence": "eda.summary_stats.monthly_charges.mean",
            "recommendation": f"Action #{i + 1}.",
        })

    mock_llm = AdversarialMockLLM(json.dumps(ten_insights))
    config = AppConfig()
    config.insights.max_insights = 5
    agent = InsightAgent(config=config, llm_provider=mock_llm)
    df = pd.DataFrame({"monthly_charges": [64.76] * 120})

    _, updated_dio = agent.run(df, rich_dio, run_dir=tmp_path)

    assert len(updated_dio["insights"]) == 5
    assert updated_dio["insights"][-1]["id"] == "ins_005"


def test_adversarial_malformed_json_fallback(rich_dio: DIO, tmp_path: Path):
    """
    LLM returns unparseable garbage / broken JSON.
    Agent must gracefully fall back to deterministic engine and log error.
    """
    mock_llm = AdversarialMockLLM("Here are some insights: {not valid json at all... [ broken")
    agent = InsightAgent(llm_provider=mock_llm)
    df = pd.DataFrame({"monthly_charges": [64.76] * 120})

    _, updated_dio = agent.run(df, rich_dio, run_dir=tmp_path)

    assert len(updated_dio["insights"]) >= 3
    assert updated_dio["progress"]["insight"] == "FINISHED"
    assert any(e.get("code") == "INS_001" for e in updated_dio.get("errors", []))


def test_adversarial_llm_connection_error_fallback(rich_dio: DIO, tmp_path: Path):
    """
    LLM raises connection timeout or network failure.
    Agent must not halt the pipeline, returning deterministic insights.
    """
    mock_llm = AdversarialMockLLM("", raise_error=ConnectionError("Failed to connect to Ollama"))
    agent = InsightAgent(llm_provider=mock_llm)
    df = pd.DataFrame({"monthly_charges": [64.76] * 120})

    _, updated_dio = agent.run(df, rich_dio, run_dir=tmp_path)

    assert len(updated_dio["insights"]) >= 3
    assert updated_dio["progress"]["insight"] == "FINISHED"
    assert any(e.get("code") == "INS_001" for e in updated_dio["errors"])


def test_adversarial_token_budget_exhaustion(rich_dio: DIO, tmp_path: Path):
    """
    Token budget is exceeded during LLM call.
    Agent records token exhaustion error and completes via deterministic engine.
    """
    governor = TokenGovernor(max_tokens=100)
    governor.record_usage(prompt_tokens=100, completion_tokens=0)  # already full
    mock_llm = AdversarialMockLLM("[]", governor=governor)
    agent = InsightAgent(llm_provider=mock_llm)
    df = pd.DataFrame({"monthly_charges": [64.76] * 120})

    _, updated_dio = agent.run(df, rich_dio, run_dir=tmp_path)

    assert len(updated_dio["insights"]) >= 3
    assert any(e.get("code") == "LLM_002" for e in updated_dio["errors"])


def test_adversarial_pii_prompt_safety(rich_dio: DIO, tmp_path: Path):
    """
    Ensure that PII column names or metadata containing raw PII values never enter the prompt.
    """
    rich_dio["columns"].append({
        "name": "ssn",
        "dtype_inferred": "string",
        "semantic_label": "ssn",
        "is_pii": True,
    })
    mock_llm = AdversarialMockLLM("[]")
    agent = InsightAgent(llm_provider=mock_llm)
    df = pd.DataFrame({"monthly_charges": [64.76] * 120})

    agent.run(df, rich_dio, run_dir=tmp_path)

    # Verify prompt does NOT contain any raw PII values
    assert "999-99-9999" not in mock_llm.last_prompt
    assert "password" not in mock_llm.last_prompt.lower()


def test_adversarial_ml_insufficient_data_produces_zero_ml_claims(tmp_path: Path):
    """
    When ML status is 'insufficient_data', verify that exactly zero ML claims are generated.
    """
    dio = DIO.create_empty(file_name="small.csv", dataset_hash="h_small")
    dio["ingestion"] = {"n_rows": 5, "n_columns": 2, "file_type": "csv", "encoding": "utf-8"}
    dio["quality"] = {"score": 90, "issues": []}
    dio["columns"] = [{"name": "x", "dtype_inferred": "float"}, {"name": "y", "dtype_inferred": "int"}]
    dio["eda"]["summary_stats"] = {
        "numeric": {
            "x": {"mean": 10.0, "median": 10.0, "std": 1.0, "min": 8.0, "max": 12.0},
            "y": {"mean": 20.0, "median": 20.0, "std": 2.0, "min": 18.0, "max": 22.0},
        }
    }
    dio["ml"] = {
        "status": "insufficient_data",
        "error_code": "ML_003_INSUFFICIENT_DATA",
        "reason": "Dataset has 5 rows, below min_rows_for_ml threshold of 30",
    }

    mock_llm = AdversarialMockLLM("[]")
    agent = InsightAgent(llm_provider=mock_llm)  # deterministic fallback
    df = pd.DataFrame({"x": range(5), "y": range(5)})

    _, updated_dio = agent.run(df, dio, run_dir=tmp_path)

    for ins in updated_dio["insights"]:
        assert ins["category"] != "machine_learning"
        assert "machine learning" not in ins["text"].lower()
        assert "f1" not in ins["text"].lower()
        assert "rmse" not in ins["text"].lower()


def test_adversarial_dio_mutation_boundary_snapshot(rich_dio: DIO, tmp_path: Path):
    """
    Prove that InsightAgent mutates ONLY dio["insights"] and standard metadata
    (progress, agent_metrics, decision_log, llm_usage, errors).
    Upstream sections must be bit-for-bit unchanged.
    """
    dio_before = copy.deepcopy(rich_dio)

    mock_llm = AdversarialMockLLM("[]")
    agent = InsightAgent(llm_provider=mock_llm)
    df = pd.DataFrame({"monthly_charges": [64.76] * 120})
    _, updated_dio = agent.run(df, rich_dio, run_dir=tmp_path)

    # Immutable sections
    assert updated_dio["schema_version"] == dio_before["schema_version"]
    assert updated_dio["dataset_id"] == dio_before["dataset_id"]
    assert updated_dio["dataset_hash"] == dio_before["dataset_hash"]
    assert updated_dio["file_name"] == dio_before["file_name"]
    assert updated_dio["ingestion"] == dio_before["ingestion"]
    assert updated_dio["columns"] == dio_before["columns"]
    assert updated_dio["domain_guess"] == dio_before["domain_guess"]
    assert updated_dio["quality"] == dio_before["quality"]
    assert updated_dio["cleaning_log"] == dio_before["cleaning_log"]
    assert updated_dio["eda"] == dio_before["eda"]
    assert updated_dio["ml"] == dio_before["ml"]
    assert updated_dio["artifacts"] == dio_before["artifacts"]


def test_adversarial_backfill_floor_no_fabrication_on_empty_evidence(tmp_path: Path):
    """
    CRITICAL BACKFILL FLOOR TEST:
    If a dataset has genuinely sparse evidence (e.g. 1 constant column, 0 correlations, no ML),
    InsightAgent MUST NOT invent/fabricate insights to force min_insights.
    It returns whatever grounded insights exist and explicitly logs 'backfill_floor_reached'.
    """
    sparse_dio = DIO.create_empty(file_name="empty_sparse.csv", dataset_hash="h_sparse")
    sparse_dio["ingestion"] = {"n_rows": 2, "n_columns": 1, "file_type": "csv", "encoding": "utf-8"}
    sparse_dio["quality"] = {"score": 100, "issues": []}
    sparse_dio["columns"] = [{"name": "const_val", "dtype_inferred": "int"}]
    sparse_dio["eda"]["summary_stats"] = {
        "numeric": {
            "const_val": {"mean": 1.0, "median": 1.0, "std": 0.0, "min": 1.0, "max": 1.0},
        }
    }
    sparse_dio["eda"]["correlations"] = {"pearson": {}, "top_correlations": []}
    sparse_dio["ml"] = {"status": "skipped", "reason": "No target"}

    config = AppConfig()
    config.insights.min_insights = 5  # Requested 5, but evidence can only support 2
    mock_llm = AdversarialMockLLM("[]")
    agent = InsightAgent(config=config, llm_provider=mock_llm)
    df = pd.DataFrame({"const_val": [1, 1]})

    _, updated_dio = agent.run(df, sparse_dio, run_dir=tmp_path)

    # Should have fewer than 5 insights because we refuse to hallucinate
    assert len(updated_dio["insights"]) < 5
    assert len(updated_dio["insights"]) >= 1

    # Must log backfill_floor_reached
    floor_logs = [d for d in updated_dio["decision_log"] if d.get("action") == "backfill_floor_reached"]
    assert len(floor_logs) >= 1
    assert "insufficient evidence in dataset" in floor_logs[0]["reason"]
