"""
tests/test_quality_scorer.py
============================
Tests for data quality scoring formula and issue list generation.
"""

from agents.intelligence.quality_scorer import calculate_quality_score


def test_quality_score_clean_dataset():
    columns_info = [
        {"name": "id", "null_pct": 0.0, "confidence": 0.95},
        {"name": "revenue", "null_pct": 0.0, "confidence": 0.90},
    ]
    date_columns = [
        {"column": "date", "detected_format": "YYYY-MM-DD", "needs_user_confirmation": False}
    ]
    res = calculate_quality_score(columns_info, date_columns, duplicate_row_pct=0.0)

    assert res["score"] == 100
    assert len(res["issues"]) == 0


def test_quality_score_deductions():
    # 50% average nulls -> -15 points (0.50 * 30)
    # 10% duplicates -> -2 points (0.10 * 20)
    # 1 out of 1 date ambiguous -> -15 points
    # 1 out of 2 cols low confidence -> -7.5 points (0.50 * 15)
    # Expected: 100 - 15 - 2 - 15 - 7.5 = 60.5 -> 60 or 61
    columns_info = [
        {"name": "user_id", "null_pct": 0.50, "confidence": 0.90},
        {"name": "weird_col", "null_pct": 0.50, "confidence": 0.50},
    ]
    date_columns = [
        {"column": "event_date", "detected_format": "ambiguous", "needs_user_confirmation": True}
    ]
    res = calculate_quality_score(columns_info, date_columns, duplicate_row_pct=0.10)

    assert res["score"] in (60, 61)
    assert len(res["issues"]) >= 3
    assert any("null values" in iss for iss in res["issues"])
    assert any("Duplicate rows" in iss for iss in res["issues"])
    assert any("Ambiguous date" in iss for iss in res["issues"])


def test_quality_score_bounds():
    # Severe dataset with maximum deductions
    columns_info = [{"name": f"col_{i}", "null_pct": 1.0, "confidence": 0.2} for i in range(5)]
    date_columns = [{"column": "d", "detected_format": "ambiguous", "needs_user_confirmation": True}]
    res = calculate_quality_score(columns_info, date_columns, duplicate_row_pct=1.0)

    # 100 - 30(nulls) - 20(dups) - 15(dates) - 15(conf) = 20
    assert res["score"] == 20
    assert 0 <= res["score"] <= 100
    assert len(res["issues"]) >= 4
