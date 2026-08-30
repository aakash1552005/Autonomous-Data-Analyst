"""
tests/test_date_resolver.py
===========================
Comprehensive date format resolution test suite.
Includes 20+ hand-picked date cases covering unambiguous DD/MM/YYYY, MM/DD/YYYY,
ISO formats, ambiguous dates, multi-column consistency, and locale hinting.
"""

import pandas as pd
import pytest
from agents.intelligence.date_resolver import resolve_date_column, resolve_all_dates


# --- 20+ Hand-Picked Date Test Cases ---

@pytest.mark.parametrize(
    "date_strings,expected_format,expected_confidence,needs_confirmation",
    [
        # 1-4: Unambiguous DD/MM/YYYY (Day > 12 in first position)
        (["25/08/2024", "13/05/2024", "04/02/2024"], "DD/MM/YYYY", 1.0, False),
        (["31-12-2023", "15-01-2023", "01-02-2023"], "DD-MM-YYYY", 1.0, False),
        (["28.02.2022", "19.11.2022", "05.06.2022"], "DD.MM.YYYY", 1.0, False),
        (["14/07/2021", "22/09/2021", "09/01/2021"], "DD/MM/YYYY", 1.0, False),

        # 5-8: Unambiguous MM/DD/YYYY (Day > 12 in second position)
        (["08/25/2024", "05/13/2024", "02/04/2024"], "MM/DD/YYYY", 1.0, False),
        (["12-31-2023", "01-15-2023", "02-01-2023"], "MM-DD-YYYY", 1.0, False),
        (["02.28.2022", "11.19.2022", "06.05.2022"], "MM.DD.YYYY", 1.0, False),
        (["07/14/2021", "09/22/2021", "01/09/2021"], "MM/DD/YYYY", 1.0, False),

        # 9-12: ISO Formats (4-digit Year first)
        (["2024-08-25", "2024-05-13", "2024-02-04"], "YYYY-MM-DD", 1.0, False),
        (["2023/12/31", "2023/01/15", "2023/02/01"], "YYYY/MM/DD", 1.0, False),
        (["2022.02.28", "2022.11.19", "2022.06.05"], "YYYY.MM.DD", 1.0, False),
        (["2021-01-01", "2021-06-15", "2021-12-31"], "YYYY-MM-DD", 1.0, False),

        # 13-16: Ambiguous dates (all components <= 12, no signals)
        (["01/02/2024", "03/04/2024", "05/06/2024"], "ambiguous", 0.5, True),
        (["07-08-2023", "09-10-2023", "11-12-2023"], "ambiguous", 0.5, True),
        (["02.03.2022", "04.05.2022", "06.07.2022"], "ambiguous", 0.5, True),
        (["01/01/2021", "02/02/2021", "03/03/2021"], "ambiguous", 0.5, True),

        # 17-20: Additional mixed and single disambiguator dates
        (["01/02/2024", "02/03/2024", "25/04/2024"], "DD/MM/YYYY", 1.0, False),
        (["01/02/2024", "02/03/2024", "04/25/2024"], "MM/DD/YYYY", 1.0, False),
        (["2020-01-01 12:00:00", "2020-05-15 14:30:00"], "YYYY-MM-DD", 1.0, False),
        (["29/02/2024", "01/03/2024"], "DD/MM/YYYY", 1.0, False),
    ],
)
def test_date_resolution_cases(date_strings, expected_format, expected_confidence, needs_confirmation):
    series = pd.Series(date_strings)
    res = resolve_date_column(series, col_name="order_date")
    assert res is not None
    assert res["detected_format"] == expected_format
    assert res["confidence"] == expected_confidence
    assert res["needs_user_confirmation"] == needs_confirmation


def test_date_resolution_locale_hinting():
    # Ambiguous date with UK locale column
    df_uk = pd.DataFrame({
        "event_date": ["05/06/2024", "07/08/2024"],
        "country": ["UK", "UK"],
    })
    res_uk = resolve_date_column(df_uk["event_date"], "event_date", dataset_df=df_uk)
    assert res_uk is not None
    assert res_uk["detected_format"] == "DD/MM/YYYY"
    assert res_uk["confidence"] == 0.70
    assert res_uk["needs_user_confirmation"] is False

    # Ambiguous date with US locale column
    df_us = pd.DataFrame({
        "event_date": ["05/06/2024", "07/08/2024"],
        "country": ["USA", "USA"],
    })
    res_us = resolve_date_column(df_us["event_date"], "event_date", dataset_df=df_us)
    assert res_us is not None
    assert res_us["detected_format"] == "MM/DD/YYYY"
    assert res_us["confidence"] == 0.70
    assert res_us["needs_user_confirmation"] is False


def test_date_resolution_multi_column_consistency():
    # First column resolved as DD/MM/YYYY (via 25/01/2024)
    # Second column is ambiguous (05/06/2024) -> should inherit DD/MM/YYYY format
    df = pd.DataFrame({
        "start_date": ["25/01/2024", "14/02/2024"],
        "end_date": ["05/06/2024", "07/08/2024"],
    })
    results = resolve_all_dates(df)
    assert len(results) == 2
    assert results[0]["column"] == "start_date"
    assert results[0]["detected_format"] == "DD/MM/YYYY"
    assert results[0]["confidence"] == 1.0

    assert results[1]["column"] == "end_date"
    assert results[1]["detected_format"] == "DD/MM/YYYY"
    assert results[1]["confidence"] == 0.75
    assert results[1]["needs_user_confirmation"] is False
