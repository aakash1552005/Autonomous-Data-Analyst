"""
tests/test_progress.py
======================
Tests for standardized progress states and DIO progress mapping.
"""

from core.base_agent import ProgressState
from core.dio import DIO


def test_progress_states_enum():
    assert ProgressState.PENDING == "PENDING"
    assert ProgressState.RUNNING == "RUNNING"
    assert ProgressState.FINISHED == "FINISHED"
    assert ProgressState.FAILED == "FAILED"
    assert ProgressState.SKIPPED == "SKIPPED"

    assert ProgressState.is_valid("PENDING") is True
    assert ProgressState.is_valid("RUNNING") is True
    assert ProgressState.is_valid("FINISHED") is True
    assert ProgressState.is_valid("FAILED") is True
    assert ProgressState.is_valid("SKIPPED") is True
    assert ProgressState.is_valid("UNKNOWN_STATE") is False


def test_dio_progress_state_transitions():
    dio = DIO.create_empty(file_name="dataset.csv")

    dio.progress["intelligence"] = ProgressState.PENDING.value
    assert dio.progress["intelligence"] == "PENDING"

    dio.progress["intelligence"] = ProgressState.RUNNING.value
    assert dio.progress["intelligence"] == "RUNNING"

    dio.progress["intelligence"] = ProgressState.FINISHED.value
    assert dio.progress["intelligence"] == "FINISHED"

    dio.progress["ml"] = ProgressState.SKIPPED.value
    assert dio.progress["ml"] == "SKIPPED"
