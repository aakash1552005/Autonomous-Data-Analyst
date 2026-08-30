"""
tests/test_base_agent.py
========================
Tests for BaseAgent interface and agent contracts.
"""

import pytest
from core.base_agent import BaseAgent
from core.dio import DIO


class ConcreteDummyAgent(BaseAgent):
    name = "dummy"

    def run(self, df, dio):
        dio["progress"][self.name] = "FINISHED"
        return df, dio


class UnnamedAgent(BaseAgent):
    name = ""  # Invalid empty name


def test_base_agent_contract():
    agent = ConcreteDummyAgent()
    assert agent.name == "dummy"
    assert "dummy" in repr(agent)

    dio = DIO.create_empty(file_name="test.csv")
    df, updated_dio = agent.run(None, dio)
    assert updated_dio["progress"]["dummy"] == "FINISHED"


def test_base_agent_must_implement_run():
    class UnimplementedAgent(BaseAgent):
        name = "unimplemented"

    agent = UnimplementedAgent()
    dio = DIO.create_empty(file_name="test.csv")
    with pytest.raises(NotImplementedError):
        agent.run(None, dio)


def test_base_agent_requires_valid_name():
    with pytest.raises(ValueError):
        UnnamedAgent()
