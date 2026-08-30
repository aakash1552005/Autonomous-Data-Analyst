"""
tests/test_llm_provider.py
==========================
Unit tests for the abstract LLMProvider interface and mock provider execution.
Runnable completely offline without external network or Ollama dependencies.
"""

import pytest
from llm.base import LLMProvider, LLMResponse, TokenGovernor, LLMTokenBudgetExceededError


class MockLLMProvider(LLMProvider):
    """Deterministic mock provider for unit testing."""
    def __init__(self, governor: TokenGovernor | None = None, response_text: str = "Mock response"):
        super().__init__(governor=governor)
        self.response_text = response_text
        self.call_count = 0

    def _call_provider(self, prompt: str, max_tokens: int, temperature: float) -> LLMResponse:
        self.call_count += 1
        prompt_tokens = len(prompt.split())
        completion_tokens = len(self.response_text.split())
        return LLMResponse(
            text=self.response_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            model="mock-model",
            provider="mock",
        )


def test_mock_llm_provider_complete():
    provider = MockLLMProvider(response_text="Customer Churn Prediction")
    response_text = provider.complete("Analyze column target", max_tokens=100)
    assert response_text == "Customer Churn Prediction"
    assert provider.call_count == 1
    assert provider.governor.total_tokens > 0
    assert provider.governor.call_count == 1


def test_mock_llm_provider_complete_with_usage():
    governor = TokenGovernor(max_tokens=5000)
    provider = MockLLMProvider(governor=governor, response_text="Retail Domain Classification")
    resp = provider.complete_with_usage("Classify domain for revenue, sku, order_date", max_tokens=200)

    assert isinstance(resp, LLMResponse)
    assert resp.text == "Retail Domain Classification"
    assert resp.provider == "mock"
    assert resp.model == "mock-model"
    assert resp.prompt_tokens == 6
    assert resp.completion_tokens == 3
    assert resp.total_tokens == 9
    assert governor.total_tokens == 9
    assert governor.remaining_tokens == 4991
