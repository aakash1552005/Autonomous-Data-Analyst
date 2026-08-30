"""
tests/test_token_governor.py
============================
Comprehensive tests for LLM Token Governor budget enforcement.
Verifies consumption tracking, budget cutoff, underlying call blocking,
and counter accuracy after rejected requests.
"""

import pytest
from llm.base import LLMProvider, LLMResponse, TokenGovernor, LLMTokenBudgetExceededError


class MonitoredMockProvider(LLMProvider):
    """Mock provider that tracks whether `_call_provider` was actually invoked."""
    def __init__(self, governor: TokenGovernor):
        super().__init__(governor=governor)
        self.underlying_network_calls = 0

    def _call_provider(self, prompt: str, max_tokens: int, temperature: float) -> LLMResponse:
        self.underlying_network_calls += 1
        return LLMResponse(
            text="Response from provider",
            prompt_tokens=50,
            completion_tokens=50,
            total_tokens=100,
            model="mock",
            provider="mock",
        )


def test_token_governor_consumption_and_cutoff():
    """
    Test dedicated token governor enforcement:
    1. Simulate consumption up to max_tokens (e.g. 20,000).
    2. Assert allowed request succeeds and increments usage.
    3. Assert request exceeding remaining budget is refused.
    4. Assert provider does NOT make the underlying call when budget is exceeded.
    5. Assert token counter remains accurate after rejected request.
    """
    governor = TokenGovernor(max_tokens=20000)
    provider = MonitoredMockProvider(governor=governor)

    # 1. Simulate initial usage: 19,500 tokens consumed
    governor.record_usage(prompt_tokens=10000, completion_tokens=9500, provider="mock", model="mock")
    assert governor.total_tokens == 19500
    assert governor.remaining_tokens == 500
    assert not governor.is_exhausted

    # 2. A request fitting within remaining budget (max_tokens=300 <= 500) -> Allowed
    resp = provider.complete_with_usage("Allowed prompt within budget", max_tokens=300)
    assert resp.text == "Response from provider"
    assert provider.underlying_network_calls == 1
    assert governor.total_tokens == 19600  # 19500 + 100
    assert governor.remaining_tokens == 400

    # 3. A request exceeding remaining budget (max_tokens=500 > 400) -> Refused
    with pytest.raises(LLMTokenBudgetExceededError) as exc_info:
        provider.complete("Exceeding prompt", max_tokens=500)

    assert "LLM token budget exceeded" in str(exc_info.value)
    # 4. Underlying provider call was NOT made
    assert provider.underlying_network_calls == 1

    # 5. Token counter remains accurate after rejection
    assert governor.total_tokens == 19600
    assert governor.remaining_tokens == 400
    assert governor.call_count == 2  # Initial record_usage + 1 successful call


def test_token_governor_exact_limit_boundary():
    """Test behavior right at the boundary of exhaustion."""
    governor = TokenGovernor(max_tokens=1000)
    provider = MonitoredMockProvider(governor=governor)

    governor.record_usage(prompt_tokens=500, completion_tokens=500)
    assert governor.is_exhausted
    assert governor.remaining_tokens == 0

    with pytest.raises(LLMTokenBudgetExceededError):
        provider.complete("Should be rejected immediately", max_tokens=1)

    assert provider.underlying_network_calls == 0
    assert governor.total_tokens == 1000
