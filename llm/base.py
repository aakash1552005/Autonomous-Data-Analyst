"""
llm/base.py
===========
Abstract LLMProvider interface, token governor, and response structures.
Enforces the global per-run token budget and guarantees provider modularity.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class LLMTokenBudgetExceededError(Exception):
    """Raised when an LLM call cannot proceed because the global budget is exhausted."""


@dataclass
class LLMResponse:
    """Standardized response container for all LLM providers."""
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    provider: str = ""


@dataclass
class TokenGovernor:
    """
    Governor tracking and bounding LLM token consumption per pipeline run.
    Hard-stops further LLM calls once the configured limit is reached.
    """
    max_tokens: int = 20000
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    call_count: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.max_tokens - self.total_tokens)

    @property
    def is_exhausted(self) -> bool:
        return self.total_tokens >= self.max_tokens

    def can_spend(self, estimated_tokens: int = 1) -> bool:
        """Check if an estimated token spend fits within the remaining budget."""
        return (self.total_tokens + estimated_tokens) <= self.max_tokens

    def record_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        provider: str = "",
        model: str = "",
    ) -> None:
        """Update token counts following a successful provider call."""
        spent = prompt_tokens + completion_tokens
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += spent
        self.call_count += 1
        self.history.append({
            "call_index": self.call_count,
            "provider": provider,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": spent,
            "cumulative_tokens": self.total_tokens,
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "call_count": self.call_count,
            "is_exhausted": self.is_exhausted,
            "remaining_tokens": self.remaining_tokens,
            "calls": self.history,
        }


class LLMProvider(ABC):
    """
    Abstract LLM Provider interface.
    All agents must interact with local or hosted LLMs solely through this interface.
    """

    def __init__(self, governor: TokenGovernor | None = None) -> None:
        self.governor = governor or TokenGovernor()

    @abstractmethod
    def _call_provider(self, prompt: str, max_tokens: int, temperature: float) -> LLMResponse:
        """Provider-specific call implementation. Must return an LLMResponse."""
        raise NotImplementedError

    def complete(self, prompt: str, max_tokens: int = 500, temperature: float = 0.2) -> str:
        """
        Generate completion text while strictly enforcing the token budget.
        """
        response = self.complete_with_usage(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
        return response.text

    def complete_with_usage(self, prompt: str, max_tokens: int = 500, temperature: float = 0.2) -> LLMResponse:
        """
        Execute completion and update the governor token records.
        If budget would be exceeded, immediately blocks the call without network interaction.
        """
        if self.governor.is_exhausted or not self.governor.can_spend(estimated_tokens=max_tokens):
            raise LLMTokenBudgetExceededError(
                f"LLM token budget exceeded: remaining={self.governor.remaining_tokens}, "
                f"requested max_tokens={max_tokens}, max_limit={self.governor.max_tokens}"
            )

        response = self._call_provider(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
        self.governor.record_usage(
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            provider=response.provider,
            model=response.model,
        )
        return response
