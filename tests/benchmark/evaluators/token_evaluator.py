"""
tests/benchmark/evaluators/token_evaluator.py
============================================
Evaluator for Metric 11: Token Usage Tracking.
Tracks LLM token consumption across pipeline stages.
Distinguishes:
  - "actual"
  - "estimated"
  - "unavailable"
Never fabricates token numbers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from core.dio import DIO


@dataclass
class TokenEvaluationResult:
    status: str  # "actual" | "estimated" | "unavailable"
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    calls_count: int
    stage_breakdown: dict[str, Any] = field(default_factory=dict)
    limitation_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TokenEvaluator:
    """Evaluates and validates LLM token telemetry across pipeline stages."""

    def evaluate(self, dio: DIO) -> TokenEvaluationResult:
        llm_usage = dio.get("llm_usage", {})

        prompt_tokens = llm_usage.get("prompt_tokens")
        completion_tokens = llm_usage.get("completion_tokens")
        total_tokens = llm_usage.get("total_tokens")
        calls_count = int(llm_usage.get("calls_count", 0))

        if total_tokens is not None and total_tokens > 0:
            return TokenEvaluationResult(
                status="actual",
                prompt_tokens=int(prompt_tokens) if prompt_tokens is not None else None,
                completion_tokens=int(completion_tokens) if completion_tokens is not None else None,
                total_tokens=int(total_tokens),
                calls_count=calls_count,
                stage_breakdown=llm_usage.get("stages", {}),
                limitation_note="Verified provider token telemetry from active LLM calls.",
            )

        # In offline/deterministic evaluation or when LLM usage is not tracked by provider
        return TokenEvaluationResult(
            status="unavailable",
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            calls_count=calls_count,
            stage_breakdown={},
            limitation_note="Token metrics unavailable: Pipeline executed with deterministic heuristic fallbacks or local provider without exact token telemetry.",
        )
