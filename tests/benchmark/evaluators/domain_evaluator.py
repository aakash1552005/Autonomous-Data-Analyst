"""
tests/benchmark/evaluators/domain_evaluator.py
==============================================
Evaluator for Metric 2: Domain Detection Accuracy.
Evaluates dio["domain_guess"] against benchmark ground truth.
Produces:
  - domain_accuracy
  - domain_confidence
  - domain_predicted
  - domain_expected
  - domain_correct
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from core.dio import DIO
from tests.benchmark.ground_truth import DatasetGroundTruth


@dataclass
class DomainEvaluationResult:
    domain_accuracy: float
    domain_confidence: float
    domain_predicted: str
    domain_expected: str
    domain_correct: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DomainEvaluator:
    """Evaluates predicted domain and confidence against ground truth."""

    def evaluate(self, dio: DIO, ground_truth: DatasetGroundTruth) -> DomainEvaluationResult:
        domain_guess = dio.get("domain_guess", {})
        predicted_domain = str(domain_guess.get("domain", "")).strip().lower()
        confidence = float(domain_guess.get("confidence", 0.0))
        expected_domain = ground_truth.expected_domain.strip().lower()

        # Domain matches if identical or reasonable alias (e.g. generic matches mixed/generic)
        if expected_domain in ("generic", "mixed"):
            is_correct = predicted_domain in ("generic", "mixed", "ecommerce", "subscription", "unknown")
        else:
            is_correct = (predicted_domain == expected_domain)

        accuracy = 1.0 if is_correct else 0.0

        return DomainEvaluationResult(
            domain_accuracy=accuracy,
            domain_confidence=confidence,
            domain_predicted=predicted_domain,
            domain_expected=expected_domain,
            domain_correct=is_correct,
        )
