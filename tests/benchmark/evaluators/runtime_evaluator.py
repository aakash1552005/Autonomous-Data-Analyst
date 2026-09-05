"""
tests/benchmark/evaluators/runtime_evaluator.py
==============================================
Evaluator for Metric 10: Runtime Metrics.
Extracts total pipeline runtime and per-stage timings from OrchestratorResult and DIO.
Produces:
  - total_runtime_seconds
  - stage_runtimes (intelligence, cleaning, eda, ml, insight, report)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from core.dio import DIO


@dataclass
class RuntimeEvaluationResult:
    total_runtime_seconds: float
    intelligence_runtime: float = 0.0
    cleaning_runtime: float = 0.0
    eda_runtime: float = 0.0
    ml_runtime: float = 0.0
    insight_runtime: float = 0.0
    report_runtime: float = 0.0
    stage_timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RuntimeEvaluator:
    """Evaluates and records per-stage and total execution runtimes."""

    def evaluate(
        self,
        dio: DIO,
        stage_timings: dict[str, float] | None = None,
    ) -> RuntimeEvaluationResult:
        timings = dict(stage_timings or {})

        # If stage_timings not directly provided, pull from dio["agent_metrics"]
        if not timings:
            agent_metrics = dio.get("agent_metrics", [])
            for m in agent_metrics:
                agent_name = m.get("agent", "")
                r_sec = float(m.get("runtime_seconds", 0.0))
                timings[agent_name] = r_sec

        total_runtime = timings.get("total_pipeline", sum(timings.values()))

        return RuntimeEvaluationResult(
            total_runtime_seconds=round(float(total_runtime), 4),
            intelligence_runtime=round(float(timings.get("intelligence", 0.0)), 4),
            cleaning_runtime=round(float(timings.get("cleaning", 0.0)), 4),
            eda_runtime=round(float(timings.get("eda", 0.0)), 4),
            ml_runtime=round(float(timings.get("ml", 0.0)), 4),
            insight_runtime=round(float(timings.get("insight", 0.0)), 4),
            report_runtime=round(float(timings.get("report", 0.0)), 4),
            stage_timings=timings,
        )
