"""
agents/insight/prompt_builder.py
================================
Constructs strictly formatted, PII-masked prompts for the LLM to generate insights.
Ensures zero raw data rows enter the prompt and formats only structured DIO statistics.
"""

from __future__ import annotations

import json
from typing import Any


def build_insight_prompt(evidence: dict[str, Any], min_insights: int = 3, max_insights: int = 6) -> str:
    """
    Build a comprehensive, strictly constrained prompt for the LLM.
    """
    evidence_json = json.dumps(evidence, indent=2, default=str)

    prompt = f"""You are a senior data scientist and business analyst analyzing structured dataset statistics.

Here is the structured statistical evidence collected from the dataset:
```json
{evidence_json}
```

Instructions:
1. Generate between {min_insights} and {max_insights} key business findings based strictly on the evidence above.
2. For EVERY finding, you MUST include specific numbers, percentages, or metrics directly present in the evidence.
3. NEVER invent, hallucinate, or extrapolate numbers that are not in the evidence.
4. For each finding, provide:
   - "category": One of "distribution", "correlation", "quality", or "machine_learning".
   - "text": A clear, plain-English summary sentence citing the exact computed numbers (e.g. mean, median, percentage, correlation, or model score).
   - "confidence": A float between 0.60 and 0.95 (0.90+ for direct statistics, 0.70-0.85 for correlations/trends, 0.60-0.70 for ML signals).
   - "evidence": The exact key or metric path referenced (e.g. "eda.summary_stats.revenue.mean").
   - "recommendation": A concise, actionable, business-oriented next step.

Respond ONLY with a valid JSON array of objects. Do not include markdown code block formatting around the JSON if possible, or wrap strictly in ```json.

Example Format:
[
  {{
    "category": "distribution",
    "text": "Average monthly charges are $64.76 with a median of $70.35 across 120 customer records.",
    "confidence": 0.90,
    "evidence": "eda.summary_stats.monthly_charges.mean",
    "recommendation": "Audit account tiering to ensure premium features correspond to higher monthly charge bands."
  }}
]
"""
    return prompt
