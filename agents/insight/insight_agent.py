"""
agents/insight/insight_agent.py
===============================
Insight & Narrative Agent (Agent 7).
Synthesizes statistical and machine learning results from the DIO into
quantifiable, grounded business insights and actionable recommendations.
Guarantees strict numerical hallucination prevention and zero raw DataFrame access.
"""

from __future__ import annotations

import datetime
import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

from core.base_agent import BaseAgent, ProgressState
from core.config import AppConfig
from core.dio import DIO
from core.logger import get_logger
from llm.base import LLMProvider, LLMTokenBudgetExceededError, TokenGovernor
from llm.ollama_client import OllamaClient
from llm.openai_client import OpenAIClient
from agents.insight.evidence_collector import collect_evidence_summary, extract_all_dio_numbers
from agents.insight.prompt_builder import build_insight_prompt
from agents.insight.hallucination_guard import verify_insight_grounding
from agents.insight.deterministic_engine import generate_deterministic_insights

logger = get_logger("agents.insight")


class InsightAgent(BaseAgent):
    """
    Agent 7: Synthesizes structured DIO outputs into grounded business insights.
    """
    name: str = "insight"

    def __init__(
        self,
        config: AppConfig | None = None,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        super().__init__()
        from core.config import load_config
        self.config = config or load_config()
        self.llm_provider = llm_provider

    def _init_llm_provider(self) -> LLMProvider | None:
        """Initialize LLM provider with token governor from config."""
        if self.llm_provider is not None:
            return self.llm_provider

        governor = TokenGovernor(max_tokens=self.config.security.max_llm_tokens_per_run)

        if self.config.llm.provider == "ollama":
            client = OllamaClient(
                model=self.config.llm.model,
                host=self.config.llm.host,
                timeout=self.config.llm.timeout_seconds,
                governor=governor,
            )
            if not client.is_available():
                logger.warning(f"InsightAgent: Ollama host {self.config.llm.host} is unreachable.")
                if self.config.llm.api_key:
                    logger.info("InsightAgent: Falling back to OpenAI provider.")
                    return OpenAIClient(
                        api_key=self.config.llm.api_key,
                        model="gpt-4o-mini",
                        governor=governor,
                    )
                return None
            return client
        elif self.config.llm.provider == "openai":
            if not self.config.llm.api_key:
                return None
            return OpenAIClient(
                api_key=self.config.llm.api_key,
                model=self.config.llm.model,
                governor=governor,
            )
        return None

    def _parse_llm_json(self, raw_text: str) -> list[dict[str, Any]]:
        """Parse LLM output JSON string, stripping markdown if present."""
        cleaned = raw_text.strip()
        if "```" in cleaned:
            # Extract content inside code fences
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
            if match:
                cleaned = match.group(1).strip()

        try:
            data = json.loads(cleaned)
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict)]
            elif isinstance(data, dict):
                # If wrapped in a dict like {"insights": [...]}
                for v in data.values():
                    if isinstance(v, list):
                        return [d for d in v if isinstance(d, dict)]
        except Exception:
            pass
        return []

    def run(
        self,
        df: pd.DataFrame,
        dio: DIO,
        run_dir: Path | str | None = None,
    ) -> tuple[pd.DataFrame, DIO]:
        """
        Execute the Insight & Narrative Agent pipeline.
        Consumes structured DIO metadata and populates dio["insights"].
        """
        start_time = time.time()
        logger.info(f"InsightAgent: Starting execution for dataset {dio.file_name}")
        dio["progress"][self.name] = ProgressState.RUNNING.value

        min_insights = self.config.insights.min_insights
        max_insights = self.config.insights.max_insights
        tolerance = self.config.insights.grounding_tolerance

        # 1. Collect grounded numerical facts across DIO
        grounded_ints, grounded_floats = extract_all_dio_numbers(dio)
        evidence_summary = collect_evidence_summary(dio)

        valid_insights: list[dict[str, Any]] = []
        llm_used = False
        llm_error_msg = ""

        # 2. Attempt LLM Generation
        provider = self._init_llm_provider()
        if provider is not None:
            prompt = build_insight_prompt(evidence_summary, min_insights, max_insights)
            try:
                resp = provider.complete_with_usage(
                    prompt=prompt,
                    max_tokens=self.config.insights.max_tokens,
                    temperature=self.config.insights.temperature,
                )

                # Record token usage in DIO
                dio["llm_usage"]["total_tokens"] += resp.total_tokens
                dio["llm_usage"]["total_calls"] += 1
                dio["llm_usage"]["prompt_tokens"] += resp.prompt_tokens
                dio["llm_usage"]["completion_tokens"] += resp.completion_tokens
                dio["llm_usage"]["calls"].append({
                    "agent": self.name,
                    "model": resp.model,
                    "prompt_tokens": resp.prompt_tokens,
                    "completion_tokens": resp.completion_tokens,
                    "total_tokens": resp.total_tokens,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                })

                # Parse LLM JSON candidates
                candidates = self._parse_llm_json(resp.text)
                llm_used = True

                if not candidates and resp.text.strip():
                    dio["errors"].append({
                        "agent": self.name,
                        "code": "INS_001",
                        "message": "Failed to parse valid JSON insight array from LLM response",
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    })

                for cand in candidates:
                    is_grounded, nums, reason = verify_insight_grounding(
                        cand,
                        grounded_ints=grounded_ints,
                        grounded_floats=grounded_floats,
                        tolerance=tolerance,
                    )
                    if is_grounded:
                        cand["grounded_numbers"] = nums
                        valid_insights.append(cand)
                    else:
                        dio["decision_log"].append({
                            "agent": self.name,
                            "action": "hallucination_guard_rejected",
                            "reason": reason,
                            "rejected_text": cand.get("text", ""),
                            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        })
                        dio["errors"].append({
                            "agent": self.name,
                            "code": "INS_002",
                            "message": f"Rejected ungrounded insight: {reason}",
                            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        })

            except LLMTokenBudgetExceededError as e:
                logger.warning(f"InsightAgent: LLM token budget exceeded: {e}")
                llm_error_msg = f"Token budget exceeded: {str(e)}"
                dio["errors"].append({
                    "agent": self.name,
                    "code": "LLM_002",
                    "message": str(e),
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                })
            except Exception as e:
                logger.warning(f"InsightAgent: LLM inference failed: {e}. Falling back to deterministic engine.")
                llm_error_msg = str(e)
                dio["errors"].append({
                    "agent": self.name,
                    "code": "INS_001",
                    "message": f"LLM insight generation failed: {str(e)}",
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                })

        # 3. Deterministic Backfill / Fallback Engine
        # If LLM wasn't used or yielded fewer than min_insights, supplement deterministically
        if len(valid_insights) < min_insights:
            det_insights = generate_deterministic_insights(dio, max_insights=max_insights)
            existing_texts = {ins["text"] for ins in valid_insights}

            for d_ins in det_insights:
                if d_ins["text"] not in existing_texts:
                    valid_insights.append(d_ins)
                    existing_texts.add(d_ins["text"])
                if len(valid_insights) >= max_insights:
                    break

            # 4. Backfill Floor / Evidence Availability Check
            # If after deterministic generation we still have fewer than min_insights,
            # we MUST NOT fabricate or pad — record explicit decision log.
            if len(valid_insights) < min_insights:
                dio["decision_log"].append({
                    "agent": self.name,
                    "action": "backfill_floor_reached",
                    "reason": f"Only {len(valid_insights)} of {min_insights} minimum insights could be grounded; insufficient evidence in dataset for additional claims.",
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                })

        # 5. Cap to max_insights & Assign IDs
        final_insights = valid_insights[:max_insights]
        for idx, ins in enumerate(final_insights, start=1):
            ins["id"] = f"ins_{idx:03d}"

        dio["insights"] = final_insights

        # 6. Record Provenance Decision Log
        dio["decision_log"].append({
            "agent": self.name,
            "action": "insights_generated",
            "count": len(final_insights),
            "llm_used": llm_used,
            "min_insights_satisfied": len(final_insights) >= min_insights,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        })

        # 7. Record Agent Metrics
        elapsed = time.time() - start_time
        dio["agent_metrics"].append({
            "agent": self.name,
            "runtime_seconds": round(elapsed, 4),
            "warnings": 1 if (len(final_insights) < min_insights or llm_error_msg) else 0,
            "errors": 1 if llm_error_msg else 0,
            "insights_count": len(final_insights),
        })

        dio["progress"][self.name] = ProgressState.FINISHED.value
        logger.info(f"InsightAgent: Generated {len(final_insights)} verified insights in {elapsed:.3f}s")
        return df, dio
