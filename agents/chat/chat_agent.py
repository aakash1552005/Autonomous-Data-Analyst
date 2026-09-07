"""
agents/chat/chat_agent.py
=========================
Agent 7: Conversational Dataset Intelligence and Interactive Q&A Agent.

Provides a dual-tier query resolution engine:
  - Tier 1: Deterministic regex classification and whitelisted pandas execution
            with a structural PII shield (Zero LLM, Zero code execution).
  - Tier 2: Bounded DIO factual retrieval and LLM synthesis with strict prompt injection
            defenses and offline heuristic fallback.

Hard guarantees:
  - NEVER mutates analytical DIO sections.
  - NEVER exposes raw PII or identifiers.
  - NEVER runs eval(), exec(), or arbitrary code.
  - NEVER crashes the analytical pipeline.
"""

from __future__ import annotations

import datetime
import logging
import re
from typing import Any
import pandas as pd

from core.base_agent import BaseAgent
from core.config import AppConfig, load_config
from core.dio import DIO
from llm.base import LLMProvider, TokenGovernor
from llm.ollama_client import OllamaClient
from llm.openai_client import OpenAIClient
from agents.chat.query_classifier import classify_query
from agents.chat.whitelist_executor import execute_whitelisted_operation, is_column_sensitive
from agents.chat.context_retriever import build_chat_context

logger = logging.getLogger(__name__)

# Prompt injection and adversarial attack indicators
INJECTION_PATTERNS = [
    r"\bignore\s+(?:all\s+)?(?:previous\s+)?instructions\b",
    r"\bignore\s+(?:the\s+)?safety\s+rules\b",
    r"\b(?:system\s+prompt|hidden\s+prompt)\b",
    r"\b(?:print\s*\(\s*df|show\s+(?:all\s+)?(?:patient|customer|client)?\s*(?:names|emails|phone|ssn|credit\s*card))\b",
    r"(?:\bexec\s*\(|\beval\s*\(|\bimport\s+os|\bos\.system|\bsubprocess|__import__|pd\.eval|pandas\.eval|\.query\s*\(|\bselect\s+.+\s+from\b|\bdrop\s+table\b|\binsert\s+into\b|\bdelete\s+from\b)",
    r"\b(?:give\s+me|reveal|dump)\s+(?:the\s+)?(?:prompt|instructions|secret)\b",
]


class ChatAgent(BaseAgent):
    """
    Agent 7: Interactive Q&A and Conversational Dataset Exploration Agent.
    """
    name = "chat"

    def __init__(
        self,
        config: AppConfig | None = None,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        super().__init__()
        self.config = config or load_config()
        self.llm_provider = llm_provider

    def _init_llm_provider(self) -> LLMProvider | None:
        """Initialize LLM client safely if available."""
        if self.llm_provider is not None:
            return self.llm_provider

        governor = TokenGovernor(max_tokens=self.config.security.max_llm_tokens_per_run)

        if self.config.llm.provider == "ollama":
            client = OllamaClient(
                model=self.config.llm.model,
                host=self.config.llm.host,
                timeout=min(self.config.llm.timeout_seconds, 15),  # Fast timeout for chat
                governor=governor,
            )
            if client.is_available():
                return client
            elif self.config.llm.api_key:
                return OpenAIClient(
                    api_key=self.config.llm.api_key,
                    model="gpt-4o-mini",
                    governor=governor,
                )
            return None
        elif self.config.llm.provider == "openai":
            if not self.config.llm.api_key:
                return None
            return OpenAIClient(
                api_key=self.config.llm.api_key,
                model=self.config.llm.model,
                governor=governor,
            )
        return None

    def run_query(
        self,
        query: str,
        df: pd.DataFrame | None,
        dio: DIO | dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute an interactive query against the dataset and analytical findings.

        Returns:
            dict containing:
              - response: Textual answer for the user
              - tier: 1 (deterministic) or 2 (context retrieval / LLM)
              - operation: Matched operation name or None
              - status: "success", "refusal", "unavailable", or "error"
              - sources: List of data sources used
        """
        # 1. Validate Input
        clean_query = query.strip() if query else ""
        if not clean_query:
            return {
                "response": "Please enter a valid question about the dataset or analysis results.",
                "tier": 1,
                "operation": None,
                "status": "unavailable",
                "sources": [],
            }

        # 2. Prompt Injection & Adversarial Security Gate
        q_lower = clean_query.lower()
        for pat in INJECTION_PATTERNS:
            if re.search(pat, q_lower):
                return {
                    "response": "Refusal: Requests to override instructions, execute system commands, or access raw sensitive personal data (PII) are strictly prohibited under system safety protocols.",
                    "tier": 1,
                    "operation": None,
                    "status": "refusal",
                    "sources": ["security_shield"],
                }

        columns_info = dio.get("columns", [])
        columns = [c.get("name", "") for c in columns_info if c.get("name")]
        if df is not None and not columns:
            columns = list(df.columns)

        try:
            # 3. Tier 1 — Deterministic Pandas Operation (if DataFrame is available)
            if df is not None and self.config.chat.allow_deterministic_aggregations:
                classified = classify_query(
                    query=clean_query,
                    columns=columns,
                    safe_operations=self.config.chat.safe_operations,
                )

                if classified.operation != "unclassified":
                    exec_result = execute_whitelisted_operation(
                        df=df,
                        columns_info=columns_info,
                        operation=classified.operation,
                        column=classified.column,
                        group_column=classified.group_column,
                        safe_operations=self.config.chat.safe_operations,
                    )
                    return {
                        "response": exec_result.result_text,
                        "tier": 1,
                        "operation": exec_result.operation,
                        "status": exec_result.status,
                        "sources": ["pandas_whitelisted_aggregation"],
                    }

            # Sensitive Column Guard: Unclassified queries targeting protected columns are refused
            for col_name in columns:
                if is_column_sensitive(col_name, columns_info):
                    pattern = rf"(?:\b|['\"]){re.escape(col_name.lower())}(?:\b|['\"])"
                    if re.search(pattern, q_lower):
                        return {
                            "response": f"Refusal: Access to column '{col_name}' is restricted because it contains sensitive personal data or identifier records.",
                            "tier": 1,
                            "operation": None,
                            "status": "refusal",
                            "sources": ["security_shield"],
                        }

            # 4. Tier 2 — Bounded DIO Retrieval & Question Answering
            chat_context = build_chat_context(
                dio=dio,
                max_context_tokens=self.config.chat.max_context_tokens,
            )

            # Try LLM if configured and online
            provider = self._init_llm_provider()
            if provider is not None:
                prompt = (
                    "You are the Autonomous Data Analyst Chat Assistant. You answer questions strictly and factually "
                    "using only the verified dataset analysis summary below.\n"
                    "Rules:\n"
                    "1. If the answer is not present in the context, respond: 'The requested information is not available in the dataset or analysis results.'\n"
                    "2. Do not invent, extrapolate, or hallucinate numbers or claims.\n"
                    "3. Never generate code, SQL queries, or shell commands.\n"
                    "4. Keep the answer concise and direct.\n\n"
                    f"--- ANALYSIS CONTEXT ---\n{chat_context}\n\n"
                    f"User Question: {clean_query}\n\n"
                    "Answer:"
                )
                try:
                    resp = provider.complete_with_usage(
                        prompt=prompt,
                        max_tokens=300,
                        temperature=self.config.chat.temperature,
                    )
                    if resp.text.strip():
                        return {
                            "response": resp.text.strip(),
                            "tier": 2,
                            "operation": None,
                            "status": "success",
                            "sources": ["llm_dio_context"],
                        }
                except Exception as e:
                    logger.warning(f"ChatAgent: LLM call failed ({e}); falling back to deterministic heuristic QA.")

            # 5. Offline Heuristic QA Fallback (Zero-LLM deterministic retrieval from DIO)
            fallback_ans = self._resolve_dio_heuristic(clean_query, dio)
            return {
                "response": fallback_ans,
                "tier": 2,
                "operation": None,
                "status": "success" if fallback_ans != "The requested information is not available in the dataset or analysis results." else "unavailable",
                "sources": ["dio_telemetry_heuristic"],
            }

        except Exception as e:
            logger.error(f"ChatAgent: Unhandled error processing query: {e}", exc_info=True)
            return {
                "response": "An internal error occurred while processing the request. The analytical pipeline remains stable.",
                "tier": 1,
                "operation": None,
                "status": "error",
                "sources": [],
            }

    def _resolve_dio_heuristic(self, query: str, dio: dict[str, Any]) -> str:
        """
        Deterministic rule-based answers from DIO telemetry when LLM is unavailable.
        """
        q = query.lower()

        # Data Quality
        if "quality" in q or "score" in q or "cleanliness" in q:
            quality = dio.get("quality", {})
            score = quality.get("score", "N/A")
            issues = quality.get("issues", [])
            issues_str = "; ".join(issues) if issues else "No major defects."
            return f"The dataset has an overall data quality score of {score}/100. Key issues identified: {issues_str}"

        # Domain
        if "domain" in q or "industry" in q or "business context" in q:
            domain_info = dio.get("domain_guess", {})
            domain = domain_info.get("domain", "Unknown").title()
            conf = domain_info.get("confidence", 0.0)
            return f"The detected domain for this dataset is {domain} (confidence: {conf:.1%})."

        # Machine Learning / Modeling
        if "model" in q or "ml" in q or "algorithm" in q or "predict" in q or "train" in q:
            ml = dio.get("ml", {})
            status = ml.get("model_training_status", "NOT_CONFIGURED")
            if status == "SKIPPED_INSUFFICIENT_DATA":
                return "Machine learning training was safely skipped due to insufficient data rows (< 30 rows)."
            elif status == "SKIPPED_NO_TARGET":
                return "Machine learning was skipped because no suitable prediction target column was identified."
            elif status == "TRAINED":
                model_name = ml.get("best_model_name", "Unknown")
                task = ml.get("task_type", "modeling")
                target = ml.get("target_column", "target")
                metrics = ml.get("metrics", {})
                metrics_str = ", ".join([f"{k}={v}" for k, v in metrics.items()]) if metrics else "computed"
                return f"Model training completed ({task}). Best performing model: {model_name} on target '{target}'. Metrics: {metrics_str}."
            return f"Machine learning status: {status}."

        # Dataset Shape / Rows / Columns
        if "rows" in q or "how many rows" in q or "records" in q or "columns" in q or "size" in q:
            ingestion = dio.get("ingestion", {})
            n_rows = ingestion.get("n_rows", "N/A")
            n_cols = ingestion.get("n_columns", "N/A")
            return f"The dataset contains {n_rows} rows and {n_cols} columns."

        # Insights / Key Findings
        if "insight" in q or "finding" in q or "recommendation" in q or "takeaway" in q:
            insights = dio.get("insights", [])
            if insights:
                texts = [f"- {ins.get('text', '')}" for ins in insights[:3]]
                return "Key analytical insights:\n" + "\n".join(texts)
            return "No executive insights were generated for this dataset."

        # Target Candidate
        if "target" in q or "candidate" in q:
            eda = dio.get("eda", {})
            candidates = eda.get("target_candidates", [])
            if candidates:
                return f"Potential target candidates identified in the dataset: {', '.join(candidates)}."
            return "No target candidate columns were flagged in this dataset."

        return "The requested information is not available in the dataset or analysis results."

    def run(
        self,
        df: pd.DataFrame,
        dio: DIO | dict[str, Any],
        query: str = "",
    ) -> tuple[pd.DataFrame, DIO | dict[str, Any]]:
        """
        BaseAgent run implementation.
        Does NOT alter analytical sections in DIO.
        """
        if query:
            res = self.run_query(query, df, dio)
            dio["decision_log"].append({
                "agent": self.name,
                "action": "chat_query_processed",
                "tier": res["tier"],
                "status": res["status"],
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            })
        return df, dio
