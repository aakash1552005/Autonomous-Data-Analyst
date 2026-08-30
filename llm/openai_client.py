"""
llm/openai_client.py
====================
Optional OpenAI provider implementation of the LLMProvider interface.
Only invoked when explicitly configured with an API key.
"""

from __future__ import annotations

import os
from typing import Any
import requests

from llm.base import LLMProvider, LLMResponse, TokenGovernor


class OpenAIClient(LLMProvider):
    """
    Optional hosted LLM provider using OpenAI-compatible chat completions API.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        timeout: int = 60,
        governor: TokenGovernor | None = None,
    ) -> None:
        super().__init__(governor=governor)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _call_provider(self, prompt: str, max_tokens: int, temperature: float) -> LLMResponse:
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not configured. Set OPENAI_API_KEY environment variable or config."
            )

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(f"Failed to connect to OpenAI API at {self.base_url}") from e
        except requests.exceptions.Timeout as e:
            raise TimeoutError(f"OpenAI request timed out after {self.timeout}s.") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error calling OpenAI API: {e}") from e

        if resp.status_code != 200:
            raise RuntimeError(f"OpenAI API returned HTTP {resp.status_code}: {resp.text}")

        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            raise ValueError(f"No completion choices returned by OpenAI: {data}")

        text = choices[0].get("message", {}).get("content", "").strip()
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", len(prompt.split()))
        completion_tokens = usage.get("completion_tokens", len(text.split()))
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

        return LLMResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            model=self.model,
            provider="openai",
        )
