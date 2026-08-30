"""
llm/ollama_client.py
====================
Ollama client implementation of the LLMProvider interface.
Connects locally to the Ollama REST API (default model: llama3.1:8b).
"""

from __future__ import annotations

from typing import Any
import requests

from llm.base import LLMProvider, LLMResponse, TokenGovernor


class OllamaClient(LLMProvider):
    """
    Default local LLM provider connecting to Ollama without any API keys.
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        host: str = "http://localhost:11434",
        timeout: int = 60,
        governor: TokenGovernor | None = None,
    ) -> None:
        super().__init__(governor=governor)
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def _call_provider(self, prompt: str, max_tokens: int, temperature: float) -> LLMResponse:
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(f"Failed to connect to Ollama at {self.host}. Is Ollama running?") from e
        except requests.exceptions.Timeout as e:
            raise TimeoutError(f"Ollama request timed out after {self.timeout}s.") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error calling Ollama API: {e}") from e

        if resp.status_code != 200:
            raise RuntimeError(f"Ollama returned HTTP error {resp.status_code}: {resp.text}")

        try:
            data = resp.json()
        except Exception as e:
            raise ValueError(f"Failed to parse JSON response from Ollama: {resp.text}") from e

        text = data.get("response", "").strip()
        prompt_tokens = int(data.get("prompt_eval_count", len(prompt.split())))
        completion_tokens = int(data.get("eval_count", len(text.split())))
        total_tokens = prompt_tokens + completion_tokens

        return LLMResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            model=self.model,
            provider="ollama",
        )

    def is_available(self) -> bool:
        """Helper to quickly check if Ollama server is reachable."""
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False
