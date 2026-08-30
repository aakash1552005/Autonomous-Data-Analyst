"""
tests/test_ollama_client.py
===========================
Tests for OllamaClient provider implementation.
Includes offline unit tests and conditionally isolated integration tests.
"""

from unittest.mock import MagicMock, patch
import pytest
import requests

from llm.base import TokenGovernor
from llm.ollama_client import OllamaClient


def is_ollama_online(host: str = "http://localhost:11434") -> bool:
    """Helper to detect if local Ollama server is up and responding."""
    try:
        r = requests.get(f"{host.rstrip('/')}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


# --- Unit Tests (Offline / Mocked) ---

def test_ollama_client_init():
    client = OllamaClient(model="llama3.1:8b", host="http://127.0.0.1:11434", timeout=30)
    assert client.model == "llama3.1:8b"
    assert client.host == "http://127.0.0.1:11434"
    assert client.timeout == 30
    assert client.governor is not None


@patch("requests.post")
def test_ollama_client_mocked_completion(mock_post: MagicMock):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "response": "revenue",
        "prompt_eval_count": 12,
        "eval_count": 2,
    }
    mock_post.return_value = mock_response

    governor = TokenGovernor(max_tokens=1000)
    client = OllamaClient(governor=governor)
    res = client.complete("What is this column: sales_amt?", max_tokens=100)

    assert res == "revenue"
    assert governor.total_tokens == 14
    assert governor.prompt_tokens == 12
    assert governor.completion_tokens == 2


@patch("requests.post")
def test_ollama_client_connection_error(mock_post: MagicMock):
    mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

    client = OllamaClient()
    with pytest.raises(ConnectionError) as exc_info:
        client.complete("Test prompt", max_tokens=50)
    assert "Failed to connect to Ollama" in str(exc_info.value)


# --- Integration Test (Conditional Skip when Ollama is not running) ---

@pytest.mark.skipif(
    not is_ollama_online(),
    reason="Ollama server is not running or unreachable at http://localhost:11434",
)
def test_ollama_integration_live_inference():
    """
    Live integration test against local Ollama + llama3.1:8b.
    Executes normally when Ollama is available; skips gracefully if unavailable.
    """
    governor = TokenGovernor(max_tokens=20000)
    client = OllamaClient(model="llama3.1:8b", host="http://127.0.0.1:11434", governor=governor)

    assert client.is_available() is True
    response = client.complete("Respond with exactly: OLLAMA_OK", max_tokens=20)

    assert "OLLAMA_OK" in response
    assert governor.total_tokens > 0
    assert governor.call_count == 1
