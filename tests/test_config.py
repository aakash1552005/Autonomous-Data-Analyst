"""
tests/test_config.py
====================
Tests for configuration loading, validation, and typed structures.
"""

import os
from pathlib import Path
import pytest
from core.config import AppConfig, LLMConfig, SecurityConfig, EDAConfig, MLConfig, load_config


def test_load_default_config(tmp_path: Path):
    yaml_content = """
max_upload_size_mb: 150
llm:
  provider: ollama
  model: llama3.1:8b
  host: http://localhost:11434
  max_tokens_per_call: 600
  temperature: 0.3
  timeout_seconds: 45
security:
  max_llm_tokens_per_run: 25000
eda:
  max_charts: 5
ml:
  train_test_split: 0.75
  random_state: 99
"""
    cfg_file = tmp_path / "custom_config.yaml"
    cfg_file.write_text(yaml_content, encoding="utf-8")

    config = load_config(cfg_file)
    assert config.max_upload_size_mb == 150
    assert config.llm.model == "llama3.1:8b"
    assert config.llm.max_tokens_per_call == 600
    assert config.llm.temperature == 0.3
    assert config.security.max_llm_tokens_per_run == 25000
    assert config.eda.max_charts == 5
    assert config.ml.train_test_split == 0.75
    assert config.ml.random_state == 99


def test_config_env_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg_file = tmp_path / "base_config.yaml"
    cfg_file.write_text("max_upload_size_mb: 100\n", encoding="utf-8")

    monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", "250")
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    monkeypatch.setenv("MAX_LLM_TOKENS_PER_RUN", "30000")

    config = load_config(cfg_file)
    assert config.max_upload_size_mb == 250
    assert config.llm.host == "http://127.0.0.1:11434"
    assert config.security.max_llm_tokens_per_run == 30000


def test_config_validation_errors():
    with pytest.raises(ValueError):
        LLMConfig(provider="invalid_provider").validate()

    with pytest.raises(ValueError):
        LLMConfig(temperature=5.0).validate()

    with pytest.raises(ValueError):
        SecurityConfig(max_llm_tokens_per_run=-1).validate()

    with pytest.raises(ValueError):
        EDAConfig(max_charts=0).validate()

    with pytest.raises(ValueError):
        MLConfig(train_test_split=1.5).validate()
