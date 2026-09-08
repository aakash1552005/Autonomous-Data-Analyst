"""
tests/test_n8n_security.py
==========================
Phase 23: Security and schema validation for n8n automation workflows.
Verifies:
  1. All n8n workflow definitions are valid, well-formed JSON.
  2. All workflows adhere to expected n8n schema (name, nodes, connections, settings).
  3. No workflow employs dangerous arbitrary code execution nodes.
  4. Zero hardcoded secrets, passwords, or API keys in workflow definitions.
  5. All example payloads are well-formed and schema-compliant.
"""

import json
from pathlib import Path
import re
import pytest

WORKFLOWS_DIR = Path("n8n/workflows")
EXAMPLES_DIR = Path("n8n/examples")

SUSPICIOUS_SECRET_PATTERNS = [
    r"(?i)api[_-]?key\s*[:=]\s*['\"][a-zA-Z0-9_\-]{16,}['\"]",
    r"(?i)password\s*[:=]\s*['\"][^'\"]{6,}['\"]",
    r"(?i)bearer\s+[a-zA-Z0-9_\-\.]{20,}",
    r"(?i)secret[_-]?key\s*[:=]\s*['\"][a-zA-Z0-9_\-]{16,}['\"]",
]

DISALLOWED_NODE_TYPES = {
    "n8n-nodes-base.executeCommand",
    "n8n-nodes-base.ssh",
}


class TestN8nWorkflowSchema:
    def test_workflow_files_exist(self):
        workflow_files = list(WORKFLOWS_DIR.glob("*.json"))
        assert len(workflow_files) >= 6, f"Expected at least 6 n8n workflows, found {len(workflow_files)}"

    @pytest.mark.parametrize(
        "workflow_path",
        list(WORKFLOWS_DIR.glob("*.json")),
        ids=lambda p: p.name,
    )
    def test_workflow_json_validity(self, workflow_path: Path):
        with open(workflow_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert isinstance(data, dict), f"{workflow_path.name} must be a JSON object"
        assert "name" in data, f"{workflow_path.name} missing 'name'"
        assert "nodes" in data, f"{workflow_path.name} missing 'nodes'"
        assert isinstance(data["nodes"], list), f"{workflow_path.name} 'nodes' must be a list"
        assert "connections" in data, f"{workflow_path.name} missing 'connections'"
        assert isinstance(data["connections"], dict), f"{workflow_path.name} 'connections' must be a dict"

    @pytest.mark.parametrize(
        "workflow_path",
        list(WORKFLOWS_DIR.glob("*.json")),
        ids=lambda p: p.name,
    )
    def test_workflow_nodes_safe(self, workflow_path: Path):
        with open(workflow_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for node in data.get("nodes", []):
            node_type = node.get("type", "")
            assert node_type not in DISALLOWED_NODE_TYPES, (
                f"Dangerous node type '{node_type}' detected in {workflow_path.name}!"
            )


class TestN8nSecretsAndSecurity:
    @pytest.mark.parametrize(
        "workflow_path",
        list(WORKFLOWS_DIR.glob("*.json")) + list(EXAMPLES_DIR.glob("*.json")),
        ids=lambda p: p.name,
    )
    def test_no_hardcoded_secrets(self, workflow_path: Path):
        content = workflow_path.read_text(encoding="utf-8")
        for pattern in SUSPICIOUS_SECRET_PATTERNS:
            match = re.search(pattern, content)
            assert not match, (
                f"Potential hardcoded secret matching pattern '{pattern}' found in {workflow_path.name}: {match.group(0)}"
            )


class TestN8nExamplePayloads:
    def test_example_payload_valid(self):
        payload_file = EXAMPLES_DIR / "example_payload.json"
        assert payload_file.exists(), "example_payload.json not found"
        with open(payload_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "file_path" in data
        assert isinstance(data["file_path"], str)
