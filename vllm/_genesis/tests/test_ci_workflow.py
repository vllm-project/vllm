# SPDX-License-Identifier: Apache-2.0
"""Pin the GitHub Actions CI contract.

The point of these tests is not to validate YAML syntax — it's to lock
in the *gates* the project has committed to running on every push/PR,
so a casual workflow edit can't silently drop one.

Today the workflow runs four gates:

  1. The pytest session suite (compat/* + dispatcher_validator + PN14 +
     PN16 + B2 wiring helper + D3 ablation bench + version sanity)
  2. `lifecycle_audit_cli --quiet` (exit 1 on unknown lifecycle state)
  3. `schema_validator` (exit 1 on malformed PATCH_REGISTRY entry)
  4. `cli self-test --quiet` (exit 1 on any structural check failure)

If you intentionally remove one of these gates, update the test to
match — the test is the contract.
"""
from __future__ import annotations

from pathlib import Path

import pytest


WORKFLOW_PATH = (
    Path(__file__).resolve().parents[3] / ".github" / "workflows" / "test.yml"
)


@pytest.fixture(scope="module")
def workflow_text() -> str:
    if not WORKFLOW_PATH.is_file():
        pytest.skip(f"workflow file not present in this checkout: {WORKFLOW_PATH}")
    return WORKFLOW_PATH.read_text(encoding="utf-8")


class TestWorkflowGates:
    def test_workflow_file_exists(self):
        assert WORKFLOW_PATH.is_file(), f"missing CI workflow at {WORKFLOW_PATH}"

    def test_runs_pytest_session_suite(self, workflow_text: str):
        # Every gate that depends on the pytest suite shares this driver.
        assert "python -m pytest" in workflow_text
        # Coverage scoped to the compat package — drift here = silent
        # coverage loss.
        assert "--cov=vllm._genesis.compat" in workflow_text

    def test_runs_lifecycle_audit_gate(self, workflow_text: str):
        assert "vllm._genesis.compat.lifecycle_audit_cli" in workflow_text

    def test_runs_schema_validator_gate(self, workflow_text: str):
        assert "vllm._genesis.compat.schema_validator" in workflow_text

    def test_runs_self_test_gate(self, workflow_text: str):
        # The self-test gate is the most recent addition and the most
        # likely to get accidentally dropped — pin it explicitly.
        assert (
            "vllm._genesis.compat.cli self-test" in workflow_text
            or "vllm._genesis.compat.self_test" in workflow_text
        ), "CI must run `genesis self-test` on every push/PR"

    def test_python_matrix_covers_310_and_312(self, workflow_text: str):
        # Drift would mean we're no longer testing the lower bound or
        # the operator-current line.
        assert '"3.10"' in workflow_text
        assert '"3.12"' in workflow_text
