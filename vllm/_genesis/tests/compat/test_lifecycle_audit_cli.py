# SPDX-License-Identifier: Apache-2.0
"""Tests for `python3 -m vllm._genesis.compat.lifecycle_audit_cli`."""
from __future__ import annotations

import json

import pytest


_FAKE_REGISTRY = {
    "P_S": {"lifecycle": "stable", "stable_since": "v1.0"},
    "P_E": {"lifecycle": "experimental",
            "experimental_note": "untested in PROD"},
    "P_D": {"lifecycle": "deprecated",
            "superseded_by": ["P_S"], "removal_planned": "v2.0"},
    "P_R": {"lifecycle": "research",
            "research_note": "kept for future hardware"},
    "P_X": {"lifecycle": "made_up_state"},  # registry error
}


@pytest.fixture
def fake_registry(monkeypatch):
    from vllm._genesis import dispatcher
    monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", _FAKE_REGISTRY)
    yield _FAKE_REGISTRY


class TestLifecycleAuditCLI:
    def test_main_returns_int(self, fake_registry):
        from vllm._genesis.compat.lifecycle_audit_cli import main
        rc = main([])
        assert isinstance(rc, int)

    def test_main_returns_nonzero_on_unknown_state(self, fake_registry):
        """A registry with `lifecycle: <unknown_state>` is a real error
        — CLI should exit non-zero so CI catches it."""
        from vllm._genesis.compat.lifecycle_audit_cli import main
        rc = main([])
        assert rc != 0

    def test_main_clean_registry_returns_zero(self, monkeypatch):
        """Healthy registry → exit 0."""
        from vllm._genesis import dispatcher
        clean = {
            "P_A": {"lifecycle": "stable"},
            "P_B": {"lifecycle": "experimental"},
            "P_C": {"lifecycle": "deprecated", "superseded_by": ["P_A"]},
        }
        monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", clean)
        from vllm._genesis.compat.lifecycle_audit_cli import main
        rc = main([])
        assert rc == 0

    def test_json_output(self, fake_registry, capsys):
        from vllm._genesis.compat.lifecycle_audit_cli import main
        main(["--json"])
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        # Each lifecycle bucket present
        assert isinstance(parsed, dict)
        # Should at least include 'entries' and a count
        assert "entries" in parsed or "by_state" in parsed

    def test_filter_by_state(self, fake_registry, capsys):
        from vllm._genesis.compat.lifecycle_audit_cli import main
        main(["--state", "stable"])
        captured = capsys.readouterr()
        # Should only show P_S, not the deprecated/experimental ones
        assert "P_S" in captured.out
        assert "P_E" not in captured.out
        assert "P_D" not in captured.out

    def test_quiet_mode_only_errors(self, fake_registry, capsys):
        """`--quiet` prints only non-ok rows (errors + warnings).
        Stable patches (severity=ok) must NOT appear AS ROWS, though
        their patch_id may still appear in another row's note (e.g. as
        `superseded_by` target of a deprecated row)."""
        from vllm._genesis.compat.lifecycle_audit_cli import main
        main(["--quiet"])
        captured = capsys.readouterr()
        # P_X (unknown state, error severity) must show
        assert "P_X" in captured.out
        # P_E (experimental, warn) and P_D (deprecated, warn) must show
        assert "P_E" in captured.out
        assert "P_D" in captured.out
        # P_S (stable, ok severity) MUST NOT have its own row, but it CAN
        # appear in P_D's superseded_by note. Detect "row-shaped" P_S
        # pattern: bullet/mark + P_S followed by spaces + em-dash.
        import re
        row_pattern = re.compile(r"[•⚠✗]\s+P_S\s+—")
        assert not row_pattern.search(captured.out), (
            "P_S has severity 'ok' and should not appear as a row in --quiet"
        )

    def test_real_registry_runs(self):
        """Smoke: against the actual PATCH_REGISTRY, just verify it
        runs without crashing."""
        from vllm._genesis.compat.lifecycle_audit_cli import main
        rc = main([])
        assert rc in (0, 1)  # both valid


class TestExitCodeContract:
    def test_explicit_unknown_state_exit_1(self, monkeypatch):
        from vllm._genesis import dispatcher
        bad = {"P_BAD": {"lifecycle": "totally_not_a_state"}}
        monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", bad)
        from vllm._genesis.compat.lifecycle_audit_cli import main
        rc = main([])
        assert rc == 1

    def test_only_warnings_returns_zero(self, monkeypatch):
        """Warnings (deprecated, experimental) should NOT fail CI;
        operators see them but they're not blockers."""
        from vllm._genesis import dispatcher
        warnings_only = {
            "P_DEP": {"lifecycle": "deprecated", "superseded_by": ["P_X"]},
            "P_EXP": {"lifecycle": "experimental"},
        }
        monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", warnings_only)
        from vllm._genesis.compat.lifecycle_audit_cli import main
        rc = main([])
        assert rc == 0
