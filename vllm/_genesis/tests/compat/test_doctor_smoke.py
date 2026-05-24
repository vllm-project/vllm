# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for vllm._genesis.compat.doctor.

Heavy CPU-only tests — verifies the doctor at least runs end-to-end
without crashing on a host with no GPU / no live model. Real GPU
validation is in CI / on the live container.
"""
from __future__ import annotations

import json



def test_collect_report_runs_without_exception():
    from vllm._genesis.compat.doctor import collect_report
    report = collect_report()
    # Has all expected sections
    assert "hardware" in report
    assert "software" in report
    assert "model_profile" in report
    assert "patches" in report
    assert "lifecycle" in report
    assert "validator" in report
    assert "recommendations" in report


def test_report_is_json_serializable():
    from vllm._genesis.compat.doctor import collect_report
    report = collect_report()
    # Round-trip via JSON
    s = json.dumps(report, default=str)
    decoded = json.loads(s)
    assert "hardware" in decoded


def test_format_text_produces_lines():
    from vllm._genesis.compat.doctor import _format_text, collect_report
    report = collect_report()
    lines = _format_text(report)
    assert len(lines) > 10  # at least a few sections rendered
    # Section headers
    joined = "\n".join(lines)
    assert "Hardware" in joined
    assert "Software" in joined
    assert "Patch registry" in joined or "Patches" in joined


def test_recommendations_always_present():
    """Even on a clean system, recommendations section returns at least
    one line (the OK marker)."""
    from vllm._genesis.compat.doctor import collect_report
    report = collect_report()
    recs = report["recommendations"]
    assert isinstance(recs, list)
    assert len(recs) >= 1


def test_main_returns_int():
    from vllm._genesis.compat.doctor import main
    rc = main(argv=["--quiet"])
    assert isinstance(rc, int)
    assert rc in (0, 1)


def test_main_json_mode(capsys):
    from vllm._genesis.compat.doctor import main
    main(argv=["--json"])
    captured = capsys.readouterr()
    # Verify output is valid JSON
    parsed = json.loads(captured.out)
    assert "hardware" in parsed


def test_environment_section_shape():
    """Environment section must always be present with stable keys."""
    from vllm._genesis.compat.doctor import _section_environment
    env = _section_environment()
    # Stable schema for downstream consumers (CI, dashboards)
    assert "is_wsl" in env
    assert "wsl_version" in env
    assert "pcie_lanes" in env
    assert "errors" in env
    # Types
    assert isinstance(env["is_wsl"], bool)
    assert isinstance(env["pcie_lanes"], list)
    assert isinstance(env["errors"], list)


def test_environment_section_in_report():
    from vllm._genesis.compat.doctor import collect_report
    report = collect_report()
    assert "environment" in report
    assert isinstance(report["environment"], dict)


def test_wsl_recommendation_fires_when_wsl_set(monkeypatch):
    """If env section reports WSL, recommendations include the WSL warning."""
    from vllm._genesis.compat import doctor as D

    real_env_fn = D._section_environment

    def fake_env():
        e = real_env_fn()
        e["is_wsl"] = True
        e["wsl_version"] = "WSL2"
        return e

    monkeypatch.setattr(D, "_section_environment", fake_env)
    report = D.collect_report()
    joined = "\n".join(report["recommendations"])
    assert "WSL" in joined
    assert "display" in joined.lower() or "overhead" in joined.lower()


def test_pcie_underlane_recommendation(monkeypatch):
    """A PCIe lane wired below max width triggers a [WARN] recommendation."""
    from vllm._genesis.compat import doctor as D

    real_env_fn = D._section_environment

    def fake_env():
        e = real_env_fn()
        e["pcie_lanes"] = [{
            "index": 0, "name": "RTX A5000",
            "gen_current": "4", "width_current": "8",
            "gen_max": "4", "width_max": "16",
        }]
        return e

    monkeypatch.setattr(D, "_section_environment", fake_env)
    report = D.collect_report()
    joined = "\n".join(report["recommendations"])
    assert "x8" in joined and "x16" in joined
