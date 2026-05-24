# SPDX-License-Identifier: Apache-2.0
"""Tests for `vllm._genesis.compat.verify` — post-install smoke test.

Verifies:
  - Quick checks run on CPU-only host without raising
  - Each individual check returns valid CheckResult
  - WARN does NOT fail overall (only FAIL does)
  - JSON output is parseable + has expected schema
  - CLI subcommand routes through unified dispatcher
  - Boot-level checks add B1/B2/B3 to the report
  - Idempotent: running verify multiple times is safe

Author: Sandermage (Sander) Barzov Aleksandr.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


# ─────────────────────────────────────────────────────────────────
# Module imports + dataclass shapes
# ─────────────────────────────────────────────────────────────────


def test_verify_module_imports():
    from vllm._genesis.compat import verify
    assert hasattr(verify, "run_verify")
    assert hasattr(verify, "render_report")
    assert hasattr(verify, "CheckResult")
    assert hasattr(verify, "VerifyReport")
    assert hasattr(verify, "main")


def test_verify_status_constants_defined():
    from vllm._genesis.compat.verify import FAIL, PASS, SKIP, WARN
    assert PASS == "PASS"
    assert WARN == "WARN"
    assert FAIL == "FAIL"
    assert SKIP == "SKIP"


def test_check_result_dataclass_shape():
    from vllm._genesis.compat.verify import PASS, CheckResult

    r = CheckResult(name="test", status=PASS, detail="all good")
    assert r.name == "test"
    assert r.status == PASS
    assert r.detail == "all good"
    assert r.duration_ms == 0  # default
    assert r.hint is None       # default


def test_verify_report_summary_counts():
    from vllm._genesis.compat.verify import (
        FAIL,
        PASS,
        SKIP,
        WARN,
        CheckResult,
        VerifyReport,
    )

    r = VerifyReport()
    r.add(CheckResult("a", PASS))
    r.add(CheckResult("b", PASS))
    r.add(CheckResult("c", WARN))
    r.add(CheckResult("d", FAIL))
    r.add(CheckResult("e", SKIP))
    assert r.n_pass == 2
    assert r.n_warn == 1
    assert r.n_fail == 1
    assert r.n_skip == 1
    assert r.overall_pass is False  # has FAIL


def test_verify_report_warn_does_not_fail_overall():
    """WARN should NOT fail overall — only FAIL does. This is critical
    for installer UX: warnings should not abort the install."""
    from vllm._genesis.compat.verify import (
        PASS,
        WARN,
        CheckResult,
        VerifyReport,
    )

    r = VerifyReport()
    r.add(CheckResult("a", PASS))
    r.add(CheckResult("b", WARN))
    assert r.overall_pass is True


# ─────────────────────────────────────────────────────────────────
# run_verify(level)
# ─────────────────────────────────────────────────────────────────


def test_run_verify_quick_does_not_raise():
    from vllm._genesis.compat.verify import run_verify

    # Should never raise — even on CPU-only / no-vllm hosts
    report = run_verify(level="quick")
    assert report is not None
    assert len(report.checks) >= 9  # C1-C9


def test_run_verify_quick_includes_all_C_checks():
    from vllm._genesis.compat.verify import run_verify

    report = run_verify(level="quick")
    names = [c.name for c in report.checks]
    for cid in ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]:
        assert any(n.startswith(cid) for n in names), (
            f"missing quick check {cid} (got: {names})"
        )


def test_run_verify_boot_includes_B_checks():
    from vllm._genesis.compat.verify import run_verify

    report = run_verify(level="boot")
    names = [c.name for c in report.checks]
    for bid in ["B1", "B2", "B3"]:
        assert any(n.startswith(bid) for n in names), (
            f"missing boot check {bid} (got: {names})"
        )


def test_run_verify_full_includes_F_checks():
    from vllm._genesis.compat.verify import run_verify

    report = run_verify(level="full")
    names = [c.name for c in report.checks]
    assert any(n.startswith("F1") for n in names), (
        f"missing full check F1 (got: {names})"
    )


def test_run_verify_unknown_level_raises():
    from vllm._genesis.compat.verify import run_verify

    with pytest.raises(ValueError, match="unknown level"):
        run_verify(level="banana")


def test_run_verify_each_check_has_duration_ms():
    """Every check should have duration_ms >= 0 after running."""
    from vllm._genesis.compat.verify import run_verify

    report = run_verify(level="quick")
    for c in report.checks:
        assert c.duration_ms >= 0, (
            f"check {c.name!r} has negative duration"
        )


def test_run_verify_idempotent():
    """Running verify twice gives same overall pass/fail (no state leak)."""
    from vllm._genesis.compat.verify import run_verify

    r1 = run_verify(level="quick")
    r2 = run_verify(level="quick")
    assert r1.overall_pass == r2.overall_pass
    assert r1.n_pass == r2.n_pass


# ─────────────────────────────────────────────────────────────────
# Specific check correctness
# ─────────────────────────────────────────────────────────────────


def test_check_genesis_importable_passes_when_installed():
    """C1 must pass — we ARE running inside an installed Genesis."""
    from vllm._genesis.compat.verify import (
        PASS,
        _check_genesis_importable,
    )

    r = _check_genesis_importable()
    assert r.status == PASS


def test_check_dispatcher_loads_passes_with_full_registry():
    """C2 must pass with all 100 patches."""
    from vllm._genesis.compat.verify import (
        PASS,
        _check_dispatcher_loads,
    )

    r = _check_dispatcher_loads()
    assert r.status == PASS, f"C2 unexpected status: {r.status} ({r.detail})"


def test_check_cli_routes_includes_required_subcommands():
    """C4 must verify presence of doctor + preset + verify."""
    from vllm._genesis.compat.verify import PASS, _check_cli_routes

    r = _check_cli_routes()
    assert r.status == PASS, f"C4 unexpected: {r.status} ({r.detail})"


def test_check_gpu_detected_warns_on_no_cuda(monkeypatch):
    """C5 should WARN (not FAIL) on CPU-only / no-CUDA hosts —
    Genesis is meaningful in install-only mode."""
    from vllm._genesis.compat import verify

    # Force detect_current_gpu to return None
    monkeypatch.setattr(
        "vllm._genesis.gpu_profile.detect_current_gpu", lambda: None
    )
    r = verify._check_gpu_detected()
    assert r.status == verify.WARN
    assert "no CUDA" in r.detail or "torch" in r.detail


def test_check_plugin_entry_point_warns_on_missing(monkeypatch):
    """C8 should WARN if plugin entry point not registered."""
    from vllm._genesis.compat import verify

    # Mock entry_points to return empty
    import importlib.metadata as _md

    orig = _md.entry_points

    class _FakeEntries:
        def __init__(self): pass
        def __iter__(self): return iter([])

    monkeypatch.setattr(_md, "entry_points",
                        lambda group=None: _FakeEntries())

    r = verify._check_plugin_entry_point()
    assert r.status in (verify.WARN, verify.FAIL)
    assert r.hint is not None  # WARN/FAIL must have a hint


# ─────────────────────────────────────────────────────────────────
# render_report
# ─────────────────────────────────────────────────────────────────


def test_render_report_contains_all_check_names():
    from vllm._genesis.compat.verify import render_report, run_verify

    report = run_verify(level="quick")
    text = render_report(report)
    for c in report.checks:
        assert c.name in text


def test_render_report_shows_summary_line():
    from vllm._genesis.compat.verify import render_report, run_verify

    report = run_verify(level="quick")
    text = render_report(report)
    assert "overall:" in text
    assert "pass" in text
    assert "warn" in text
    assert "fail" in text


def test_render_report_shows_hints_for_warn_and_fail():
    from vllm._genesis.compat.verify import (
        FAIL,
        WARN,
        CheckResult,
        VerifyReport,
        render_report,
    )

    r = VerifyReport()
    r.add(CheckResult(
        "X1 test warn", WARN, "synthetic", hint="do this thing"
    ))
    r.add(CheckResult(
        "X2 test fail", FAIL, "synthetic", hint="fix that thing"
    ))
    text = render_report(r)
    assert "do this thing" in text
    assert "fix that thing" in text


# ─────────────────────────────────────────────────────────────────
# JSON output
# ─────────────────────────────────────────────────────────────────


def test_to_dict_returns_serializable():
    from vllm._genesis.compat.verify import run_verify

    report = run_verify(level="quick")
    d = report.to_dict()
    # Must be JSON-serializable
    s = json.dumps(d)
    parsed = json.loads(s)
    assert "checks" in parsed
    assert "summary" in parsed
    assert "overall_pass" in parsed["summary"]


def test_to_dict_summary_counts_match_attributes():
    from vllm._genesis.compat.verify import run_verify

    report = run_verify(level="quick")
    d = report.to_dict()
    assert d["summary"]["pass"] == report.n_pass
    assert d["summary"]["warn"] == report.n_warn
    assert d["summary"]["fail"] == report.n_fail
    assert d["summary"]["skip"] == report.n_skip


# ─────────────────────────────────────────────────────────────────
# CLI smoke
# ─────────────────────────────────────────────────────────────────


def _run_cli(*args: str) -> tuple[int, str, str]:
    """Run verify CLI as subprocess; return (exit_code, stdout, stderr)."""
    repo_root = Path(__file__).resolve().parents[4]
    proc = subprocess.run(
        [sys.executable, "-m", "vllm._genesis.compat.verify", *args],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
    )
    return proc.returncode, proc.stdout, proc.stderr


def test_cli_default_runs_quick_and_exits_clean_on_cpu_only_host():
    """The default `verify` (no flags) must not exit with code != 0
    on a CPU-only host — only WARNs are expected, not FAILs."""
    rc, out, err = _run_cli()
    assert rc == 0, f"verify exit non-zero on CPU host: rc={rc}, err={err}"
    assert "Genesis verify" in out
    assert "overall:" in out


def test_cli_json_emits_valid_json():
    rc, out, err = _run_cli("--json")
    # JSON output should always be valid even on warn/fail
    data = json.loads(out)
    assert "checks" in data
    assert "summary" in data
    assert isinstance(data["checks"], list)
    assert len(data["checks"]) >= 9


def test_cli_quick_flag_explicit():
    rc, out, err = _run_cli("--quick")
    assert rc == 0
    assert "Genesis verify" in out


def test_cli_boot_flag_includes_B_checks():
    rc, out, err = _run_cli("--boot", "--json")
    data = json.loads(out)
    names = [c["name"] for c in data["checks"]]
    assert any(n.startswith("B1") for n in names)


def test_cli_help_runs_without_error():
    rc, out, err = _run_cli("--help")
    assert rc == 0
    assert "Genesis post-install smoke test" in out or "verify" in out


def test_cli_invalid_flag_exits_nonzero():
    rc, out, err = _run_cli("--nonexistent-flag")
    assert rc != 0


# ─────────────────────────────────────────────────────────────────
# Integration with unified `genesis` CLI dispatcher
# ─────────────────────────────────────────────────────────────────


def test_verify_registered_in_unified_cli():
    from vllm._genesis.compat.cli import KNOWN_SUBCOMMANDS
    assert "verify" in KNOWN_SUBCOMMANDS, (
        "verify must be a subcommand of `genesis ...`"
    )


def test_unified_cli_routes_verify_subcommand():
    """`python3 -m vllm._genesis.compat.cli verify --quick` must work
    (test the dispatcher path, not just direct module invocation)."""
    repo_root = Path(__file__).resolve().parents[4]
    proc = subprocess.run(
        [
            sys.executable, "-m", "vllm._genesis.compat.cli",
            "verify", "--quick",
        ],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
    )
    assert proc.returncode == 0, (
        f"unified CLI verify routing failed: rc={proc.returncode}, "
        f"err={proc.stderr}"
    )
    assert "Genesis verify" in proc.stdout
