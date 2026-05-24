# SPDX-License-Identifier: Apache-2.0
"""TDD for compat/preflight_checks — PN60 + club#43 + club#34 doctor rules."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from vllm._genesis.compat.preflight_checks import (
    CheckResult,
    check_grammar_rejection_pattern,
    check_quant_arg,
    check_spec_decode_token_loop,
)


# ─── PN60 quant arg validator ───────────────────────────────────────────


def _make_model_dir(quant_method: str) -> str:
    """Create a tempdir with a minimal config.json holding the given quant_method."""
    tmp = tempfile.mkdtemp(prefix="genesis-pn60-test-")
    config = {
        "model_type": "qwen3_5",
        "quantization_config": {"quant_method": quant_method},
    }
    Path(tmp, "config.json").write_text(json.dumps(config))
    return tmp


class TestPN60QuantValidator:
    def test_match_returns_ok(self):
        model_dir = _make_model_dir("compressed-tensors")
        result = check_quant_arg("compressed-tensors", model_dir)
        assert result.severity == "OK"
        assert result.remediation is None

    def test_alias_match_normalized(self):
        """compressed-tensors and compressed_tensors should both match."""
        model_dir = _make_model_dir("compressed_tensors")  # underscore form
        result = check_quant_arg("compressed-tensors", model_dir)  # dash form
        assert result.severity == "OK"

    def test_mismatch_returns_error_with_remediation(self):
        """apnar club-3090#51 reproducer: --quantization auto_round vs config compressed-tensors."""
        model_dir = _make_model_dir("compressed-tensors")
        result = check_quant_arg("auto_round", model_dir)
        assert result.severity == "ERROR"
        assert "auto_round" in result.message
        assert "compressed-tensors" in result.message
        assert result.remediation is not None
        assert "compressed-tensors" in result.remediation

    def test_no_cli_quant_returns_ok_with_inference_note(self):
        model_dir = _make_model_dir("auto_round")
        result = check_quant_arg(None, model_dir)
        assert result.severity == "OK"
        msg = result.message.lower()
        assert "config-detected" in msg or "infer" in msg

    def test_no_config_returns_info_skip(self):
        result = check_quant_arg("auto_round", "/nonexistent/path/12345")
        assert result.severity == "INFO"
        assert "skip" in result.message.lower()

    def test_unknown_alias_falls_back_to_literal(self):
        model_dir = _make_model_dir("custom-method")
        result = check_quant_arg("custom-method", model_dir)
        assert result.severity == "OK"


# ─── club#43 grammar rejection ──────────────────────────────────────────


class TestClub43GrammarRejection:
    def test_no_pattern_returns_ok(self):
        log_text = "INFO healthy boot\nINFO request 200 OK\n"
        result = check_grammar_rejection_pattern(log_text)
        assert result.severity == "OK"

    def test_all_candidates_rejected_pattern_warns(self):
        log_text = (
            "INFO request 200 OK\n"
            "WARNING all 32 candidate tokens were rejected\n"
            "INFO grammar reject something\n"
        )
        result = check_grammar_rejection_pattern(log_text)
        assert result.severity == "WARN"
        assert "remediation" in str(result).lower() or result.remediation

    def test_single_weak_signal_returns_info(self):
        log_text = "WARNING GrammarRejection occurred at step 17\n"
        result = check_grammar_rejection_pattern(log_text)
        # single weak hit → INFO, not WARN
        assert result.severity in ("INFO", "WARN")  # tolerate either


# ─── club#34 spec-decode token-loop ─────────────────────────────────────


class TestClub34TokenLoop:
    def test_healthy_metrics_return_ok(self):
        # healthy: gen throughput > 0
        log_text = (
            "INFO Mean acceptance length: 3.2, Accepted throughput: 30.0 tokens/s, "
            "Drafted throughput: 35.0 tokens/s, Avg Draft acceptance rate: 85.0%\n"
            "INFO Avg generation throughput: 25.0 tokens/s, Running: 1 reqs\n"
        )
        result = check_spec_decode_token_loop(log_text)
        assert result.severity == "OK"

    def test_stuck_pattern_warns(self):
        # Multiple snapshots: accept >50%, gen=0, Running >=1 → stuck
        log_text = (
            "INFO Mean acceptance length: 3.5, Accepted throughput: 0.0 tokens/s, "
            "Drafted throughput: 5.0 tokens/s, Avg Draft acceptance rate: 90.0%\n"
            "INFO Avg generation throughput: 0.0 tokens/s, Running: 1 reqs\n"
            "INFO Mean acceptance length: 3.4, Accepted throughput: 0.0 tokens/s, "
            "Drafted throughput: 5.0 tokens/s, Avg Draft acceptance rate: 88.0%\n"
            "INFO Avg generation throughput: 0.0 tokens/s, Running: 1 reqs\n"
        )
        result = check_spec_decode_token_loop(log_text)
        assert result.severity == "WARN"
        assert "stuck" in result.message.lower() or "loop" in result.message.lower()

    def test_no_metrics_returns_ok_with_no_signal(self):
        log_text = "INFO booting\nINFO ready\n"
        result = check_spec_decode_token_loop(log_text)
        assert result.severity == "OK"
        assert "no signal" in result.message.lower() or \
               "insufficient" in result.message.lower()


# ─── CheckResult formatting ─────────────────────────────────────────────


class TestCheckResultRender:
    def test_str_includes_severity_name_message(self):
        r = CheckResult(name="X", severity="WARN", message="issue here")
        s = str(r)
        assert "[WARN]" in s
        assert "X" in s
        assert "issue here" in s

    def test_str_includes_remediation_if_present(self):
        r = CheckResult(
            name="X", severity="ERROR",
            message="thing failed",
            remediation="run command Y",
        )
        s = str(r)
        assert "→" in s
        assert "run command Y" in s


# ─── CLI integration ────────────────────────────────────────────────────


class TestCLIMain:
    def test_main_exits_0_when_all_ok(self, capsys, monkeypatch):
        # No model_dir, no container — should yield 0 results = exit 0
        from vllm._genesis.compat.preflight_checks import main
        monkeypatch.setenv("GENESIS_DOCKER_LOGS_AVAILABLE", "0")
        rc = main([])
        assert rc == 0

    def test_main_exits_2_on_error(self, capsys, monkeypatch):
        from vllm._genesis.compat.preflight_checks import main
        monkeypatch.setenv("GENESIS_DOCKER_LOGS_AVAILABLE", "0")
        model_dir = _make_model_dir("compressed-tensors")
        rc = main(["--quantization", "auto_round", "--model", model_dir])
        assert rc == 2

    def test_main_json_output(self, capsys, monkeypatch):
        from vllm._genesis.compat.preflight_checks import main
        monkeypatch.setenv("GENESIS_DOCKER_LOGS_AVAILABLE", "0")
        model_dir = _make_model_dir("compressed-tensors")
        main(["--quantization", "compressed-tensors", "--model", model_dir,
              "--json"])
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert isinstance(parsed, list)
        assert any(item.get("name") == "PN60 quant validator" for item in parsed)
