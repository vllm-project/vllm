# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import binascii
import os
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from ci_otel import Span, encode_request, export_spans, new_context  # noqa: E402


def _encoded(value: str) -> str:
    return binascii.b2a_base64(value.encode(), newline=False).decode()


def test_new_context_continues_w3c_traceparent(monkeypatch):
    monkeypatch.setenv(
        "TRACEPARENT",
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
    )

    trace_id, span_id, parent_span_id = new_context()

    assert trace_id == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert len(span_id) == 16
    assert parent_span_id == "00f067aa0ba902b7"


def test_otlp_payload_contains_span_and_build_identity(monkeypatch):
    monkeypatch.setenv("BUILDKITE_ORGANIZATION_SLUG", "vllm")
    monkeypatch.setenv("BUILDKITE_PIPELINE_SLUG", "ci")
    monkeypatch.setenv("BUILDKITE_BUILD_ID", "build-id")
    monkeypatch.setenv("BUILDKITE_BUILD_NUMBER", "42")
    monkeypatch.setenv("BUILDKITE_BRANCH", "main")
    monkeypatch.setenv("BUILDKITE_JOB_ID", "job-id")
    span = Span(
        trace_id="01" * 16,
        span_id="02" * 8,
        parent_span_id="03" * 8,
        name="ci.command",
        start_ns=100,
        end_ns=200,
        attributes={"ci.span.kind": "command"},
    )

    payload = encode_request([span])

    assert b"ci.command" in payload
    assert b"ci.span.kind" in payload
    assert b"buildkite.job.id" in payload
    assert b"job-id" in payload


def test_export_is_disabled_outside_buildkite(monkeypatch):
    monkeypatch.delenv("BUILDKITE", raising=False)

    assert export_spans([]) is False


def test_shell_wrapper_preserves_command_state_and_quoting():
    script = SCRIPTS_DIR / "ci_otel.sh"
    first = "ci_otel_run 1 {} {}".format(
        _encoded("export VALUE=ready"),
        _encoded("export VALUE=ready"),
    )
    second_command = 'test "$VALUE" = ready'
    second = f"ci_otel_run 2 {_encoded('check VALUE')} {_encoded(second_command)}"
    result = subprocess.run(
        ["bash", "-c", f'source "{script}"; {first}; {second}'],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "VLLM_CI_OTEL_DIR": str(SCRIPTS_DIR)},
    )

    assert result.returncode == 0, result.stderr
