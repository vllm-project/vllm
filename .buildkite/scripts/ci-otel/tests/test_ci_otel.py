# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import binascii
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

import ci_otel  # noqa: E402
import ci_pytest_otel  # noqa: E402
from ci_otel import (  # noqa: E402
    Span,
    encode_request,
    export_spans,
    load_spans,
    new_context,
    record_spans,
)


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


def test_spans_are_spooled_without_requesting_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("CI_INFRA_OTEL_SPOOL_DIR", str(tmp_path))
    monkeypatch.setattr(
        ci_otel,
        "_oidc_token",
        lambda deadline: (_ for _ in ()).throw(AssertionError("unexpected upload")),
    )
    span = Span(
        trace_id="01" * 16,
        span_id="02" * 8,
        parent_span_id=None,
        name="ci.command",
        start_ns=100,
        end_ns=200,
        attributes={"ci.command.index": 1},
    )

    assert record_spans([span]) is True
    assert load_spans() == [span]


def test_export_mints_one_token_for_multiple_batches(monkeypatch):
    for name, value in {
        "BUILDKITE": "true",
        "BUILDKITE_ORGANIZATION_SLUG": "vllm",
        "BUILDKITE_PIPELINE_SLUG": "ci",
        "BUILDKITE_BUILD_ID": "build-id",
        "BUILDKITE_BUILD_NUMBER": "42",
        "BUILDKITE_JOB_ID": "job-id",
        "BUILDKITE_BRANCH": "main",
    }.items():
        monkeypatch.setenv(name, value)
    token_calls = []
    request_timeouts = []
    monkeypatch.setattr(ci_otel, "MAX_BATCH_SIZE", 1)
    monkeypatch.setattr(
        ci_otel,
        "_oidc_token",
        lambda deadline: token_calls.append(deadline) or "token",
    )

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def open_request(request, timeout):
        request_timeouts.append(timeout)
        return Response()

    monkeypatch.setattr(ci_otel.urllib.request, "urlopen", open_request)
    spans = [
        Span(
            trace_id="01" * 16,
            span_id=f"{index:016x}",
            parent_span_id=None,
            name="pytest.test",
            start_ns=100,
            end_ns=200,
            attributes={"test.nodeid": f"test_{index}"},
        )
        for index in (1, 2)
    ]

    assert export_spans(spans, timeout_seconds=0.5) is True
    assert len(token_calls) == 1
    assert len(request_timeouts) == 2
    assert all(0 < timeout <= 0.5 for timeout in request_timeouts)


def test_export_deadline_bounds_oidc_request(monkeypatch):
    for name, value in {
        "BUILDKITE": "true",
        "BUILDKITE_ORGANIZATION_SLUG": "vllm",
        "BUILDKITE_PIPELINE_SLUG": "ci",
        "BUILDKITE_BUILD_ID": "build-id",
        "BUILDKITE_BUILD_NUMBER": "42",
        "BUILDKITE_JOB_ID": "job-id",
        "BUILDKITE_BRANCH": "main",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("BUILDKITE_AGENT_ACCESS_TOKEN", "agent-token")

    def open_request(request, timeout):
        assert 0 < timeout <= 0.1
        raise TimeoutError("request timed out")

    monkeypatch.setattr(ci_otel.urllib.request, "urlopen", open_request)
    span = Span(
        trace_id="01" * 16,
        span_id="02" * 8,
        parent_span_id=None,
        name="ci.command",
        start_ns=100,
        end_ns=200,
        attributes={},
    )

    started = time.monotonic()
    assert export_spans([span], timeout_seconds=0.1) is False

    assert time.monotonic() - started < 0.5


def test_oidc_token_uses_buildkite_job_api(monkeypatch):
    monkeypatch.setenv("BUILDKITE_AGENT_ACCESS_TOKEN", "agent-token")
    monkeypatch.setenv("BUILDKITE_JOB_ID", "job-id")
    monkeypatch.setenv("BUILDKITE_AGENT_ENDPOINT", "https://agent.example/v3/")
    observed = {}

    class Result:
        def read(self):
            return b'{"token":"oidc-token"}'

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def open_request(request, timeout):
        observed["request"] = request
        observed["timeout"] = timeout
        return Result()

    monkeypatch.setattr(ci_otel.urllib.request, "urlopen", open_request)

    assert ci_otel._oidc_token(time.monotonic() + 1) == "oidc-token"
    request = observed["request"]
    assert request.full_url == "https://agent.example/v3/jobs/job-id/oidc/tokens"
    assert request.get_header("Authorization") == "Token agent-token"
    assert json.loads(request.data) == {
        "audience": "https://ci.vllm.ai/api/otel",
        "lifetime": 300,
    }
    assert 0 < observed["timeout"] <= 1


def test_oidc_token_requires_job_credentials(monkeypatch):
    monkeypatch.delenv("BUILDKITE_AGENT_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("BUILDKITE_JOB_ID", raising=False)

    try:
        ci_otel._oidc_token(time.monotonic() + 1)
    except RuntimeError as error:
        assert str(error) == "Buildkite job credentials are unavailable"
    else:
        raise AssertionError("missing job credentials should be reported")


def test_safe_oidc_claims_reports_identity_without_token():
    payload = {
        "iss": "https://agent.buildkite.com",
        "aud": "https://ci.vllm.ai/api/otel",
        "organization_slug": "vllm",
        "pipeline_slug": "ci",
        "build_branch": "khluu/otel",
        "build_number": 42,
        "job_id": "job-id",
        "private": "must-not-be-logged",
    }
    encoded = binascii.b2a_base64(json.dumps(payload).encode(), newline=False).rstrip(
        b"="
    )
    encoded = encoded.translate(bytes.maketrans(b"+/", b"-_"))
    token = f"header.{encoded.decode()}.signature"

    claims = ci_otel._safe_oidc_claims(token)

    assert '"build_branch":"khluu/otel"' in claims
    assert '"build_source":null' in claims
    assert "must-not-be-logged" not in claims
    assert token not in claims


def test_invalid_upload_timeout_cannot_break_plugin_import():
    result = subprocess.run(
        [sys.executable, "-c", "import ci_otel, ci_pytest_otel"],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(SCRIPTS_DIR),
            "CI_INFRA_OTEL_UPLOAD_TIMEOUT": "not-a-number",
        },
    )

    assert result.returncode == 0, result.stderr


def test_export_contains_unexpected_internal_errors(monkeypatch):
    for name, value in {
        "BUILDKITE": "true",
        "BUILDKITE_ORGANIZATION_SLUG": "vllm",
        "BUILDKITE_PIPELINE_SLUG": "ci",
        "BUILDKITE_BUILD_ID": "build-id",
        "BUILDKITE_BUILD_NUMBER": "42",
        "BUILDKITE_JOB_ID": "job-id",
        "BUILDKITE_BRANCH": "main",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        ci_otel,
        "_oidc_token",
        lambda deadline: (_ for _ in ()).throw(ValueError("broken exporter")),
    )
    span = Span(
        trace_id="01" * 16,
        span_id="02" * 8,
        parent_span_id=None,
        name="ci.command",
        start_ns=100,
        end_ns=200,
        attributes={},
    )

    assert export_spans([span]) is False


def test_record_spans_contains_serialization_errors(monkeypatch, tmp_path):
    monkeypatch.setenv("CI_INFRA_OTEL_SPOOL_DIR", str(tmp_path))
    span = Span(
        trace_id="01" * 16,
        span_id="02" * 8,
        parent_span_id=None,
        name="ci.command",
        start_ns=100,
        end_ns=200,
        attributes={"invalid": object()},  # type: ignore[dict-item]
    )

    assert record_spans([span]) is False


def test_shell_wrapper_preserves_command_state_and_quoting(tmp_path):
    script = SCRIPTS_DIR / "ci_otel.sh"
    first = f"ci_otel_start 1 {_encoded('export VALUE=ready')}"
    second = f"ci_otel_start 2 {_encoded('check VALUE')}"
    shell = (
        f'source "{script}"; {first}; export VALUE=ready; ci_otel_finish 0; '
        f'{second}; test "$VALUE" = ready; ci_otel_finish 0'
    )
    result = subprocess.run(
        ["bash", "-c", shell],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "CI_INFRA_OTEL_DIR": str(SCRIPTS_DIR),
            "CI_INFRA_OTEL_SPOOL_DIR": str(tmp_path),
        },
    )

    assert result.returncode == 0, result.stderr


def test_shell_wrapper_expands_runtime_environment_arguments(tmp_path):
    script = SCRIPTS_DIR / "ci_otel.sh"
    start = f"ci_otel_start 1 {_encoded('check generated variables')}"
    command = 'test "$REGISTRY" = registry.example.com && test "$COMMIT" = abc123'
    result = subprocess.run(
        ["bash", "-c", f'source "{script}"; {start}; {command}; ci_otel_finish $?'],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "REGISTRY": "registry.example.com",
            "COMMIT": "abc123",
            "CI_INFRA_OTEL_DIR": str(SCRIPTS_DIR),
            "CI_INFRA_OTEL_SPOOL_DIR": str(tmp_path),
        },
    )

    assert result.returncode == 0, result.stderr


def test_shell_wrapper_preserves_failure_status(tmp_path):
    script = SCRIPTS_DIR / "ci_otel.sh"
    start = f"ci_otel_start 1 {_encoded('false')}"
    result = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{script}"; {start}; false; status=$?; '
            'ci_otel_finish "$status"; (exit "$status")',
        ],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "CI_INFRA_OTEL_DIR": str(SCRIPTS_DIR),
            "CI_INFRA_OTEL_SPOOL_DIR": str(tmp_path),
        },
    )

    assert result.returncode == 1, result.stderr


def test_shell_wrapper_runs_command_when_context_creation_fails(tmp_path):
    script = SCRIPTS_DIR / "ci_otel.sh"
    start = f"ci_otel_start 1 {_encoded('write marker')}"
    result = subprocess.run(
        [
            "/bin/sh",
            "-c",
            f'. "{script}"; python3() {{ return 1; }}; {start} || :; '
            'printf ran > "$OUTPUT_FILE"',
        ],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "OUTPUT_FILE": str(tmp_path / "ran"),
            "CI_INFRA_OTEL_DIR": str(SCRIPTS_DIR),
            "CI_INFRA_OTEL_RUNTIME_DIR": str(tmp_path / "runtime"),
            "CI_INFRA_OTEL_SPOOL_DIR": str(tmp_path / "spans"),
        },
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "ran").read_text(encoding="utf-8") == "ran"


def test_shell_wrapper_ignores_recording_failure(tmp_path):
    script = SCRIPTS_DIR / "ci_otel.sh"
    start = f"ci_otel_start 1 {_encoded('true')}"
    shell = f"""
      . "{script}"
      python3() {{
        if [ "$2" = "new-context" ]; then
          echo 01010101010101010101010101010101 0202020202020202 -
          return 0
        fi
        return 99
      }}
      {start}
      true
      ci_otel_finish 0
    """
    result = subprocess.run(
        ["/bin/sh", "-e", "-c", shell],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "CI_INFRA_OTEL_DIR": str(SCRIPTS_DIR),
            "CI_INFRA_OTEL_SPOOL_DIR": str(tmp_path),
        },
    )

    assert result.returncode == 0, result.stderr


def test_exit_flush_failure_preserves_success_and_failure(tmp_path):
    script = SCRIPTS_DIR / "ci_otel.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python3"
    fake_python.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    fake_python.chmod(0o755)

    def run_with_status(status: int):
        shell = f'. "{script}"; PATH="{fake_bin}"; export PATH; exit {status}'
        return subprocess.run(
            ["/bin/sh", "-c", shell],
            check=False,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "CI_INFRA_OTEL_DIR": str(SCRIPTS_DIR),
                "CI_INFRA_OTEL_SPOOL_DIR": str(tmp_path / f"spans-{status}"),
            },
        )

    assert run_with_status(0).returncode == 0
    assert run_with_status(7).returncode == 7


def test_pytest_hooks_contain_all_tracing_errors(monkeypatch):
    monkeypatch.setenv("CI_INFRA_TRACE_ID", "01" * 16)
    monkeypatch.setenv("CI_INFRA_COMMAND_SPAN_ID", "02" * 8)
    ci_pytest_otel._runs.clear()
    ci_pytest_otel._spans.clear()
    monkeypatch.setattr(
        ci_pytest_otel.time,
        "time_ns",
        lambda: (_ for _ in ()).throw(RuntimeError("broken clock")),
    )

    ci_pytest_otel.pytest_runtest_logstart("test_example", ("test.py", 1, "test"))
    ci_pytest_otel.pytest_runtest_logreport(object())
    ci_pytest_otel.pytest_runtest_logfinish("test_example", ("test.py", 1, "test"))
    ci_pytest_otel.pytest_sessionfinish(None, 0)


def test_real_pytest_passes_when_otel_spool_is_unwritable(tmp_path):
    test_file = tmp_path / "test_sample.py"
    test_file.write_text("def test_sample():\n    assert True\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(test_file)],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(SCRIPTS_DIR),
            "PYTEST_ADDOPTS": "-p ci_pytest_otel",
            "CI_INFRA_TRACE_ID": "01" * 16,
            "CI_INFRA_COMMAND_SPAN_ID": "02" * 8,
            "CI_INFRA_OTEL_SPOOL_DIR": "/dev/null/spans",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "1 passed" in result.stdout


def test_pytest_shim_traces_with_command_level_pythonpath(tmp_path):
    test_file = tmp_path / "test_sample.py"
    test_file.write_text("def test_sample():\n    assert True\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    pytest_executable = bin_dir / "pytest"
    pytest_executable.write_text(
        f'#!/bin/sh\nexec {shlex.quote(sys.executable)} -m pytest "$@"\n',
        encoding="utf-8",
    )
    pytest_executable.chmod(0o755)
    encoded_label = _encoded("pytest with PYTHONPATH override")
    shell = (
        f'. "{SCRIPTS_DIR / "ci_otel.sh"}"\n'
        f"ci_otel_start 1 {encoded_label}\n"
        f"PYTHONPATH={shlex.quote(str(workspace))} "
        f"pytest -q {shlex.quote(str(test_file))}\n"
        "status=$?\n"
        'ci_otel_finish "$status"\n'
        'grep -q \'"name":"pytest.test"\' '
        '"$CI_INFRA_OTEL_SPOOL_DIR"/spans-*.jsonl\n'
        '(exit "$status")'
    )
    result = subprocess.run(
        ["/bin/sh", "-e", "-c", shell],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "CI_INFRA_OTEL_DIR": str(SCRIPTS_DIR),
            "CI_INFRA_OTEL_SPOOL_DIR": str(tmp_path / "spans"),
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "PYTEST_ADDOPTS": "",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "1 passed" in result.stdout
    assert "No module named 'ci_pytest_otel'" not in result.stderr
