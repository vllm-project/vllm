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
from ci_otel import Span  # noqa: E402


def _span(**attributes) -> Span:
    return Span(
        trace_id="01" * 16,
        span_id="02" * 8,
        parent_span_id=None,
        name="ci.command",
        start_ns=100,
        end_ns=200,
        attributes=attributes,
    )


def _encoded(value: str) -> str:
    return binascii.b2a_base64(value.encode(), newline=False).decode()


def test_context_continues_traceparent(monkeypatch):
    monkeypatch.setenv(
        "TRACEPARENT",
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
    )

    trace_id, span_id, parent_span_id = ci_otel.new_context()

    assert trace_id == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert len(span_id) == 16
    assert parent_span_id == "00f067aa0ba902b7"


def test_otlp_payload_contains_dashboard_identity(monkeypatch):
    monkeypatch.setenv("BUILDKITE_ORGANIZATION_SLUG", "vllm")
    monkeypatch.setenv("BUILDKITE_PIPELINE_SLUG", "ci")
    monkeypatch.setenv("BUILDKITE_BUILD_ID", "build-id")
    monkeypatch.setenv("BUILDKITE_BUILD_NUMBER", "42")
    monkeypatch.setenv("BUILDKITE_JOB_ID", "job-id")

    payload = ci_otel.encode_request([_span(**{"ci.span.kind": "command"})])

    for value in (b"ci.command", b"ci.span.kind", b"buildkite.job.id", b"job-id"):
        assert value in payload


def test_spool_round_trip_is_local(monkeypatch, tmp_path):
    monkeypatch.setenv("CI_INFRA_OTEL_SPOOL_DIR", str(tmp_path))
    monkeypatch.setattr(
        ci_otel,
        "_oidc_token",
        lambda deadline: (_ for _ in ()).throw(AssertionError("unexpected upload")),
    )
    span = _span(**{"ci.command.index": 1})

    assert ci_otel.record_spans([span]) is True
    assert ci_otel.load_spans() == [span]


def test_export_uses_buildkite_oidc_and_otlp(monkeypatch):
    for name, value in {
        "BUILDKITE": "true",
        "BUILDKITE_AGENT_ACCESS_TOKEN": "agent-token",
        "BUILDKITE_AGENT_ENDPOINT": "https://agent.example/v3",
        "BUILDKITE_JOB_ID": "job-id",
    }.items():
        monkeypatch.setenv(name, value)
    requests = []

    class Response:
        status = 200

        def __init__(self, body=b""):
            self.body = body

        def read(self):
            return self.body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def open_request(request, timeout):
        requests.append((request, timeout))
        if request.full_url.endswith("/oidc/tokens"):
            return Response(b'{"token":"oidc-token"}')
        return Response()

    monkeypatch.setattr(ci_otel.urllib.request, "urlopen", open_request)

    assert ci_otel.export_spans([_span()], timeout_seconds=0.5) is True
    oidc_request, upload_request = (request for request, _ in requests)
    assert oidc_request.full_url.endswith("/jobs/job-id/oidc/tokens")
    assert oidc_request.get_header("Authorization") == "Token agent-token"
    assert json.loads(oidc_request.data)["audience"] == ci_otel.AUDIENCE
    assert upload_request.get_header("Authorization") == "Bearer oidc-token"
    assert upload_request.get_header("Content-type") == "application/x-protobuf"
    assert all(0 < timeout <= 0.5 for _, timeout in requests)


def test_export_failure_is_soft_and_bounded(monkeypatch):
    monkeypatch.setenv("BUILDKITE", "true")
    monkeypatch.setattr(
        ci_otel,
        "_oidc_token",
        lambda deadline: (_ for _ in ()).throw(TimeoutError("unavailable")),
    )

    started = time.monotonic()
    assert ci_otel.export_spans([_span()], timeout_seconds=0.1) is False
    assert time.monotonic() - started < 0.5


def test_shell_wrapper_records_commands_without_changing_shell_state(tmp_path):
    first = f"ci_otel_start 1 {_encoded('export VALUE=ready')}"
    second = f"ci_otel_start 2 {_encoded('check VALUE')}"
    shell = (
        f'. "{SCRIPTS_DIR / "ci_otel.sh"}"; {first}; '
        f"export VALUE=ready; ci_otel_finish 0; {second}; "
        'test "$VALUE" = ready; ci_otel_finish 0'
    )

    result = subprocess.run(
        ["/bin/sh", "-c", shell],
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
    records = "".join(path.read_text() for path in tmp_path.glob("spans-*.jsonl"))
    assert "export VALUE=ready" in records
    assert "check VALUE" in records


def test_shell_wrapper_preserves_failure_status(tmp_path):
    shell = (
        f'. "{SCRIPTS_DIR / "ci_otel.sh"}"; '
        f"ci_otel_start 1 {_encoded('false')}; false; status=$?; "
        'ci_otel_finish "$status"; exit "$status"'
    )

    result = subprocess.run(
        ["/bin/sh", "-c", shell],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "CI_INFRA_OTEL_DIR": str(SCRIPTS_DIR),
            "CI_INFRA_OTEL_SPOOL_DIR": str(tmp_path),
        },
    )

    assert result.returncode == 1


def test_missing_helpers_do_not_block_the_test_command(tmp_path):
    output = tmp_path / "ran"
    shell = (
        f'CI_INFRA_OTEL_DIR="{tmp_path / "missing"}"; export CI_INFRA_OTEL_DIR; '
        f'. "{SCRIPTS_DIR / "ci_otel.sh"}"; printf ran > "{output}"'
    )

    result = subprocess.run(
        ["/bin/sh", "-e", "-c", shell],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert output.read_text() == "ran"


def test_flush_failure_preserves_job_status(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python3"
    fake_python.write_text("#!/bin/sh\nexit 99\n")
    fake_python.chmod(0o755)

    def run(status: int):
        shell = (
            f'. "{SCRIPTS_DIR / "ci_otel.sh"}"; '
            f'PATH="{fake_bin}"; export PATH; exit {status}'
        )
        return subprocess.run(
            ["/bin/sh", "-c", shell],
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "CI_INFRA_OTEL_DIR": str(SCRIPTS_DIR)},
        )

    assert run(0).returncode == 0
    assert run(7).returncode == 7


def test_pytest_shim_records_distinct_test_intervals_with_pythonpath_override(
    tmp_path,
):
    test_file = tmp_path / "test_sample.py"
    test_file.write_text(
        "import time\n"
        "def test_first(): time.sleep(0.02)\n"
        "def test_second(): time.sleep(0.02)\n"
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    pytest_executable = bin_dir / "pytest"
    pytest_executable.write_text(
        f'#!/bin/sh\nexec {shlex.quote(sys.executable)} -m pytest "$@"\n'
    )
    pytest_executable.chmod(0o755)
    shell = (
        f'. "{SCRIPTS_DIR / "ci_otel.sh"}"\n'
        f"ci_otel_start 1 {_encoded('pytest tests')}\n"
        f"PYTHONPATH={shlex.quote(str(workspace))} "
        f"pytest -q {shlex.quote(str(test_file))}\n"
        "status=$?\n"
        'ci_otel_finish "$status"\n'
        'exit "$status"'
    )

    result = subprocess.run(
        ["/bin/sh", "-c", shell],
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
    records = [
        json.loads(line)
        for path in (tmp_path / "spans").glob("spans-*.jsonl")
        for line in path.read_text().splitlines()
    ]
    tests = [record for record in records if record["name"] == "pytest.test"]
    assert len(tests) == 2
    assert tests[0]["start_ns"] < tests[1]["start_ns"]
    assert tests[0]["end_ns"] <= tests[1]["start_ns"]


def test_pytest_plugin_failure_cannot_fail_pytest(tmp_path):
    test_file = tmp_path / "test_sample.py"
    test_file.write_text("def test_sample(): assert True\n")

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "ci_otel", str(test_file)],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(SCRIPTS_DIR),
            "CI_INFRA_TRACE_ID": "01" * 16,
            "CI_INFRA_COMMAND_SPAN_ID": "02" * 8,
            "CI_INFRA_OTEL_SPOOL_DIR": "/dev/null/spans",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "1 passed" in result.stdout
