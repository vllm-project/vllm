# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import pytest

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


def _quoted(value: str) -> str:
    return shlex.quote(value)


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


def test_otlp_payload_accepts_negative_integer_attributes():
    payload = ci_otel.encode_request([_span(**{"process.exit.code": -1})])

    assert b"process.exit.code" in payload


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


def test_buildx_ref_supports_bake_metadata(tmp_path):
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps({"test-ci": {"buildx.build.ref": "builder/node/record-id"}})
    )

    assert ci_otel.buildx_ref(str(metadata), "test-ci") == "builder/node/record-id"


def test_buildkit_trace_groups_stages_and_includes_descendant_time(monkeypatch):
    monkeypatch.setenv("BUILDKITE_BUILD_ID", "build-id")
    payload = {
        "data": [
            {
                "spans": [
                    {
                        "spanID": "01" * 8,
                        "operationName": "[base 1/2] FROM alpine:3.21",
                        "startTime": 100,
                        "duration": 10,
                        "tags": [{"key": "vertex", "value": "sha256:from"}],
                    },
                    {
                        "spanID": "02" * 8,
                        "operationName": "registry pull",
                        "references": [{"refType": "CHILD_OF", "spanID": "01" * 8}],
                        "startTime": 105,
                        "duration": 95,
                    },
                    {
                        "spanID": "03" * 8,
                        "operationName": "[base 2/2] RUN uv pip install vllm",
                        "startTime": 210,
                        "duration": 40,
                        "tags": [{"key": "vertex", "value": "sha256:run"}],
                    },
                    {
                        "spanID": "04" * 8,
                        "operationName": "[test 1/1] COPY --from=base /opt /opt",
                        "startTime": 260,
                        "duration": 20,
                        "tags": [{"key": "vertex", "value": "sha256:copy"}],
                    },
                    {
                        "spanID": "05" * 8,
                        "operationName": "[internal] load build context",
                        "startTime": 90,
                        "duration": 30,
                        "tags": [{"key": "vertex", "value": "sha256:context"}],
                    },
                    {
                        "spanID": "06" * 8,
                        "operationName": (
                            "cache request: [base 2/2] RUN uv pip install vllm"
                        ),
                        "startTime": 205,
                        "duration": 2,
                        "tags": [{"key": "vertex", "value": "sha256:run"}],
                    },
                ]
            }
        ]
    }

    spans = ci_otel.buildkit_spans(payload)
    stages = {
        span.attributes.get("docker.stage"): span
        for span in spans
        if span.name == "docker.stage"
    }
    instructions = [span for span in spans if span.name == "docker.instruction"]

    assert set(stages) == {"base", "test"}
    assert stages["base"].start_ns == 100_000
    assert stages["base"].end_ns == 250_000
    assert len(instructions) == 3
    assert len([span for span in spans if span.name == "docker.internal"]) == 1
    from_span = next(
        span for span in instructions if span.attributes["docker.instruction"] == "FROM"
    )
    assert from_span.end_ns == 200_000
    assert from_span.parent_span_id == stages["base"].span_id
    assert all(span.trace_id == stages["base"].trace_id for span in spans)


def test_record_buildkit_trace_spools_normalized_spans(monkeypatch, tmp_path):
    trace_file = tmp_path / "trace.json"
    trace_file.write_text(
        json.dumps(
            {
                "data": [
                    {
                        "spans": [
                            {
                                "spanID": "01" * 8,
                                "operationName": "[base 1/1] RUN true",
                                "startTime": 100,
                                "duration": 20,
                            }
                        ]
                    }
                ]
            }
        )
    )
    monkeypatch.setenv("CI_INFRA_OTEL_SPOOL_DIR", str(tmp_path / "spool"))

    assert ci_otel.record_buildkit_trace(str(trace_file)) is True
    assert {span.attributes["ci.span.kind"] for span in ci_otel.load_spans()} == {
        "docker-stage",
        "docker-instruction",
    }


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


def test_export_batches_with_one_oidc_token(monkeypatch):
    monkeypatch.setenv("BUILDKITE", "true")
    monkeypatch.setattr(ci_otel, "MAX_BATCH_SIZE", 2)
    token_calls = []
    requests = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def token(deadline):
        token_calls.append(deadline)
        return "token"

    def open_request(request, timeout):
        requests.append(request)
        return Response()

    monkeypatch.setattr(ci_otel, "_oidc_token", token)
    monkeypatch.setattr(ci_otel.urllib.request, "urlopen", open_request)

    assert ci_otel.export_spans([_span() for _ in range(5)]) is True
    assert len(token_calls) == 1
    assert len(requests) == 3


def test_successful_flush_removes_spool_and_does_not_duplicate(monkeypatch, tmp_path):
    monkeypatch.setenv("CI_INFRA_OTEL_SPOOL_DIR", str(tmp_path))
    uploaded = []
    assert ci_otel.record_spans([_span()]) is True
    monkeypatch.setattr(
        ci_otel, "export_spans", lambda spans, timeout: uploaded.extend(spans) or True
    )

    assert ci_otel.flush_spans() is True
    assert len(uploaded) == 1
    assert list(tmp_path.glob("spans-*.jsonl")) == []
    assert ci_otel.flush_spans() is True
    assert len(uploaded) == 1


def test_failed_flush_preserves_spool(monkeypatch, tmp_path):
    monkeypatch.setenv("CI_INFRA_OTEL_SPOOL_DIR", str(tmp_path))
    assert ci_otel.record_spans([_span()]) is True
    monkeypatch.setattr(ci_otel, "export_spans", lambda spans, timeout: False)

    assert ci_otel.flush_spans() is False
    assert len(list(tmp_path.glob("spans-*.jsonl"))) == 1


def test_cli_rejects_bad_arguments_without_traceback(capsys):
    assert ci_otel.main(["bogus"]) == 2
    assert ci_otel.main(["record-command", "too", "short"]) == 2
    assert (
        ci_otel.main(["record-command", "t", "s", "-", "not-an-int", "1", "0", "label"])
        == 1
    )

    stderr = capsys.readouterr().err
    assert "usage:" in stderr
    assert "CI timing helper failed:" in stderr
    assert "Traceback" not in stderr


def test_pytest_spans_are_spooled_incrementally(monkeypatch):
    recorded = []
    monkeypatch.setenv("CI_INFRA_TRACE_ID", "01" * 16)
    monkeypatch.setenv("CI_INFRA_COMMAND_SPAN_ID", "02" * 8)
    monkeypatch.setattr(ci_otel, "TEST_SPOOL_BATCH_SIZE", 1)
    monkeypatch.setattr(
        ci_otel, "record_spans", lambda spans: recorded.extend(spans) or True
    )
    ci_otel._test_runs.clear()
    ci_otel._test_spans.clear()
    ci_otel._test_runs["test_sample.py::test_one"] = (time.time_ns(), "passed")

    ci_otel.pytest_runtest_logfinish("test_sample.py::test_one", ("", 0, ""))

    assert len(recorded) == 1
    assert ci_otel._test_spans == []


def test_shell_wrapper_records_commands_without_changing_shell_state(tmp_path):
    first = f"ci_otel_start 1 {_quoted('export VALUE=ready')}"
    second = f"ci_otel_start 2 {_quoted('check VALUE')}"
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
        f"ci_otel_start 1 {_quoted('false')}; false; status=$?; "
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


def test_ci_otel_run_records_command_and_preserves_status(tmp_path):
    shell = (
        f'. "{SCRIPTS_DIR / "ci_otel.sh"}"; '
        f"ci_otel_run 1 {_quoted('true')} true; "
        f"ci_otel_run 2 {_quoted('false')} false; "
        'echo "status=$?"'
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
    assert "status=1" in result.stdout
    records = "".join(path.read_text() for path in tmp_path.glob("spans-*.jsonl"))
    assert '"ci.command.index":1' in records
    assert '"ci.command.index":2' in records
    assert '"process.exit.code":0' in records
    assert '"process.exit.code":1' in records


def test_ci_otel_run_fail_open_when_tracing_unavailable(tmp_path):
    shell = (
        f'CI_INFRA_OTEL_DIR="{tmp_path / "missing"}"; export CI_INFRA_OTEL_DIR; '
        f'. "{SCRIPTS_DIR / "ci_otel.sh"}"; '
        f"ci_otel_run 1 {_quoted('echo ran')} echo ran"
    )

    result = subprocess.run(
        ["/bin/sh", "-e", "-c", shell],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ran"


def test_ci_otel_run_handles_assignment_prefixed_commands(tmp_path):
    shell = (
        f'. "{SCRIPTS_DIR / "ci_otel.sh"}"; '
        f"ci_otel_run 1 {_quoted('MY_VAR=hello env')} "
        "MY_VAR=hello env"
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
    assert "MY_VAR=hello" in result.stdout


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


def test_unset_helper_dir_does_not_block_the_test_command():
    shell = f'unset CI_INFRA_OTEL_DIR; . "{SCRIPTS_DIR / "ci_otel.sh"}"; echo ran'

    result = subprocess.run(
        ["/bin/bash", "-e", "-c", shell],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ran"


def test_helper_dir_assignment_is_exported():
    shell = (
        "unset CI_INFRA_OTEL_DIR; "
        f'CI_INFRA_OTEL_DIR="{SCRIPTS_DIR}" . "{SCRIPTS_DIR / "ci_otel.sh"}"; '
        f'/bin/sh -c \'test "$CI_INFRA_OTEL_DIR" = "{SCRIPTS_DIR}"\''
    )

    result = subprocess.run(
        ["/bin/sh", "-c", shell], check=False, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("failure", ["new-context", "record-command"])
def test_helper_failure_cannot_change_job_status(tmp_path, failure):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python3"
    fake_python.write_text(
        "#!/bin/sh\n"
        '[ "$1" = "-c" ] && exit 0\n'
        '[ "$2" = "new-context" ] && {\n'
        '  [ "$CI_OTEL_TEST_FAILURE" = "new-context" ] && exit 99\n'
        '  echo "01010101010101010101010101010101 0202020202020202 - 1"\n'
        "  exit 0\n"
        "}\n"
        '[ "$2" = "record-command" ] && exit 99\n'
        "exit 99\n"
    )
    fake_python.chmod(0o755)
    shell = (
        f'PATH="{fake_bin}:$PATH"; . "{SCRIPTS_DIR / "ci_otel.sh"}"; '
        f"ci_otel_start 1 {_quoted('true')}; ci_otel_finish 0; echo ran"
    )

    result = subprocess.run(
        ["/bin/sh", "-e", "-c", shell],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "CI_INFRA_OTEL_DIR": str(SCRIPTS_DIR),
            "CI_INFRA_OTEL_RUNTIME_DIR": str(tmp_path / "runtime"),
            "CI_OTEL_TEST_FAILURE": failure,
        },
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ran"


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


def test_shell_wrapper_uses_private_trace_ids(tmp_path):
    shell = (
        f'. "{SCRIPTS_DIR / "ci_otel.sh"}"; '
        f"ci_otel_start 1 {_quoted('true')}; "
        'expected="$CI_INFRA_TRACE_ID"; '
        "CI_INFRA_TRACE_ID=corrupted; CI_INFRA_COMMAND_SPAN_ID=corrupted; "
        'ci_otel_finish 0; printf "%s" "$expected"'
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
    record = json.loads(next(tmp_path.glob("spans-*.jsonl")).read_text())
    assert record["trace_id"] == result.stdout
    assert record["span_id"] != "corrupted"


def test_shell_setup_is_idempotent(tmp_path):
    runtime = tmp_path / "runtime"
    shell = (
        f'. "{SCRIPTS_DIR / "ci_otel.sh"}"; '
        f'. "{SCRIPTS_DIR / "ci_otel.sh"}"; '
        'printf "%s\n%s" "$PATH" "$CI_INFRA_OTEL_SHIM_PATHS"'
    )

    result = subprocess.run(
        ["/bin/sh", "-c", shell],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "CI_INFRA_OTEL_DIR": str(SCRIPTS_DIR),
            "CI_INFRA_OTEL_RUNTIME_DIR": str(runtime),
        },
    )

    assert result.returncode == 0, result.stderr
    path, shim_paths = result.stdout.splitlines()
    assert path.split(":").count(str(runtime / "bin")) == 1
    assert shim_paths.split(":").count(str(runtime / "bin")) == 1


def test_helper_owned_runtime_is_removed_on_exit(tmp_path):
    runtime_file = tmp_path / "runtime"
    shell = (
        f'. "{SCRIPTS_DIR / "ci_otel.sh"}"; '
        f'printf "%s" "$CI_INFRA_OTEL_RUNTIME_DIR" > "{runtime_file}"'
    )

    result = subprocess.run(
        ["/bin/sh", "-c", shell],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "CI_INFRA_OTEL_DIR": str(SCRIPTS_DIR)},
    )

    assert result.returncode == 0, result.stderr
    assert not Path(runtime_file.read_text()).exists()


def test_caller_owned_runtime_is_preserved(tmp_path):
    runtime = tmp_path / "runtime"
    shell = f'. "{SCRIPTS_DIR / "ci_otel.sh"}"'

    result = subprocess.run(
        ["/bin/sh", "-c", shell],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "CI_INFRA_OTEL_DIR": str(SCRIPTS_DIR),
            "CI_INFRA_OTEL_RUNTIME_DIR": str(runtime),
        },
    )

    assert result.returncode == 0, result.stderr
    assert runtime.exists()


def test_pytest_shim_recovers_from_stale_recorded_path(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    real_pytest = bin_dir / "pytest"
    real_pytest.write_text("#!/bin/sh\nprintf 'CURRENT-PYTEST'")
    real_pytest.chmod(0o755)

    result = subprocess.run(
        [str(SCRIPTS_DIR / "ci_pytest.sh"), "--version"],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": str(bin_dir),
            "CI_INFRA_OTEL_REAL_PYTEST": "/missing/pytest",
        },
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "CURRENT-PYTEST"


def test_path_change_after_source_uses_new_pytest(tmp_path):
    old_bin = tmp_path / "old"
    new_bin = tmp_path / "new"
    runtime = tmp_path / "runtime"
    old_bin.mkdir()
    new_bin.mkdir()
    for directory, output in ((old_bin, "OLD"), (new_bin, "NEW")):
        executable = directory / "pytest"
        executable.write_text(f"#!/bin/sh\nprintf '{output}'")
        executable.chmod(0o755)
    shell = (
        f'PATH="{old_bin}:$PATH"; . "{SCRIPTS_DIR / "ci_otel.sh"}"; '
        f'PATH="{new_bin}:$PATH"; hash -r; pytest'
    )

    result = subprocess.run(
        ["/bin/sh", "-c", shell],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "CI_INFRA_OTEL_DIR": str(SCRIPTS_DIR),
            "CI_INFRA_OTEL_RUNTIME_DIR": str(runtime),
        },
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "NEW"


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
        f"ci_otel_start 1 {_quoted('pytest tests')}\n"
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
