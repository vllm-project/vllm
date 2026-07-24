# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import os
import shlex
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
from huggingface_hub import constants as hf_constants
from huggingface_hub.utils import _http as hf_http
from vllm_test_utils.hf_hub import (
    HF_HUB_RETRY_EXIT_STATUS,
    HFHubRequestMonitor,
    HFHubTestGroup,
)

ROOT = Path(__file__).parents[2]
TEST_UTILS = ROOT / "tests/vllm_test_utils"
AMD_TEST_RUNNER = ROOT / ".buildkite/scripts/hardware_ci/run-amd-test.sh"


class _StatusHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("location", "/200")
            self.end_headers()
            return
        status = int(self.path.removeprefix("/").partition("?")[0])
        body = str(status).encode()
        self.send_response(status)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture(scope="module")
def http_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StatusHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host = server.server_address[0]
    port = server.server_address[1]
    if isinstance(host, bytes):
        host = host.decode()
    yield f"http://{host}:{port}"
    server.shutdown()
    thread.join()


@pytest.fixture(autouse=True)
def restore_hf_clients():
    sync_factory = hf_http._GLOBAL_CLIENT_FACTORY
    async_factory = hf_http._GLOBAL_ASYNC_CLIENT_FACTORY
    offline = hf_constants.HF_HUB_OFFLINE
    yield
    hf_http.set_client_factory(sync_factory)
    hf_http.set_async_client_factory(async_factory)
    hf_constants.HF_HUB_OFFLINE = offline


def _plugin_environment(*, offline: bool) -> dict[str, str]:
    environment = os.environ.copy()
    pythonpath = [str(TEST_UTILS)]
    if environment.get("PYTHONPATH"):
        pythonpath.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(pythonpath)
    value = "1" if offline else "0"
    environment.update(
        {
            "HF_HUB_OFFLINE": value,
            "HF_DATASETS_OFFLINE": value,
            "TRANSFORMERS_OFFLINE": value,
        }
    )
    environment.pop("PYTEST_ADDOPTS", None)
    return environment


def _run_plugin_pytest(
    test_dir: Path,
    request_log: Path,
    *args: str,
    offline: bool = False,
) -> subprocess.CompletedProcess[str]:
    request_log.touch()
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "vllm_test_utils.hf_hub",
            f"--hf-request-log={request_log}",
            "-q",
            *args,
        ],
        cwd=test_dir,
        env=_plugin_environment(offline=offline),
        check=False,
        text=True,
        capture_output=True,
    )


def test_listener_counts_successful_and_failed_sync_and_async_requests():
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/timeout":
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(200, request=request)

    hf_http.set_client_factory(
        lambda: httpx.Client(transport=httpx.MockTransport(respond))
    )
    hf_http.set_async_client_factory(
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(respond))
    )
    monitor = HFHubRequestMonitor()
    monitor.install()

    assert hf_http.get_session().get("https://huggingface.co/success").is_success
    with pytest.raises(httpx.ReadTimeout):
        hf_http.get_session().get("https://huggingface.co/timeout")

    async def send_async_request() -> httpx.Response:
        async with hf_http.get_async_session() as client:
            return await client.get("https://huggingface.co/async")

    assert asyncio.run(send_async_request()).is_success
    assert monitor.counts == {"<session>": 3}

    monitor.uninstall()
    assert not hf_http.get_session().event_hooks["request"]


def test_group_aggregates_invocations_and_prints_only_the_table(
    monkeypatch, tmp_path, capfd, http_server
):
    (tmp_path / "conftest.py").write_text(
        f"""
from huggingface_hub.utils import _http

_http.get_session().get("{http_server}/200?token=hf_secret")
"""
    )
    (tmp_path / "test_a.py").write_text(
        f"""
from huggingface_hub.utils import _http

def test_a():
    assert _http.get_session().get("{http_server}/200").is_success
    assert _http.get_session().get("{http_server}/503").status_code == 503
"""
    )
    (tmp_path / "test_b.py").write_text(
        f"""
from huggingface_hub.utils import _http

def test_b():
    assert _http.get_session().get("{http_server}/400").status_code == 400
"""
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(TEST_UTILS))
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    pytest_command = shlex.join([sys.executable, "-m", "pytest", "-q"])

    status = HFHubTestGroup(
        f"{pytest_command} test_a.py && {pytest_command} test_b.py"
    ).run()

    output = capfd.readouterr().out
    assert status == 0
    assert output.count("Hugging Face Hub HTTP requests") == 1
    assert "       2  <session>" in output
    assert "       2  test_a.py::test_a" in output
    assert "       1  test_b.py::test_b" in output
    assert "       5  TOTAL" in output
    assert "hf_secret" not in output


def test_offline_request_attempt_is_counted(monkeypatch, tmp_path, capfd):
    (tmp_path / "test_offline.py").write_text(
        """
from huggingface_hub.errors import OfflineModeIsEnabled
from huggingface_hub.utils import _http

def test_offline():
    with __import__("pytest").raises(OfflineModeIsEnabled):
        _http.get_session().get("https://huggingface.co/api/models/test")
"""
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(TEST_UTILS))
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    command = shlex.join([sys.executable, "-m", "pytest", "-q", "test_offline.py"])

    status = HFHubTestGroup(command, offline=True).run()

    output = capfd.readouterr().out
    assert status == 0
    assert "       1  test_offline.py::test_offline" in output
    assert "       1  TOTAL" in output


@pytest.mark.parametrize(
    ("status", "expected_status"),
    [
        (400, 1),
        (404, 1),
        (408, HF_HUB_RETRY_EXIT_STATUS),
        (429, HF_HUB_RETRY_EXIT_STATUS),
        (500, HF_HUB_RETRY_EXIT_STATUS),
        (501, 1),
        (502, HF_HUB_RETRY_EXIT_STATUS),
        (503, HF_HUB_RETRY_EXIT_STATUS),
        (504, HF_HUB_RETRY_EXIT_STATUS),
    ],
)
def test_online_retry_statuses_are_strict(
    tmp_path, http_server, status, expected_status
):
    (tmp_path / "test_failure.py").write_text(
        f"""
from huggingface_hub.utils import _http

def test_failure():
    _http.get_session().get("{http_server}/{status}").raise_for_status()
"""
    )
    request_log = tmp_path / "requests.log"

    result = _run_plugin_pytest(tmp_path, request_log, "test_failure.py")

    assert result.returncode == expected_status, result
    assert request_log.read_text() == "1\ttest_failure.py::test_failure\n"


@pytest.mark.parametrize(
    ("client", "url_kind", "expected_status"),
    [
        ("hub", "connection", HF_HUB_RETRY_EXIT_STATUS),
        ("hub", "unsupported", 1),
        ("httpx", "connection", 1),
    ],
)
def test_online_retry_requires_a_transient_hub_request(
    tmp_path, client, url_kind, expected_status
):
    with socket.socket() as unused_socket:
        unused_socket.bind(("127.0.0.1", 0))
        _, unused_port = unused_socket.getsockname()
    url = (
        f"http://127.0.0.1:{unused_port}"
        if url_kind == "connection"
        else "ftp://huggingface.co/model"
    )
    (tmp_path / "test_failure.py").write_text(
        f"""
import httpx
from huggingface_hub.utils import _http

def test_failure():
    client = _http.get_session() if "{client}" == "hub" else httpx.Client()
    client.get("{url}")
"""
    )
    request_log = tmp_path / "requests.log"

    result = _run_plugin_pytest(tmp_path, request_log, "test_failure.py")

    assert result.returncode == expected_status, result
    expected_count = 1 if client == "hub" else 0
    assert request_log.read_text() == (
        f"{expected_count}\ttest_failure.py::test_failure\n" if expected_count else ""
    )


@pytest.mark.parametrize(
    ("failure", "expected_status"),
    [
        ("offline", HF_HUB_RETRY_EXIT_STATUS),
        ("cache-miss", HF_HUB_RETRY_EXIT_STATUS),
        ("collection-cache-miss", HF_HUB_RETRY_EXIT_STATUS),
        ("initial-cache-miss", HF_HUB_RETRY_EXIT_STATUS),
        ("unrelated", 1),
    ],
)
def test_offline_retry_is_limited_to_hub_cache_failures(
    monkeypatch, tmp_path, failure, expected_status
):
    statements = {
        "offline": (
            "from huggingface_hub.utils import _http\n"
            "    _http.get_session().get('https://huggingface.co/model')"
        ),
        "cache-miss": (
            "from huggingface_hub.errors import LocalEntryNotFoundError\n"
            "    raise LocalEntryNotFoundError('not cached')"
        ),
        "unrelated": "raise AssertionError('unrelated failure')",
    }
    test_path = tmp_path / "test_failure.py"
    if failure == "collection-cache-miss":
        test_path.write_text(
            "from huggingface_hub.errors import LocalEntryNotFoundError\n"
            "raise LocalEntryNotFoundError('not cached')\n"
        )
    elif failure == "initial-cache-miss":
        (tmp_path / "conftest.py").write_text(
            "from huggingface_hub.errors import LocalEntryNotFoundError\n"
            "raise LocalEntryNotFoundError('not cached')\n"
        )
        test_path.touch()
    else:
        test_path.write_text(f"def test_failure():\n    {statements[failure]}\n")
    if failure == "initial-cache-miss":
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("PYTHONPATH", str(TEST_UTILS))
        monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
        command = shlex.join([sys.executable, "-m", "pytest", "-q", test_path.name])
        assert HFHubTestGroup(command, offline=True).run() == expected_status
        return
    request_log = tmp_path / "requests.log"

    result = _run_plugin_pytest(tmp_path, request_log, "test_failure.py", offline=True)

    assert result.returncode == expected_status, result


def test_forked_request_keeps_count_and_retry_attribution(tmp_path, http_server):
    (tmp_path / "conftest.py").write_text(
        f"""
from huggingface_hub.utils import _http

_http.get_session().get("{http_server}/200")
"""
    )
    (tmp_path / "test_forked.py").write_text(
        f"""
import pytest
from huggingface_hub.utils import _http

@pytest.mark.forked
def test_failure():
    _http.get_session().get("{http_server}/503").raise_for_status()
"""
    )
    request_log = tmp_path / "requests.log"

    result = _run_plugin_pytest(tmp_path, request_log, "test_forked.py")

    assert result.returncode == HF_HUB_RETRY_EXIT_STATUS, result
    assert request_log.read_text() == (
        "1\ttest_forked.py::test_failure\n1\t<session>\n"
    )


@pytest.mark.parametrize(("offline", "expected_offline"), [(True, "1"), (False, "0")])
def test_group_preserves_quoting_and_unrelated_status(
    monkeypatch, tmp_path, capfd, offline, expected_offline
):
    offline_output = tmp_path / "offline"
    quoted_output = tmp_path / "quoted"
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    command = (
        f"printf '%s' \"$HF_HUB_OFFLINE\" > {shlex.quote(str(offline_output))}; "
        f"printf '%s' 'quoted value' > {shlex.quote(str(quoted_output))}; "
        "exit 42"
    )

    status = HFHubTestGroup(command, offline=offline).run()

    assert status == 42
    assert offline_output.read_text() == expected_offline
    assert quoted_output.read_text() == "quoted value"
    assert capfd.readouterr().out.count("Hugging Face Hub HTTP requests") == 1


@pytest.mark.parametrize(("pipefail", "expected_status"), [(False, 0), (True, 1)])
def test_group_preserves_callers_pipeline_behavior(pipefail, expected_status):
    assert HFHubTestGroup("false | true", pipefail=pipefail).run() == expected_status


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents)
    path.chmod(0o755)


def _run_stubbed_amd_docker(
    tmp_path: Path, *, mode: str | None, retry_count: str = "0"
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker_args = tmp_path / "docker-args"
    _write_executable(
        bin_dir / "docker",
        """#!/bin/bash
if [[ "$1" == "run" ]]; then
  shift
  printf '%s\\0' "$@" > "$DOCKER_ARGS"
fi
""",
    )
    _write_executable(bin_dir / "rocminfo", "#!/bin/bash\nexit 0\n")
    _write_executable(
        bin_dir / "getent",
        """#!/bin/bash
if [[ "$1" == "group" && "$2" == "render" ]]; then
  echo 'render:x:123:'
  exit 0
fi
exit 1
""",
    )
    environment = {
        **os.environ,
        "BUILDKITE_COMMIT": "test",
        "DOCKER_ARGS": str(docker_args),
        "HOME": str(tmp_path),
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "ROCM_DOCKER_TTY": "0",
        "BUILDKITE_RETRY_COUNT": retry_count,
        "VLLM_CI_USE_ARTIFACTS": "0",
        "VLLM_TEST_COMMANDS": "false | true",
    }
    if mode is None:
        environment.pop("VLLM_CI_HF_HUB_MODE", None)
    else:
        environment["VLLM_CI_HF_HUB_MODE"] = mode

    result = subprocess.run(
        ["bash", str(AMD_TEST_RUNNER)],
        cwd=ROOT,
        env=environment,
        check=False,
        text=True,
        capture_output=True,
    )
    arguments = docker_args.read_bytes().rstrip(b"\0").decode().split("\0")
    return result, arguments


@pytest.mark.parametrize("mode", [None, ""])
def test_disabled_amd_runner_preserves_legacy_docker_command(tmp_path, mode):
    result, arguments = _run_stubbed_amd_docker(tmp_path, mode=mode)

    assert result.returncode == 0, result
    assert "PYTHONPATH=/vllm-workspace" in arguments
    assert arguments[-3] == "/bin/bash"
    assert arguments[-2] == "-c"
    assert arguments[-1].endswith(" && false | true")
    assert "Hugging Face Hub HTTP requests" not in result.stdout


@pytest.mark.parametrize(
    ("mode", "retry_count", "offline"),
    [
        ("offline-first", "0", True),
        ("offline-first", "1", False),
        ("online", "0", False),
    ],
)
def test_enabled_amd_docker_runner_wraps_only_the_test_command(
    tmp_path, mode, retry_count, offline
):
    result, arguments = _run_stubbed_amd_docker(
        tmp_path, mode=mode, retry_count=retry_count
    )

    assert result.returncode == 0, result
    assert (
        "PYTHONPATH=/vllm-workspace/tests/vllm_test_utils:/vllm-workspace" in arguments
    )
    expected_runner = ["python3", "-m", "vllm_test_utils.hf_hub"]
    if offline:
        expected_runner.append("--offline")
    assert arguments[-len(expected_runner) - 2 : -2] == expected_runner
    assert arguments[-2] == "--"
    assert arguments[-1].endswith(" && false | true")


def test_amd_runner_rejects_invalid_enabled_mode():
    environment = {**os.environ, "VLLM_CI_HF_HUB_MODE": "invalid"}

    result = subprocess.run(
        ["bash", str(AMD_TEST_RUNNER)],
        cwd=ROOT,
        env=environment,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "Invalid VLLM_CI_HF_HUB_MODE: invalid" in result.stderr
