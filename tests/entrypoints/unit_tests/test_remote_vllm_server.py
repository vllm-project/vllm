# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest

import tests.utils as test_utils
from tests.utils import RemoteLaunchRenderServer, RemoteOpenAIServer


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["--api-key", "key1", "key2"], ["--api-key", "***", "***"]),
        (["--api_key", "key1", "key2"], ["--api_key", "***", "***"]),
        (["--api-key=key1", "key2"], ["--api-key=***", "***"]),
        (["--api_key=key1", "key2"], ["--api_key=***", "***"]),
        (["--hf-token", "token"], ["--hf-token", "***"]),
        (["--hf_token", "token"], ["--hf_token", "***"]),
        (["--hf-token=token"], ["--hf-token=***"]),
        (["--hf_token=token"], ["--hf_token=***"]),
    ],
)
def test_redact_sensitive_cli_arg_variants(args: list[str], expected: list[str]):
    assert test_utils._redact_sensitive_cli_args(args) == expected


@pytest.mark.parametrize(
    ("server_cls", "command"),
    [
        (RemoteOpenAIServer, ["vllm", "serve"]),
        (RemoteLaunchRenderServer, ["vllm", "launch", "render"]),
    ],
)
def test_server_redacts_sensitive_values(
    server_cls,
    command: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    server = object.__new__(server_cls)
    popen = Mock()
    monkeypatch.setattr(test_utils.subprocess, "Popen", popen)
    monkeypatch.setenv("INHERITED_SECRET", "inherited-secret")
    serve_args = [
        "--api_key",
        "api-secret-1",
        "api-secret-2",
        "--hf-token=hf-secret",
        "--max-num-seqs",
        "2",
    ]

    server._start_server(
        "test-model", serve_args, {"OVERRIDE_SECRET": "override-secret"}
    )

    stdout = capsys.readouterr().out
    assert "api-secret-1" not in stdout
    assert "api-secret-2" not in stdout
    assert "hf-secret" not in stdout
    assert "inherited-secret" not in stdout
    assert "override-secret" not in stdout
    assert "--api_key *** ***" in stdout
    assert "--hf-token=***" in stdout
    assert "--max-num-seqs 2" in stdout
    assert popen.call_args.args[0] == [*command, "test-model", *serve_args]
    child_env = popen.call_args.kwargs["env"]
    assert child_env["INHERITED_SECRET"] == "inherited-secret"
    assert child_env["OVERRIDE_SECRET"] == "override-secret"


def test_openai_server_shutdown_wait_covers_engine_cleanup(
    monkeypatch: pytest.MonkeyPatch,
):
    server = object.__new__(RemoteOpenAIServer)
    server._request_shutdown_timeout = 0.0
    server.proc = Mock(pid=1234)
    engine_timeout_args = []

    def get_engine_timeout(request_timeout, process_timeout):
        engine_timeout_args.append((request_timeout, process_timeout))
        return 60.0

    monkeypatch.setattr(
        test_utils, "get_engine_process_shutdown_timeout", get_engine_timeout
    )
    monkeypatch.setattr(test_utils.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(server, "_kill_process_group_survivors", Mock())

    server._terminate_process_tree()

    assert engine_timeout_args == [(0.0, 0.0)]
    server.proc.wait.assert_called_once_with(timeout=75.0)
