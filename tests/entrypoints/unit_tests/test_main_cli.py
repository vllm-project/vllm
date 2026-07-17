# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest

import vllm.entrypoints.cli.main as cli_main


def test_bench_serve_execs_rust_cli(monkeypatch: pytest.MonkeyPatch, tmp_path):
    argv = [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "openai-chat",
        "--request-rate",
        "inf",
    ]
    rust_cli = tmp_path / "vllm-rs"
    rust_cli.touch()
    execv = Mock()
    log_info = Mock()
    monkeypatch.setattr(cli_main.sys, "argv", argv)
    monkeypatch.setattr(cli_main, "_RUST_CLI_PATH", rust_cli)
    monkeypatch.setattr(cli_main.os, "execv", execv)
    monkeypatch.setattr(cli_main.logger, "info", log_info)

    cli_main._maybe_exec_rust_bench()

    rust_cli_str = str(rust_cli)
    log_info.assert_called_once_with(
        "Delegating `vllm bench serve` to Rust binary at %s.", rust_cli_str
    )
    execv.assert_called_once_with(rust_cli_str, [rust_cli_str, *argv[1:]])


def test_bench_serve_falls_back_when_rust_cli_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    rust_cli = tmp_path / "vllm-rs"
    execv = Mock()
    log_warning = Mock()
    monkeypatch.setattr(cli_main.sys, "argv", ["vllm", "bench", "serve"])
    monkeypatch.setattr(cli_main, "_RUST_CLI_PATH", rust_cli)
    monkeypatch.setattr(cli_main.os, "execv", execv)
    monkeypatch.setattr(cli_main.logger, "warning", log_warning)

    cli_main._maybe_exec_rust_bench()

    log_warning.assert_called_once_with(
        "Rust benchmark binary not found at %s; falling back to Python.", rust_cli
    )
    execv.assert_not_called()


@pytest.mark.parametrize(
    "argv",
    [
        ["vllm", "bench", "latency"],
        ["vllm", "bench", "serve", "--omni"],
    ],
)
def test_other_bench_routes_do_not_exec_rust_cli(
    monkeypatch: pytest.MonkeyPatch, argv: list[str]
):
    execv = Mock()
    monkeypatch.setattr(cli_main.sys, "argv", argv)
    monkeypatch.setattr(cli_main.os, "execv", execv)

    cli_main._maybe_exec_rust_bench()

    execv.assert_not_called()
