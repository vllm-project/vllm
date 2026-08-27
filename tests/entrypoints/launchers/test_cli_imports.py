# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import importlib
import subprocess
import sys

import pytest

from vllm.entrypoints.cli._utils import is_serve_help


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        pytest.param(["serve", "--help"], True, id="plain-help"),
        pytest.param(["serve", "--help=all"], True, id="full-help"),
        pytest.param(["--help", "serve"], True, id="global-help"),
        pytest.param(["serve", "--helpful"], False, id="help-prefix"),
        pytest.param(["serve", "--", "--help=all"], False, id="end-options"),
        pytest.param(["collect-env", "--help"], False, id="other-command"),
    ],
)
def test_is_serve_help(argv: list[str], expected: bool):
    assert is_serve_help(argv) is expected


def test_bench_delegates_before_parser_build(monkeypatch):
    cli_main = importlib.import_module("vllm.entrypoints.cli.main")
    events = []

    class BenchmarkDelegated(Exception):
        pass

    def maybe_exec_rust_bench():
        events.append("bench")
        raise BenchmarkDelegated

    def import_module(name):
        return argparse.Namespace(maybe_exec_rust_bench=maybe_exec_rust_bench)

    def fail_if_parser_built(name):
        raise AssertionError("parser built")

    monkeypatch.setattr(sys, "argv", ["vllm", "bench"])
    monkeypatch.setattr(cli_main, "cli_env_setup", lambda: events.append("cli-env"))
    monkeypatch.setattr(
        cli_main.importlib,
        "import_module",
        import_module,
    )
    monkeypatch.setattr(cli_main, "_build_parser", fail_if_parser_built)

    with pytest.raises(BenchmarkDelegated):
        cli_main.main()

    assert events == ["cli-env", "bench"]


@pytest.mark.parametrize(
    ("help_arg", "expected_output"),
    [
        pytest.param("--help", "Config Groups:", id="grouped"),
        pytest.param("--help=all", "--max-model-len", id="full"),
        pytest.param("--help=ModelConfig", "--max-model-len", id="section"),
    ],
)
def test_serve_help_uses_the_canonical_parser_without_torch(
    help_arg: str, expected_output: str
):
    script = f"""
import importlib.abc
import os
import sys

class RejectTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise AssertionError(f"unexpected import: {{fullname}}")
        if fullname in {{
            "vllm.platforms.cpu",
            "vllm.platforms.cuda",
            "vllm.platforms.rocm",
            "vllm.platforms.tpu",
            "vllm.platforms.xpu",
            "vllm.platforms.zen_cpu",
        }}:
            raise AssertionError(f"unexpected platform probe: {{fullname}}")
        return None

sys.meta_path.insert(0, RejectTorch())
os.environ.pop("VLLM_WORKER_MULTIPROC_METHOD", None)
sys.argv = ["vllm", "serve", {help_arg!r}]

from vllm.entrypoints.cli.main import main

try:
    main()
except SystemExit as exc:
    assert exc.code == 0

assert "VLLM_WORKER_MULTIPROC_METHOD" not in os.environ
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
    assert expected_output in result.stdout
