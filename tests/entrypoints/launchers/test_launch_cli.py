# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for CLI launch and runtime dispatch."""

import argparse
import subprocess
import sys
from unittest.mock import patch

import pytest

from vllm.entrypoints.cli.launch import (
    LaunchSubcommand,
    RenderSubcommand,
    cmd_init,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser


@pytest.fixture
def launch_parser():
    parser = FlexibleArgumentParser(description="test")
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    LaunchSubcommand().subparser_init(subparsers)
    return parser


def test_subcommand_name():
    assert LaunchSubcommand().name == "launch"


def test_cmd_init_returns_subcommand():
    result = cmd_init()
    assert len(result) == 1
    assert isinstance(result[0], LaunchSubcommand)


# -- Parsing: `vllm launch render` --


def test_parse_launch_render(launch_parser):
    args = launch_parser.parse_args(["launch", "render", "--model", "test-model"])
    assert args.launch_component == "render"


def test_parse_launch_requires_component(launch_parser):
    with pytest.raises(SystemExit):
        launch_parser.parse_args(["launch", "--model", "test-model"])


def test_parse_launch_invalid_component(launch_parser):
    with pytest.raises(SystemExit):
        launch_parser.parse_args(["launch", "unknown", "--model", "test-model"])


# -- Dispatch --


def test_cmd_launch_render_calls_run():
    args = argparse.Namespace(model_tag=None, model="test-model")
    with patch("vllm.entrypoints.cli.launch.uvloop.run") as mock_uvloop_run:
        RenderSubcommand.cmd(args)
        mock_uvloop_run.assert_called_once()


def test_cmd_launch_model_tag_overrides():
    args = argparse.Namespace(
        model_tag="tag-model",
        model="original-model",
        launch_command=lambda a: None,
    )
    LaunchSubcommand.cmd(args)
    assert args.model == "tag-model"


def test_cmd_launch_model_tag_none():
    args = argparse.Namespace(
        model_tag=None,
        model="original-model",
        launch_command=lambda a: None,
    )
    LaunchSubcommand.cmd(args)
    assert args.model == "original-model"


def test_cmd_dispatches():
    called = {}

    def fake_dispatch(args):
        called["args"] = args

    args = argparse.Namespace(launch_command=fake_dispatch)
    LaunchSubcommand.cmd(args)
    assert "args" in called


# -- Module registration --


def test_subparser_init_returns_parser():
    parser = FlexibleArgumentParser(description="test")
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    result = LaunchSubcommand().subparser_init(subparsers)
    assert isinstance(result, FlexibleArgumentParser)


def test_launch_registered_in_main():
    """Verify that launch module is importable as a CLI module."""
    import vllm.entrypoints.cli.launch as launch_module

    assert hasattr(launch_module, "cmd_init")
    subcmds = launch_module.cmd_init()
    assert any(s.name == "launch" for s in subcmds)


def _run_python(*args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run([sys.executable, *args], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result


def test_serve_parser_uses_explicit_args_not_host_sys_argv():
    from vllm.entrypoints.cli.serve import ServeSubcommand

    with patch.object(sys, "argv", ["foreign-host", "serve", "--help"]):
        parser = FlexibleArgumentParser()
        subparsers = parser.add_subparsers(dest="subparser", required=True)
        selected_command = ServeSubcommand()
        selected_command.subparser_init(subparsers)
        args = parser.parse_args(["serve", "--host", "127.0.0.1"])

    assert selected_command.name == "serve"
    assert args.host == "127.0.0.1"


@pytest.mark.parametrize(
    "runtime_import",
    [
        pytest.param("import vllm.compilation.compiler_interface", id="compilation"),
        pytest.param(
            "import vllm.entrypoints.launchers.api_server.app_state",
            id="api-server-state",
        ),
        pytest.param("import vllm.v1.worker.gpu_model_runner", id="v1"),
        pytest.param("import vllm.v1.worker.gpu.model_runner", id="v2"),
        pytest.param("import vllm\nvllm.RequestOutput", id="lazy-export"),
    ],
)
def test_runtime_imports_preserve_environment_overrides(runtime_import):
    script = f"""
import os

{runtime_import}
from torch._inductor.lowering import FALLBACK_ALLOW_LIST

assert "vllm::runtime_override_probe" in FALLBACK_ALLOW_LIST
assert os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] == "1"
assert os.environ["TORCHINDUCTOR_COMPILE_THREADS"] == "1"
"""
    _run_python("-c", script)
