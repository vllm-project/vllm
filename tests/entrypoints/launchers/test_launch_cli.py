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


def test_top_level_help_does_not_import_runtime_modules():
    script = """
import sys

sys.argv = ["vllm", "--help"]
from vllm.entrypoints.cli.main import main

exit_code = None
try:
    main()
except SystemExit as exc:
    exit_code = exc.code

assert exit_code == 0
blocked_prefixes = (
    "torch",
    "uvloop",
    "vllm.env_override",
    "vllm.entrypoints.openai.api_server",
    "vllm.v1.executor",
    "vllm.v1.metrics",
)
loaded = sorted(
    prefix
    for prefix in blocked_prefixes
    if any(
        name == prefix or name.startswith(f"{prefix}.") for name in sys.modules
    )
)
assert not loaded, loaded
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_serve_help_is_compact_and_import_light():
    script = """
import os
import sys

os.environ.pop("VLLM_WORKER_MULTIPROC_METHOD", None)
sys.argv = ["vllm", "serve", "--help"]
from vllm.entrypoints.cli.main import main

exit_code = None
try:
    main()
except SystemExit as exc:
    exit_code = exc.code

assert exit_code == 0
assert "VLLM_WORKER_MULTIPROC_METHOD" not in os.environ
blocked_prefixes = (
    "torch",
    "uvloop",
    "vllm.env_override",
    "vllm.entrypoints.cli.serve",
    "vllm.entrypoints.launchers.api_server",
    "vllm.entrypoints.openai.api_server",
    "vllm.entrypoints.openai.cli_args",
    "vllm.entrypoints.serve.utils.api_utils",
    "vllm.v1.executor",
    "vllm.v1.metrics",
)
loaded = sorted(
    prefix
    for prefix in blocked_prefixes
    if any(
        name == prefix or name.startswith(f"{prefix}.") for name in sys.modules
    )
)
assert not loaded, loaded
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    for argument in (
        "model_tag",
        "--headless",
        "--api-server-count",
        "-asc",
        "--config",
        "--grpc",
    ):
        assert argument in result.stdout
    for non_core_argument in ("--host", "--port", "--max-model-len"):
        assert non_core_argument not in result.stdout
    assert "Config Groups:" not in result.stdout
    assert "vllm serve --help=all" in result.stdout


def test_serve_help_all_uses_canonical_parser():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            "--help=all",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    for argument in ("model_tag", "--grpc", "--host", "--max-model-len"):
        assert argument in result.stdout


@pytest.mark.parametrize(
    ("argv", "module_name"),
    [
        pytest.param(
            ["serve", "--help=all"],
            "vllm.entrypoints.cli.serve",
            id="serve",
        ),
        pytest.param(
            ["bench", "--help"],
            "vllm.entrypoints.cli.benchmark.main",
            id="bench",
        ),
    ],
)
def test_runtime_cli_sets_environment_before_loading_selected_command(
    argv, module_name
):
    script = f"""
import importlib.abc
import os
import sys

os.environ.pop("VLLM_WORKER_MULTIPROC_METHOD", None)

class RuntimeImportOrderGuard(importlib.abc.MetaPathFinder):
    torch_seen = False
    selected_seen = False

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch":
            self.torch_seen = True
            assert os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") == "spawn"
        if fullname == {module_name!r}:
            self.selected_seen = True
            assert os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") == "spawn"
            assert self.torch_seen
            assert "vllm.env_override" in sys.modules
            assert "vllm.entrypoints.serve.utils.api_utils" not in sys.modules
        return None

guard = RuntimeImportOrderGuard()
sys.meta_path.insert(0, guard)
sys.argv = ["vllm", *{argv!r}]

from vllm.entrypoints.cli.main import main

try:
    main()
except SystemExit as exc:
    assert exc.code == 0

assert guard.torch_seen
assert guard.selected_seen
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


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


def test_public_lazy_exports_apply_runtime_environment_once():
    script = """
import sys

import vllm

assert "torch" not in sys.modules
vllm.RequestOutput
override_module = sys.modules["vllm.env_override"]
vllm.SamplingParams

from torch._inductor.lowering import FALLBACK_ALLOW_LIST

assert sys.modules["vllm.env_override"] is override_module
assert "vllm::runtime_override_probe" in FALLBACK_ALLOW_LIST
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "runtime_import",
    [
        pytest.param("import vllm.compilation.compiler_interface", id="compilation"),
        pytest.param("import vllm.config", id="config"),
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
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
