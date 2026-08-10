# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for public vLLM CLI parser and dispatch paths."""

import argparse
import sys
from unittest.mock import patch

import pytest

from vllm.entrypoints.cli import main as cli_main
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


def test_prepare_model_info_registered_in_main(capsys):
    with (
        patch.object(sys, "argv", ["vllm", "prepare-model-info", "Qwen3ForCausalLM"]),
        patch("vllm.entrypoints.cli.benchmark.main.maybe_exec_rust_bench"),
        patch(
            "vllm.model_executor.models.ModelRegistry.prepare_model_info"
        ) as prepare_model_info,
    ):
        cli_main.main()

    prepare_model_info.assert_called_once_with("Qwen3ForCausalLM")
    assert capsys.readouterr().out == "prepared model info for Qwen3ForCausalLM\n"
