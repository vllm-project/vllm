# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the `vllm launch` CLI subcommand."""

import argparse
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


# -- Parsing: `--server` argument --


def test_parse_render_server_default(launch_parser):
    args = launch_parser.parse_args(["launch", "render", "--model", "m"])
    assert args.server == "http"


def test_parse_render_server_http(launch_parser):
    args = launch_parser.parse_args(
        ["launch", "render", "--model", "m", "--server", "http"]
    )
    assert args.server == "http"


def test_parse_render_server_grpc(launch_parser):
    args = launch_parser.parse_args(
        ["launch", "render", "--model", "m", "--server", "grpc"]
    )
    assert args.server == "grpc"


def test_parse_render_server_invalid(launch_parser):
    with pytest.raises(SystemExit):
        launch_parser.parse_args(
            ["launch", "render", "--model", "m", "--server", "invalid"]
        )


# -- Dispatch --


def test_cmd_launch_render_calls_http():
    args = argparse.Namespace(server="http")
    with (
        patch("vllm.entrypoints.cli.launch.run_launch_http") as mock_http,
        patch("vllm.entrypoints.cli.launch.uvloop.run"),
    ):
        RenderSubcommand.cmd(args)
        mock_http.assert_called_once_with(args)


def test_cmd_launch_render_calls_grpc():
    args = argparse.Namespace(server="grpc")
    with (
        patch("vllm.entrypoints.cli.launch.run_launch_grpc") as mock_grpc,
        patch("vllm.entrypoints.cli.launch.uvloop.run"),
    ):
        RenderSubcommand.cmd(args)
        mock_grpc.assert_called_once_with(args)


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
