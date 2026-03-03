# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the `vllm launch` CLI subcommand."""

import argparse
from unittest.mock import patch

import pytest

from vllm.entrypoints.cli.launch import (
    LaunchSubcommand,
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


def test_parse_model(launch_parser):
    args = launch_parser.parse_args(["launch", "--model", "test-model"])
    assert args.model == "test-model"


def test_cmd_calls_run():
    args = argparse.Namespace(model_tag=None, model="test-model")
    with patch("vllm.entrypoints.cli.launch.uvloop.run") as mock_uvloop_run:
        LaunchSubcommand.cmd(args)
        mock_uvloop_run.assert_called_once()


def test_model_tag_overrides_model():
    args = argparse.Namespace(model_tag="tag-model", model="original-model")
    with patch("vllm.entrypoints.cli.launch.uvloop.run"):
        LaunchSubcommand.cmd(args)
        assert args.model == "tag-model"


def test_model_tag_none_keeps_model():
    args = argparse.Namespace(model_tag=None, model="original-model")
    with patch("vllm.entrypoints.cli.launch.uvloop.run"):
        LaunchSubcommand.cmd(args)
        assert args.model == "original-model"


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
