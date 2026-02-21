# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the `vllm online` CLI subcommand."""

import argparse
from unittest.mock import patch

import pytest

from vllm.entrypoints.cli.online import (
    OnlineSubcommand,
    cmd_init,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser


@pytest.fixture
def online_parser():
    parser = FlexibleArgumentParser(description="test")
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    OnlineSubcommand().subparser_init(subparsers)
    return parser


def test_subcommand_name():
    assert OnlineSubcommand().name == "online"


def test_cmd_init_returns_subcommand():
    result = cmd_init()
    assert len(result) == 1
    assert isinstance(result[0], OnlineSubcommand)


def test_parse_default_server(online_parser):
    args = online_parser.parse_args(["online", "--model", "test-model"])
    assert args.server == "fastapi"


def test_parse_server_fastapi(online_parser):
    args = online_parser.parse_args(
        ["online", "--model", "test-model", "--server", "fastapi"]
    )
    assert args.server == "fastapi"


def test_parse_server_grpc(online_parser):
    args = online_parser.parse_args(
        ["online", "--model", "test-model", "--server", "grpc"]
    )
    assert args.server == "grpc"


def test_parse_invalid_server_rejected(online_parser):
    with pytest.raises(SystemExit):
        online_parser.parse_args(
            ["online", "--model", "test-model", "--server", "unknown"]
        )


def test_cmd_fastapi_calls_run():
    args = argparse.Namespace(model_tag=None, model="test-model", server="fastapi")
    with patch("vllm.entrypoints.cli.online.uvloop.run") as mock_uvloop_run:
        OnlineSubcommand.cmd(args)
        mock_uvloop_run.assert_called_once()


def test_cmd_unknown_server_raises():
    args = argparse.Namespace(model_tag=None, model="test-model", server="unknown")
    with pytest.raises(ValueError, match="Unknown server type"):
        OnlineSubcommand.cmd(args)


def test_model_tag_overrides_model():
    args = argparse.Namespace(
        model_tag="tag-model", model="original-model", server="fastapi"
    )
    with patch("vllm.entrypoints.cli.online.uvloop.run"):
        OnlineSubcommand.cmd(args)
        assert args.model == "tag-model"


def test_model_tag_none_keeps_model():
    args = argparse.Namespace(model_tag=None, model="original-model", server="fastapi")
    with patch("vllm.entrypoints.cli.online.uvloop.run"):
        OnlineSubcommand.cmd(args)
        assert args.model == "original-model"


def test_subparser_init_returns_parser():
    parser = FlexibleArgumentParser(description="test")
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    result = OnlineSubcommand().subparser_init(subparsers)
    assert isinstance(result, FlexibleArgumentParser)


def test_online_registered_in_main():
    """Verify that online module is importable as a CLI module."""
    import vllm.entrypoints.cli.online as online_module

    assert hasattr(online_module, "cmd_init")
    subcmds = online_module.cmd_init()
    assert any(s.name == "online" for s in subcmds)
