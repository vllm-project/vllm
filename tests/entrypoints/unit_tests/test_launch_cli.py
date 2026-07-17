# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the `vllm launch` CLI subcommand."""

import argparse
import sys
from unittest.mock import patch

import pytest

from vllm.entrypoints.cli.launch import (
    LaunchSubcommand,
    RenderSubcommand,
    cmd_init,
)
from vllm.entrypoints.cli.snapshot import SnapshotSubcommand
from vllm.entrypoints.snapshot import maybe_restore_serve
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


# -- `vllm snapshot` subcommand (folded here per the no-new-test-file rule; this
#    file now also covers the snapshot CLI surface) --


@pytest.fixture
def snapshot_parser():
    parser = FlexibleArgumentParser(description="test")
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    SnapshotSubcommand().subparser_init(subparsers)
    return parser


def test_snapshot_registered_in_main():
    import vllm.entrypoints.cli.snapshot as snapshot_module

    assert hasattr(snapshot_module, "cmd_init")
    subcmds = snapshot_module.cmd_init()
    assert any(s.name == "snapshot" for s in subcmds)


def test_parse_snapshot_create_flags(snapshot_parser):
    args = snapshot_parser.parse_args(["snapshot", "create", "--dry-run", "--force"])
    assert args.dry_run is True
    assert args.force is True


def test_restore_hook_noop_when_disabled(monkeypatch, caplog):
    monkeypatch.delenv("VLLM_SNAPSHOT", raising=False)
    with caplog.at_level("INFO", logger="vllm.entrypoints.snapshot"):
        assert maybe_restore_serve() is None
    assert not any(
        "snapshot restore" in record.getMessage() for record in caplog.records
    )


def test_restore_hook_cold_fallback_logs_miss(monkeypatch, caplog, tmp_path):
    # Enabled + `serve` + no matching snapshot: the hook must fall back to a cold
    # start (return None) and log exactly one miss line. Platform is pinned so
    # the linux-only gate runs off-linux too, and the lookup key is stubbed so
    # the test stays on the no-snapshot path even in editable or RECORD-less
    # environments where key computation itself would refuse.
    import vllm.entrypoints.snapshot as snapshot_module

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "argv", ["vllm", "serve", "some-model"])
    monkeypatch.setenv("VLLM_SNAPSHOT", "1")
    monkeypatch.delenv("VLLM_SNAPSHOT_RESTORED", raising=False)
    monkeypatch.setenv("VLLM_SNAPSHOT_ROOT", str(tmp_path))
    monkeypatch.setattr(snapshot_module, "lookup_key", lambda env: {"stub": 1})
    with caplog.at_level("INFO", logger="vllm.entrypoints.snapshot"):
        assert maybe_restore_serve() is None
    misses = [
        record
        for record in caplog.records
        if "snapshot restore miss" in record.getMessage()
    ]
    assert len(misses) == 1
    assert "no snapshot" in misses[0].getMessage()
