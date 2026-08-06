# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import subprocess
import sys

import pytest

from vllm.entrypoints.cli import snapshot as snapshot_cli
from vllm.utils.argparse_utils import FlexibleArgumentParser


def parse_snapshot(*argv: str):
    parser = FlexibleArgumentParser()
    subparsers = parser.add_subparsers(dest="subparser", required=True)
    snapshot_cli.SnapshotSubcommand().subparser_init(subparsers)
    return parser.parse_args(["snapshot", *argv])


def test_snapshot_help_is_registered():
    result = subprocess.run(
        [sys.executable, "-m", "vllm.entrypoints.cli.main", "snapshot", "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "create" in result.stdout
    assert "inspect" in result.stdout
    assert "restore" in result.stdout


def test_snapshot_create_parses_model_and_directory():
    args = parse_snapshot(
        "create", "Qwen/Qwen3-0.6B", "--snapshot-dir", "/tmp/qwen-snapshot"
    )

    assert args.snapshot_action == "create"
    assert args.model_tag == "Qwen/Qwen3-0.6B"
    assert args.snapshot_dir == "/tmp/qwen-snapshot"


def test_snapshot_inspect_parses_directory():
    args = parse_snapshot("inspect", "/tmp/qwen-snapshot")

    assert args.snapshot_action == "inspect"
    assert args.snapshot_dir == "/tmp/qwen-snapshot"


def test_snapshot_restore_parses_listener():
    args = parse_snapshot(
        "restore", "/tmp/qwen-snapshot", "--host", "127.0.0.1", "--port", "9000"
    )

    assert args.snapshot_action == "restore"
    assert args.snapshot_dir == "/tmp/qwen-snapshot"
    assert args.host == "127.0.0.1"
    assert args.port == 9000


def test_create_rejects_tp2():
    args = parse_snapshot(
        "create",
        "Qwen/Qwen3-0.6B",
        "--snapshot-dir",
        "/tmp/qwen-snapshot",
        "--tensor-parallel-size",
        "2",
    )

    with pytest.raises(ValueError, match="tensor parallel size 1"):
        snapshot_cli.validate_create_args(args)
