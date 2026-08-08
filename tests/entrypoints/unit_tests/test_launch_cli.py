# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for vLLM CLI subcommands."""

import argparse
import builtins
import os
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from vllm.entrypoints.cli import snapshot as snapshot_cli
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


def parse_snapshot(*argv: str):
    parser = FlexibleArgumentParser()
    subparsers = parser.add_subparsers(dest="subparser", required=True)
    snapshot_cli.SnapshotSubcommand().subparser_init(subparsers)
    return parser.parse_args(["snapshot", *argv])


def test_snapshot_restore_dispatches_without_runtime_imports(monkeypatch):
    dispatched: list[list[str]] = []
    snapshot_module = types.ModuleType("vllm_cli.snapshot")
    snapshot_module.capture_snapshot_environment = lambda _env: None  # type: ignore[attr-defined]
    snapshot_module.main = lambda argv: dispatched.append(argv)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm_cli.snapshot", snapshot_module)

    real_import = builtins.__import__

    def reject_runtime_import(name, *args, **kwargs):
        if name == "vllm" or name.startswith("vllm.") or name == "torch":
            raise AssertionError(f"lightweight restore imported {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_runtime_import)

    from vllm_cli.main import main

    main(["snapshot", "restore", "/tmp/qwen-snapshot"])

    assert dispatched == [["restore", "/tmp/qwen-snapshot"]]


def test_non_snapshot_commands_use_the_existing_vllm_cli(monkeypatch):
    received_argv: list[list[str]] = []

    def fake_vllm_main() -> None:
        received_argv.append(sys.argv.copy())

    monkeypatch.setattr("vllm.entrypoints.cli.main.main", fake_vllm_main)

    from vllm_cli.main import main

    main(["serve", "Qwen/Qwen3-0.6B"])

    assert received_argv == [
        [sys.argv[0], "serve", "Qwen/Qwen3-0.6B"],
    ]


def test_snapshot_environment_is_captured_before_runtime_imports(monkeypatch):
    from vllm_cli.snapshot import (
        capture_snapshot_environment,
        snapshot_environment,
    )

    prior_environment = snapshot_environment()
    monkeypatch.setenv("VLLM_USER_SETTING", "configured")
    monkeypatch.delenv("TRITON_CACHE_AUTOTUNING", raising=False)

    def fake_vllm_main() -> None:
        os.environ["TRITON_CACHE_AUTOTUNING"] = "1"

    monkeypatch.setattr("vllm.entrypoints.cli.main.main", fake_vllm_main)

    from vllm_cli.main import main

    try:
        main(["snapshot", "create", "Qwen/Qwen3-0.6B"])

        assert snapshot_environment()["VLLM_USER_SETTING"] == "configured"
        assert "TRITON_CACHE_AUTOTUNING" not in snapshot_environment()
    finally:
        capture_snapshot_environment(prior_environment)


def test_snapshot_restore_parser_stays_lightweight(monkeypatch, tmp_path: Path):
    from vllm_cli.snapshot import cli

    restored: list[argparse.Namespace] = []
    monkeypatch.setattr(cli, "restore_snapshot", restored.append)
    artifact = tmp_path / "snapshot"

    cli.main(
        [
            "restore",
            str(artifact),
            "--host",
            "127.0.0.1",
            "--port",
            "9000",
        ]
    )

    assert len(restored) == 1
    assert restored[0].snapshot_dir == str(artifact)
    assert restored[0].host == "127.0.0.1"
    assert restored[0].port == 9000


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
        "--revision",
        "model-sha",
        "--tensor-parallel-size",
        "2",
    )

    with pytest.raises(ValueError, match="tensor parallel size 1"):
        snapshot_cli.validate_create_args(args)


def test_create_requires_pinned_model_revision():
    args = parse_snapshot(
        "create", "Qwen/Qwen3-0.6B", "--snapshot-dir", "/tmp/qwen-snapshot"
    )

    with pytest.raises(ValueError, match="--revision"):
        snapshot_cli.validate_create_args(args)


@pytest.mark.parametrize(
    ("extra_args", "message"),
    [
        (("--api-key", "secret"), "API authentication"),
        (("--uds", "/tmp/vllm.sock"), "Unix domain sockets"),
        (
            ("--ssl-keyfile", "/tmp/key.pem", "--ssl-certfile", "/tmp/cert.pem"),
            "TLS",
        ),
        (("--middleware", "example.middleware"), "custom middleware"),
    ],
)
def test_create_rejects_frontends_the_restore_oracle_cannot_reach(
    extra_args: tuple[str, ...], message: str
):
    args = parse_snapshot(
        "create",
        "Qwen/Qwen3-0.6B",
        "--snapshot-dir",
        "/tmp/qwen-snapshot",
        "--revision",
        "model-sha",
        *extra_args,
    )

    with pytest.raises(ValueError, match=message):
        snapshot_cli.validate_create_args(args)


def test_create_rejects_environment_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_API_KEY", "secret")
    args = parse_snapshot(
        "create",
        "Qwen/Qwen3-0.6B",
        "--snapshot-dir",
        "/tmp/qwen-snapshot",
        "--revision",
        "model-sha",
    )

    with pytest.raises(ValueError, match="API authentication"):
        snapshot_cli.validate_create_args(args)
