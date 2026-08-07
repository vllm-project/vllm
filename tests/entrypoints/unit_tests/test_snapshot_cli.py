# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import builtins
import subprocess
import sys
import types
from pathlib import Path

import pytest

from vllm.entrypoints.cli import snapshot as snapshot_cli
from vllm.utils.argparse_utils import FlexibleArgumentParser


def parse_snapshot(*argv: str):
    parser = FlexibleArgumentParser()
    subparsers = parser.add_subparsers(dest="subparser", required=True)
    snapshot_cli.SnapshotSubcommand().subparser_init(subparsers)
    return parser.parse_args(["snapshot", *argv])


def test_snapshot_restore_dispatches_without_runtime_imports(monkeypatch):
    dispatched: list[list[str]] = []
    snapshot_module = types.ModuleType("vllm_cli.snapshot")
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
