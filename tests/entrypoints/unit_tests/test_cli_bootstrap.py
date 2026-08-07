# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import builtins
import sys
import types
from pathlib import Path


def test_snapshot_restore_dispatches_without_importing_vllm(monkeypatch):
    dispatched: list[list[str]] = []
    snapshot_module = types.ModuleType("vllm_cli.snapshot")
    snapshot_module.main = lambda argv: dispatched.append(argv)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm_cli.snapshot", snapshot_module)

    real_import = builtins.__import__

    def reject_vllm_import(name, *args, **kwargs):
        if name == "vllm" or name.startswith("vllm."):
            raise AssertionError(f"lightweight restore imported {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_vllm_import)

    from vllm_cli.main import main

    main(["snapshot", "restore", "/tmp/qwen-snapshot"])

    assert dispatched == [["restore", "/tmp/qwen-snapshot"]]


def test_snapshot_restore_parser_stays_lightweight(monkeypatch, tmp_path: Path):
    real_import = builtins.__import__

    def reject_vllm_import(name, *args, **kwargs):
        if name == "vllm" or name.startswith("vllm."):
            raise AssertionError(f"lightweight restore imported {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_vllm_import)

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


def test_snapshot_restore_controller_imports_without_runtime(monkeypatch):
    real_import = builtins.__import__

    def reject_runtime_import(name, *args, **kwargs):
        if name == "vllm" or name.startswith("vllm.") or name == "torch":
            raise AssertionError(f"lightweight restore imported {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_runtime_import)

    from vllm_cli.snapshot.controller import restore_snapshot

    assert callable(restore_snapshot)
