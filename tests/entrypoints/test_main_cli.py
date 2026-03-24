# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from vllm.entrypoints.cli import main as cli_main


class _FakeCommand:
    name = "collect-env"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        return None

    def validate(self, args: argparse.Namespace) -> None:
        return None

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> argparse.ArgumentParser:
        return subparsers.add_parser(
            self.name,
            help="Collect environment information.",
            description="Collect environment information.",
        )


def test_requested_subcommand_skips_global_flags():
    assert cli_main._requested_subcommand(["--help"]) is None
    assert cli_main._requested_subcommand(["--version"]) is None
    assert cli_main._requested_subcommand(["collect-env", "--help"]) == "collect-env"


def test_root_help_does_not_import_subcommand_modules(monkeypatch, capsys):
    imported_modules: list[str] = []

    def fail_import(name: str):
        imported_modules.append(name)
        raise AssertionError(f"unexpected subcommand import: {name}")

    monkeypatch.setattr(cli_main.importlib, "import_module", fail_import)
    monkeypatch.setattr(cli_main.sys, "argv", ["vllm", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main()

    assert exc_info.value.code == 0
    assert imported_modules == []
    assert "collect-env" in capsys.readouterr().out


def test_selected_subcommand_only_imports_requested_module(monkeypatch):
    imported_modules: list[str] = []

    def fake_import(name: str):
        imported_modules.append(name)
        if name != "vllm.entrypoints.cli.collect_env":
            raise AssertionError(f"unexpected subcommand import: {name}")
        return SimpleNamespace(cmd_init=lambda: [_FakeCommand()])

    monkeypatch.setattr(cli_main.importlib, "import_module", fake_import)
    monkeypatch.setattr(cli_main.sys, "argv", ["vllm", "collect-env", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main()

    assert exc_info.value.code == 0
    assert imported_modules == ["vllm.entrypoints.cli.collect_env"]
