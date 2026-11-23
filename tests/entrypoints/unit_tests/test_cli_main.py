# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.metadata
import pkgutil
import subprocess
import sys
from textwrap import dedent

import pytest

import vllm.entrypoints.cli as cli
from vllm.entrypoints.cli.main import _SUBCOMMAND_MODULES


def test_subcommand_modules_match_command_factories():
    commands = {}
    for info in pkgutil.walk_packages(cli.__path__, cli.__name__ + "."):
        module = importlib.import_module(info.name)
        if hasattr(module, "cmd_init"):
            for command in module.cmd_init():
                commands[command.name] = info.name
    assert commands == _SUBCOMMAND_MODULES


@pytest.mark.parametrize("command", [None, "serve", "bench"])
def test_help_matches_eager_registration(command, monkeypatch):
    from vllm.entrypoints.common.startup import VLLM_SUBCMD_PARSER_EPILOG
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    argv = ["vllm", "--help"] if command is None else ["vllm", command, "--help"]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.setenv("PYTHON_COLORS", "0")
    monkeypatch.setenv("VLLM_LOGGING_LEVEL", "CRITICAL")

    parser = FlexibleArgumentParser(
        prog="vllm",
        description="vLLM CLI",
        epilog=VLLM_SUBCMD_PARSER_EPILOG.format(subcmd="[subcommand]"),
    )
    parser.add_argument(
        "-v", "--version", action="version", version=importlib.metadata.version("vllm")
    )
    subparsers = parser.add_subparsers(dest="subparser")
    expected_parser = parser
    for name in dict.fromkeys(_SUBCOMMAND_MODULES.values()):
        module = importlib.import_module(name)
        for cmd in module.cmd_init():
            child = cmd.subparser_init(subparsers)
            if cmd.name == command:
                expected_parser = child

    entrypoint = (
        "import sys; sys.argv[0] = 'vllm'; "
        "from vllm.entrypoints.cli.main import main; main()"
    )
    output = subprocess.check_output(
        [sys.executable, "-c", entrypoint, *argv[1:]], timeout=60
    )
    assert output == expected_parser.format_help().encode()


@pytest.mark.parametrize(
    "argv", [["collect-env", "--help"], ["serve", "--help"], ["-v"], ["--version"]]
)
def test_cli_imports_only_requested_subcommand(argv, monkeypatch):
    script = dedent("""
        import sys
        from vllm.entrypoints.cli.main import _SUBCOMMAND_MODULES, main

        try:
            main()
        except SystemExit as exc:
            assert exc.code == 0
        else:
            raise AssertionError("help and version flags should exit")

        requested = _SUBCOMMAND_MODULES.get(sys.argv[1])
        loaded = sys.modules.keys() & _SUBCOMMAND_MODULES.values()
        assert loaded == ({requested} if requested else set()), loaded
    """)
    monkeypatch.setenv("VLLM_LOGGING_LEVEL", "CRITICAL")
    output = subprocess.check_output([sys.executable, "-c", script, *argv], timeout=60)
    if argv[0] in ("-v", "--version"):
        assert output == f"{importlib.metadata.version('vllm')}\n".encode()


@pytest.mark.parametrize(
    ("argv", "diagnostic"),
    [
        (["collect-env", "--invalid"], b"unrecognized arguments: --invalid"),
        (["no-such-cmd"], b"argument subparser: invalid choice: 'no-such-cmd'"),
    ],
)
def test_invalid_arguments_list_all_commands(argv, diagnostic):
    result = subprocess.run(
        [sys.executable, "-m", "vllm.entrypoints.cli.main", *argv],
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 2
    assert diagnostic in result.stderr
    assert ",".join(_SUBCOMMAND_MODULES).encode() in result.stderr
