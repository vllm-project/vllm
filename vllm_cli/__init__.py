# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.metadata
import json
import sys
from importlib.resources import files

_HELP_PATHS = {
    (): "top",
    ("-h",): "top",
    ("--help",): "top",
    ("serve", "-h"): "serve",
    ("serve", "--help"): "serve",
}


def _help() -> dict:
    return json.loads(files(__package__).joinpath("_help.json").read_text())


def _runtime() -> None:
    from vllm.entrypoints.cli.main import main

    main()


def _serve_help(argument: str) -> None:
    data = _help()
    query = argument.split("=", 1)[1].lower().replace("_", "-")
    output = data["help"]["all"] if query == "all" else data["queries"].get(query)
    if output is None:
        _runtime()
    else:
        print(output, end="")


def _is_serve_help(args: tuple[str, ...]) -> bool:
    return len(args) == 2 and args[0] == "serve" and args[1].startswith("--help=")


def main() -> None:
    args = tuple(sys.argv[1:])
    if help_path := _HELP_PATHS.get(args):
        print(_help()["help"][help_path], end="")
    elif args in (("-v",), ("--version",)):
        print(importlib.metadata.version("vllm"))
    elif _is_serve_help(args):
        _serve_help(args[1])
    else:
        _runtime()
