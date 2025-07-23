# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import typing

from vllm.collect_env import main as collect_env_main
from vllm.entrypoints.cli.types import CLISubcommand

if typing.TYPE_CHECKING:
    from vllm.utils import FlexibleArgumentParser


class CollectEnvSubcommand(CLISubcommand):
    """The `collect-env` subcommand for the vLLM CLI. """
    name = "collect-env"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        """Collect information about the environment."""
        collect_env_main()

    def subparser_init(
            self,
            subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        return subparsers.add_parser(
            "collect-env",
            help="Start collecting environment information.",
            description="Start collecting environment information.",
            usage="vllm collect-env")


def cmd_init() -> list[CLISubcommand]:
    return [CollectEnvSubcommand()]
