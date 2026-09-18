# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import typing

from vllm.entrypoints.cli.types import CLISubcommand

if typing.TYPE_CHECKING:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
else:
    FlexibleArgumentParser = argparse.ArgumentParser


class WeightCacheDaemonSubcommand(CLISubcommand):
    """The ``weight-cache-daemon`` subcommand for the vLLM CLI."""

    name = "weight-cache-daemon"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        from vllm.model_executor.model_loader.weight_cache.daemon import run

        run(args)

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        from vllm.model_executor.model_loader.weight_cache.daemon import add_cli_args

        parser = subparsers.add_parser(
            self.name,
            help="Launch weight cache daemons for fast engine restarts.",
            description="Launch weight cache daemons (one per TP rank).",
            usage="vllm weight-cache-daemon [options]",
        )
        add_cli_args(parser)
        return parser


def cmd_init() -> list[CLISubcommand]:
    return [WeightCacheDaemonSubcommand()]
