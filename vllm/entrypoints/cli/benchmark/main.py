# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import typing

from vllm.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.utils import (VLLM_SUBCMD_PARSER_EPILOG,
                                    show_filtered_argument_or_group_from_help)

if typing.TYPE_CHECKING:
    from vllm.utils import FlexibleArgumentParser


class BenchmarkSubcommand(CLISubcommand):
    """ The `bench` subcommand for the vLLM CLI. """

    name = "bench"
    help = "vLLM bench subcommand."

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        args.dispatch_function(args)

    def validate(self, args: argparse.Namespace) -> None:
        pass

    def subparser_init(
            self,
            subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        bench_parser = subparsers.add_parser(
            self.name,
            help=self.help,
            description=self.help,
            usage="vllm bench <bench_type> [options]")
        bench_subparsers = bench_parser.add_subparsers(required=True,
                                                       dest="bench_type")

        for cmd_cls in BenchmarkSubcommandBase.__subclasses__():
            cmd_subparser = bench_subparsers.add_parser(
                cmd_cls.name,
                help=cmd_cls.help,
                description=cmd_cls.help,
                usage=f"vllm bench {cmd_cls.name} [options]",
            )
            cmd_subparser.set_defaults(dispatch_function=cmd_cls.cmd)
            cmd_cls.add_cli_args(cmd_subparser)
            show_filtered_argument_or_group_from_help(cmd_subparser,
                                                      ["bench", cmd_cls.name])
            cmd_subparser.epilog = VLLM_SUBCMD_PARSER_EPILOG
        return bench_parser


def cmd_init() -> list[CLISubcommand]:
    return [BenchmarkSubcommand()]
