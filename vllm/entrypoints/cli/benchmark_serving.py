# SPDX-License-Identifier: Apache-2.0
import argparse

from vllm.benchmarks.benchmark_serving import add_options, main
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.utils import FlexibleArgumentParser


class BenchmarkServingSubcommand(CLISubcommand):
    """ The `serving` subcommand for vllm bench. """

    def __init__(self):
        self.name = "serving"
        super().__init__()

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        main(args)

    def subparser_init(
            self,
            subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        parser = subparsers.add_parser(
            "serving",
            help="Benchmark the online serving throughput.",
            usage="vllm bench serving [options]")
        add_options(parser)
        return parser


def cmd_init() -> list[CLISubcommand]:
    return [BenchmarkServingSubcommand()]
