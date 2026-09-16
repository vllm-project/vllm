# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse

from vllm.benchmarks.throughput import add_cli_args, main
from vllm.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase
from vllm.logger import configure_logging_from_args
from vllm.utils.argparse_utils import FlexibleArgumentParser


class BenchmarkThroughputSubcommand(BenchmarkSubcommandBase):
    """The `throughput` subcommand for `vllm bench`."""

    name = "throughput"
    help = "Benchmark offline inference throughput."

    @classmethod
    def add_cli_args(cls, parser: FlexibleArgumentParser) -> None:
        add_cli_args(parser)

    def post_parse(self, args: argparse.Namespace) -> None:
        configure_logging_from_args(args)

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        main(args)
