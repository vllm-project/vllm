# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse

from vllm.benchmarks.startup import add_cli_args, main
from vllm.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase
from vllm.logger import configure_logging_from_args
from vllm.utils.argparse_utils import FlexibleArgumentParser


class BenchmarkStartupSubcommand(BenchmarkSubcommandBase):
    """The `startup` subcommand for `vllm bench`."""

    name = "startup"
    help = "Benchmark the startup time of vLLM models."

    @classmethod
    def add_cli_args(cls, parser: FlexibleArgumentParser) -> None:
        add_cli_args(parser)

    def post_parse(self, args: argparse.Namespace) -> None:
        configure_logging_from_args(args)

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        main(args)
