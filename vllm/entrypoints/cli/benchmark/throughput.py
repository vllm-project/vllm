# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse

from vllm.benchmarks.throughput import add_cli_args, main
from vllm.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase


class BenchmarkThroughputSubcommand(BenchmarkSubcommandBase):
    """The `throughput` subcommand for `vllm bench`."""

    name = "throughput"
    help = "Benchmark offline inference throughput."

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        # Type ignore since ArgumentParser is compatible with FlexibleArgumentParser
        add_cli_args(parser)  # type: ignore[arg-type]

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        main(args)
