# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse

from vllm.benchmarks.sweep.cli import add_cli_args, main
from vllm.foundation.utilities.argparse_utils import FlexibleArgumentParser
from vllm.frontend.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase


class BenchmarkSweepSubcommand(BenchmarkSubcommandBase):
    """The `sweep` subcommand for `vllm bench`."""

    name = "sweep"
    help = "Benchmark for a parameter sweep."

    @classmethod
    def add_cli_args(cls, parser: FlexibleArgumentParser) -> None:
        add_cli_args(parser)

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        main(args)
