# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse

from vllm.benchmarks.throughput import add_cli_args, main
from vllm.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase
from vllm.entrypoints.cli.types import CLISubcommand


class BenchmarkThroughputSubcommand(BenchmarkSubcommandBase):
    """ The `throughput` subcommand for vllm bench. """

    def __init__(self):
        self.name = "throughput"
        super().__init__()

    @property
    def help(self) -> str:
        return "Benchmark offline inference throughput."

    def add_cli_args(self, parser: argparse.ArgumentParser) -> None:
        add_cli_args(parser)

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        main(args)


def cmd_init() -> list[CLISubcommand]:
    return [BenchmarkThroughputSubcommand()]
