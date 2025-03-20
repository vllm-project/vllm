# SPDX-License-Identifier: Apache-2.0
import argparse

from vllm.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase
from vllm.entrypoints.cli.types import CLISubcommand


class BenchmarkServingSubcommand(BenchmarkSubcommandBase):
    """ The `serve` subcommand for vllm bench. """

    def __init__(self):
        self.name = "serve"
        super().__init__()

    @property
    def help(self) -> str:
        return "Benchmark the online serving throughput."

    def add_cli_args(self, parser: argparse.ArgumentParser) -> None:
        from vllm.benchmarks.serve import add_cli_args
        add_cli_args(parser)

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        from vllm.benchmarks.serve import main
        main(args)

    @staticmethod
    def lazy_cmd(args: argparse.Namespace) -> None:
        cmd(args)


def cmd_init() -> list[CLISubcommand]:
    return [BenchmarkServingSubcommand()]
