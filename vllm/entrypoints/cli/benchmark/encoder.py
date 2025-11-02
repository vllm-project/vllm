# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse

from vllm.benchmarks.encoder import add_cli_args, main
from vllm.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase


class BenchmarkEncoderSubcommand(BenchmarkSubcommandBase):
    """The `encoder` subcommand for vllm bench."""

    name = "encoder"
    help = "Benchmark standalone multimodal encoder forward latency."

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        add_cli_args(parser)

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        main(args)
