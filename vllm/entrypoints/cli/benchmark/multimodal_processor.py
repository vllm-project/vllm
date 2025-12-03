# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse

from vllm.benchmarks.multimodal_processor import add_cli_args, main
from vllm.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase


class BenchmarkMultimodalProcessorSubcommand(BenchmarkSubcommandBase):
    """The `multimodal-processor` subcommand for `vllm bench`."""

    name = "multimodal-processor"
    help = "Benchmark multimodal processor latency across different configurations."

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        add_cli_args(parser)

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        main(args)

