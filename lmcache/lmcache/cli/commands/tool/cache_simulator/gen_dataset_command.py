# SPDX-License-Identifier: Apache-2.0
"""``lmcache tool cache-simulator gen-dataset`` subcommand."""

# Standard
import argparse

# First Party
from lmcache.cli.commands.base import BaseCommand


class GenDatasetCommand(BaseCommand):
    """
    Generate a vllm bench serve custom dataset (JSONL)
    from lookup-hash JSONL logs.
    """

    def name(self) -> str:
        return "gen-dataset"

    def help(self) -> str:
        return (
            "Generate a vllm bench serve custom dataset (JSONL) from "
            "lookup-hash JSONL logs, preserving prefix-sharing structure."
        )

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        # First Party
        from lmcache.tools.cache_simulator.gen_bench_dataset import (
            add_gen_dataset_arguments,
        )

        add_gen_dataset_arguments(parser)

    def register(self, subparsers: argparse._SubParsersAction) -> None:
        # Skip _add_output_args — this command defines its own --output.
        # Gracefully skip if optional dependencies are missing.
        try:
            parser = subparsers.add_parser(self.name(), help=self.help())
            self.add_arguments(parser)
            parser.set_defaults(func=self.execute)
        except ImportError:
            return

    def execute(self, args: argparse.Namespace) -> None:
        # First Party
        from lmcache.tools.cache_simulator.gen_bench_dataset import run_gen_dataset

        run_gen_dataset(args)
