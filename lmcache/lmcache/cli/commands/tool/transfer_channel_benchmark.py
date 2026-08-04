# SPDX-License-Identifier: Apache-2.0
"""``lmcache tool transfer-channel-benchmark`` — transfer channel throughput benchmark.

Exposes :class:`TransferChannelBenchmarkCommand` for auto-discovery by
:class:`~lmcache.cli.commands.base.CompositeCommand`.

Argument definitions and execution logic live in the benchmark module:

* :func:`~lmcache.tools.transfer_channel_benchmark.config.add_benchmark_arguments`
* :func:`~lmcache.tools.transfer_channel_benchmark.benchmark.run_benchmark`
"""

# Standard
import argparse
import sys

# First Party
from lmcache.cli.commands.base import BaseCommand


class TransferChannelBenchmarkCommand(BaseCommand):
    """Benchmark transfer channel read throughput (server/client)."""

    def name(self) -> str:
        return "transfer-channel-benchmark"

    def help(self) -> str:
        return "Benchmark transfer channel read throughput (server/client)."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add benchmark-specific arguments.

        Only the torch-free ``config`` module is imported here, so registering
        this tool does not require torch or the distributed runtime.

        Args:
            parser: The ``ArgumentParser`` for this subcommand.
        """
        # First Party
        from lmcache.tools.transfer_channel_benchmark.config import (
            add_benchmark_arguments,
        )

        add_benchmark_arguments(parser)

    def register(self, subparsers: argparse._SubParsersAction) -> None:
        """Register without the common output args (--format/--output/--quiet).

        The benchmark manages its own output lifecycle and does not use the
        generic metrics output flags.
        """
        parser = subparsers.add_parser(
            self.name(),
            help=self.help(),
            description=(
                "Throughput benchmark for the LMCache transfer channel. Run one "
                "process with --role server and another with --role client."
            ),
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self.add_arguments(parser)
        parser.set_defaults(func=self.execute)

    def execute(self, args: argparse.Namespace) -> None:
        """Run the benchmark and exit non-zero on failure.

        Args:
            args: Parsed CLI arguments.
        """
        # First Party
        from lmcache.tools.transfer_channel_benchmark.benchmark import (
            run_benchmark,
        )

        if not run_benchmark(args):
            sys.exit(1)
