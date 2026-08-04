# SPDX-License-Identifier: Apache-2.0
"""``lmcache mock`` — example command that demonstrates the CLI framework.

This command does not connect to any server.  It generates fake metrics
to exercise argument parsing, the ``Metrics`` logger, and both terminal
and JSON output paths.
"""

# Standard
import argparse

# First Party
from lmcache.cli.commands.base import BaseCommand


class MockCommand(BaseCommand):
    """Mock command used as a reference implementation for new commands."""

    def name(self) -> str:
        """Return the subcommand name.

        Returns:
            The string ``"mock"``.
        """
        return "mock"

    def help(self) -> str:
        """Return short help text.

        Returns:
            Help string shown by ``lmcache -h``.
        """
        return "Run a mock command (example/test)."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add mock-specific arguments.

        Args:
            parser: The ``ArgumentParser`` for this subcommand.
        """
        parser.add_argument(
            "--name",
            type=str,
            default="default",
            help="Name tag for this mock run.",
        )
        parser.add_argument(
            "--num-items",
            type=int,
            default=10,
            help="Number of fake items to process.",
        )

    def execute(self, args: argparse.Namespace) -> None:
        """Execute the mock command.

        Args:
            args: Parsed CLI arguments containing ``name``, ``num_items``,
                and optionally ``output``.
        """
        metrics = self.create_metrics("Mock Result", args, width=40)

        metrics.add_section("input", "Input Parameters")
        metrics["input"].add("name", "Name", args.name)
        metrics["input"].add("num_items", "Num items", args.num_items)

        metrics.add_section("mock", "Mock Metrics")
        metrics["mock"].add("items_processed", "Items processed", 42)
        metrics["mock"].add("total_time_ms", "Total time (ms)", 12.34)
        metrics["mock"].add("throughput", "Throughput (items/s)", 3403.73)

        metrics.add_section("validation", "Validation")
        metrics["validation"].add("status", "Status", "OK")

        metrics.emit()
