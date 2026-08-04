# SPDX-License-Identifier: Apache-2.0
"""``lmcache tool cache-simulator sweep`` subcommand."""

# Standard
import argparse

# First Party
from lmcache.cli.commands.base import BaseCommand


class SweepCommand(BaseCommand):
    """
    Sweep across a range of cache capacities and save a hit-rate vs capacity PNG.
    """

    def name(self) -> str:
        return "sweep"

    def help(self) -> str:
        return (
            "Sweep across a range of cache capacities and save a "
            "hit-rate vs capacity PNG."
        )

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        # First Party
        from lmcache.tools.cache_simulator.plot_hit_rate import add_sweep_arguments

        add_sweep_arguments(parser)

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
        from lmcache.tools.cache_simulator.plot_hit_rate import run_sweep

        run_sweep(args)
