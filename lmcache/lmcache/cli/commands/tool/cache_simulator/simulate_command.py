# SPDX-License-Identifier: Apache-2.0
"""``lmcache tool cache-simulator simulate`` subcommand."""

# Standard
import argparse

# First Party
from lmcache.cli.commands.base import BaseCommand


class SimulateCommand(BaseCommand):
    """
    Replay logs at a fixed cache capacity; print a text report and s
    ave a 7-panel statistics PNG.
    """

    def name(self) -> str:
        return "simulate"

    def help(self) -> str:
        return (
            "Replay logs at a fixed cache capacity; print a text report "
            "and save a 7-panel statistics PNG."
        )

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        # First Party
        from lmcache.tools.cache_simulator.simulator import add_simulate_arguments

        add_simulate_arguments(parser)

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
        from lmcache.tools.cache_simulator.simulator import run_simulate

        run_simulate(args)
