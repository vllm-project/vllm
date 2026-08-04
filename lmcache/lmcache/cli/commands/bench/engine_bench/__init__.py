# SPDX-License-Identifier: Apache-2.0
"""``lmcache bench engine`` subpackage.

Exposes :class:`EngineBenchCommand` for auto-discovery by
:class:`~lmcache.cli.commands.base.CompositeCommand`.
"""

# Standard
import argparse

# First Party
from lmcache.cli.commands.base import BaseCommand


class EngineBenchCommand(BaseCommand):
    """Benchmark an inference engine."""

    def name(self) -> str:
        return "engine"

    def help(self) -> str:
        return "Benchmark an inference engine."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        # First Party
        from lmcache.cli.commands.bench.engine_bench.command import (
            add_engine_arguments,
        )

        add_engine_arguments(parser)

    def register(self, subparsers: argparse._SubParsersAction) -> None:
        """Register without the common output args (--format/--output).

        The engine bench manages its own output lifecycle (CSV/JSON export,
        --quiet) so the generic _add_output_args flags are not applicable
        and would only clutter ``-h`` output.
        """
        parser = subparsers.add_parser(self.name(), help=self.help())
        self.add_arguments(parser)
        parser.set_defaults(func=self.execute)

    def execute(self, args: argparse.Namespace) -> None:
        # First Party
        from lmcache.cli.commands.bench.engine_bench.command import (
            run_engine_bench,
        )

        run_engine_bench(self, args)
