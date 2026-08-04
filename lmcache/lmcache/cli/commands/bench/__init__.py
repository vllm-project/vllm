# SPDX-License-Identifier: Apache-2.0
"""``lmcache bench`` command — sustained performance benchmarking.

Sub-subcommands are auto-discovered from sub-packages in this package.
To add a new benchmark, create a sub-package with a concrete
:class:`BaseCommand` subclass in its ``__init__.py`` — no edits to
this file are required.
"""

# First Party
from lmcache.cli.commands.base import CompositeCommand


class BenchCommand(CompositeCommand):
    """CLI command for sustained performance benchmarking."""

    def name(self) -> str:
        return "bench"

    def help(self) -> str:
        return "Run sustained performance benchmarks."
