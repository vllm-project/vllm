# SPDX-License-Identifier: Apache-2.0
"""``lmcache tool`` command — offline analysis utilities.

Sub-subcommands are auto-discovered from modules in this package.
To add a new tool, create a module defining a concrete
:class:`BaseCommand` subclass — no edits to this file are required.
"""

# First Party
from lmcache.cli.commands.base import CompositeCommand


class ToolCommand(CompositeCommand):
    """CLI command for offline analysis tools bundled with LMCache."""

    def name(self) -> str:
        return "tool"

    def help(self) -> str:
        return "Run offline analysis tools."
