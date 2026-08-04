# SPDX-License-Identifier: Apache-2.0
"""``lmcache query`` command — single-shot inference request interface.

Sub-subcommands are auto-discovered from modules in this package.
To add a new subcommand, create a module defining a concrete
:class:`BaseCommand` subclass — no edits to this file are required.
"""

# First Party
from lmcache.cli.commands.base import CompositeCommand


class QueryCommand(CompositeCommand):
    """CLI command that sends one request to a serving engine."""

    def name(self) -> str:
        return "query"

    def help(self) -> str:
        return "Run one inference request and report metrics."
