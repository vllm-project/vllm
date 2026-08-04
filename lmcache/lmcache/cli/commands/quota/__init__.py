# SPDX-License-Identifier: Apache-2.0
"""``lmcache quota`` command — per-salt quota management.

Sub-subcommands are auto-discovered from modules in this package.
To add a new subcommand, create a module defining a concrete
:class:`BaseCommand` subclass — no edits to this file are required.
"""

# First Party
from lmcache.cli.commands.base import CompositeCommand


class QuotaCommand(CompositeCommand):
    """CLI command for per-salt quota management on LMCache server."""

    def name(self) -> str:
        return "quota"

    def help(self) -> str:
        return "Manage per-salt cache quotas."
