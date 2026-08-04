# SPDX-License-Identifier: Apache-2.0
"""``lmcache tool cache-simulator`` command group.

Sub-subcommands (simulate, sweep, gen-dataset) are auto-discovered from
modules in this package.  To add a new action, create a module defining
a concrete :class:`BaseCommand` subclass — no edits to this file are
required.
"""

# First Party
from lmcache.cli.commands.base import CompositeCommand


class CacheSimulatorCommand(CompositeCommand):
    """Simulate KV-cache token hit rate from lookup-hash JSONL logs."""

    def name(self) -> str:
        return "cache-simulator"

    def help(self) -> str:
        return "Simulate KV-cache token hit rate from lookup-hash JSONL logs."
