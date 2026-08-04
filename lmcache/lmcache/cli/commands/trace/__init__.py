# SPDX-License-Identifier: Apache-2.0
"""``lmcache trace`` command — inspect and replay storage-level trace files.

Sub-subcommands are auto-discovered from modules in this package.
To add a new subcommand, create a module defining a concrete
:class:`BaseCommand` subclass — no edits to this file are required.

Trace *capture* is not a ``trace`` subcommand — recording is bound to
the live process via ``lmcache server --trace-level storage
[--trace-output ...]``.
"""

# First Party
from lmcache.cli.commands.base import CompositeCommand


class TraceCommand(CompositeCommand):
    """Subcommand group for trace inspection and replay."""

    def name(self) -> str:
        return "trace"

    def help(self) -> str:
        return "Inspect and replay LMCache storage-level trace files."
