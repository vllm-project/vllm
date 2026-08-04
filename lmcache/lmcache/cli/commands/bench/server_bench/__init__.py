# SPDX-License-Identifier: Apache-2.0
"""``lmcache bench server`` subpackage.

Exposes :class:`ServerBenchCommand` for auto-discovery by
:class:`~lmcache.cli.commands.base.CompositeCommand`.
"""

# Standard
import argparse

# First Party
from lmcache.cli.commands.base import BaseCommand


class ServerBenchCommand(BaseCommand):
    """End-to-end test for LMCache MP cache server."""

    def name(self) -> str:
        return "server"

    def help(self) -> str:
        return "End-to-end test for LMCache MP cache server (GPU mode)."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        # First Party
        from lmcache.cli.commands.bench.server_bench.command import (
            add_server_arguments,
        )

        add_server_arguments(parser)

    def execute(self, args: argparse.Namespace) -> None:
        # Standard
        import copy
        import sys

        # First Party
        from lmcache.cli.commands.bench.server_bench.command import (
            run_server_bench,
        )

        # A comma-separated --flamegraph-mode drives the load once per mode
        # (the modes cannot share a recording window), each rendered to its
        # own default path.
        modes = [args.flamegraph_mode]
        if getattr(args, "flamegraph", "off") == "on" and "," in args.flamegraph_mode:
            modes = list(
                dict.fromkeys(
                    m.strip() for m in args.flamegraph_mode.split(",") if m.strip()
                )
            )
            if args.flamegraph_output:
                print(
                    "Error: --flamegraph-output takes a single path; with "
                    "several modes each graph uses its own default path.",
                    file=sys.stderr,
                )
                sys.exit(2)

        for mode in modes:
            run_args = copy.copy(args)
            run_args.flamegraph_mode = mode
            run_server_bench(self, run_args)
