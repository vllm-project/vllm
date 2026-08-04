# SPDX-License-Identifier: Apache-2.0
"""``lmcache bench l2`` subpackage.

Exposes :class:`L2AdapterBenchCommand` for auto-discovery by
:class:`~lmcache.cli.commands.base.CompositeCommand`.
"""

# Standard
import argparse

# First Party
from lmcache.cli.commands.base import BaseCommand


class L2AdapterBenchCommand(BaseCommand):
    """Benchmark an L2 adapter (store / lookup / load)."""

    def name(self) -> str:
        return "l2"

    def help(self) -> str:
        return "Benchmark an L2 adapter (store / lookup / load)."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        # First Party
        from lmcache.cli.commands.bench.l2_adapter_bench.command import (
            add_l2_arguments,
        )

        add_l2_arguments(parser)

    def execute(self, args: argparse.Namespace) -> None:
        # Standard
        import copy
        import sys

        # First Party
        from lmcache.cli.commands.bench.l2_adapter_bench.command import (
            run_l2_adapter_bench,
        )

        # A comma-separated --flamegraph-mode profiles one benchmark run per
        # mode (the modes cannot share a recording window), each rendered to
        # its own default path.
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
            run_l2_adapter_bench(self, run_args)
