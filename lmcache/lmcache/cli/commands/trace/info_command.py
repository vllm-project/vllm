# SPDX-License-Identifier: Apache-2.0
"""``lmcache trace info`` — print a summary of a trace file."""

# Future
from __future__ import annotations

# Standard
from collections import Counter
import argparse

# First Party
from lmcache.cli.commands.base import BaseCommand


class InfoCommand(BaseCommand):
    """Print a summary of a trace file."""

    def name(self) -> str:
        return "info"

    def help(self) -> str:
        return "Print a summary of a trace file."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "trace_path",
            metavar="FILE",
            help="Path to a .lct trace file.",
        )

    def execute(self, args: argparse.Namespace) -> None:
        # First Party
        from lmcache.v1.mp_observability.trace.reader import TraceReader

        with TraceReader(args.trace_path) as r:
            header = r.header
            counts: Counter[str] = Counter()
            max_mono = 0.0
            for record in r.records():
                counts[record.qualname] += 1
                if record.t_mono > max_mono:
                    max_mono = record.t_mono

        print(f"Trace file: {args.trace_path}")
        print(f"  level:                {header.level}")
        print(f"  format_version:       {header.format_version}")
        print(f"  trace_schema_version: {header.trace_schema_version}")
        print(f"  duration:             {max_mono:.3f}s")
        print(f"  sm_config_digest:     {header.sm_config_digest or '(none)'}")
        print(f"  total_records:        {sum(counts.values())}")
        if counts:
            print("  ops:")
            for qn in sorted(counts):
                print(f"    {qn}: {counts[qn]}")
        else:
            print("  ops: (none)")
