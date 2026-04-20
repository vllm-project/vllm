# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`vllm snapshot` CLI subcommand glue.

Delegates to vllm.snapshot.cli for the actual work; this file only
provides the `CLISubcommand` wrapper so the subcommand shows up in
`vllm --help` alongside serve / chat / etc.
"""

from __future__ import annotations

import argparse

from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG


class SnapshotSubcommand(CLISubcommand):
    name = "snapshot"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        # Re-dispatch through the snapshot module's own parser so we
        # can use its richer subcommand tree without duplicating code.
        import sys

        from vllm.snapshot.cli import main as snapshot_main

        # Build fresh argv for the snapshot module
        inner_argv: list[str] = []
        sub = getattr(args, "snapshot_sub", None)
        if sub:
            inner_argv.append(sub)
        for flag in ("force", "dry_run", "all"):
            if getattr(args, flag, False):
                inner_argv.append(f"--{flag.replace('_', '-')}")
        if getattr(args, "key", None):
            inner_argv.extend(["--key", args.key])
        rc = snapshot_main(inner_argv)
        sys.exit(rc)

    @staticmethod
    def subparser_init(
        subparsers: argparse._SubParsersAction,
    ) -> argparse.ArgumentParser:
        p = subparsers.add_parser(
            "snapshot",
            help="Manage CRIU + cuda-checkpoint startup snapshots",
            description="Per-(vllm, python, torch, cuda-driver, gpu-arch) "
            "snapshot that skips Python imports + CUDA context init on "
            "subsequent `vllm serve` invocations.",
            epilog=VLLM_SUBCMD_PARSER_EPILOG.format(subcmd="snapshot"),
        )
        sub = p.add_subparsers(dest="snapshot_sub", required=True)
        c = sub.add_parser("create", help="Create a snapshot for current version")
        c.add_argument("--force", action="store_true",
                       help="Rebuild even if snapshot already exists")
        c.add_argument("--dry-run", action="store_true",
                       help="Skip CUDA + binary invocations; log only")
        sub.add_parser("list", help="List existing snapshots")
        d = sub.add_parser("drop", help="Remove snapshot(s)")
        d.add_argument("--all", action="store_true", help="Remove all")
        d.add_argument("--key", help="Remove specific key digest")
        return p


def cmd_init() -> list[CLISubcommand]:
    return [SnapshotSubcommand()]
