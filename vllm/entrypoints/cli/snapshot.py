# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import typing

from vllm.entrypoints.cli.types import CLISubcommand

if typing.TYPE_CHECKING:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
else:
    FlexibleArgumentParser = argparse.ArgumentParser


def _run_create(args: argparse.Namespace) -> None:
    from vllm.entrypoints.snapshot import create_snapshot

    try:
        create_snapshot(force=args.force, dry_run=args.dry_run)
    except (RuntimeError, ValueError, OSError) as error:
        raise SystemExit(f"vllm snapshot create failed: {error}") from error


class SnapshotSubcommand(CLISubcommand):
    """The `snapshot` subcommand: create an imports snapshot for serve."""

    name = "snapshot"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        args.snapshot_command(args)

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        snapshot_parser = subparsers.add_parser(
            "snapshot",
            help="Create and restore CRIU imports snapshots.",
            description="Create and restore CRIU imports snapshots.",
            usage="vllm snapshot <command> [options]",
        )
        commands = snapshot_parser.add_subparsers(
            required=True, dest="snapshot_command_name"
        )
        create = commands.add_parser(
            "create",
            help="Create an imports snapshot of the serve envelope.",
            description="Create an imports snapshot of the serve envelope.",
            usage="vllm snapshot create [options]",
        )
        create.add_argument(
            "--dry-run",
            action="store_true",
            help="Run preflight and print the dump command without dumping.",
        )
        create.add_argument(
            "--force",
            action="store_true",
            help="Replace an existing valid snapshot for this environment.",
        )
        create.set_defaults(snapshot_command=_run_create)
        return snapshot_parser


def cmd_init() -> list[CLISubcommand]:
    return [SnapshotSubcommand()]
