# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import os
import platform
import typing
from pathlib import Path

from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.openai.cli_args import make_arg_parser

if typing.TYPE_CHECKING:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
else:
    FlexibleArgumentParser = argparse.ArgumentParser


def validate_create_args(args: argparse.Namespace) -> None:
    if not args.revision:
        raise ValueError("snapshot create requires an explicit --revision")
    if args.tensor_parallel_size != 1:
        raise ValueError("snapshot create currently requires tensor parallel size 1")
    if args.pipeline_parallel_size != 1:
        raise ValueError("snapshot create currently requires pipeline parallel size 1")
    if args.data_parallel_size != 1:
        raise ValueError("snapshot create currently requires data parallel size 1")
    if args.api_server_count not in (None, 1):
        raise ValueError("snapshot create currently supports one API server")
    if args.grpc:
        raise ValueError("snapshot create currently supports the HTTP frontend only")
    if args.headless:
        raise ValueError("snapshot create requires an API frontend")
    if args.api_key or os.environ.get("VLLM_API_KEY"):
        raise ValueError("snapshot create does not support API authentication")
    if args.uds:
        raise ValueError("snapshot create does not support Unix domain sockets")
    if any(
        (
            args.ssl_keyfile,
            args.ssl_certfile,
            args.ssl_ca_certs,
            args.enable_ssl_refresh,
            args.ssl_ciphers,
        )
    ):
        raise ValueError("snapshot create does not support TLS")
    if args.middleware:
        raise ValueError("snapshot create does not support custom middleware")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("snapshot create requires Linux x86_64")


def run_create(args: argparse.Namespace) -> None:
    validate_create_args(args)
    from vllm_cli.snapshot.controller import create_snapshot

    create_snapshot(args)


def run_inspect(args: argparse.Namespace) -> None:
    from vllm_cli.snapshot.manifest import inspect_snapshot

    print(json.dumps(inspect_snapshot(Path(args.snapshot_dir)), indent=2))


def run_restore(args: argparse.Namespace) -> None:
    from vllm_cli.snapshot.controller import restore_snapshot

    restore_snapshot(args)


class SnapshotSubcommand(CLISubcommand):
    """The `snapshot` subcommand for the vLLM CLI."""

    name = "snapshot"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        args.snapshot_dispatch(args)

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        parser = subparsers.add_parser(
            self.name,
            help="Create, inspect, or restore an initialized vLLM snapshot.",
            usage="vllm snapshot <create|inspect|restore> [options]",
        )
        actions = parser.add_subparsers(required=True, dest="snapshot_action")

        create_parser = actions.add_parser(
            "create", help="Create a snapshot from an initialized TP1 engine."
        )
        create_parser = make_arg_parser(create_parser)
        create_parser.add_argument("--snapshot-dir", required=True)
        create_parser.set_defaults(snapshot_dispatch=run_create)

        inspect_parser = actions.add_parser(
            "inspect", help="Inspect a snapshot without restoring it."
        )
        inspect_parser.add_argument("snapshot_dir")
        inspect_parser.set_defaults(snapshot_dispatch=run_inspect)

        restore_parser = actions.add_parser(
            "restore", help="Restore a same-host TP1 snapshot."
        )
        restore_parser.add_argument("snapshot_dir")
        restore_parser.add_argument("--host", default=None)
        restore_parser.add_argument("--port", type=int, default=8000)
        restore_parser.set_defaults(snapshot_dispatch=run_restore)

        return parser


def cmd_init() -> list[CLISubcommand]:
    return [SnapshotSubcommand()]
