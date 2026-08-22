# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import os
import platform
import typing
from pathlib import Path

from vllm.entrypoints.cli.types import CLISubcommand

if typing.TYPE_CHECKING:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
else:
    FlexibleArgumentParser = argparse.ArgumentParser


def validate_create_args(args: argparse.Namespace) -> None:
    model = str(getattr(args, "model_tag", None) or args.model)
    revision = str(getattr(args, "revision", None) or "")
    if (
        Path(model).exists()
        or len(revision) != 40
        or any(character not in "0123456789abcdefABCDEF" for character in revision)
    ):
        raise ValueError(
            "snapshot create requires an immutable remote model and "
            "40-character --revision"
        )
    parallel_sizes = (
        ("tensor parallel", args.tensor_parallel_size),
        ("pipeline parallel", args.pipeline_parallel_size),
        ("data parallel", args.data_parallel_size),
        ("prefill context parallel", args.prefill_context_parallel_size),
    )
    for name, size in parallel_sizes:
        if size != 1:
            raise ValueError(f"snapshot create currently requires {name} size 1")
    unsupported_frontend = (
        args.api_server_count not in (None, 1)
        or args.grpc
        or args.headless
        or args.api_key
        or os.environ.get("VLLM_API_KEY")
        or args.uds
        or args.middleware
    )
    if unsupported_frontend or any(
        (
            args.ssl_keyfile,
            args.ssl_certfile,
            args.ssl_ca_certs,
            args.enable_ssl_refresh,
            args.ssl_ciphers,
        )
    ):
        raise ValueError(
            "snapshot create requires one unauthenticated plaintext HTTP server"
        )
    if args.logprobs_mode in {"raw_logits", "processed_logits"}:
        raise ValueError("snapshot canary requires a log-probability mode")
    if getattr(args, "speculative_config", None) is not None:
        raise ValueError("snapshot create does not support speculative decoding")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("snapshot create requires Linux x86_64")


def run_create(args: argparse.Namespace) -> None:
    validate_create_args(args)
    from vllm.snapshot.controller import create_snapshot

    create_snapshot(args)


def run_inspect(args: argparse.Namespace) -> None:
    from vllm.snapshot.manifest import inspect_snapshot

    print(json.dumps(inspect_snapshot(Path(args.snapshot_dir)), indent=2))


def run_restore(args: argparse.Namespace) -> None:
    from vllm.snapshot.controller import restore_snapshot

    restore_snapshot(args)


class SnapshotSubcommand(CLISubcommand):
    """The `snapshot` subcommand for the vLLM CLI."""

    name = "snapshot"

    def __init__(self, *, create_requested: bool = False) -> None:
        self.create_requested = create_requested

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
        if self.create_requested:
            from vllm.entrypoints.openai.cli_args import make_arg_parser

            create_parser = make_arg_parser(create_parser)
        else:
            create_parser.add_argument("model_tag", nargs="?")
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


def cmd_init(*, create_requested: bool = False) -> list[CLISubcommand]:
    return [SnapshotSubcommand(create_requested=create_requested)]
