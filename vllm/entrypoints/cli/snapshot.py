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
    include_model_state = bool(getattr(args, "include_model_state", False))
    model = str(getattr(args, "model_tag", None) or args.model)
    revision = str(getattr(args, "revision", None) or "")
    if not include_model_state and (
        Path(model).exists()
        or len(revision) != 40
        or any(character not in "0123456789abcdefABCDEF" for character in revision)
    ):
        raise ValueError(
            "snapshot create compact mode requires an immutable remote model "
            "and 40-character --revision; use --include-model-state for local "
            "or mutable sources"
        )
    if args.tensor_parallel_size != 1:
        raise ValueError("snapshot create currently requires tensor parallel size 1")
    if args.pipeline_parallel_size != 1:
        raise ValueError("snapshot create currently requires pipeline parallel size 1")
    if args.data_parallel_size != 1:
        raise ValueError("snapshot create currently requires data parallel size 1")
    if args.prefill_context_parallel_size != 1:
        raise ValueError(
            "snapshot create currently requires prefill context parallel size 1"
        )
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
    if args.logprobs_mode in {"raw_logits", "processed_logits"}:
        raise ValueError("snapshot canary requires a log-probability mode")
    if (
        not include_model_state
        and getattr(args, "speculative_config", None) is not None
    ):
        raise ValueError(
            "snapshot create with speculative decoding currently requires "
            "--include-model-state"
        )
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


def _add_restore_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("snapshot_dir")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=8000)


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
        create_parser.add_argument(
            "--include-model-state",
            action="store_true",
            help=(
                "Include initialized model and KV state in the snapshot instead "
                "of releasing reloadable allocations before capture."
            ),
        )
        create_parser.set_defaults(snapshot_dispatch=run_create)

        inspect_parser = actions.add_parser(
            "inspect", help="Inspect a snapshot without restoring it."
        )
        inspect_parser.add_argument("snapshot_dir")
        inspect_parser.set_defaults(snapshot_dispatch=run_inspect)

        restore_parser = actions.add_parser(
            "restore", help="Restore a same-host TP1 snapshot."
        )
        _add_restore_arguments(restore_parser)
        restore_parser.set_defaults(snapshot_dispatch=run_restore)

        return parser


def cmd_init(*, create_requested: bool = False) -> list[CLISubcommand]:
    return [SnapshotSubcommand(create_requested=create_requested)]
