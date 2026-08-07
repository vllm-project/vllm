# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse


def restore_snapshot(args: argparse.Namespace) -> None:
    from vllm_cli.snapshot.controller import restore_snapshot as restore

    restore(args)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="vllm snapshot")
    actions = parser.add_subparsers(required=True, dest="snapshot_action")
    restore_parser = actions.add_parser(
        "restore", help="Restore a same-host TP1 snapshot."
    )
    restore_parser.add_argument("snapshot_dir")
    restore_parser.add_argument("--host", default=None)
    restore_parser.add_argument("--port", type=int, default=8000)
    restore_parser.set_defaults(snapshot_dispatch=restore_snapshot)
    args = parser.parse_args(argv)
    args.snapshot_dispatch(args)
