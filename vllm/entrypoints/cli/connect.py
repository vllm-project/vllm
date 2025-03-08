# SPDX-License-Identifier: Apache-2.0

import argparse

import uvloop

from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.disagg_connector import run_disagg_connector
from vllm.utils import FlexibleArgumentParser


class ConnectSubcommand(CLISubcommand):
    """The `connect` subcommand for the vLLM CLI. """

    def __init__(self):
        self.name = "connect"
        super().__init__()

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        uvloop.run(run_disagg_connector(args))

    def validate(self, args: argparse.Namespace) -> None:
        validate_connect_parsed_args(args)

    def subparser_init(
            self,
            subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        connect_parser = subparsers.add_parser(
            "connect",
            help=
            "Start the vLLM OpenAI Compatible API Server which connect to other"
            "servers disaggreate prefill and decode",
            usage="vllm connect [options]")

        return make_connect_arg_parser(connect_parser)


def cmd_init() -> list[CLISubcommand]:
    return [ConnectSubcommand()]


def make_connect_arg_parser(
        parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
    parser.add_argument("--port",
                        type=int,
                        default=7001,
                        help="The fastapi server port default 7001")
    # support ipc only now, support tcp later(with auth)
    parser.add_argument(
        "--protocol",
        type=str,
        choices=["ipc"],
        default="ipc",
        help="The zmq socket addr protocol IPC (Inter-Process Communication)")
    # security concern only support ipc now
    parser.add_argument("--prefill-addr",
                        type=str,
                        required=True,
                        help="The zmq ipc prefill address")
    parser.add_argument("--decode-addr",
                        type=str,
                        required=True,
                        help="The zmq ipc decode address")

    return parser


def validate_connect_parsed_args(args: argparse.Namespace):
    """Quick checks for connect args that raise prior to loading."""
    if hasattr(args, "subparser") and args.subparser != "connect":
        return
