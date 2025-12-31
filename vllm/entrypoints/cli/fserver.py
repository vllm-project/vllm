# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM AFD FFN Server CLI command."""

import argparse

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.afd_ffn_server import main
from vllm.entrypoints.cli.types import CLISubcommand


class FServerCommand(CLISubcommand):
    """Command for running vLLM AFD FFN Server."""

    def __init__(self):
        self.name = "fserver"
        super().__init__()

    def subparser_init(self, subparsers):
        """Initialize the fserver subparser."""
        parser = subparsers.add_parser(
            self.name,
            help="Start vLLM AFD FFN Server",
            description="Start vLLM AFD FFN Server for Attention-FFN Disaggregation",
            usage="vllm fserver MODEL --afd-config CONFIG [options]",
        )

        # Add model as positional argument (like vllm serve)
        parser.add_argument("model", type=str, help="Model name or path")

        # Use AsyncEngineArgs to add all vLLM engine arguments
        parser = AsyncEngineArgs.add_cli_args(parser)

        return parser

    def validate(self, args: argparse.Namespace) -> None:
        """Validate arguments for fserver command."""
        # Validate that afd_config is provided
        if not hasattr(args, "afd_config") or not args.afd_config:
            raise ValueError("--afd-config is required for FFN server")

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        """Run the fserver command."""
        # Call the main function from afd_ffn_server directly with parsed args
        main(args)


def cmd_init() -> list[CLISubcommand]:
    """Initialize fserver command."""
    return [FServerCommand()]
