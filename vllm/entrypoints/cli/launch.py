# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse

import uvloop

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.openai.api_server import (
    build_and_serve,
    setup_server,
)
from vllm.entrypoints.openai.cli_args import (
    make_arg_parser,
    validate_parsed_serve_args,
)
from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG
from vllm.logger import init_logger
from vllm.utils.argparse_utils import FlexibleArgumentParser

logger = init_logger(__name__)

DESCRIPTION = """Launch individual vLLM components.

Search by using: `--help=<ConfigGroup>` to explore options by section (e.g.,
--help=ModelConfig, --help=Frontend)
  Use `--help=all` to show all available flags at once.
"""


class LaunchSubcommand(CLISubcommand):
    """The `launch` subcommand for the vLLM CLI."""

    name = "launch"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        if hasattr(args, "model_tag") and args.model_tag is not None:
            args.model = args.model_tag

        uvloop.run(run_launch_fastapi(args))

    def validate(self, args: argparse.Namespace) -> None:
        validate_parsed_serve_args(args)

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        launch_parser = subparsers.add_parser(
            self.name,
            help="Launch individual vLLM components.",
            description=DESCRIPTION,
            usage="vllm launch [model_tag] [options]",
        )

        launch_parser = make_arg_parser(launch_parser)
        launch_parser.epilog = VLLM_SUBCMD_PARSER_EPILOG.format(subcmd=self.name)
        return launch_parser


def cmd_init() -> list[CLISubcommand]:
    return [LaunchSubcommand()]


async def run_launch_fastapi(args: argparse.Namespace) -> None:
    """Run the online serving layer with FastAPI (no GPU inference)."""
    from vllm.config import VllmConfig
    from vllm.v1.engine.launch import LaunchEngineClient

    # 1. Socket binding
    listen_address, sock = setup_server(args)

    # 2. Create LaunchEngineClient (no GPU)
    engine_args = AsyncEngineArgs.from_cli_args(args)
    model_config = engine_args.create_model_config()
    vllm_config = VllmConfig(model_config=model_config)
    engine_client = LaunchEngineClient.from_vllm_config(vllm_config)

    # 3. Build app, initialize state, and start serving
    shutdown_task = await build_and_serve(engine_client, listen_address, sock, args)
    try:
        await shutdown_task
    finally:
        sock.close()
