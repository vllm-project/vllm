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

DESCRIPTION = "Launch individual vLLM components."


class LaunchSubcommand(CLISubcommand):
    """The `launch` subcommand for the vLLM CLI.

    Uses nested sub-subcommands so each component can define its own
    arguments independently (e.g. ``vllm launch render``).
    """

    name = "launch"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        args.dispatch_function(args)

    def validate(self, args: argparse.Namespace) -> None:
        validate_parsed_serve_args(args)

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        launch_parser = subparsers.add_parser(
            self.name,
            help=DESCRIPTION,
            description=DESCRIPTION,
            usage=f"vllm {self.name} <component> [options]",
        )
        launch_subparsers = launch_parser.add_subparsers(
            required=True, dest="launch_component"
        )

        # -- vllm launch render --
        render_parser = launch_subparsers.add_parser(
            "render",
            help="Launch a GPU-less rendering server "
            "(preprocessing and postprocessing only).",
            description="Launch a GPU-less rendering server "
            "(preprocessing and postprocessing only).",
            usage=f"vllm {self.name} render [model_tag] [options]",
        )
        render_parser.set_defaults(dispatch_function=_cmd_launch_render)
        render_parser = make_arg_parser(render_parser)
        render_parser.epilog = VLLM_SUBCMD_PARSER_EPILOG.format(
            subcmd=f"{self.name} render"
        )

        return launch_parser


def _cmd_launch_render(args: argparse.Namespace) -> None:
    """Dispatch for ``vllm launch render``."""
    if hasattr(args, "model_tag") and args.model_tag is not None:
        args.model = args.model_tag

    uvloop.run(run_launch_fastapi(args))


def cmd_init() -> list[CLISubcommand]:
    return [LaunchSubcommand()]


async def run_launch_fastapi(args: argparse.Namespace) -> None:
    """Run the online serving layer with FastAPI (no GPU inference)."""
    from vllm.config import VllmConfig
    from vllm.v1.engine.async_renderer import AsyncRenderer

    # 1. Socket binding
    listen_address, sock = setup_server(args)

    # 2. Create RendererClient (CPU-only, no engine needed)
    engine_args = AsyncEngineArgs.from_cli_args(args)
    model_config = engine_args.create_model_config()
    vllm_config = VllmConfig(model_config=model_config)
    renderer_client = AsyncRenderer.from_vllm_config(vllm_config=vllm_config)

    # 3. Build app, initialize state, and start serving
    shutdown_task = await build_and_serve(
        renderer_client=renderer_client,
        engine_client=None,
        listen_address=listen_address,
        sock=sock,
        args=args,
    )
    try:
        await shutdown_task
    finally:
        renderer_client.shutdown()
        sock.close()
