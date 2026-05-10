# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse

import uvloop

from vllm import envs
from vllm.config import VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.openai.api_server import (
    build_and_serve_renderer,
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


class LaunchSubcommandBase(CLISubcommand):
    """The base class of subcommands for `vllm launch`."""

    help: str

    @classmethod
    def add_cli_args(cls, parser: FlexibleArgumentParser) -> None:
        """Add the CLI arguments to the parser.

        By default, adds the standard vLLM serving arguments.
        Subclasses can override to add component-specific arguments.
        """
        make_arg_parser(parser)

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        raise NotImplementedError


class RenderSubcommand(LaunchSubcommandBase):
    """The `render` subcommand for `vllm launch`."""

    name = "render"
    help = "Launch a GPU-less rendering server (preprocessing and postprocessing only)."

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        uvloop.run(run_launch_fastapi(args))


class LaunchSubcommand(CLISubcommand):
    """The `launch` subcommand for the vLLM CLI.

    Uses nested sub-subcommands so each component can define its own
    arguments independently (e.g. ``vllm launch render``).
    """

    name = "launch"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        if hasattr(args, "model_tag") and args.model_tag is not None:
            args.model = args.model_tag

        args.launch_command(args)

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

        for cmd_cls in LaunchSubcommandBase.__subclasses__():
            cmd_subparser = launch_subparsers.add_parser(
                cmd_cls.name,
                help=cmd_cls.help,
                description=cmd_cls.help,
                usage=f"vllm {self.name} {cmd_cls.name} [options]",
            )
            cmd_subparser.set_defaults(launch_command=cmd_cls.cmd)
            cmd_cls.add_cli_args(cmd_subparser)
            cmd_subparser.epilog = VLLM_SUBCMD_PARSER_EPILOG.format(
                subcmd=f"{self.name} {cmd_cls.name}"
            )

        return launch_parser


def cmd_init() -> list[CLISubcommand]:
    return [LaunchSubcommand()]


async def run_launch_fastapi(args: argparse.Namespace) -> None:
    """Run the online serving layer with FastAPI (no GPU inference)."""
    # 1. Socket binding
    listen_address, sock = setup_server(args)

    # 2. Build and serve the API server
    engine_args = AsyncEngineArgs.from_cli_args(args)
    model_config = engine_args.create_model_config()

    # Render servers preprocess data only — no inference, no quantized kernels.
    # Clear quantization so VllmConfig skips quant dtype/capability validation.
    model_config.quantization = None

    # Render servers never allocate KV cache; suppress the spurious CPU KV
    # cache space warning from CpuPlatform.check_and_update_config.
    envs.VLLM_CPU_KVCACHE_SPACE = 0

    vllm_config = VllmConfig(model_config=model_config)
    shutdown_task = await build_and_serve_renderer(
        vllm_config, listen_address, sock, args
    )
    try:
        await shutdown_task
    finally:
        sock.close()
