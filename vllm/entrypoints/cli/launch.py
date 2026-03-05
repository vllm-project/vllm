# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import asyncio
import signal
import time

import grpc
import uvloop
from grpc_reflection.v1alpha import reflection
from starlette.datastructures import State

from vllm.config import VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.openai.api_server import (
    build_and_serve,
    init_app_state,
    setup_server,
)
from vllm.entrypoints.openai.cli_args import (
    make_arg_parser,
    validate_parsed_serve_args,
)
from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG
from vllm.grpc import (  # type: ignore[attr-defined]
    vllm_render_pb2,
    vllm_render_pb2_grpc,
)
from vllm.grpc.render_servicer import RenderGrpcServicer
from vllm.logger import init_logger
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.engine.launch import LaunchEngineClient

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

    @classmethod
    def add_cli_args(cls, parser: FlexibleArgumentParser) -> None:
        super().add_cli_args(parser)
        parser.add_argument(
            "--server",
            choices=["http", "grpc"],
            default="http",
            help="Server protocol to use (default: http).",
        )

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        server = getattr(args, "server", "http")
        if server == "http":
            uvloop.run(run_launch_http(args))
        else:
            uvloop.run(run_launch_grpc(args))


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


async def run_launch_grpc(args: argparse.Namespace) -> None:
    """Run the render serving layer with gRPC (no GPU inference)."""
    # 1. Create LaunchEngineClient (no GPU)
    engine_args = AsyncEngineArgs.from_cli_args(args)
    model_config = engine_args.create_model_config()
    vllm_config = VllmConfig(model_config=model_config)
    engine_client = LaunchEngineClient.from_vllm_config(vllm_config)

    # 2. Initialize app state (reuses init_app_state like HTTP path)
    state = State()
    await init_app_state(engine_client, state, args, ("generate",))

    # 3. Create servicer and gRPC server
    start_time = time.time()
    servicer = RenderGrpcServicer(state, start_time)
    server = grpc.aio.server(
        options=[
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
        ],
    )
    vllm_render_pb2_grpc.add_VllmRenderServicer_to_server(servicer, server)

    # 4. Enable reflection
    service_names = (
        vllm_render_pb2.DESCRIPTOR.services_by_name["VllmRender"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(service_names, server)

    # 5. Bind and start
    host = args.host or "0.0.0.0"
    port = args.port
    address = f"{host}:{port}"
    server.add_insecure_port(address)
    await server.start()
    logger.info("gRPC render server started on %s", address)

    # 6. Wait for shutdown signal
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def signal_handler():
        logger.info("Received shutdown signal")
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await stop_event.wait()
    finally:
        await server.stop(grace=5.0)
        logger.info("gRPC render server stopped")


async def run_launch_http(args: argparse.Namespace) -> None:
    """Run the online serving layer with FastAPI (no GPU inference)."""
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
