# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import asyncio
import signal
import socket
import time
from argparse import Namespace

import grpc
from grpc_reflection.v1alpha import reflection
from starlette.datastructures import State

from vllm import AsyncEngineArgs, envs
from vllm.config import VllmConfig
from vllm.entrypoints.grpc import (  # type: ignore[attr-defined]
    vllm_render_pb2,
    vllm_render_pb2_grpc,
)
from vllm.entrypoints.grpc.auth import build_auth_interceptors
from vllm.entrypoints.grpc.render_servicer import RenderGrpcServicer
from vllm.logger import init_logger

from ..app import build_app
from ..launcher import serve_http, setup_server
from ..utils.server_utils import get_uvicorn_log_config
from .app_state import init_render_app_state

logger = init_logger("vllm.entrypoints.launchers.render.entry")


async def build_and_serve_renderer(
    vllm_config: VllmConfig,
    listen_address: str,
    sock: socket.socket,
    args: Namespace,
    **uvicorn_kwargs,
) -> asyncio.Task:
    """Build FastAPI app for a CPU-only render server, initialize state, and
    start serving.

    Returns the shutdown task for the caller to await.
    """

    # Get uvicorn log config (from file or with endpoint filter)
    log_config = get_uvicorn_log_config(args)
    if log_config is not None:
        uvicorn_kwargs["log_config"] = log_config

    app = build_app(args, ("render",))
    await init_render_app_state(vllm_config, app.state, args)

    logger.info("Starting vLLM server on %s", listen_address)

    return await serve_http(
        app,
        sock=sock,
        enable_ssl_refresh=args.enable_ssl_refresh,
        host=args.host,
        port=args.port,
        log_level=args.uvicorn_log_level,
        # NOTE: When the 'disable_uvicorn_access_log' value is True,
        # no access log will be output.
        access_log=not args.disable_uvicorn_access_log,
        timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
        ssl_ciphers=args.ssl_ciphers,
        h11_max_incomplete_event_size=args.h11_max_incomplete_event_size,
        h11_max_header_count=args.h11_max_header_count,
        **uvicorn_kwargs,
    )


def _prepare_render_model_config(args: argparse.Namespace) -> VllmConfig:
    """Build a VllmConfig suitable for a GPU-less render server.

    Render servers preprocess data only — no inference, no quantized kernels,
    and no KV cache. Clear quantization so VllmConfig skips quant
    dtype/capability validation, and zero VLLM_CPU_KVCACHE_SPACE to suppress
    the spurious CPU KV cache space warning from
    CpuPlatform.check_and_update_config.
    """
    engine_args = AsyncEngineArgs.from_cli_args(args)
    model_config = engine_args.create_model_config()
    model_config.quantization = None
    envs.VLLM_CPU_KVCACHE_SPACE = 0
    return VllmConfig(model_config=model_config)


async def run_launch_fastapi(args: argparse.Namespace) -> None:
    """Run the online serving layer with FastAPI (no GPU inference)."""

    # Interrupt initialization if SIGTERM arrives before uvicorn installs
    # its own signal handlers. Once uvicorn is running it replaces this.
    def _interrupt_init(*_) -> None:
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, _interrupt_init)

    # 1. Socket binding
    listen_address, sock = setup_server(args, reuse_port=False)

    # 2. Build and serve the API server
    vllm_config = _prepare_render_model_config(args)
    shutdown_task = await build_and_serve_renderer(
        vllm_config, listen_address, sock, args
    )
    try:
        await shutdown_task
    finally:
        sock.close()


async def run_launch_grpc(args: argparse.Namespace) -> None:
    """Run the render serving layer with gRPC (no GPU inference)."""
    # 1. Create VllmConfig
    vllm_config = _prepare_render_model_config(args)

    # 2. Initialize app state
    state = State()
    await init_render_app_state(vllm_config, state, args)

    # 3. Create servicer and gRPC server
    start_time = time.time()
    servicer = RenderGrpcServicer(state, start_time)
    server = grpc.aio.server(
        # Enforce the same --api-key / VLLM_API_KEY auth as the HTTP server
        # (no-op when no key is configured).
        interceptors=build_auth_interceptors(args),
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


if __name__ == "__main__":
    import uvloop

    from vllm.entrypoints.openai.cli_args import (
        make_arg_parser,
        validate_parsed_serve_args,
    )
    from vllm.entrypoints.serve.utils.api_utils import cli_env_setup
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    cli_env_setup()
    parser = FlexibleArgumentParser(
        description="Starts a GPU-less rendering server "
        "for preprocessing and postprocessing only"
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    uvloop.run(run_launch_fastapi(args))
