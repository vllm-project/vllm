# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import asyncio
import signal
import socket
from argparse import Namespace

from vllm import AsyncEngineArgs, envs
from vllm.config import VllmConfig
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
