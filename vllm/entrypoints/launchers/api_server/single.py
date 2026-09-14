# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-process vLLM API server launcher.

Top-level structure
-------------------
::

    main()  # CLI entrypoint: parse, validate, dispatch.
    run_single_api_server()  # Enter uvloop and run the lifecycle.
    run_server_with_exit_stack()  # Wrap run_server with an ExitStack and
    # uniform error logging.
    run_server()  # The real orchestration (steps below).

``run_server`` flow
-------------------
1. Process-level setup (one-shot side effects).
2. Bind the listen socket BEFORE the engine starts, so that port conflicts
   fail fast, and we avoid races with ray.
3. Build the ``AsyncLLM`` engine client from the resolved engine config.
4. Build the FastAPI app and initialise its state.
5. Build the uvicorn server (``NoSignalServer``; we handle signals here).
6. Install SIGINT/SIGTERM handlers (this replaces the pre-init handler from
   step 1) and start the graceful shutdown task.
7. Start serving, plus the watchdog and the optional SSL refresher.
8. Await ``server_task``; log graceful stop, then let the ``ExitStack``
   unwind all resources in LIFO order on any exit path.

All long-lived resources (socket, engine, tasks, SSL refresher, signal
handlers) are registered on the ``ExitStack`` owned by
``run_server_with_exit_stack`` so they are released in reverse order when
the server exits for any reason.
"""

import argparse
import asyncio
import contextlib
import signal
from typing import Any

import uvicorn
import uvloop
from fastapi import FastAPI

from vllm import AsyncEngineArgs, envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext
from vllm.utils.system_utils import decorate_logs, set_ulimit
from vllm.v1.engine.async_llm import AsyncLLM

from ..app import build_app
from ..launcher import (
    NoSignalServer,
    validate_api_server_args,
    watchdog_loop,
)
from ..utils.constants import (
    H11_MAX_HEADER_COUNT_DEFAULT,
    H11_MAX_INCOMPLETE_EVENT_SIZE_DEFAULT,
)
from ..utils.listen import cleanup_listen_socket, setup_listen_address
from ..utils.server_utils import get_uvicorn_log_config
from ..utils.setup_utils import (
    init_parser_plugin,
    log_non_default_args,
    log_routes,
    log_version_and_model,
    setup_forkserver,
    setup_interrupt_handler,
)
from ..utils.ssl import start_ssl_refresher_if_needed
from .app_state import init_app_state

logger = init_logger("vllm.entrypoints.launchers.api_server.single")


async def build_async_engine_client(
    vllm_config: VllmConfig,
    engine_args: AsyncEngineArgs,
    *,
    client_config: dict[str, Any] | None = None,
    usage_context: UsageContext = UsageContext.OPENAI_API_SERVER,
) -> AsyncLLM:
    """Create the AsyncLLM engine client from the resolved config."""
    setup_forkserver()

    # Don't mutate the input client_config.
    client_config = dict(client_config) if client_config else {}
    client_count = client_config.pop("client_count", 1)
    client_index = client_config.pop("client_index", 0)

    if client_count != 1 or client_index != 0:
        engine_args._api_process_count = client_count
        engine_args._api_process_rank = client_index

    return AsyncLLM.from_vllm_config(
        vllm_config=vllm_config,
        usage_context=usage_context,
        enable_log_requests=engine_args.enable_log_requests,
        aggregate_engine_logging=engine_args.aggregate_engine_logging,
        disable_log_stats=engine_args.disable_log_stats,
        client_addresses=client_config,
        client_count=client_count,
        client_index=client_index,
    )


async def init_app(args: argparse.Namespace, engine_client: AsyncLLM) -> FastAPI:
    """Build the FastAPI app and initialise its state."""
    supported_tasks = await engine_client.get_supported_tasks()
    model_config = engine_client.model_config
    logger.info("Supported tasks: %s", supported_tasks)
    app = build_app(args, supported_tasks, model_config)
    await init_app_state(engine_client, app.state, args, supported_tasks)
    log_routes(app)
    return app


def init_uvicorn(
    args: argparse.Namespace,
    app: FastAPI,
    uvicorn_kwargs: dict[str, Any] | None = None,
):
    """Build a uvicorn.Config for the given FastAPI app.

    The input ``uvicorn_kwargs`` is copied before mutation; the caller's
    dict is left untouched.
    """
    # Copy so we never mutate the caller's dict.
    uvicorn_kwargs = dict(uvicorn_kwargs) if uvicorn_kwargs else {}

    log_config = get_uvicorn_log_config(args)
    if log_config is not None:
        uvicorn_kwargs["log_config"] = log_config

    # Extract header limit options if present.
    h11_max_incomplete_event_size = uvicorn_kwargs.pop(
        "h11_max_incomplete_event_size", None
    )
    h11_max_header_count = uvicorn_kwargs.pop("h11_max_header_count", None)

    if h11_max_incomplete_event_size is None:
        h11_max_incomplete_event_size = H11_MAX_INCOMPLETE_EVENT_SIZE_DEFAULT
    if h11_max_header_count is None:
        h11_max_header_count = H11_MAX_HEADER_COUNT_DEFAULT

    config = uvicorn.Config(app, **uvicorn_kwargs)
    config.h11_max_incomplete_event_size = h11_max_incomplete_event_size
    config.h11_max_header_count = h11_max_header_count
    return config


def _install_signal_handlers(
    loop: asyncio.AbstractEventLoop,
    server: NoSignalServer,
    exit_stack: contextlib.ExitStack,
) -> None:
    """Install SIGINT/SIGTERM handlers and start the shutdown task."""
    shutdown_event = asyncio.Event()
    received_signum: int | None = None

    def _on_signal(signum: int) -> None:
        nonlocal received_signum
        if received_signum is None:
            received_signum = signum
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        exit_stack.callback(loop.remove_signal_handler, sig)
        loop.add_signal_handler(sig, _on_signal, sig)

    async def handle_shutdown() -> None:
        await shutdown_event.wait()
        if received_signum is not None:
            signame = signal.Signals(received_signum).name
            logger.info_once(
                "[shutdown] API server: shutdown triggered by signal %s (%d); "
                "signalling HTTP server shutdown",
                signame,
                received_signum,
            )
        else:
            # Defensive: shutdown_event is only set from the handlers above,
            # so this branch should be unreachable.
            logger.info_once(
                "[shutdown] API server: shutdown triggered; "
                "signalling HTTP server shutdown"
            )
        server.should_exit = True

    shutdown_task = loop.create_task(handle_shutdown())
    exit_stack.callback(shutdown_task.cancel)


async def run_server(
    args: argparse.Namespace,
    exit_stack: contextlib.ExitStack,
    *,
    client_config: dict[str, Any] | None = None,
    usage_context: UsageContext = UsageContext.OPENAI_API_SERVER,
    vllm_version=None,
):
    """Orchestrate engine build, app build, and HTTP serving.

    See the module docstring for the full step-by-step flow. All long-lived
    resources are registered on ``exit_stack`` and released in LIFO order.
    """

    # --- 1. Process-level setup (one-shot side effects) --------------------
    setup_interrupt_handler()
    set_ulimit()

    decorate_logs("APIServer", skip_if_decorated=True)
    log_version_and_model(logger, vllm_version, args.model)
    log_non_default_args(args)

    validate_api_server_args(args)
    init_parser_plugin(args)

    # --- 2. Bind the socket before starting the engine ---------------------
    # Port conflicts raise a clear OSError from setup_listen_address.
    listen_address, sock = setup_listen_address(args, reuse_port=False)
    exit_stack.callback(cleanup_listen_socket, sock, args.uds or None)

    # --- 3. Build the AsyncLLM engine client -------------------------------
    engine_args = AsyncEngineArgs.from_cli_args(args)
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)

    engine_client = await build_async_engine_client(
        vllm_config=vllm_config,
        engine_args=engine_args,
        client_config=client_config,
        usage_context=usage_context,
    )
    # NOTE: KeyboardInterrupt during build_async_engine_client is not
    # currently propagated. Ideally build_async_engine_client should be
    # split into "build" and "init" phases.
    exit_stack.callback(engine_client.shutdown, vllm_config.shutdown_timeout)

    logger.info("Starting vLLM server on %s", listen_address)

    # --- 4. Build the FastAPI app ------------------------------------------
    app = await init_app(args, engine_client)

    # --- 5. Build the uvicorn server ---------------------------------------
    uvicorn_kwargs = dict(
        host=args.host,
        port=args.port,
        log_level=args.uvicorn_log_level,
        # When disable_uvicorn_access_log is True, no access log is emitted.
        access_log=not args.disable_uvicorn_access_log,
        timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
        ssl_ciphers=args.ssl_ciphers,
        h11_max_incomplete_event_size=getattr(
            args, "h11_max_incomplete_event_size", None
        ),
        h11_max_header_count=getattr(args, "h11_max_header_count", None),
    )

    config = init_uvicorn(args, app, uvicorn_kwargs)
    server = NoSignalServer(config)

    # Used by terminate_if_errored.
    app.state.server = server

    loop = asyncio.get_running_loop()

    # --- 6. Install signal handlers before we start serving ----------------
    # This replaces the pre-init handler installed in step 1.
    _install_signal_handlers(loop, server, exit_stack)

    # --- 7. Start serving, watchdog, and (optional) SSL refresher ----------
    server_task = loop.create_task(server.serve(sockets=[sock]))
    exit_stack.callback(server_task.cancel)

    watchdog_task = loop.create_task(watchdog_loop(server, engine_client))
    exit_stack.callback(watchdog_task.cancel)

    start_ssl_refresher_if_needed(args, config, exit_stack)

    # --- 8. Wait until the server stops; ExitStack cleans everything up ----
    await server_task
    logger.info_once("[shutdown] API server: HTTP server stopped gracefully")


async def run_server_with_exit_stack(args: argparse.Namespace):
    """Wrap run_server with an ExitStack and uniform error logging."""
    with contextlib.ExitStack() as exit_stack:
        try:
            await run_server(args, exit_stack)
        except asyncio.CancelledError:
            logger.info_once("[shutdown] API server: cancelled.")
            raise
        except KeyboardInterrupt:
            logger.info_once("[shutdown] API server: interrupted by user.")
        except Exception:
            # Unexpected error: keep the traceback for debugging and
            # propagate so the caller sees a non-zero exit code.
            logger.exception("[shutdown] API server: unexpected error during shutdown")
            raise


def run_single_api_server(args: argparse.Namespace):
    """Enter uvloop and run the whole server lifecycle."""
    uvloop.run(run_server_with_exit_stack(args))


def main():
    """CLI entrypoint: parse and validate args, then dispatch."""
    from vllm.entrypoints.serve.utils.api_utils import cli_env_setup
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    from ..cli_args import (
        make_arg_parser,
        validate_parsed_serve_args,
    )

    # NOTE(simon):
    # Keep this section in sync with vllm/entrypoints/cli/main.py.
    cli_env_setup()
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    run_single_api_server(args)


if __name__ == "__main__":
    main()
