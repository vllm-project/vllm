# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lean entrypoint for a headless vLLM engine (no API server).

`vllm serve --headless` builds the full OpenAI-server CLI schema
(tool/reasoning-parser plugins, MCP, chat templates, ...), none of which a
headless engine reads. This builds a parser from `AsyncEngineArgs.add_cli_args`
only. `run_headless` is implemented here (and re-exported from
`vllm.entrypoints.cli.serve` for `vllm serve --headless`) rather than the
other way around, so this module never risks pulling in
`vllm.entrypoints.cli.serve`'s OpenAI-server-only imports, even if more are
added there in the future.

Uses `FlexibleArgumentParser.parse_args` (not `parse_known_args`) so
`--config` files, dash/underscore normalization, and dotted-JSON options
behave exactly like `vllm serve`, and unknown args still error.

Not a public `vllm` subcommand; invoked via
`python -m vllm.entrypoints.cli.headless_engine` by `vllm-rs serve`'s managed
headless-engine mode (`rust/src/managed-engine`).
"""

import argparse
import signal

import vllm
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext
from vllm.utils.network_utils import get_tcp_uri
from vllm.v1.engine.utils import CoreEngineProcManager
from vllm.v1.executor import Executor
from vllm.v1.executor.multiproc_executor import MultiprocExecutor

logger = init_logger(__name__)


def main(argv: list[str] | None = None) -> None:
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser(
        description="Launch a headless vLLM engine (no API server).",
        add_json_tip=False,
    )
    parser = AsyncEngineArgs.add_cli_args(parser)

    args = parser.parse_args(argv)

    # Headless mode never starts API servers.
    args.api_server_count = 0

    run_headless(args)


def run_headless(args: argparse.Namespace) -> None:
    if args.api_server_count > 1:
        raise ValueError("api_server_count can't be set in headless mode")

    # Create the EngineConfig.
    engine_args = vllm.AsyncEngineArgs.from_cli_args(args)
    usage_context = UsageContext.OPENAI_API_SERVER
    vllm_config = engine_args.create_engine_config(
        usage_context=usage_context, headless=True
    )

    if engine_args.data_parallel_hybrid_lb:
        raise ValueError("data_parallel_hybrid_lb is not applicable in headless mode")

    parallel_config = vllm_config.parallel_config
    local_engine_count = parallel_config.data_parallel_size_local

    if local_engine_count <= 0:
        raise ValueError("data_parallel_size_local must be > 0 in headless mode")

    shutdown_requested = False

    # Catch SIGTERM and SIGINT to allow graceful shutdown.
    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        logger.debug("Received %d signal.", signum)
        if not shutdown_requested:
            shutdown_requested = True
            raise SystemExit

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    if parallel_config.node_rank_within_dp > 0:
        from vllm.version import __version__ as VLLM_VERSION

        # Run headless workers (for multi-node PP/TP).
        host = parallel_config.master_addr
        head_node_address = f"{host}:{parallel_config.master_port}"
        logger.info(
            "Launching vLLM (v%s) headless multiproc executor, "
            "with head node address %s for torch.distributed process group.",
            VLLM_VERSION,
            head_node_address,
        )

        executor = MultiprocExecutor(vllm_config, monitor_workers=False)
        executor.start_worker_monitor(inline=True)
        return

    host = parallel_config.data_parallel_master_ip
    port = parallel_config.data_parallel_rpc_port
    handshake_address = get_tcp_uri(host, port)

    logger.info(
        "Launching %d data parallel engine(s) in headless mode, "
        "with head node address %s.",
        local_engine_count,
        handshake_address,
    )

    # Create the engines.
    engine_manager = CoreEngineProcManager(
        local_engine_count=local_engine_count,
        start_index=vllm_config.parallel_config.data_parallel_rank,
        local_start_index=0,
        vllm_config=vllm_config,
        local_client=False,
        handshake_address=handshake_address,
        executor_class=Executor.get_class(vllm_config),
        log_stats=not engine_args.disable_log_stats,
    )

    try:
        engine_manager.monitor_engine_liveness()
    finally:
        timeout = None
        if shutdown_requested:
            timeout = vllm_config.shutdown_timeout
            logger.info("Waiting up to %d seconds for processes to exit", timeout)
        engine_manager.shutdown(timeout=timeout)
        logger.info("Shutting down.")


if __name__ == "__main__":
    main()
