# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import signal

import uvloop

import vllm
import vllm.envs as envs
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.openai.api_server import (
    run_server,
    run_server_worker,
    setup_server,
)
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.network_utils import get_tcp_uri
from vllm.utils.system_utils import decorate_logs, set_process_title
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.utils import CoreEngineProcManager, launch_core_engines
from vllm.v1.executor import Executor
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.metrics.prometheus import setup_multiprocess_prometheus
from vllm.v1.utils import APIServerProcessManager, wait_for_completion_or_failure

logger = init_logger(__name__)

DESCRIPTION = """Launch a local OpenAI-compatible API server to serve LLM
completions via HTTP. Defaults to Qwen/Qwen3-0.6B if no model is specified.

Search by using: `--help=<ConfigGroup>` to explore options by section (e.g.,
--help=ModelConfig, --help=Frontend)
  Use `--help=all` to show all available flags at once.
"""


class ServeSubcommand(CLISubcommand):
    """The `serve` subcommand for the vLLM CLI."""

    name = "serve"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        # If model is specified in CLI (as positional arg), it takes precedence
        if hasattr(args, "model_tag") and args.model_tag is not None:
            args.model = args.model_tag

        if args.headless or args.api_server_count < 1:
            run_headless(args)
        else:
            if args.api_server_count > 1:
                run_multi_api_server(args)
            else:
                # Single API server (this process).
                uvloop.run(run_server(args))

    def validate(self, args: argparse.Namespace) -> None:
        validate_parsed_serve_args(args)

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        serve_parser = subparsers.add_parser(
            self.name,
            help="Launch a local OpenAI-compatible API server to serve LLM "
            "completions via HTTP.",
            description=DESCRIPTION,
            usage="vllm serve [model_tag] [options]",
        )

        serve_parser = make_arg_parser(serve_parser)
        serve_parser.epilog = VLLM_SUBCMD_PARSER_EPILOG.format(subcmd=self.name)
        return serve_parser


def cmd_init() -> list[CLISubcommand]:
    return [ServeSubcommand()]


def run_headless(args: argparse.Namespace):
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
        target_fn=EngineCoreProc.run_engine_core,
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
        engine_manager.join_first()
    finally:
        logger.info("Shutting down.")
        engine_manager.close()


def run_multi_api_server(args: argparse.Namespace):
    assert not args.headless
    num_api_servers: int = args.api_server_count
    assert num_api_servers > 0

    if num_api_servers > 1:
        setup_multiprocess_prometheus()

    listen_address, sock = setup_server(args)

    engine_args = vllm.AsyncEngineArgs.from_cli_args(args)
    engine_args._api_process_count = num_api_servers
    engine_args._api_process_rank = -1

    usage_context = UsageContext.OPENAI_API_SERVER
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)

    if num_api_servers > 1 and envs.VLLM_ALLOW_RUNTIME_LORA_UPDATING:
        raise ValueError(
            "VLLM_ALLOW_RUNTIME_LORA_UPDATING cannot be used with api_server_count > 1"
        )

    executor_class = Executor.get_class(vllm_config)
    log_stats = not engine_args.disable_log_stats

    parallel_config = vllm_config.parallel_config
    dp_rank = parallel_config.data_parallel_rank
    external_dp_lb = parallel_config.data_parallel_external_lb
    hybrid_dp_lb = parallel_config.data_parallel_hybrid_lb
    assert external_dp_lb or hybrid_dp_lb or dp_rank == 0

    api_server_manager: APIServerProcessManager | None = None

    with launch_core_engines(
        vllm_config, executor_class, log_stats, num_api_servers
    ) as (local_engine_manager, coordinator, addresses):
        # Construct common args for the APIServerProcessManager up-front.
        api_server_manager_kwargs = dict(
            target_server_fn=run_api_server_worker_proc,
            listen_address=listen_address,
            sock=sock,
            args=args,
            num_servers=num_api_servers,
            input_addresses=addresses.inputs,
            output_addresses=addresses.outputs,
            stats_update_address=coordinator.get_stats_publish_address()
            if coordinator
            else None,
        )

        # For dp ranks > 0 in external/hybrid DP LB modes, we must delay the
        # start of the API servers until the local engine is started
        # (after the launcher context manager exits),
        # since we get the front-end stats update address from the coordinator
        # via the handshake with the local engine.
        if dp_rank == 0 or not (external_dp_lb or hybrid_dp_lb):
            # Start API servers using the manager.
            api_server_manager = APIServerProcessManager(**api_server_manager_kwargs)

    # Start API servers now if they weren't already started.
    if api_server_manager is None:
        api_server_manager_kwargs["stats_update_address"] = (
            addresses.frontend_stats_publish_address
        )
        api_server_manager = APIServerProcessManager(**api_server_manager_kwargs)

    # Wait for API servers
    wait_for_completion_or_failure(
        api_server_manager=api_server_manager,
        engine_manager=local_engine_manager,
        coordinator=coordinator,
    )


def run_api_server_worker_proc(
    listen_address, sock, args, client_config=None, **uvicorn_kwargs
) -> None:
    """Entrypoint for individual API server worker processes."""
    client_config = client_config or {}
    server_index = client_config.get("client_index", 0)

    # Set process title and add process-specific prefix to stdout and stderr.
    set_process_title("APIServer", str(server_index))
    decorate_logs()

    uvloop.run(
        run_server_worker(listen_address, sock, args, client_config, **uvicorn_kwargs)
    )
