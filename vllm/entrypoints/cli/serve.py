# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import os
import signal
import sys

import uvloop
import zmq

import vllm
import vllm.envs as envs
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.openai.api_server import (run_server, run_server_worker,
                                                setup_server)
from vllm.entrypoints.openai.cli_args import (make_arg_parser,
                                              validate_parsed_serve_args)
from vllm.entrypoints.utils import (VLLM_SUBCMD_PARSER_EPILOG,
                                    show_filtered_argument_or_group_from_help)
from vllm.executor.multiproc_worker_utils import _add_prefix
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser, get_tcp_uri, zmq_socket_ctx
from vllm.v1.engine.coordinator import DPCoordinator
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.core_client import CoreEngineProcManager
from vllm.v1.executor.abstract import Executor
from vllm.v1.metrics.prometheus import setup_multiprocess_prometheus
from vllm.v1.utils import (APIServerProcessManager, CoreEngine,
                           CoreEngineActorManager, EngineZmqAddresses,
                           get_engine_client_zmq_addr,
                           wait_for_completion_or_failure,
                           wait_for_engine_startup)

logger = init_logger(__name__)


class ServeSubcommand(CLISubcommand):
    """The `serve` subcommand for the vLLM CLI. """
    name = "serve"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        # If model is specified in CLI (as positional arg), it takes precedence
        if hasattr(args, 'model_tag') and args.model_tag is not None:
            args.model = args.model_tag

        if args.headless or args.api_server_count < 1:
            run_headless(args)
        elif args.api_server_count > 1:
            run_multi_api_server(args)
        else:
            # Single API server (this process).
            uvloop.run(run_server(args))

    def validate(self, args: argparse.Namespace) -> None:
        validate_parsed_serve_args(args)

    def subparser_init(
            self,
            subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        serve_parser = subparsers.add_parser(
            "serve",
            help="Start the vLLM OpenAI Compatible API server.",
            description="Start the vLLM OpenAI Compatible API server.",
            usage="vllm serve [model_tag] [options]")
        serve_parser.add_argument("model_tag",
                                  type=str,
                                  nargs='?',
                                  help="The model tag to serve "
                                  "(optional if specified in config)")
        serve_parser.add_argument(
            "--headless",
            action='store_true',
            default=False,
            help="Run in headless mode. See multi-node data parallel "
            "documentation for more details.")
        serve_parser.add_argument(
            '--data-parallel-start-rank',
            '-dpr',
            type=int,
            default=0,
            help='Starting data parallel rank for secondary nodes.')
        serve_parser.add_argument('--api-server-count',
                                  '-asc',
                                  type=int,
                                  default=1,
                                  help='How many API server processes to run.')
        serve_parser.add_argument(
            "--config",
            type=str,
            default='',
            required=False,
            help="Read CLI options from a config file. "
            "Must be a YAML with the following options: "
            "https://docs.vllm.ai/en/latest/configuration/serve_args.html")

        serve_parser = make_arg_parser(serve_parser)
        show_filtered_argument_or_group_from_help(serve_parser, "serve")
        serve_parser.epilog = VLLM_SUBCMD_PARSER_EPILOG
        return serve_parser


def cmd_init() -> list[CLISubcommand]:
    return [ServeSubcommand()]


def run_headless(args: argparse.Namespace):

    if args.api_server_count > 1:
        raise ValueError("api_server_count can't be set in headless mode")

    # Create the EngineConfig.
    engine_args = vllm.AsyncEngineArgs.from_cli_args(args)
    usage_context = UsageContext.OPENAI_API_SERVER
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)

    if not envs.VLLM_USE_V1:
        raise ValueError("Headless mode is only supported for V1")

    parallel_config = vllm_config.parallel_config
    local_engine_count = parallel_config.data_parallel_size_local
    host = parallel_config.data_parallel_master_ip
    port = engine_args.data_parallel_rpc_port  # add to config too
    handshake_address = get_tcp_uri(host, port)

    if local_engine_count <= 0:
        raise ValueError("data_parallel_size_local must be > 0 in "
                         "headless mode")

    # Catch SIGTERM and SIGINT to allow graceful shutdown.
    def signal_handler(signum, frame):
        logger.debug("Received %d signal.", signum)
        raise SystemExit

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    logger.info(
        "Launching %d data parallel engine(s) in headless mode, "
        "with head node address %s.", local_engine_count, handshake_address)

    # Create the engines.
    engine_manager = CoreEngineProcManager(
        target_fn=EngineCoreProc.run_engine_core,
        local_engine_count=local_engine_count,
        start_index=args.data_parallel_start_rank,
        local_start_index=0,
        vllm_config=vllm_config,
        on_head_node=False,
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
    num_api_servers = args.api_server_count
    assert num_api_servers > 0

    if num_api_servers > 1:
        setup_multiprocess_prometheus()

    listen_address, sock = setup_server(args)

    engine_args = vllm.AsyncEngineArgs.from_cli_args(args)
    usage_context = UsageContext.OPENAI_API_SERVER
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)
    model_config = vllm_config.model_config

    if num_api_servers > 1:
        if not envs.VLLM_USE_V1:
            raise ValueError("api_server_count > 1 is only supported for V1")

        if envs.VLLM_ALLOW_RUNTIME_LORA_UPDATING:
            raise ValueError("VLLM_ALLOW_RUNTIME_LORA_UPDATING cannot be used "
                             "with api_server_count > 1")

        if model_config.is_multimodal_model and not (
                model_config.disable_mm_preprocessor_cache):
            logger.warning(
                "Multi-model preprocessor cache will be disabled for"
                " api_server_count > 1")
            model_config.disable_mm_preprocessor_cache = True

    parallel_config = vllm_config.parallel_config

    assert parallel_config.data_parallel_rank == 0

    dp_size = parallel_config.data_parallel_size
    local_engine_count = parallel_config.data_parallel_size_local
    host = parallel_config.data_parallel_master_ip
    local_only = local_engine_count == dp_size

    # Set up input and output addresses.
    input_addresses = [
        get_engine_client_zmq_addr(local_only, host)
        for _ in range(num_api_servers)
    ]
    output_addresses = [
        get_engine_client_zmq_addr(local_only, host)
        for _ in range(num_api_servers)
    ]

    addresses = EngineZmqAddresses(
        inputs=input_addresses,
        outputs=output_addresses,
    )

    # Set up coordinator for dp > 1.
    coordinator = None
    stats_update_address = None
    if dp_size > 1:
        coordinator = DPCoordinator(parallel_config)
        addresses.coordinator_input, addresses.coordinator_output = (
            coordinator.get_engine_socket_addresses())
        stats_update_address = coordinator.get_stats_publish_address()
        logger.info("Started DP Coordinator process (PID: %d)",
                    coordinator.proc.pid)

    if parallel_config.data_parallel_backend == "ray":
        logger.info("Starting ray-based data parallel backend")

        engine_actor_manager = CoreEngineActorManager(
            vllm_config=vllm_config,
            addresses=addresses,
            executor_class=Executor.get_class(vllm_config),
            log_stats=not engine_args.disable_log_stats,
        )
        # Start API servers using the manager
        api_server_manager = APIServerProcessManager(
            target_server_fn=run_api_server_worker_proc,
            listen_address=listen_address,
            sock=sock,
            args=args,
            num_servers=num_api_servers,
            input_addresses=input_addresses,
            output_addresses=output_addresses,
            stats_update_address=stats_update_address)

        wait_for_completion_or_failure(api_server_manager=api_server_manager,
                                       engine_manager=engine_actor_manager,
                                       coordinator=coordinator)
        return

    handshake_address = get_engine_client_zmq_addr(
        local_only, host, parallel_config.data_parallel_rpc_port)

    with zmq_socket_ctx(handshake_address, zmq.ROUTER,
                        bind=True) as handshake_socket:

        # Start local engines.
        if not local_engine_count:
            local_engine_manager = None
        else:
            local_engine_manager = CoreEngineProcManager(
                EngineCoreProc.run_engine_core,
                vllm_config=vllm_config,
                executor_class=Executor.get_class(vllm_config),
                log_stats=not engine_args.disable_log_stats,
                handshake_address=handshake_address,
                on_head_node=True,
                local_engine_count=local_engine_count,
                start_index=0,
                local_start_index=0)

        # Start API servers using the manager
        api_server_manager = APIServerProcessManager(
            target_server_fn=run_api_server_worker_proc,
            listen_address=listen_address,
            sock=sock,
            args=args,
            num_servers=num_api_servers,
            input_addresses=input_addresses,
            output_addresses=output_addresses,
            stats_update_address=stats_update_address)

        # Wait for engine handshakes to complete.
        core_engines = [
            CoreEngine(index=i, local=(i < local_engine_count))
            for i in range(dp_size)
        ]
        wait_for_engine_startup(
            handshake_socket,
            addresses,
            core_engines,
            parallel_config,
            vllm_config.cache_config,
            local_engine_manager,
            coordinator.proc if coordinator else None,
        )

        # Wait for API servers
        wait_for_completion_or_failure(api_server_manager=api_server_manager,
                                       engine_manager=local_engine_manager,
                                       coordinator=coordinator)


def run_api_server_worker_proc(listen_address,
                               sock,
                               args,
                               client_config=None,
                               **uvicorn_kwargs) -> None:
    """Entrypoint for individual API server worker processes."""

    # Add process-specific prefix to stdout and stderr.
    from multiprocessing import current_process
    process_name = current_process().name
    pid = os.getpid()
    _add_prefix(sys.stdout, process_name, pid)
    _add_prefix(sys.stderr, process_name, pid)

    uvloop.run(
        run_server_worker(listen_address, sock, args, client_config,
                          **uvicorn_kwargs))
