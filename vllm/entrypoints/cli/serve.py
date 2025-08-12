# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import os
import signal
import sys
from typing import Optional

import uvloop

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
from vllm.utils import FlexibleArgumentParser, get_tcp_uri
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.utils import CoreEngineProcManager, launch_core_engines
from vllm.v1.executor.abstract import Executor
from vllm.v1.metrics.prometheus import setup_multiprocess_prometheus
from vllm.v1.utils import (APIServerProcessManager,
                           wait_for_completion_or_failure)

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
        else:
            if args.data_parallel_start_rank:
                raise ValueError("data_parallel_start_rank is only "
                                 "applicable in headless mode")
            if args.api_server_count > 1:
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
        show_filtered_argument_or_group_from_help(serve_parser, ["serve"])
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

    if engine_args.data_parallel_rank is not None:
        raise ValueError("data_parallel_rank is not applicable in "
                         "headless mode")

    parallel_config = vllm_config.parallel_config
    local_engine_count = parallel_config.data_parallel_size_local

    if local_engine_count <= 0:
        raise ValueError("data_parallel_size_local must be > 0 in "
                         "headless mode")

    host = parallel_config.data_parallel_master_ip
    port = engine_args.data_parallel_rpc_port  # add to config too
    handshake_address = get_tcp_uri(host, port)

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

    executor_class = Executor.get_class(vllm_config)
    log_stats = not engine_args.disable_log_stats

    parallel_config = vllm_config.parallel_config
    dp_rank = parallel_config.data_parallel_rank
    external_dp_lb = parallel_config.data_parallel_external_lb
    assert external_dp_lb or dp_rank == 0

    api_server_manager: Optional[APIServerProcessManager] = None

    with launch_core_engines(vllm_config, executor_class, log_stats,
                             num_api_servers) as (local_engine_manager,
                                                  coordinator, addresses):

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
            if coordinator else None)

        # For dp ranks > 0 in external DP LB mode, we must delay the
        # start of the API servers until the local engine is started
        # (after the launcher context manager exits),
        # since we get the front-end stats update address from the coordinator
        # via the handshake with the local engine.
        if dp_rank == 0 or not external_dp_lb:
            # Start API servers using the manager.
            api_server_manager = APIServerProcessManager(
                **api_server_manager_kwargs)

    # Start API servers now if they weren't already started.
    if api_server_manager is None:
        api_server_manager_kwargs["stats_update_address"] = (
            addresses.frontend_stats_publish_address)
        api_server_manager = APIServerProcessManager(
            **api_server_manager_kwargs)

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
