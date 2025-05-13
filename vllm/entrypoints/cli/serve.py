# SPDX-License-Identifier: Apache-2.0

import argparse
import multiprocessing
import os
import signal
import sys
import time
import weakref
from multiprocessing import connection
from multiprocessing.context import SpawnProcess
from typing import Any, Optional

import uvloop
import zmq

import vllm.envs as envs
from vllm import AsyncEngineArgs
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.openai.api_server import (run_server, run_server_worker,
                                                setup_server)
from vllm.entrypoints.openai.cli_args import (make_arg_parser,
                                              validate_parsed_serve_args)
from vllm.executor.multiproc_worker_utils import _add_prefix
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser, get_tcp_uri, zmq_socket_ctx
from vllm.v1.engine.coordinator import DPCoordinator
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.core_client import CoreEngineProcManager
from vllm.v1.executor.abstract import Executor
from vllm.v1.utils import (CoreEngine, get_engine_client_zmq_addr,
                           wait_for_engine_startup)

logger = init_logger(__name__)


class APIServerProcessManager:
    """Manages a group of API server processes.
    
    Handles creation, monitoring, and termination of API server worker 
    processes.
    """

    def __init__(
        self,
        listen_address: str,
        sock: Any,
        args: argparse.Namespace,
        num_servers: int,
        input_addresses: list[str],
        output_addresses: list[str],
        stats_update_address: Optional[str] = None,
    ):
        """Initialize and start API server worker processes.
        
        Args:
            listen_address: Address to listen for client connections
            sock: Socket for client connections
            args: Command line arguments
            num_servers: Number of API server processes to start
            input_addresses: Input addresses for each API server
            output_addresses: Output addresses for each API server
            stats_update_address: Optional stats update address
        """
        self.listen_address = listen_address
        self.sock = sock
        self.args = args

        # Start API servers
        spawn_context = multiprocessing.get_context("spawn")
        self.processes: list[SpawnProcess] = []

        for i, in_addr, out_addr in zip(range(num_servers), input_addresses,
                                        output_addresses):
            client_config = {
                "input_address": in_addr,
                "output_address": out_addr,
                "client_index": i
            }
            if stats_update_address is not None:
                client_config["stats_update_address"] = stats_update_address

            proc = spawn_context.Process(target=run_api_server_worker,
                                         name=f"ApiServer_{i}",
                                         args=(listen_address, sock, args,
                                               client_config))
            self.processes.append(proc)
            proc.start()

        logger.info("Started %d API server processes", len(self.processes))
        # Create a finalizer to ensure processes are terminated when this
        # object is garbage collected
        self._finalizer = weakref.finalize(self, self._shutdown_all_processes,
                                           self.processes)

    def wait_for_completion_or_failure(self) -> None:
        """Wait for all processes to complete or detect if any fail.
        
        Raises an exception if any process exits with a non-zero status.
        """
        try:
            # Create a mapping of sentinels to their corresponding processes
            # for efficient lookup
            sentinel_to_proc = {proc.sentinel: proc for proc in self.processes}
            sentinels = list(sentinel_to_proc.keys())

            # Check if any process terminates
            while sentinels:
                # Wait for any process to terminate
                ready_sentinels = connection.wait(sentinels)
                if not ready_sentinels:
                    continue

                # Process any terminated processes
                for sentinel in ready_sentinels:
                    proc = sentinel_to_proc[sentinel]
                    sentinels.remove(sentinel)

                    # Check if process exited with error
                    if proc.exitcode != 0:
                        logger.error(
                            "API server process %s (PID: %d) "
                            "died with exit code %d", proc.name, proc.pid,
                            proc.exitcode)
                        raise RuntimeError(
                            "API server process %s died with exit code %d. "
                            "All API server processes will be terminated.",
                            proc.name, proc.exitcode)

            # If we've processed all sentinels, all processes completed
            # successfully
            logger.info("All API server processes completed successfully")

        except Exception:
            # If any exception occurs, terminate all processes before re-raising
            self.terminate()
            raise

    def terminate(self) -> None:
        """Gracefully terminate all API server processes.
        
        First tries SIGTERM, then SIGKILL after a timeout.
        """
        self._shutdown_all_processes(self.processes)

    @staticmethod
    def _shutdown_all_processes(processes: list[SpawnProcess]) -> None:
        """Static method to terminate all processes.
        
        This is separate to allow it to be used as a finalizer.
        
        Args:
            processes: List of SpawnProcess objects to terminate
        """
        logger.warning("Terminating all API server processes...")
        for proc in processes:
            if proc.is_alive():
                logger.info("Terminating %s (PID: %d)", proc.name, proc.pid)
                proc.terminate()

        # Wait for graceful termination with timeout
        termination_timeout = 5.0  # 5 seconds timeout
        termination_deadline = time.time() + termination_timeout

        # Wait for processes to terminate gracefully
        while time.time() < termination_deadline:
            if all(not proc.is_alive() for proc in processes):
                break
            time.sleep(0.1)

        # Force kill any processes that didn't terminate gracefully
        for proc in processes:
            if proc.is_alive():
                logger.warning("Force killing %s (PID: %d)", proc.name,
                               proc.pid)
                proc.kill()
                proc.join(1.0)  # Short timeout for killed processes

        logger.info("All API server processes have been terminated")


class ServeSubcommand(CLISubcommand):
    """The `serve` subcommand for the vLLM CLI. """

    def __init__(self):
        self.name = "serve"
        super().__init__()

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
            help="Read CLI options from a config file."
            "Must be a YAML with the following options:"
            "https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#cli-reference"
        )

        return make_arg_parser(serve_parser)


def cmd_init() -> list[CLISubcommand]:
    return [ServeSubcommand()]


def run_headless(args: argparse.Namespace):

    if args.api_server_count > 1:
        raise RuntimeError("api_server_count can't be set in headless mode")

    # Create the EngineConfig.
    engine_args = AsyncEngineArgs.from_cli_args(args)
    usage_context = UsageContext.OPENAI_API_SERVER
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)

    if not envs.VLLM_USE_V1:
        raise RuntimeError("Headless mode is only supported for V1")

    parallel_config = vllm_config.parallel_config
    local_engine_count = parallel_config.data_parallel_size_local
    host = parallel_config.data_parallel_master_ip
    port = engine_args.data_parallel_rpc_port  # add to config too
    handshake_address = get_tcp_uri(host, port)

    if local_engine_count <= 0:
        raise RuntimeError("data_parallel_size_local must be > 0 in "
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
    # assert num_api_servers > 1

    listen_address, sock = setup_server(args)

    engine_args = AsyncEngineArgs.from_cli_args(args)
    usage_context = UsageContext.OPENAI_API_SERVER
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)
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

    addresses: dict[str, Any] = {
        "input_addresses": input_addresses,
        "output_addresses": output_addresses,
    }

    # Set up coordinator for dp > 1.
    coordinator = None
    stats_update_address = None
    if dp_size > 1:
        # TODO "ready" event for coordinator
        coordinator = DPCoordinator(parallel_config)
        addresses.update(coordinator.get_engine_socket_addresses())
        stats_update_address = coordinator.get_stats_publish_address()

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

        # Wait for API server processes to complete or fail
        try:
            api_server_manager.wait_for_completion_or_failure()
        except KeyboardInterrupt:
            logger.info(
                "Received KeyboardInterrupt, shutting down API servers...")
            api_server_manager.terminate()
            raise
        except Exception as e:
            logger.exception(
                "Exception occurred while running API servers: %s", str(e))
            # The manager will have already terminated the processes
            raise


def run_api_server_worker(listen_address,
                          sock,
                          args,
                          client_config=None,
                          **uvicorn_kwargs) -> None:

    # Add process-specific prefix to stdout and stderr.
    from multiprocessing import current_process
    process_name = current_process().name
    pid = os.getpid()
    _add_prefix(sys.stdout, process_name, pid)
    _add_prefix(sys.stderr, process_name, pid)

    uvloop.run(
        run_server_worker(listen_address, sock, args, client_config,
                          **uvicorn_kwargs))
