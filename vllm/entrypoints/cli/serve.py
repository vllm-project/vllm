# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import copy
import signal
import socket

import uvloop

import vllm
import vllm.envs as envs
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.openai.api_server import (
    create_server_socket,
    run_server,
    run_server_worker,
    setup_server,
)
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.openai.frontend import main_frontend
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
from vllm.v1.utils import (
    APIServerProcessManager,
    shutdown,
    wait_for_completion_or_failure,
)

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

        if args.headless:
            if args.api_server_count is not None and args.api_server_count > 0:
                raise ValueError(
                    f"--api-server-count={args.api_server_count} cannot be "
                    "used with --headless (no API servers are started in "
                    "headless mode)."
                )
            # Default to 0 in headless mode (no API servers)
            args.api_server_count = 0

        # Detect LB mode for defaulting api_server_count.
        # External LB: --data-parallel-external-lb or --data-parallel-rank
        # Hybrid LB: --data-parallel-hybrid-lb or --data-parallel-start-rank
        is_external_lb = (
            args.data_parallel_external_lb or args.data_parallel_rank is not None
        )
        is_hybrid_lb = (
            args.data_parallel_hybrid_lb or args.data_parallel_start_rank is not None
        )

        if is_external_lb and is_hybrid_lb:
            raise ValueError(
                "Cannot use both external and hybrid data parallel load "
                "balancing modes. External LB is enabled via "
                "--data-parallel-external-lb or --data-parallel-rank. "
                "Hybrid LB is enabled via --data-parallel-hybrid-lb or "
                "--data-parallel-start-rank. Use one mode or the other."
            )

        # Default api_server_count if not explicitly set.
        # - External LB: Leave as 1 (external LB handles distribution)
        # - Hybrid LB: Use local DP size (internal LB for local ranks only)
        # - Internal LB: Use full DP size
        if args.api_server_count is None:
            if is_external_lb:
                args.api_server_count = 1
            elif is_hybrid_lb:
                args.api_server_count = args.data_parallel_size_local or 1
                if args.api_server_count > 1:
                    logger.info(
                        "Defaulting api_server_count to data_parallel_size_local "
                        "(%d) for hybrid LB mode.",
                        args.api_server_count,
                    )
            else:
                args.api_server_count = args.data_parallel_size
                if args.api_server_count > 1:
                    logger.info(
                        "Defaulting api_server_count to data_parallel_size (%d).",
                        args.api_server_count,
                    )

        if args.api_server_count < 1:
            run_headless(args)
        elif args.api_server_count > 1:
            if args.multi_server_frontend:
                run_multi_api_server_with_frontend(args)
            else:
                run_multi_api_server(args)
        else:
            # Single API server (this process).
            args.api_server_count = None
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
    assert parallel_config.local_engines_only or dp_rank == 0

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
        if dp_rank == 0 or not parallel_config.local_engines_only:
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


def run_multi_api_server_with_frontend(args: argparse.Namespace) -> None:
    """Launch N backend vLLM API servers each on their own port, plus a
    lightweight frontend process on the main port (``args.port``) that:

      1. Exposes ``/health`` by aggregating the ``/health`` endpoints of all
         N backends.  K8s liveness and startup probes should be aimed at
         the frontend port so that the pod is only considered live once every
         backend has finished loading the model.

      2. Monitors the N backend processes and triggers a pod-level shutdown
         (exits with code 1) if any backend crashes, ensuring K8s will
         restart the pod rather than leaving it in a partially-healthy state.

    Port layout
    -----------
    * Frontend  →  ``args.port``          (the K8s-facing port)
    * Backend i →  ``args.port + i + 1``  (internal to the pod)

    This is different from the default multi-server mode (``--api-server-count
    > 1`` without ``--multi-server-frontend``) where all N servers share the
    same port via ``SO_REUSEPORT``.  Here each server gets its own dedicated
    port and only the frontend is externally visible.
    """
    assert not args.headless
    num_api_servers: int = args.api_server_count
    assert num_api_servers > 1, (
        "run_multi_api_server_with_frontend requires api_server_count > 1"
    )

    if num_api_servers > 1:
        setup_multiprocess_prometheus()

    host = args.host or ""

    # ------------------------------------------------------------------
    # Pre-bind all sockets in the parent process to avoid race conditions
    # with Ray / torch.distributed (same pattern as the existing code).
    # ------------------------------------------------------------------

    # Frontend socket: the pod-level port that K8s probes.
    frontend_port = args.port
    frontend_sock = create_server_socket((host, frontend_port))
    logger.info("Frontend will listen on %s:%d", host or "0.0.0.0", frontend_port)

    # Backend sockets: one dedicated port per backend.
    backend_base_port = args.port + 1
    backend_socks: list[socket.socket] = []
    backend_ports: list[int] = []
    for i in range(num_api_servers):
        bp = backend_base_port + i
        backend_socks.append(create_server_socket((host, bp)))
        backend_ports.append(bp)
    logger.info(
        "Backends will listen on ports %s",
        ", ".join(str(p) for p in backend_ports),
    )

    # ------------------------------------------------------------------
    # Build VllmConfig (needed for engine launch + executor selection).
    # ------------------------------------------------------------------
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
    assert parallel_config.local_engines_only or dp_rank == 0

    # ------------------------------------------------------------------
    # Spawn core engine(s) and then the per-port backend API servers.
    # ------------------------------------------------------------------
    api_server_manager: APIServerProcessManager | None = None

    # local_engine_manager / coordinator / addresses are set inside the
    # `with` block; capture them so the finally clause can clean up.
    local_engine_manager = None
    coordinator = None
    addresses = None

    with launch_core_engines(
        vllm_config, executor_class, log_stats, num_api_servers
    ) as (_lem, _coord, _addrs):
        local_engine_manager = _lem
        coordinator = _coord
        addresses = _addrs

        stats_addr = coordinator.get_stats_publish_address() if coordinator else None

        if dp_rank == 0 or not parallel_config.local_engines_only:
            api_server_manager = _start_per_port_backends(
                args=args,
                backend_ports=backend_ports,
                backend_socks=backend_socks,
                addresses=addresses,
                num_api_servers=num_api_servers,
                stats_update_address=stats_addr,
            )

    # For external/hybrid DP where dp_rank > 0 we need to wait until the
    # local engine has started before we can get the stats address.
    if api_server_manager is None:
        assert addresses is not None
        api_server_manager = _start_per_port_backends(
            args=args,
            backend_ports=backend_ports,
            backend_socks=backend_socks,
            addresses=addresses,
            num_api_servers=num_api_servers,
            stats_update_address=addresses.frontend_stats_publish_address,
        )

    # ------------------------------------------------------------------
    # Run the frontend in this (main) process.
    # Blocks until shutdown (clean or crash-triggered).
    # ------------------------------------------------------------------
    backend_urls = [f"http://127.0.0.1:{p}" for p in backend_ports]
    try:
        main_frontend(
            host=args.host or "0.0.0.0",
            port=frontend_port,
            sock=frontend_sock,
            backend_urls=backend_urls,
            processes=api_server_manager.processes,
            log_level=args.uvicorn_log_level,
        )
    finally:
        # Always clean up backends on exit.
        frontend_sock.close()
        for s in backend_socks:
            s.close()
        api_server_manager.close()
        if local_engine_manager is not None:
            local_engine_manager.close()
        if coordinator is not None:
            coordinator.close()


def _start_per_port_backends(
    args: argparse.Namespace,
    backend_ports: list[int],
    backend_socks: list[socket.socket],
    addresses,
    num_api_servers: int,
    stats_update_address: str | None,
) -> APIServerProcessManager:
    """Spawn N backend API server processes, each on its own port.

    We reuse :class:`~vllm.v1.utils.APIServerProcessManager` for process
    lifecycle management, but each backend gets a *unique* socket/port
    instead of sharing one via ``SO_REUSEPORT``.

    Args:
        args: Parsed CLI arguments (``port`` will be overridden per-backend).
        backend_ports: Port numbers for each backend (len == num_api_servers).
        backend_socks: Pre-bound sockets for each backend.
        addresses: ZMQ addresses returned by :func:`launch_core_engines`.
        num_api_servers: Total number of backend servers to spawn.
        stats_update_address: Optional ZMQ stats address for DP coordinator.

    Returns:
        An :class:`~vllm.v1.utils.APIServerProcessManager` whose
        ``processes`` list holds the N spawned backend processes.
    """
    from multiprocessing import get_context as mp_get_context

    spawn_context = mp_get_context("spawn")
    processes = []

    for i, in_addr, out_addr in zip(
        range(num_api_servers), addresses.inputs, addresses.outputs
    ):
        backend_port = backend_ports[i]
        backend_sock = backend_socks[i]

        # Give each backend its own port so it doesn't clash with the frontend
        # or the other backends.
        backend_args = copy.copy(args)
        backend_args.port = backend_port

        client_config: dict = {
            "input_address": in_addr,
            "output_address": out_addr,
            "client_count": num_api_servers,
            "client_index": i,
        }
        if stats_update_address is not None:
            client_config["stats_update_address"] = stats_update_address

        listen_address = f"http://{backend_args.host or '0.0.0.0'}:{backend_port}"

        proc = spawn_context.Process(
            target=run_api_server_worker_proc,
            name=f"ApiServer_{i}",
            args=(listen_address, backend_sock, backend_args, client_config),
        )
        proc.start()
        processes.append(proc)
        logger.info(
            "Started backend API server %d (PID %d) on port %d",
            i,
            proc.pid,
            backend_port,
        )

    logger.info("Started %d backend API server processes", len(processes))

    # Wrap in APIServerProcessManager for shutdown / finalization.
    # We construct it manually here because its __init__ creates *and starts*
    # processes; we've already started them above.
    import weakref

    manager = object.__new__(APIServerProcessManager)
    manager.processes = processes
    manager._finalizer = weakref.finalize(manager, shutdown, processes)
    return manager
