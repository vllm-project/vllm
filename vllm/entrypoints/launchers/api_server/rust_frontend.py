# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Launcher for the Rust frontend process.

Shutdown ordering
-----------------
All teardown goes through a shared ``contextlib.ExitStack``. ``ExitStack``
runs callbacks in **LIFO** order (last registered, first executed), so the
callbacks are registered in the *reverse* of the desired execution order.
The desired execution order is::

    set_timeout -> _api_shutdown -> _engine_shutdown -> _coord_shutdown
                -> grpc_sock cleanup -> sock cleanup

which requires registering them as::

    sock cleanup -> grpc_sock cleanup -> _coord_shutdown -> _engine_shutdown
                -> _api_shutdown -> set_timeout

The reasoning behind the desired order:

* ``set_timeout`` runs first so every later shutdown hook can compute its
  remaining budget via ``to_timeout(signal_utils.shutdown_by)``.
* ``_api_shutdown`` runs next: stop the frontend from accepting new requests.
* ``_engine_shutdown`` then stops the engines feeding that frontend.
* ``_coord_shutdown`` then stops the DP coordinator.
* The ``grpc_sock`` / ``sock`` cleanups run last: the listening sockets are
  used for the entire lifetime of the process and are only released once
  every user of them is gone.
"""

import argparse
import contextlib

import vllm
from vllm import envs
from vllm.config import VllmConfig
from vllm.entrypoints.common.cli_args import propagate_flash_late_interaction
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.coordinator import DPCoordinator
from vllm.v1.engine.utils import (
    CoreEngineActorManager,
    CoreEngineProcManager,
    bind_engine_zmq_listeners,
    launch_core_engines,
)
from vllm.v1.executor import Executor
from vllm.v1.utils import RustFrontendProcessManager, wait_for_completion_or_failure

from ..launcher import setup_server
from ..utils import signal_utils
from ..utils.listen import (
    cleanup_listen_socket,
    setup_listen_address,
)
from ..utils.signal_utils import set_signal_handler, set_timeout, to_timeout

logger = init_logger(__name__)


def _run_rust_frontend(
    rust_frontend_path: str,
    args: argparse.Namespace,
    num_api_servers: int,
    shutdown_timeout_ref: list[float | None],
    exit_stack: contextlib.ExitStack,
) -> None:
    set_signal_handler()

    # Bind the HTTP and gRPC listen sockets before create_engine_config().
    # Otherwise get_open_ports_list() (DP>1) may pick --port / --grpc-port
    # as the DP master port while they are still unbound, and the DP
    # process-group init then fails with EADDRINUSE. This preserves the
    # "bind before the engine starts" ordering required by #8204.
    listen_address, sock = setup_server(
        args,
        reuse_port=num_api_servers > 1,
    )
    exit_stack.callback(cleanup_listen_socket, sock, args.uds or None)

    # `--grpc-port` is only accepted for the Rust frontend, which inherits
    # this listener like the HTTP one. gRPC follows the HTTP TCP host, or
    # IPv4 loopback when HTTP uses a Unix socket.
    grpc_sock = None
    if args.grpc_port is not None:
        grpc_host = "127.0.0.1" if args.uds else (args.host or "")
        _, grpc_sock = setup_listen_address(
            host=grpc_host, port=args.grpc_port, reuse_port=False
        )
        exit_stack.callback(cleanup_listen_socket, grpc_sock)

    engine_args = vllm.AsyncEngineArgs.from_cli_args(args)
    propagate_flash_late_interaction(args, engine_args)
    engine_args._api_process_count = num_api_servers
    engine_args._api_process_rank = -1

    usage_context = UsageContext.OPENAI_API_SERVER
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)
    shutdown_timeout_ref[0] = vllm_config.shutdown_timeout

    executor_class = Executor.get_class(vllm_config)
    log_stats = not args.disable_log_stats

    parallel_config = vllm_config.parallel_config
    dp_rank = parallel_config.data_parallel_rank
    assert parallel_config.local_engines_only or dp_rank == 0

    zmq_listeners = bind_engine_zmq_listeners(vllm_config, num_api_servers)
    addresses = zmq_listeners.addresses

    # Nested ExitStack so we can control exactly when
    # `launch_core_engines.__exit__` runs.
    engine_stack = exit_stack.enter_context(contextlib.ExitStack())
    engine_launch = engine_stack.enter_context(
        launch_core_engines(
            vllm_config,
            executor_class,
            log_stats,
            addresses,
        )
    )

    local_engine_manager = engine_launch.engine_manager
    coordinator = engine_launch.coordinator
    stats_update_address = (
        coordinator.get_stats_publish_address() if coordinator else None
    )

    if parallel_config.local_engines_only:
        expected_engine_start_index = parallel_config.data_parallel_rank
        expected_engine_count = parallel_config.data_parallel_size_local
    else:
        expected_engine_start_index = 0
        expected_engine_count = parallel_config.data_parallel_size

    api_server_manager = RustFrontendProcessManager(
        binary_path=rust_frontend_path,
        sock=sock,
        grpc_sock=grpc_sock,
        args=args,
        input_listener=zmq_listeners.inputs[0],
        output_listener=zmq_listeners.outputs[0],
        engine_start_index=expected_engine_start_index,
        engine_count=expected_engine_count,
        data_parallel_size=parallel_config.data_parallel_size,
        stats_update_address=stats_update_address,
    )
    engine_launch.watched_frontend_processes = api_server_manager.processes

    # Equivalent to leaving `with launch_core_engines(...)` in the original
    # code.
    engine_stack.close()

    if coordinator is not None:
        exit_stack.callback(_coord_shutdown, coordinator)
    if local_engine_manager is not None:
        exit_stack.callback(_engine_shutdown, local_engine_manager)
    exit_stack.callback(_api_shutdown, api_server_manager, vllm_config)

    wait_for_completion_or_failure(
        api_server_manager=api_server_manager,
        engine_manager=local_engine_manager,
        coordinator=coordinator,
    )


def _coord_shutdown(coordinator: DPCoordinator):
    coordinator.shutdown(timeout=to_timeout(signal_utils.shutdown_by))


def _engine_shutdown(
    local_engine_manager: CoreEngineProcManager | CoreEngineActorManager,
):
    local_engine_manager.shutdown(timeout=to_timeout(signal_utils.shutdown_by))


def _api_shutdown(
    api_server_manager: RustFrontendProcessManager, vllm_config: VllmConfig
):
    timeout = vllm_config.shutdown_timeout if signal_utils.shutdown_requested else None
    api_server_manager.shutdown(timeout=timeout)


def run_rust_frontend(args: argparse.Namespace) -> None:
    assert not args.headless

    rust_frontend_path = (
        envs.VLLM_RUST_FRONTEND_PATH if envs.VLLM_USE_RUST_FRONTEND else None
    )
    assert rust_frontend_path is not None

    num_api_servers: int = args.api_server_count
    if num_api_servers != 1:
        logger.warning(
            "Ignoring --api-server-count=%d when using rust front-end process",
            args.api_server_count,
        )
        num_api_servers = 1

    shutdown_timeout_ref: list[float | None] = [None]

    with contextlib.ExitStack() as exit_stack:
        try:
            _run_rust_frontend(
                rust_frontend_path,
                args,
                num_api_servers,
                shutdown_timeout_ref,
                exit_stack,
            )
        except KeyboardInterrupt:
            logger.info_once("[shutdown] API server: interrupted by user.")
        except Exception:
            logger.exception("[shutdown] API server: unexpected error during shutdown")
            raise
        finally:
            exit_stack.callback(set_timeout, shutdown_timeout_ref[0])
