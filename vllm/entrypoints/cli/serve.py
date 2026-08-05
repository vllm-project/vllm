# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import signal
import time

import uvloop

import vllm
import vllm.envs as envs
from vllm.entrypoints.cli.headless_engine import run_headless
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.serve.utils.api_utils import VLLM_SUBCMD_PARSER_EPILOG
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.executor import Executor

# NOTE: cli_args/api_server/dp_supervisor/v1.utils/v1.metrics.prometheus
# imports live inside the functions that use them, not here, so
# `run_headless` (defined in `vllm.entrypoints.cli.headless_engine`, and
# re-exported here for `vllm serve --headless`) can be imported without
# pulling in the OpenAI-server-only machinery it never uses.

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

        if getattr(args, "grpc", False):
            from vllm.entrypoints.grpc_server import serve_grpc

            uvloop.run(serve_grpc(args))
            return

        rust_frontend_path = (
            envs.VLLM_RUST_FRONTEND_PATH if envs.VLLM_USE_RUST_FRONTEND else None
        )

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
        # Multi-port: --data-parallel-multi-port-external-lb
        # External LB: --data-parallel-external-lb or --data-parallel-rank
        # Hybrid LB: --data-parallel-hybrid-lb or --data-parallel-start-rank
        is_external_lb = (
            args.data_parallel_external_lb or args.data_parallel_rank is not None
        )

        # If --data_parallel_multi_port_external_lb and --data_parallel_hybrid_lb
        # are unset, default to hybrid if --data-parallel-start-rank is set
        is_hybrid_lb = is_multi_port = False
        if (
            not args.data_parallel_hybrid_lb
            and not args.data_parallel_multi_port_external_lb
        ):
            is_hybrid_lb = args.data_parallel_start_rank is not None
        else:
            is_hybrid_lb = args.data_parallel_hybrid_lb
            is_multi_port = args.data_parallel_multi_port_external_lb

        if sum([is_multi_port, is_external_lb, is_hybrid_lb]) > 1:
            raise ValueError(
                "Cannot use more than one data parallel load balancing mode. "
                "Choose one of: --data-parallel-multi-port-external-lb, "
                "--data-parallel-external-lb (or --data-parallel-rank), "
                "--data-parallel-hybrid-lb (or --data-parallel-start-rank)."
            )

        # Default api_server_count if not explicitly set.
        # - Multi-port: 1 (supervisor spawns one server per local DP rank)
        # - Rust frontend: 1 (not applicable as it's multithreaded)
        # - External LB: 1 (external LB handles distribution)
        # - Hybrid LB: Use local DP size (internal LB for local ranks only)
        # - Internal LB: Use full DP size
        if args.api_server_count is None:
            if is_multi_port or is_external_lb or rust_frontend_path:
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
        elif rust_frontend_path and args.api_server_count > 1:
            logger.warning(
                "Ignoring --api-server-count=%d when using rust front-end process",
                args.api_server_count,
            )
            args.api_server_count = 1

        # Elastic EP currently only supports running with at most one API server.
        if getattr(args, "enable_elastic_ep", False) and args.api_server_count > 1:
            logger.warning(
                "Elastic EP only supports running with with at most one API server. "
                "Capping api_server_count from %d to 1.",
                args.api_server_count,
            )
            args.api_server_count = 1

        if is_multi_port:
            from vllm.entrypoints.openai.dp_supervisor import run_dp_supervisor

            run_dp_supervisor(args)
        elif args.api_server_count < 1:
            run_headless(args)
        elif args.api_server_count > 1 or rust_frontend_path:
            run_multi_api_server(args)
        else:
            # Single API server (this process).
            from vllm.entrypoints.launchers.api_server.entry import run_server

            args.api_server_count = None
            uvloop.run(run_server(args))

    def validate(self, args: argparse.Namespace) -> None:
        from vllm.entrypoints.openai.cli_args import validate_parsed_serve_args

        validate_parsed_serve_args(args)

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        from vllm.entrypoints.openai.cli_args import make_arg_parser

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


def run_multi_api_server(args: argparse.Namespace):
    from vllm.entrypoints.launchers.api_server.entry import setup_server
    from vllm.v1.metrics.prometheus import setup_multiprocess_prometheus
    from vllm.v1.utils import (
        APIServerProcessManager,
        RustFrontendProcessManager,
        wait_for_completion_or_failure,
    )

    assert not args.headless
    rust_frontend_path = (
        envs.VLLM_RUST_FRONTEND_PATH if envs.VLLM_USE_RUST_FRONTEND else None
    )
    num_api_servers: int = args.api_server_count
    assert num_api_servers > 0

    if rust_frontend_path and num_api_servers > 1:
        raise ValueError(
            "VLLM_RUST_FRONTEND_PATH does not support api_server_count > 1"
        )

    if num_api_servers > 1:
        setup_multiprocess_prometheus()

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

    listen_address, sock = setup_server(args, reuse_port=num_api_servers > 1)

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

    api_server_manager: APIServerProcessManager | RustFrontendProcessManager | None = (
        None
    )

    from vllm.v1.engine.utils import get_engine_zmq_addresses, launch_core_engines

    # Defer port allocation to the child's bind() to avoid TOCTOU, except
    # for Rust front-end and Ray DP, which can't see the post-bind rebind
    # (CLI-arg subprocess / pickled-into-actor snapshot respectively) and
    # so pre-allocate driver-side -- reintroducing the original race only
    # there.
    is_ray_dp = parallel_config.data_parallel_backend == "ray"
    addresses = get_engine_zmq_addresses(
        vllm_config,
        num_api_servers,
        defer_api_server_ports=not (rust_frontend_path or is_ray_dp),
    )

    with launch_core_engines(
        vllm_config, executor_class, log_stats, addresses
    ) as engine_launch:
        local_engine_manager = engine_launch.engine_manager
        coordinator = engine_launch.coordinator
        addresses = engine_launch.addresses
        stats_update_address = (
            coordinator.get_stats_publish_address() if coordinator else None
        )

        if rust_frontend_path:
            if parallel_config.local_engines_only:
                expected_engine_start_index = parallel_config.data_parallel_rank
                expected_engine_count = parallel_config.data_parallel_size_local
            else:
                expected_engine_start_index = 0
                expected_engine_count = parallel_config.data_parallel_size
            # Start rust front-end process.
            api_server_manager = RustFrontendProcessManager(
                binary_path=rust_frontend_path,
                sock=sock,
                args=args,
                input_address=addresses.inputs[0],
                output_address=addresses.outputs[0],
                engine_start_index=expected_engine_start_index,
                engine_count=expected_engine_count,
                data_parallel_size=parallel_config.data_parallel_size,
                stats_update_address=stats_update_address,
            )
        else:
            # Start API server(s).
            api_server_manager = APIServerProcessManager(
                listen_address=listen_address,
                sock=sock,
                args=args,
                num_servers=num_api_servers,
                input_addresses=addresses.inputs,
                output_addresses=addresses.outputs,
                stats_update_address=stats_update_address,
                tensor_queue=engine_launch.tensor_queue,
            )

            if not is_ray_dp:
                # Forward each child's bound endpoints to the engine handshake
                # (runs on ``with`` exit). Skipped for Ray DP, where addresses
                # are pre-allocated above and Ray actors already hold them.
                actual_inputs, actual_outputs = (
                    api_server_manager.gather_actual_addresses()
                )
                addresses.inputs = actual_inputs
                addresses.outputs = actual_outputs

        # Set frontend processes to watch during engine startup.
        # If any of these processes exit before the engines are up, the engine startup
        # will be aborted with an error.
        engine_launch.watched_frontend_processes = api_server_manager.processes

    # Wait for API servers.
    try:
        wait_for_completion_or_failure(
            api_server_manager=api_server_manager,
            engine_manager=local_engine_manager,
            coordinator=coordinator,
        )
    finally:
        timeout = shutdown_by = None
        if shutdown_requested:
            timeout = vllm_config.shutdown_timeout
            shutdown_by = time.monotonic() + timeout
            logger.info("Waiting up to %d seconds for processes to exit", timeout)

        def to_timeout(deadline: float | None) -> float | None:
            return (
                deadline if deadline is None else max(deadline - time.monotonic(), 0.0)
            )

        api_server_manager.shutdown(timeout=timeout)
        if local_engine_manager:
            local_engine_manager.shutdown(timeout=to_timeout(shutdown_by))
        if coordinator:
            coordinator.shutdown(timeout=to_timeout(shutdown_by))
