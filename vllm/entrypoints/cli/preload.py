# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The `preload` subcommand for the vLLM CLI.

Launches one weight cache daemon per GPU; each daemon holds the
post-quantized, TP-sharded weights of its rank in GPU memory and serves
CUDA IPC handles to vLLM engines over a Unix domain socket, so restarting
engines can map the weights via zero-copy IPC instead of reloading from disk:

    vllm preload --model /path/to/model --tensor-parallel-size 4

Engines then load from the daemons with:

    vllm serve /path/to/model --tensor-parallel-size 4 --load-format ipc_cache

Tensor, expert and data parallelism are supported, including across nodes;
see vllm/model_executor/model_loader/weight_cache/daemon.py for the rank
layout and rendezvous details.
"""

import argparse
import multiprocessing
import queue
import signal
import sys
import threading
import time
import typing
from collections.abc import Sequence
from copy import deepcopy
from itertools import product
from multiprocessing.connection import wait
from multiprocessing.process import BaseProcess

from fastapi import FastAPI
from fastapi.responses import Response

from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.common.startup import VLLM_SUBCMD_PARSER_EPILOG
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_cache.protocol import (
    check_ipc_platform_support,
)
from vllm.usage.usage_lib import UsageContext
from vllm.utils.network_utils import get_distributed_init_method, get_open_port

if typing.TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.utils.argparse_utils import FlexibleArgumentParser
else:
    FlexibleArgumentParser = argparse.ArgumentParser

logger = init_logger(__name__)


def _prepare_cudagraphs(engine_args: EngineArgs, socket_dir: str | None) -> None:
    """Run one disposable SAVE engine against the already-serving daemon."""
    from vllm.compilation.foundry_adapter import install_hooks
    from vllm.v1.engine.llm_engine import LLMEngine

    engine_args = deepcopy(engine_args)
    engine_args.load_format = "ipc_cache"
    engine_args.model_loader_extra_config = {
        "mode": "zero_copy",
        "fallback": False,
        **({"socket_dir": socket_dir} if socket_dir is not None else {}),
    }
    install_hooks(engine_args.compilation_config)
    engine = LLMEngine.from_engine_args(
        engine_args,
        usage_context=UsageContext.OPENAI_API_SERVER,
        enable_multiprocessing=True,
    )
    try:
        logger.info("Foundry preparation engine completed SAVE")
    finally:
        engine.engine_core.shutdown()


class _HealthState:
    def __init__(self, processes: Sequence[BaseProcess]) -> None:
        self.processes = processes
        self._ready = False
        self._lock = threading.Lock()

    def mark_ready(self) -> None:
        with self._lock:
            self._ready = True

    @property
    def is_ready(self) -> bool:
        with self._lock:
            ready = self._ready
        return ready and not any(proc.exitcode is not None for proc in self.processes)


def _build_health_app(state: _HealthState) -> FastAPI:
    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)

    @app.get("/health", include_in_schema=False)
    async def health() -> Response:
        return Response(status_code=200 if state.is_ready else 503)

    return app


def _start_health_server(
    host: str, port: int, state: _HealthState
) -> tuple[typing.Any, threading.Thread]:
    import uvicorn

    config = uvicorn.Config(
        _build_health_app(state),
        host=host,
        port=port,
        log_level="error",
        access_log=False,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(
        target=server.run,
        name="vllm-weight-cache-health",
        daemon=True,
    )
    thread.start()
    return server, thread


def _validate_seed_args(
    args: argparse.Namespace, *, world_size: int, num_groups: int
) -> None:
    """Validate the seed CLI options before any daemon spawns."""
    for flag, host, port in (
        ("--seed-listen", args.seed_listen_host, args.seed_listen_port),
        ("--seed-source", args.seed_source_host, args.seed_source_port),
    ):
        if (host is None) != (port is None):
            raise ValueError(f"{flag}-host and {flag}-port must be provided together")
    has_endpoint = any(
        value is not None
        for value in (
            args.seed_listen_host,
            args.seed_listen_port,
            args.seed_source_host,
            args.seed_source_port,
        )
    )
    if has_endpoint != bool(args.seed_token):
        raise ValueError("Preload daemons must set --seed-token to enable D2D transfer")
    for base_port in (args.seed_listen_port, args.seed_source_port):
        if base_port is None:
            continue
        max_port = base_port + num_groups * world_size - 1
        if not 1 <= base_port <= 65535 or max_port > 65535:
            raise ValueError(
                f"Daemon-to-daemon port range {base_port}-{max_port} exceeds 65535"
            )


class PreloadSubcommand(CLISubcommand):
    """The `preload` subcommand for the vLLM CLI."""

    name = "preload"

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> None:
        parser.add_argument(
            "--prepare-cudagraphs",
            action="store_true",
            help="Prepare persisted CUDA graphs using one temporary IPC engine "
            "before reporting ready. Requires a Foundry SAVE compilation "
            "configuration and a single-GPU engine.",
        )
        EngineArgs.add_cli_args(parser)
        parser.add_argument(
            "--weight-cache-socket-dir",
            type=str,
            default=None,
            help="Directory for the daemon Unix sockets (default: tempdir).",
        )
        parser.add_argument(
            "--weight-cache-master-port",
            type=int,
            default=None,
            help="Rendezvous port for the daemon's own TP group. Must differ "
            "from the engine's --master-port (the daemons may still be loading "
            "when the engine starts) and match across nodes. Required when "
            "--nnodes > 1; defaults to a free port for single-node.",
        )
        parser.add_argument(
            "--weight-cache-draft-master-port",
            type=int,
            default=None,
            help="Rendezvous port for the MTP draft daemon group. Defaults to "
            "--weight-cache-master-port + 1 for multi-node, or a free port "
            "for single-node.",
        )
        parser.add_argument(
            "--weight-cache-health-port",
            type=int,
            default=None,
            help="Port for the parent /health endpoint; disabled by default.",
        )
        parser.add_argument(
            "--weight-cache-health-host",
            type=str,
            default="0.0.0.0",
            help="Host for the parent /health endpoint (default: 0.0.0.0).",
        )
        parser.add_argument(
            "--seed-listen-host",
            type=str,
            default=None,
            help="Host to bind for authenticated daemon-to-daemon requests.",
        )
        parser.add_argument(
            "--seed-listen-port",
            type=int,
            default=None,
            help="Base port for daemon-to-daemon requests; each daemon binds "
            "base_port + group offset + its global rank.",
        )
        parser.add_argument(
            "--seed-source-host",
            type=str,
            default=None,
            help="Host of a source daemon replica to seed warmup caches from.",
        )
        parser.add_argument(
            "--seed-source-port",
            type=int,
            default=None,
            help="Base port of the source daemon replica; each daemon dials "
            "base_port + group offset + its global rank.",
        )
        parser.add_argument(
            "--seed-token",
            type=str,
            default=None,
            help="Shared secret for daemon-to-daemon requests. The JSON "
            "connection is unencrypted; use a trusted network.",
        )
        parser.add_argument(
            "--warmup-cache-stores",
            type=str,
            default=None,
            help="Comma-separated shared store backends mirroring the "
            "warmup caches (e.g. 'mooncake', connected via "
            "MOONCAKE_CONFIG_PATH). The local disk cache is always the "
            "first tier; these add shared tiers that reads fall back to "
            "and tuned tables are published to.",
        )

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        # Imported here so `vllm --help` and friends stay free of torch.
        from vllm.model_executor.model_loader.weight_cache.daemon import (
            _reject_unsupported_parallelism,
            _run_daemon,
            get_draft_daemon_config,
            plan_local_ranks,
        )
        from vllm.model_executor.model_loader.weight_cache.seed import SeedConfig
        from vllm.model_executor.model_loader.weight_cache.utils import (
            format_daemon_role,
        )

        engine_args = EngineArgs.from_cli_args(args)
        # Warmup context for flashinfer autotune.
        usage_context = (
            UsageContext.OPENAI_API_SERVER
            if engine_args.kernel_config.enable_flashinfer_autotune is not False
            else None
        )
        vllm_config = engine_args.create_engine_config(usage_context=usage_context)
        if vllm_config.compilation_config.graph_extension_config_path:
            from vllm.compilation.foundry_adapter.config import (
                CUDAGraphExtensionMode,
                get_graph_extension_mode,
            )
            from vllm.compilation.foundry_adapter.runtime import setup_ld_preload_env

            if get_graph_extension_mode() != CUDAGraphExtensionMode.SAVE:
                raise ValueError("vllm preload requires Foundry SAVE mode")
            setup_ld_preload_env()
        if vllm_config.load_config.load_format == "ipc_cache":
            raise ValueError(
                "The weight cache daemon itself must load from disk; use the "
                "default --load-format"
            )
        if args.prepare_cudagraphs:
            parallel = vllm_config.parallel_config
            compilation = vllm_config.compilation_config
            if (
                not compilation.graph_extension_config_path
                or vllm_config.model_config.enforce_eager
                or not compilation.cudagraph_mode.has_full_cudagraphs()
                or parallel.world_size != 1
                or parallel.data_parallel_size != 1
                or parallel.nnodes != 1
                or vllm_config.speculative_config is not None
            ):
                raise ValueError(
                    "--prepare-cudagraphs requires a single-GPU engine with "
                    "FULL CUDA graphs, a Foundry SAVE configuration, and no "
                    "speculative decoding or --enforce-eager"
                )
        # Config-only so it can fail before any model loading; the quant
        # method check needs the created model and runs in
        # WeightCacheDaemon.get_model.
        check_ipc_platform_support()
        parallel_config = vllm_config.parallel_config
        _reject_unsupported_parallelism(parallel_config)
        tp_size = parallel_config.tensor_parallel_size
        pp_size = parallel_config.pipeline_parallel_size
        dp_size = parallel_config.data_parallel_size
        world_size = dp_size * pp_size * tp_size

        placements = plan_local_ranks(parallel_config)
        local_world_size = len(placements)
        if dp_size == 1:
            nnodes = parallel_config.nnodes
            node_rank = parallel_config.node_rank
            master_addr = parallel_config.master_addr
        else:
            master_addr = parallel_config.data_parallel_master_ip
            if parallel_config.nnodes > 1:
                nnodes = parallel_config.nnodes
                node_rank = parallel_config.node_rank
            else:
                nnodes = dp_size // parallel_config.data_parallel_size_local
                node_rank = parallel_config.data_parallel_rank // (
                    parallel_config.data_parallel_size_local
                )
            if nnodes > 1 and master_addr in ("127.0.0.1", "localhost"):
                raise ValueError(
                    "Data parallelism across nodes requires a reachable "
                    "--data-parallel-address for the daemon rendezvous"
                )

        # The daemon forms its own world group and holds it open while
        # serving, so it needs a rendezvous port distinct from the engine's.
        # All nodes must agree on it; single-node can auto-pick a free port.
        # The master address is the engine's --master-addr (TP across nodes)
        # or --data-parallel-address.
        if nnodes > 1 and args.weight_cache_master_port is None:
            raise ValueError(
                "--weight-cache-master-port is required when the daemons span nodes"
            )
        master_port = args.weight_cache_master_port or get_open_port()
        distributed_init_method = get_distributed_init_method(master_addr, master_port)

        # (is_draft, vllm_config, rendezvous) per daemon group.
        groups: list[tuple[bool, VllmConfig, str]] = [
            (False, vllm_config, distributed_init_method)
        ]
        draft_vllm_config = get_draft_daemon_config(vllm_config)
        if draft_vllm_config is not None:
            draft_master_port = args.weight_cache_draft_master_port or (
                master_port + 1 if nnodes > 1 else get_open_port()
            )
            if draft_master_port == master_port:
                raise ValueError(
                    "--weight-cache-draft-master-port must differ from "
                    "--weight-cache-master-port"
                )
            groups.append(
                (
                    True,
                    draft_vllm_config,
                    get_distributed_init_method(master_addr, draft_master_port),
                )
            )

        ctx = multiprocessing.get_context("spawn")
        ready_queue: multiprocessing.Queue[tuple[str, int]] = ctx.Queue()
        # Local index == device index; global rank enumerates DP, PP, then TP.
        expected_ready = {
            (
                format_daemon_role(is_draft),
                dp_rank * pp_size * tp_size + pp_rank * tp_size + tp_rank,
            )
            for (is_draft, _, _), (_, dp_rank, pp_rank, tp_rank) in product(
                groups, placements
            )
        }
        _validate_seed_args(args, world_size=world_size, num_groups=len(groups))
        warmup_cache_stores = [
            name.strip()
            for name in (args.warmup_cache_stores or "").split(",")
            if name.strip()
        ]

        def _seed_endpoint(
            host: str | None, base_port: int | None, offset: int
        ) -> tuple[str, int] | None:
            # Validated above: host and port are provided together.
            if host is None or base_port is None:
                return None
            return host, base_port + offset

        procs: list[BaseProcess] = []
        for group_index, (is_draft, config, init_method) in enumerate(groups):
            # Draft daemons live past the target group's port block so both
            # roles can be reached over a single seed host:base_port.
            port_offset = group_index * world_size
            for local_rank, dp_rank, pp_rank, tp_rank in placements:
                global_rank = dp_rank * pp_size * tp_size + pp_rank * tp_size + tp_rank
                offset = port_offset + global_rank
                seed_config = SeedConfig(
                    listen=_seed_endpoint(
                        args.seed_listen_host, args.seed_listen_port, offset
                    ),
                    source=_seed_endpoint(
                        args.seed_source_host, args.seed_source_port, offset
                    ),
                    token=args.seed_token,
                )
                procs.append(
                    ctx.Process(
                        target=_run_daemon,
                        args=(
                            global_rank,
                            local_rank,
                            config,
                            init_method,
                            args.weight_cache_socket_dir,
                            ready_queue,
                            is_draft,
                            dp_rank,
                            pp_rank,
                            seed_config,
                            warmup_cache_stores,
                        ),
                        name=(
                            f"vllm-weight-cache-{format_daemon_role(is_draft)}-"
                            f"{global_rank}"
                        ),
                    )
                )
        health_state = _HealthState(procs)
        health_server = None
        health_thread = None
        if args.weight_cache_health_port is not None:
            health_server, health_thread = _start_health_server(
                args.weight_cache_health_host,
                args.weight_cache_health_port,
                health_state,
            )
        shutdown_requested = False
        exit_code = 0

        def _shutdown(signum, frame):
            nonlocal shutdown_requested, exit_code
            shutdown_requested = True
            exit_code = 128 + signum

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        started: list[BaseProcess] = []
        preparation: BaseProcess | None = None
        preparation_start = 0.0
        try:
            for proc in procs:
                if shutdown_requested:
                    break
                proc.start()
                started.append(proc)

            ready: set[tuple[str, int]] = set()
            while not shutdown_requested:
                dead = [p for p in procs if p.exitcode is not None]
                if dead:
                    logger.error(
                        "Weight cache daemon rank(s) exited during %s "
                        "(exitcodes=%s); shutting down.",
                        "serving" if ready == expected_ready else "startup",
                        [p.exitcode for p in dead],
                    )
                    # Preserve the failure before terminating healthy siblings.
                    # A clean rank exit is also unexpected while serving.
                    failure = next((code for p in dead if (code := p.exitcode)), 1)
                    exit_code = 128 - failure if failure < 0 else failure
                    break
                if health_state.is_ready:
                    wait([p.sentinel for p in procs], timeout=1.0)
                    continue
                if ready != expected_ready:
                    try:
                        ready.add(ready_queue.get(timeout=1.0))
                    except queue.Empty:
                        continue
                    if ready != expected_ready:
                        continue
                if args.prepare_cudagraphs:
                    if preparation is None:
                        preparation_start = time.perf_counter()
                        preparation = ctx.Process(
                            target=_prepare_cudagraphs,
                            args=(engine_args, args.weight_cache_socket_dir),
                            name="vllm-cudagraph-save",
                        )
                        logger.info("Preparing CUDA graphs with one SAVE engine")
                        preparation.start()
                        started.append(preparation)
                    if preparation.exitcode is None:
                        wait(
                            [p.sentinel for p in procs] + [preparation.sentinel],
                            timeout=1.0,
                        )
                        continue
                    preparation.join()
                    if preparation.exitcode != 0:
                        logger.error(
                            "CUDA graph preparation failed: exitcode=%s",
                            preparation.exitcode,
                        )
                        code = preparation.exitcode
                        exit_code = 128 - code if code < 0 else code
                        break
                    logger.info(
                        "CUDA graph preparation completed in %.3f s",
                        time.perf_counter() - preparation_start,
                    )
                health_state.mark_ready()
                socket_dir_msg = (
                    args.weight_cache_socket_dir or "the default socket dir"
                )
                logger.info_once(
                    "===== Weight cache daemon READY: node %d/%d serving %d "
                    "local rank(s) x %d role(s) in %s =====",
                    node_rank,
                    nnodes,
                    local_world_size,
                    len(groups),
                    socket_dir_msg,
                )
        finally:
            if health_server is not None:
                health_server.should_exit = True
            for proc in started:
                if proc.is_alive():
                    proc.terminate()
            deadline = time.monotonic() + 10.0
            for proc in started:
                proc.join(timeout=max(0.0, deadline - time.monotonic()))
            for proc in started:
                if proc.is_alive():
                    proc.kill()
            for proc in started:
                proc.join()
            if health_thread is not None:
                health_thread.join()
            ready_queue.close()
        sys.exit(exit_code)

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        preload_parser = subparsers.add_parser(
            self.name,
            help="Launch weight cache daemons (one per GPU) for fast engine restarts.",
            description="Launch weight cache daemons (one per GPU) that "
            "keep post-quantized weights in GPU memory and serve them to "
            "engines over CUDA IPC.",
            usage="vllm preload --model <model> [options]",
        )
        self.add_cli_args(preload_parser)
        preload_parser.epilog = VLLM_SUBCMD_PARSER_EPILOG.format(subcmd=self.name)
        return preload_parser


def cmd_init() -> list[CLISubcommand]:
    return [PreloadSubcommand()]
