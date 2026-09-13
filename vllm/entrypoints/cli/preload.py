# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The `preload` subcommand for the vLLM CLI.

Launches one weight cache daemon per TP rank; each daemon holds the
post-quantized, TP-sharded weights of its rank in GPU memory and serves
CUDA IPC handles to vLLM engines over a Unix domain socket, so restarting
engines can map the weights via zero-copy IPC instead of reloading from disk:

    vllm preload --model /path/to/model --tensor-parallel-size 4

Engines then load from the daemons with:

    vllm serve /path/to/model --tensor-parallel-size 4 --load-format ipc_cache
"""

import argparse
import multiprocessing
import queue
import signal
import sys
import typing

from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.serve.utils.api_utils import VLLM_SUBCMD_PARSER_EPILOG
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_cache.protocol import (
    check_ipc_platform_support,
)
from vllm.utils.network_utils import get_distributed_init_method, get_open_port

if typing.TYPE_CHECKING:
    from vllm.config import ParallelConfig, VllmConfig
    from vllm.utils.argparse_utils import FlexibleArgumentParser
else:
    FlexibleArgumentParser = argparse.ArgumentParser

logger = init_logger(__name__)


def _reject_unsupported_parallelism(parallel_config: "ParallelConfig") -> None:
    """Reject parallelism modes other than tensor/expert parallelism."""
    unsupported = {
        "pipeline parallelism": parallel_config.pipeline_parallel_size > 1,
        "data parallelism": parallel_config.data_parallel_size > 1,
    }
    for name, enabled in unsupported.items():
        if enabled:
            raise ValueError(
                f"The weight cache daemon only supports tensor and expert "
                f"parallelism; {name} is not supported"
            )


def _run_daemon(
    tp_rank: int,
    vllm_config: "VllmConfig",
    distributed_init_method: str,
    socket_dir: str | None,
    ready_queue: "multiprocessing.Queue[int]",
) -> None:
    from vllm.model_executor.model_loader.weight_cache.daemon import WeightCacheDaemon

    daemon = WeightCacheDaemon(
        vllm_config, tp_rank, distributed_init_method, socket_dir
    )
    daemon.load_model()
    daemon.serve_forever(ready_callback=lambda: ready_queue.put(tp_rank))


class PreloadSubcommand(CLISubcommand):
    """The `preload` subcommand for the vLLM CLI."""

    name = "preload"

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> None:
        EngineArgs.add_cli_args(parser)
        parser.add_argument(
            "--weight-cache-socket-dir",
            type=str,
            default=None,
            help="Directory for the daemon Unix sockets (default: tempdir).",
        )

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        engine_args = EngineArgs.from_cli_args(args)
        vllm_config = engine_args.create_engine_config()
        if vllm_config.load_config.load_format == "ipc_cache":
            raise ValueError(
                "The weight cache daemon itself must load from disk; use the "
                "default --load-format"
            )
        # Config-only so it can fail before any model loading; the quant method
        # check needs the created model and runs in get_daemon_model.
        check_ipc_platform_support()
        parallel_config = vllm_config.parallel_config
        _reject_unsupported_parallelism(parallel_config)
        tp_size = parallel_config.tensor_parallel_size

        distributed_init_method = get_distributed_init_method(
            "127.0.0.1", get_open_port()
        )
        ctx = multiprocessing.get_context("spawn")
        ready_queue: multiprocessing.Queue[int] = ctx.Queue()
        procs = [
            ctx.Process(
                target=_run_daemon,
                args=(
                    rank,
                    vllm_config,
                    distributed_init_method,
                    args.weight_cache_socket_dir,
                    ready_queue,
                ),
                name=f"vllm-weight-cache-daemon-{rank}",
            )
            for rank in range(tp_size)
        ]
        for proc in procs:
            proc.start()

        def _shutdown(signum, frame):
            for proc in procs:
                proc.terminate()

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        ready_ranks: set[int] = set()
        while len(ready_ranks) < tp_size:
            try:
                ready_ranks.add(ready_queue.get(timeout=1.0))
            except queue.Empty:
                dead = [p for p in procs if p.exitcode is not None]
                if dead:
                    logger.error(
                        "Weight cache daemon rank(s) exited during startup "
                        "(exitcodes=%s); shutting down.",
                        [p.exitcode for p in dead],
                    )
                    for proc in procs:
                        proc.terminate()
                    for proc in procs:
                        proc.join()
                    sys.exit(max((p.exitcode or 0) for p in procs))
        logger.info(
            "Weight cache daemon ready: all %d ranks serving in %s",
            tp_size,
            args.weight_cache_socket_dir or "the default socket dir",
        )
        for proc in procs:
            proc.join()
        sys.exit(max(proc.exitcode or 0 for proc in procs))

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        preload_parser = subparsers.add_parser(
            self.name,
            help="Launch weight cache daemons (one per TP rank) for fast "
            "engine restarts.",
            description="Launch weight cache daemons (one per TP rank) that "
            "keep post-quantized weights in GPU memory and serve them to "
            "engines over CUDA IPC.",
            usage="vllm preload --model <model> [options]",
        )
        self.add_cli_args(preload_parser)
        preload_parser.epilog = VLLM_SUBCMD_PARSER_EPILOG.format(subcmd=self.name)
        return preload_parser


def cmd_init() -> list[CLISubcommand]:
    return [PreloadSubcommand()]
