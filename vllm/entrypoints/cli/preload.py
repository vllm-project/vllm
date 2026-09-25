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
import typing
from itertools import product

from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.serve.utils.api_utils import VLLM_SUBCMD_PARSER_EPILOG
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_cache.protocol import (
    check_ipc_platform_support,
)
from vllm.utils.network_utils import get_distributed_init_method, get_open_port

if typing.TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.utils.argparse_utils import FlexibleArgumentParser
else:
    FlexibleArgumentParser = argparse.ArgumentParser

logger = init_logger(__name__)


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
        parser.add_argument(
            "--weight-cache-master-port",
            type=int,
            default=None,
            help="Rendezvous port for the daemon's own TP group. Must differ "
            "from the engine's --master-port (the daemon holds its group open "
            "while serving) and match across nodes. Required when --nnodes > "
            "1; defaults to a free port for single-node.",
        )
        parser.add_argument(
            "--weight-cache-draft-master-port",
            type=int,
            default=None,
            help="Rendezvous port for the MTP draft daemon group. Defaults to "
            "--weight-cache-master-port + 1 for multi-node, or a free port "
            "for single-node.",
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
        from vllm.model_executor.model_loader.weight_cache.utils import (
            format_daemon_role,
        )

        engine_args = EngineArgs.from_cli_args(args)
        vllm_config = engine_args.create_engine_config()
        if vllm_config.load_config.load_format == "ipc_cache":
            raise ValueError(
                "The weight cache daemon itself must load from disk; use the "
                "default --load-format"
            )
        # Config-only so it can fail before any model loading; the quant
        # method check needs the created model and runs in
        # WeightCacheDaemon.get_model.
        check_ipc_platform_support()
        parallel_config = vllm_config.parallel_config
        _reject_unsupported_parallelism(parallel_config)
        tp_size = parallel_config.tensor_parallel_size
        dp_size = parallel_config.data_parallel_size

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
        # Local index == device index; global rank enumerates DP then TP.
        expected_ready = {
            (format_daemon_role(is_draft), dp_rank * tp_size + tp_rank)
            for (is_draft, _, _), (_, dp_rank, tp_rank) in product(groups, placements)
        }
        procs = [
            ctx.Process(
                target=_run_daemon,
                args=(
                    tp_rank,
                    local_rank,
                    config,
                    init_method,
                    args.weight_cache_socket_dir,
                    ready_queue,
                    is_draft,
                    dp_rank,
                ),
                name=f"vllm-weight-cache-{format_daemon_role(is_draft)}-"
                f"{dp_rank * tp_size + tp_rank}",
            )
            for (is_draft, config, init_method), (
                local_rank,
                dp_rank,
                tp_rank,
            ) in product(groups, placements)
        ]
        for proc in procs:
            proc.start()

        def _shutdown(signum, frame):
            for proc in procs:
                proc.terminate()

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        ready: set[tuple[str, int]] = set()
        while len(ready) < len(expected_ready):
            try:
                ready.add(ready_queue.get(timeout=1.0))
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
        socket_dir_msg = args.weight_cache_socket_dir or "the default socket dir"
        logger.info_once(
            "===== Weight cache daemon READY: node %d/%d serving %d local "
            "rank(s) x %d role(s) in %s =====",
            node_rank,
            nnodes,
            local_world_size,
            len(groups),
            socket_dir_msg,
        )

        for proc in procs:
            proc.join()
        sys.exit(max(proc.exitcode or 0 for proc in procs))

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
