# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight cache daemon for fast engine restarts.

One daemon process per GPU holds the post-quantized, TP-sharded weights of its
rank in GPU memory and serves CUDA IPC handles to vLLM engines over a Unix
domain socket. Restarting engines map the weights via zero-copy IPC instead of
reloading from disk.

Launch one daemon per TP rank with a single command:

    python -m vllm.model_executor.model_loader.weight_cache.daemon \\
        --model /path/to/model --tensor-parallel-size 4

Engines then load from the daemons with:

    vllm serve /path/to/model --tensor-parallel-size 4 \\
        --load-format ipc_cache

Only tensor parallelism is supported; pipeline, data, and expert parallelism
are rejected at launch.
"""

import multiprocessing
import os
import signal
import socket
import sys

import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_cache.protocol import (
    CacheConfig,
    TensorEntry,
    ensure_private_socket_dir,
    get_physical_device_id,
    get_socket_path,
    recv_msg,
    send_msg,
    verify_peer_is_owner,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)


def export_entries(
    model: torch.nn.Module,
) -> tuple[dict[str, TensorEntry], dict[str, str]]:
    """Export a model's tensors, preserving tied-parameter aliases.

    ``named_parameters``/``named_buffers`` are iterated with
    ``remove_duplicate=False`` so tied weights (e.g. ``lm_head.weight`` sharing
    storage with ``embed_tokens.weight``) are not silently dropped. Each unique
    tensor is exported once; every additional name that refers to the same
    tensor object is recorded in the returned alias map so the client can
    re-establish the shared identity instead of allocating uninitialized
    memory for it.

    Returns:
        A ``(entries, aliases)`` pair where ``entries`` maps a canonical name to
        its ``TensorEntry`` and ``aliases`` maps each duplicate name to its
        canonical name.
    """
    entries: dict[str, TensorEntry] = {}
    aliases: dict[str, str] = {}
    canonical_by_id: dict[int, str] = {}

    def _add(name: str, tensor: torch.Tensor, kind: str) -> None:
        canonical = canonical_by_id.get(id(tensor))
        if canonical is not None:
            aliases[name] = canonical
            return
        canonical_by_id[id(tensor)] = name
        entries[name] = TensorEntry.from_tensor(tensor, kind)

    for name, param in model.named_parameters(remove_duplicate=False):
        _add(name, param, "param")
    # named_buffers includes non-persistent buffers (e.g. rotary embedding
    # caches) that state_dict would miss.
    for name, buffer in model.named_buffers(remove_duplicate=False):
        if name in entries or name in aliases:
            continue
        _add(name, buffer, "buffer")
    return entries, aliases


class WeightCacheDaemon:
    """Per-GPU process that loads one TP shard and serves CUDA IPC handles."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        tp_rank: int,
        distributed_init_method: str,
        socket_dir: str | None = None,
    ):
        self.vllm_config = vllm_config
        self.tp_rank = tp_rank
        self.distributed_init_method = distributed_init_method
        self.socket_dir = socket_dir
        self.model: torch.nn.Module | None = None
        self.entries: dict[str, TensorEntry] = {}
        self.aliases: dict[str, str] = {}
        # Fingerprint before loading: process_weights_after_loading may
        # mutate hf_config.quantization_config.
        self.cache_config = CacheConfig.from_model_config(
            vllm_config.model_config,
            tp_size=vllm_config.parallel_config.tensor_parallel_size,
            tp_rank=tp_rank,
        )

    def load_model(self) -> None:
        from vllm.model_executor.model_loader import get_model

        tp_size = self.cache_config.tp_size
        torch.cuda.set_device(self.tp_rank)
        init_distributed_environment(
            world_size=tp_size,
            rank=self.tp_rank,
            distributed_init_method=self.distributed_init_method,
            local_rank=self.tp_rank,
            backend=current_platform.dist_backend,
        )
        with set_current_vllm_config(self.vllm_config):
            ensure_model_parallel_initialized(tp_size, 1)
            self.model = get_model(vllm_config=self.vllm_config)
        self._export_entries()
        logger.info(
            "Weight cache daemon rank %d cached %d tensors",
            self.tp_rank,
            len(self.entries),
        )

    def _export_entries(self) -> None:
        assert self.model is not None
        self.entries, self.aliases = export_entries(self.model)

    def serve_forever(self) -> None:
        """Serve requests until terminated.

        The socket is only bound once the model is fully cached, so clients
        get a connection error (and fall back to disk) until the daemon is
        ready.
        """
        socket_path = self._socket_path()
        ensure_private_socket_dir(
            os.path.dirname(socket_path), strict_perms=self.socket_dir is None
        )
        if os.path.exists(socket_path):
            os.unlink(socket_path)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(socket_path)
        os.chmod(socket_path, 0o600)
        server.listen()
        logger.info(
            "Weight cache daemon rank %d serving on %s", self.tp_rank, socket_path
        )
        try:
            while True:
                conn, _ = server.accept()
                with conn:
                    try:
                        verify_peer_is_owner(conn)
                        self._handle_connection(conn)
                    except (ConnectionError, EOFError):
                        logger.warning("Client disconnected mid-request")
                    except Exception:
                        # A single malformed or malicious request must not take
                        # down the daemon for every other engine on this GPU.
                        logger.exception(
                            "Error handling weight cache client; continuing"
                        )
        finally:
            server.close()
            if os.path.exists(socket_path):
                os.unlink(socket_path)

    def _socket_path(self) -> str:
        device_index = torch.cuda.current_device()
        gpu_id = get_physical_device_id(device_index)
        if gpu_id is None:
            gpu_id = device_index
        return get_socket_path(gpu_id, self.socket_dir)

    def _handle_connection(self, conn: socket.socket) -> None:
        request = recv_msg(conn)
        cmd = request.get("cmd")
        if cmd == "ping":
            send_msg(
                conn,
                {
                    "status": "ok",
                    "cache_config": self.cache_config,
                    "gpu_uuid": self._gpu_uuid(),
                },
            )
        elif cmd == "get_state":
            self._handle_get_state(conn, request)
        elif cmd == "release":
            self._handle_release(conn)
        else:
            send_msg(conn, {"status": "error", "message": f"Unknown command {cmd!r}"})

    def _handle_get_state(self, conn: socket.socket, request: dict) -> None:
        client_config = request.get("cache_config")
        if not isinstance(client_config, CacheConfig):
            send_msg(conn, {"status": "error", "message": "Missing cache_config"})
            return
        mismatched = self.cache_config.mismatched_fields(client_config)
        if mismatched:
            logger.warning("CacheConfig mismatch on fields: %s", mismatched)
            send_msg(conn, {"status": "mismatch", "fields": mismatched})
            return
        if not self.entries:
            send_msg(conn, {"status": "error", "message": "Weights were released"})
            return
        send_msg(
            conn,
            {
                "status": "ok",
                "entries": self.entries,
                "aliases": self.aliases,
                "gpu_uuid": self._gpu_uuid(),
            },
        )

    def _handle_release(self, conn: socket.socket) -> None:
        self.entries.clear()
        self.aliases.clear()
        self.model = None
        torch.cuda.empty_cache()
        logger.info("Weight cache daemon rank %d released cached weights", self.tp_rank)
        send_msg(conn, {"status": "ok"})

    def _gpu_uuid(self) -> str:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        return str(props.uuid)


def _run_daemon(
    tp_rank: int,
    vllm_config: VllmConfig,
    distributed_init_method: str,
    socket_dir: str | None,
) -> None:
    daemon = WeightCacheDaemon(
        vllm_config, tp_rank, distributed_init_method, socket_dir
    )
    daemon.load_model()
    daemon.serve_forever()


def main() -> None:
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.utils.network_utils import get_distributed_init_method, get_open_port

    parser = FlexibleArgumentParser(
        description="Launch weight cache daemons (one per TP rank)."
    )
    EngineArgs.add_cli_args(parser)
    parser.add_argument(
        "--weight-cache-socket-dir",
        type=str,
        default=None,
        help="Directory for the daemon Unix sockets (default: tempdir).",
    )
    args = parser.parse_args()
    engine_args = EngineArgs.from_cli_args(args)
    vllm_config = engine_args.create_engine_config()
    if vllm_config.load_config.load_format == "ipc_cache":
        raise ValueError(
            "The weight cache daemon itself must load from disk; use the "
            "default --load-format"
        )
    parallel_config = vllm_config.parallel_config
    if parallel_config.pipeline_parallel_size > 1:
        raise ValueError(
            "The weight cache daemon only supports tensor parallelism; "
            "pipeline parallelism is not supported"
        )
    if parallel_config.data_parallel_size > 1:
        raise ValueError(
            "The weight cache daemon only supports tensor parallelism; "
            "data parallelism is not supported"
        )
    if getattr(parallel_config, "enable_expert_parallel", False):
        raise ValueError(
            "The weight cache daemon only supports tensor parallelism; "
            "expert parallelism is not supported"
        )
    tp_size = parallel_config.tensor_parallel_size

    distributed_init_method = get_distributed_init_method("127.0.0.1", get_open_port())
    ctx = multiprocessing.get_context("spawn")
    procs = [
        ctx.Process(
            target=_run_daemon,
            args=(
                rank,
                vllm_config,
                distributed_init_method,
                args.weight_cache_socket_dir,
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
    for proc in procs:
        proc.join()
    sys.exit(max(proc.exitcode or 0 for proc in procs))


if __name__ == "__main__":
    main()
