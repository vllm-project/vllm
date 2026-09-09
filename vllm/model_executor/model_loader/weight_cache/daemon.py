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

A second replica can seed its daemons from the first replica without
reading the checkpoint again. Pass one source Unix socket per TP rank for
same-host peer copies, or one ``host:port`` per rank with
``--weight-cache-seed-backend rdma`` for cross-host Mooncake copies. The source
must expose ``--weight-cache-listen-addr host:base_port`` and
``--weight-cache-seed-token`` for cross-host transfers.

Engines then load from the daemons with:

    vllm serve /path/to/model --tensor-parallel-size 4 \\
        --load-format ipc_cache

Only tensor parallelism is supported; pipeline, data, and expert parallelism
are rejected at launch.
"""

import contextlib
import fcntl
import hmac
import multiprocessing
import os
import queue
import select
import signal
import socket
import sys
from collections.abc import Callable
from typing import Any

import torch

from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_cache.protocol import (
    TensorEntry,
    WeightCacheKey,
    WeightCacheUnavailableError,
    check_ipc_quant_support,
    ensure_private_socket_dir,
    get_physical_device_id,
    get_socket_path,
    recv_msg,
    send_msg,
    verify_peer_is_owner,
    verify_socket_owner,
)
from vllm.model_executor.model_loader.weight_cache.seed import (
    PEER_IPC_SEED_SOURCE,
    build_manifest,
    get_seed_source,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)


def _split_seed_addresses(value: str | None) -> list[str]:
    return [address.strip() for address in (value or "").split(",") if address.strip()]


def _seed_address(value: str | None, tp_rank: int) -> str | None:
    addresses = _split_seed_addresses(value)
    if not addresses:
        return None
    if len(addresses) == 1:
        return addresses[0]
    if tp_rank >= len(addresses):
        raise ValueError(
            f"--weight-cache-seed has {len(addresses)} addresses but TP rank "
            f"{tp_rank} needs an address; pass one address per TP rank"
        )
    return addresses[tp_rank]


def _listen_address(value: str | None, tp_rank: int) -> tuple[str, int] | None:
    if not value:
        return None
    host, separator, port = value.rpartition(":")
    if not separator or not host or not port.isdigit():
        raise ValueError(
            f"--weight-cache-listen-addr must be host:base_port, got {value!r}"
        )
    return host, int(port) + tp_rank


def _report_ready(message: str) -> None:
    """Write a readiness message straight to the original stderr descriptor.

    Loading the model pulls in FlashInfer's CuTeDSL JIT compiler, which swaps
    ``sys.stdout``/``sys.stderr`` for in-memory buffers while compiling kernels
    across worker threads. That save/restore races on the interpreter-global
    streams and can leave them (and vLLM's logging handler, which caches the
    original stream object) detached, silently swallowing everything logged
    afterwards -- including the daemon's readiness announcement. Since operators
    rely on that line to know the daemon is serving, write it directly to file
    descriptor 2 so it bypasses the logging machinery entirely.
    """
    with contextlib.suppress(OSError):
        os.write(2, (message + "\n").encode())


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
        seed_addr: str | None = None,
        seed_backend: str = PEER_IPC_SEED_SOURCE,
        listen_addr: tuple[str, int] | None = None,
        seed_token: str | None = None,
    ):
        self.vllm_config = vllm_config
        self.tp_rank = tp_rank
        self.distributed_init_method = distributed_init_method
        self.socket_dir = socket_dir
        self.seed_addr = seed_addr
        self.seed_backend = seed_backend
        self.listen_addr = listen_addr
        self.seed_token = seed_token
        self.model: torch.nn.Module | None = None
        self.entries: dict[str, TensorEntry] = {}
        self.aliases: dict[str, str] = {}
        self.state_tensors: dict[str, torch.Tensor] = {}
        self._seed_sources: dict[str, Any] = {}
        # Fingerprint before loading: process_weights_after_loading may
        # mutate hf_config.quantization_config.
        self.cache_config = WeightCacheKey.from_model_config(
            vllm_config.model_config,
            tp_size=vllm_config.parallel_config.tensor_parallel_size,
            tp_rank=tp_rank,
        )

    def load_model(self) -> None:
        from vllm.model_executor.model_loader import get_model

        torch.accelerator.set_device_index(self.tp_rank)
        if self.seed_addr is not None:
            self._load_from_seed()
        else:
            tp_size = self.cache_config.tp_size
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
        self.state_tensors = {}
        for name, tensor in self.model.named_parameters(remove_duplicate=False):
            if name in self.entries:
                self.state_tensors[name] = tensor.detach()
        for name, tensor in self.model.named_buffers(remove_duplicate=False):
            if name in self.entries:
                self.state_tensors[name] = tensor.detach()

    def _connect_seed(self) -> tuple[socket.socket, bool]:
        assert self.seed_addr is not None
        if self.seed_addr.startswith("/") or self.seed_addr.startswith("./"):
            verify_socket_owner(self.seed_addr, strict_perms=False)
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            target: Any = self.seed_addr
            remote = False
        else:
            host, separator, port = self.seed_addr.rpartition(":")
            if not separator or not host or not port.isdigit():
                raise ValueError(
                    "Seed address must be an absolute Unix socket path or host:port"
                )
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            target = (host, int(port))
            remote = True
        sock.settimeout(300.0)
        try:
            sock.connect(target)
        except OSError:
            sock.close()
            raise
        return sock, remote

    def _load_from_seed(self) -> None:
        sock, remote = self._connect_seed()
        try:
            request: dict[str, Any] = {
                "cmd": "fetch_manifest",
                "seed_backend": self.seed_backend,
            }
            if remote:
                request["token"] = self.seed_token
            send_msg(sock, request)
            response = recv_msg(sock)
        finally:
            sock.close()
        if response.get("status") != "ok":
            raise RuntimeError(
                f"Seed daemon {self.seed_addr} refused the mirror request: "
                f"{response.get('message', response)}"
            )
        source_config = response.get("cache_config")
        if not isinstance(source_config, WeightCacheKey):
            raise RuntimeError("Seed daemon returned no compatible cache config")
        mismatched = self.cache_config.mismatched_fields(source_config)
        if mismatched:
            raise RuntimeError(
                f"Seed daemon cache differs on fields {mismatched}; refusing "
                "to mirror a different weight shard"
            )
        source = self._seed_sources.get(self.seed_backend)
        if source is None:
            source = get_seed_source(self.seed_backend)
            self._seed_sources[self.seed_backend] = source
        manifest = response["manifest"]
        tensors = source.fill(
            manifest, response["seed"], torch.device("cuda", self.tp_rank)
        )
        if set(tensors) != set(manifest):
            raise RuntimeError("Seed backend did not fill the complete tensor manifest")
        self.entries = {
            name: TensorEntry.from_tensor(
                tensors[name], "param" if metadata["is_param"] else "buffer"
            )
            for name, metadata in manifest.items()
        }
        self.aliases = response.get("aliases", {})
        self.state_tensors = tensors
        self.cache_config = source_config
        logger.info(
            "Weight cache daemon rank %d mirrored %d tensors from %s",
            self.tp_rank,
            len(self.entries),
            self.seed_addr,
        )

    def serve_forever(self, ready_callback: Callable[[], None] | None = None) -> None:
        """Serve requests until terminated.

        The socket is only bound once the model is fully cached, so clients
        get a connection error (and fall back to disk) until the daemon is
        ready.

        Args:
            ready_callback: Invoked once the socket is bound and listening,
                so the launcher can report overall readiness.
        """
        socket_path = self._socket_path()
        ensure_private_socket_dir(
            os.path.dirname(socket_path), strict_perms=self.socket_dir is None
        )
        # Hold an exclusive per-GPU lock for the daemon's lifetime so a second
        # daemon cannot remove this daemon's live socket and hijack the path.
        lock_fd = self._acquire_gpu_lock(socket_path)
        if os.path.exists(socket_path):
            os.unlink(socket_path)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(socket_path)
        os.chmod(socket_path, 0o600)
        server.listen()
        tcp_server = None
        if self.listen_addr is not None:
            tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            tcp_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            tcp_server.bind(self.listen_addr)
            tcp_server.listen()
        listeners = [server] + ([tcp_server] if tcp_server is not None else [])
        logger.info(
            "Weight cache daemon rank %d serving on %s", self.tp_rank, socket_path
        )
        print(
            f"Weight cache daemon rank {self.tp_rank} ready: serving on {socket_path}"
        )
        if ready_callback is not None:
            ready_callback()
        try:
            while True:
                readable, _, _ = select.select(listeners, [], [], 1.0)
                for listener in readable:
                    conn, _ = listener.accept()
                    with conn:
                        try:
                            remote = listener is tcp_server
                            if not remote:
                                verify_peer_is_owner(conn)
                            self._handle_connection(conn, remote=remote)
                        except (ConnectionError, EOFError):
                            logger.warning("Client disconnected mid-request")
                        except Exception:
                            # A malformed peer must not take down all clients.
                            logger.exception(
                                "Error handling weight cache client; continuing"
                            )
        finally:
            server.close()
            if tcp_server is not None:
                tcp_server.close()
            for source in self._seed_sources.values():
                source.close()
            if os.path.exists(socket_path):
                os.unlink(socket_path)
            os.close(lock_fd)

    def _acquire_gpu_lock(self, socket_path: str) -> int:
        """Take an exclusive lock guarding this GPU's socket path.

        The lock is advisory and released automatically when the daemon exits
        (or crashes), so a stale socket is only ever removed by whoever owns
        the lock. A running daemon holding it makes a second daemon fail fast
        instead of clobbering the live socket.
        """
        lock_fd = os.open(f"{socket_path}.lock", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            os.close(lock_fd)
            raise WeightCacheUnavailableError(
                f"Another weight cache daemon already owns {socket_path}"
            ) from e
        return lock_fd

    def _socket_path(self) -> str:
        device_index = torch.accelerator.current_device_index()
        gpu_id = get_physical_device_id(device_index)
        if gpu_id is None:
            gpu_id = device_index
        return get_socket_path(gpu_id, self.socket_dir)

    def _handle_connection(self, conn: socket.socket, *, remote: bool = False) -> None:
        request = recv_msg(conn)
        if remote:
            presented = request.get("token")
            if (
                not self.seed_token
                or not isinstance(presented, str)
                or not hmac.compare_digest(presented, self.seed_token)
                or request.get("cmd") != "fetch_manifest"
            ):
                send_msg(
                    conn, {"status": "error", "message": "Unauthorized seed request"}
                )
                return
        cmd = request.get("cmd")
        if cmd == "get_state":
            self._handle_get_state(conn, request)
        elif cmd == "fetch_manifest":
            self._handle_fetch_manifest(conn, request)
        elif cmd == "release":
            self._handle_release(conn)
        else:
            send_msg(conn, {"status": "error", "message": f"Unknown command {cmd!r}"})

    def _handle_get_state(self, conn: socket.socket, request: dict) -> None:
        client_config = request.get("cache_config")
        if not isinstance(client_config, WeightCacheKey):
            send_msg(conn, {"status": "error", "message": "Missing cache_config"})
            return
        mismatched = self.cache_config.mismatched_fields(client_config)
        if mismatched:
            logger.warning("WeightCacheKey mismatch on fields: %s", mismatched)
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

    def _handle_fetch_manifest(self, conn: socket.socket, request: dict) -> None:
        if not self.entries:
            send_msg(conn, {"status": "error", "message": "Weights were released"})
            return
        backend = request.get("seed_backend", PEER_IPC_SEED_SOURCE)
        source = self._seed_sources.get(backend)
        if source is None:
            source = get_seed_source(backend)
            self._seed_sources[backend] = source
        try:
            seed = source.prepare_seed(
                self.entries,
                self.state_tensors,
                get_physical_device_id(torch.accelerator.current_device_index()),
            )
        except Exception as error:
            send_msg(conn, {"status": "error", "message": str(error)})
            return
        send_msg(
            conn,
            {
                "status": "ok",
                "cache_config": self.cache_config,
                "manifest": build_manifest(self.entries),
                "aliases": self.aliases,
                "seed": seed,
            },
        )

    def _handle_release(self, conn: socket.socket) -> None:
        self.entries.clear()
        self.aliases.clear()
        self.state_tensors.clear()
        self.model = None
        torch.accelerator.empty_cache()
        logger.info("Weight cache daemon rank %d released cached weights", self.tp_rank)
        send_msg(conn, {"status": "ok"})

    def _gpu_uuid(self) -> str:
        props = torch.cuda.get_device_properties(
            torch.accelerator.current_device_index()
        )
        return str(props.uuid)


def _run_daemon(
    tp_rank: int,
    vllm_config: VllmConfig,
    distributed_init_method: str,
    socket_dir: str | None,
    seed_addr: str | None,
    seed_backend: str,
    listen_addr: tuple[str, int] | None,
    seed_token: str | None,
    ready_queue: "multiprocessing.Queue[int]",
) -> None:
    daemon = WeightCacheDaemon(
        vllm_config,
        tp_rank,
        distributed_init_method,
        socket_dir,
        seed_addr,
        seed_backend,
        listen_addr,
        seed_token,
    )
    daemon.load_model()
    daemon.serve_forever(ready_callback=lambda: ready_queue.put(tp_rank))


def _reject_unsupported_parallelism(parallel_config: ParallelConfig) -> None:
    """Reject every parallelism mode other than tensor parallelism."""
    unsupported = {
        "pipeline parallelism": parallel_config.pipeline_parallel_size > 1,
        "data parallelism": parallel_config.data_parallel_size > 1,
        "expert parallelism": parallel_config.enable_expert_parallel,
    }
    for name, enabled in unsupported.items():
        if enabled:
            raise ValueError(
                f"The weight cache daemon only supports tensor parallelism; "
                f"{name} is not supported"
            )


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
    parser.add_argument(
        "--weight-cache-seed",
        type=str,
        default=None,
        help="Comma-separated source daemon socket/host addresses, one per TP rank.",
    )
    parser.add_argument(
        "--weight-cache-seed-backend",
        choices=[PEER_IPC_SEED_SOURCE, "rdma"],
        default=PEER_IPC_SEED_SOURCE,
        help="Daemon-to-daemon mover: peer_ipc or rdma.",
    )
    parser.add_argument(
        "--weight-cache-listen-addr",
        type=str,
        default=None,
        help="host:base_port for the cross-node seed control plane.",
    )
    parser.add_argument(
        "--weight-cache-seed-token",
        type=str,
        default=None,
        help="Shared secret required by the cross-node seed control plane.",
    )
    args = parser.parse_args()
    engine_args = EngineArgs.from_cli_args(args)
    vllm_config = engine_args.create_engine_config()
    if args.weight_cache_listen_addr and not args.weight_cache_seed_token:
        raise ValueError(
            "--weight-cache-listen-addr requires --weight-cache-seed-token"
        )
    if vllm_config.load_config.load_format == "ipc_cache":
        raise ValueError(
            "The weight cache daemon itself must load from disk; use the "
            "default --load-format"
        )
    # Checked before loading anything: an unsupported quantization method would
    # otherwise only surface in the engine, after a full load.
    check_ipc_quant_support(vllm_config.model_config, where="daemon")
    parallel_config = vllm_config.parallel_config
    _reject_unsupported_parallelism(parallel_config)
    tp_size = parallel_config.tensor_parallel_size

    distributed_init_method = get_distributed_init_method("127.0.0.1", get_open_port())
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
                _seed_address(args.weight_cache_seed, rank),
                args.weight_cache_seed_backend,
                _listen_address(args.weight_cache_listen_addr, rank),
                args.weight_cache_seed_token,
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
    logger.info_once(
        "===== Weight cache daemon READY: all %d ranks serving in %s =====",
        tp_size,
        args.weight_cache_socket_dir or "the default socket dir",
    )
    _report_ready(
        f"===== Weight cache daemon READY: all {tp_size} rank(s) serving in "
        f"{args.weight_cache_socket_dir or 'the default socket dir'} ====="
    )

    for proc in procs:
        proc.join()
    sys.exit(max(proc.exitcode or 0 for proc in procs))


if __name__ == "__main__":
    main()
