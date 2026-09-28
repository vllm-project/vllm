# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight cache daemon for fast engine restarts.

One daemon process per GPU holds the post-quantized, TP-sharded weights of its
rank in GPU memory and serves CUDA IPC handles to vLLM engines over a Unix
domain socket. Restarting engines map the weights via zero-copy IPC instead of
reloading from disk.

Launch one daemon per TP rank with a single command:

    vllm preload \\
        --model /path/to/model --tensor-parallel-size 4

Engines then load from the daemons with:

    vllm serve /path/to/model --tensor-parallel-size 4 \\
        --load-format ipc_cache

Tensor, expert and data parallelism are supported; pipeline parallelism is
rejected at launch.

For multi-node tensor parallelism, run one launcher per node with a shared
rendezvous so the global TP group forms across nodes (CUDA IPC handles are
node-local, so each node serves only its local GPUs' shards). Reuse the same
``--nnodes``/``--node-rank``/``--master-addr`` flags you pass the engine, plus a
``--weight-cache-master-port`` distinct from the engine's ``--master-port``:

    # node 0 (8 local GPUs)
    vllm preload \\
        --model /path/to/model --tensor-parallel-size 16 \\
        --nnodes 2 --node-rank 0 --master-addr 10.0.0.1 \\
        --weight-cache-master-port 29600
    # node 1 (8 local GPUs)
    vllm preload \\
        --model /path/to/model --tensor-parallel-size 16 \\
        --nnodes 2 --node-rank 1 --master-addr 10.0.0.1 \\
        --weight-cache-master-port 29600

The global TP rank of local GPU ``i`` on node ``r`` is
``r * (tp_size // nnodes) + i``, matching vLLM's contiguous per-node rank
assignment, so each engine worker maps its shard from the daemon on its own
node.

For data parallelism (e.g. a TP1 x DP16 x EP decode fleet) run one launcher per
node with the engine's DP placement flags. Local GPU ``i`` serves DP rank
``start_rank + i // tp_size`` and TP rank ``i % tp_size``, and all
``dp_size * tp_size`` daemons form one world group on
``--data-parallel-address``/``--weight-cache-master-port`` so the expert
shards are laid out exactly as in the engine:

    # node r (4 local GPUs)
    vllm preload \\
        --model /path/to/model --tensor-parallel-size 1 --enable-expert-parallel \\
        --data-parallel-size 16 --data-parallel-size-local 4 \\
        --data-parallel-start-rank 4r --data-parallel-address 10.0.0.1 \\
        --weight-cache-master-port 29600

Data parallelism also combines with multi-node tensor parallelism: pass both
flag sets (``--nnodes``/``--node-rank``/``--master-addr`` and the DP flags with
a reachable ``--data-parallel-address``). Each node then serves a contiguous
block of the ``dp_size * tp_size`` global ranks, e.g. TP8 x DP2 on 4 nodes:

    # node r (4 local GPUs)
    vllm preload \\
        --model /path/to/model --tensor-parallel-size 8 --enable-expert-parallel \\
        --nnodes 4 --node-rank r --master-addr 10.0.0.1 \\
        --data-parallel-size 2 --data-parallel-address 10.0.0.1 \\
        --weight-cache-master-port 29600

With MTP, EAGLE or EAGLE3 speculative decoding the launcher additionally
starts a draft daemon group that caches the draft model. It uses its own cache
key, Unix sockets (``*_draft.sock``) and rendezvous port
(``--weight-cache-draft-master-port``, default ``--weight-cache-master-port +
1``), so each process serves exactly one model role. Other draft types are not
cached and keep loading from disk in the engine.
"""

import contextlib
import fcntl
import multiprocessing
import os
import socket
from collections.abc import Callable

import torch

from vllm.config import (
    ParallelConfig,
    VllmConfig,
    replace,
    set_current_vllm_config,
)
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.model_loader.weight_cache.protocol import (
    TensorEntry,
    WeightCacheKey,
    WeightCacheUnavailableError,
    check_ipc_quant_support,
    ensure_private_socket_dir,
    get_current_device_uuid,
    get_socket_path,
    recv_msg,
    send_msg,
    verify_peer_is_owner,
)
from vllm.model_executor.model_loader.weight_cache.utils import (
    export_model_attrs,
    format_daemon_role,
    is_draft_model_cacheable,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.worker.workspace import init_workspace_manager

logger = init_logger("vllm.model_executor.model_loader.weight_cache.daemon")


def export_entries(
    model: torch.nn.Module,
) -> tuple[dict[str, TensorEntry], dict[str, str]]:
    """Export a model's tensors, preserving tied-parameter aliases.

    ``named_parameters``/``named_buffers`` are iterated with
    ``remove_duplicate=False`` so tied weights (e.g. ``lm_head.weight`` sharing
    storage with ``embed_tokens.weight``) are not silently dropped. Each unique
    tensor is exported once per call; every additional name that refers to the same
    tensor object is recorded in the returned alias map so the client can
    re-establish the shared identity instead of allocating uninitialized
    memory for it.

    CUDA reduction arguments must be exported separately for each consumer so
    that PyTorch registers a reference for each IPC mapping's lifetime.

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
        local_rank: int,
        distributed_init_method: str,
        socket_dir: str | None = None,
        is_draft: bool = False,
        dp_rank: int = 0,
    ):
        parallel_config = vllm_config.parallel_config
        if parallel_config.data_parallel_size > 1:
            # The engine sets these on each DP rank's workers; the MoE/EP
            # layers read them to place their expert shards.
            tp_size = parallel_config.tensor_parallel_size
            if parallel_config.nnodes > 1:
                world_size = parallel_config.data_parallel_size * tp_size
                local_world = world_size // parallel_config.nnodes
                start_dp_rank = parallel_config.node_rank * local_world // tp_size
            else:
                start_dp_rank = parallel_config.data_parallel_rank
            parallel_config = replace(
                parallel_config,
                data_parallel_rank=dp_rank,
                data_parallel_rank_local=dp_rank - start_dp_rank,
            )
            vllm_config = replace(vllm_config, parallel_config=parallel_config)
        self.vllm_config = vllm_config
        # A draft daemon builds the draft with the target's VllmConfig, like
        # the engine does; only the ModelConfig differs.
        if is_draft:
            spec = vllm_config.speculative_config
            assert spec is not None and spec.draft_model_config is not None
            self.model_config = spec.draft_model_config
        else:
            self.model_config = vllm_config.model_config
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.tp_size = parallel_config.tensor_parallel_size
        self.dp_size = parallel_config.data_parallel_size
        self.global_rank = dp_rank * self.tp_size + tp_rank
        self.world_size = self.dp_size * self.tp_size
        self.local_rank = local_rank
        self.distributed_init_method = distributed_init_method
        self.socket_dir = socket_dir
        self.is_draft = is_draft
        self.role = format_daemon_role(is_draft)
        self.model: torch.nn.Module | None = None
        # Fingerprint before loading: process_weights_after_loading may
        # mutate hf_config.quantization_config.
        self.cache_config = WeightCacheKey.from_model_config(
            self.model_config,
            tp_size=self.tp_size,
            tp_rank=tp_rank,
            dp_size=self.dp_size,
            dp_rank=dp_rank,
            is_draft=is_draft,
        )

    def load_model(self) -> None:
        torch.accelerator.set_device_index(self.local_rank)
        # Outside the config context so the daemon keeps its own rendezvous
        # instead of the engine's; model parallel groups (TP/DP/EP) are then
        # carved out of this world group like in the engine.
        init_distributed_environment(
            world_size=self.world_size,
            rank=self.global_rank,
            distributed_init_method=self.distributed_init_method,
            local_rank=self.local_rank,
            backend=current_platform.dist_backend,
        )
        with set_current_vllm_config(self.vllm_config):
            ensure_model_parallel_initialized(self.tp_size, 1)
            self.model = self.get_model()
        logger.info(
            "Weight cache %s daemon rank %d loaded model",
            self.role,
            self.global_rank,
        )

    def get_model(self) -> torch.nn.Module:
        """Load the daemon's model, composed from the configured loader.

        Runs the quantization check after model creation but before the slow
        weight load, so an unsupported method fails fast. Online quantization
        always fails the check, so load_model's finalize step for it is
        unnecessary here.
        """
        vllm_config = self.vllm_config
        model_config = self.model_config
        load_config = vllm_config.load_config
        loader = get_model_loader(load_config)
        device_config = vllm_config.device_config
        target_device = torch.device(
            device_config.device if load_config.device is None else load_config.device
        )
        # Attention backends may take workspace at construction, like in the
        # engine's worker init.
        init_workspace_manager(target_device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = loader.create_model(vllm_config, model_config)
            check_ipc_quant_support(model)
            loader.load_weights(model, model_config)
            process_weights_after_loading(model, model_config, target_device)
        return model.eval()

    def serve_forever(self, ready_callback: Callable[[], None] | None = None) -> None:
        """Serve requests until terminated.

        The socket is only bound once the model is fully cached, so clients
        get a connection error (and fall back to disk) until the daemon is
        ready.

        Args:
            ready_callback: Invoked once the socket is bound and listening,
                so the launcher can report overall readiness.

        """
        socket_path = self._socket_path
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
        logger.info(
            "Weight cache %s daemon rank %d serving on %s",
            self.role,
            self.global_rank,
            socket_path,
        )
        if ready_callback is not None:
            ready_callback()
        try:
            while True:
                conn, _ = server.accept()
                with conn:
                    try:
                        verify_peer_is_owner(conn)
                        self._handle_connection(conn)
                    except (ConnectionError, EOFError):
                        logger.warning("Client disconnected mid-request")
                    except Exception as e:
                        # Report the error back instead of just closing the
                        # socket, but don't let it take the daemon down.
                        logger.exception(
                            "Error handling weight cache client; continuing"
                        )
                        with contextlib.suppress(OSError):
                            send_msg(conn, {"status": "error", "message": str(e)})
        finally:
            server.close()
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

    @property
    def _socket_path(self) -> str:
        return get_socket_path(
            get_current_device_uuid(),
            self.socket_dir,
            is_draft=self.is_draft,
        )

    def _handle_connection(self, conn: socket.socket) -> None:
        request = recv_msg(conn)
        cmd = request.get("cmd")
        if cmd == "get_state":
            self._handle_get_state(conn, request)
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
        if self.model is None:
            send_msg(conn, {"status": "error", "message": "Weights were released"})
            return
        gpu_uuid = get_current_device_uuid()
        entries, aliases = export_entries(self.model)
        send_msg(
            conn,
            {
                "status": "ok",
                "entries": entries,
                "aliases": aliases,
                "attrs": export_model_attrs(self.model),
                "gpu_uuid": gpu_uuid,
            },
        )
        logger.info_once(
            "Weight cache %s daemon rank %d sent %d tensors (+%d aliases) to engine",
            self.role,
            self.global_rank,
            len(entries),
            len(aliases),
        )

    def _handle_release(self, conn: socket.socket) -> None:
        self.model = None
        torch.accelerator.empty_cache()
        logger.info(
            "Weight cache %s daemon rank %d released cached weights",
            self.role,
            self.global_rank,
        )
        send_msg(conn, {"status": "ok"})


def _run_daemon(
    tp_rank: int,
    local_rank: int,
    vllm_config: VllmConfig,
    distributed_init_method: str,
    socket_dir: str | None,
    ready_queue: "multiprocessing.Queue[tuple[str, int]]",
    is_draft: bool = False,
    dp_rank: int = 0,
) -> None:
    daemon = WeightCacheDaemon(
        vllm_config,
        tp_rank,
        local_rank,
        distributed_init_method,
        socket_dir,
        is_draft,
        dp_rank,
    )
    daemon.load_model()
    daemon.serve_forever(
        ready_callback=lambda: ready_queue.put((daemon.role, daemon.global_rank))
    )


def plan_local_ranks(parallel_config: ParallelConfig) -> list[tuple[int, int, int]]:
    """``(local_rank, dp_rank, tp_rank)`` for every GPU this launcher serves.

    Global ranks enumerate DP then TP: ``global = dp_rank * tp_size + tp_rank``.
    With ``--nnodes`` the engine hands each node a contiguous block of global
    ranks, so node ``r`` serves ``node_rank * local + i``; without it a
    launcher's block starts at ``--data-parallel-start-rank * tp_size``.
    """
    tp_size = parallel_config.tensor_parallel_size
    world_size = parallel_config.data_parallel_size * tp_size
    if parallel_config.nnodes > 1:
        local_world_size = world_size // parallel_config.nnodes
        base = parallel_config.node_rank * local_world_size
    else:
        local_world_size = parallel_config.data_parallel_size_local * tp_size
        base = parallel_config.data_parallel_rank * tp_size
    return [
        (i, (base + i) // tp_size, (base + i) % tp_size)
        for i in range(local_world_size)
    ]


def get_draft_daemon_config(vllm_config: VllmConfig) -> VllmConfig | None:
    """VllmConfig for the draft daemon group"""
    if not is_draft_model_cacheable(vllm_config.speculative_config):
        return None
    assert vllm_config.speculative_config is not None
    return vllm_config.speculative_config.apply_draft_overrides(vllm_config)


def _reject_unsupported_parallelism(parallel_config: ParallelConfig) -> None:
    """Reject pipeline parallelism and placements the daemon cannot map."""
    if parallel_config.pipeline_parallel_size > 1:
        raise ValueError(
            "The weight cache daemon only supports tensor, expert and data "
            "parallelism; pipeline parallelism is not supported"
        )
    dp_size = parallel_config.data_parallel_size
    tp_size = parallel_config.tensor_parallel_size
    if parallel_config.nnodes > 1:
        world_size = dp_size * tp_size
        if world_size % parallel_config.nnodes != 0:
            raise ValueError(
                f"--nnodes ({parallel_config.nnodes}) must evenly divide the "
                f"daemon world size ({world_size} = data-parallel-size "
                f"{dp_size} x tensor-parallel-size {tp_size})"
            )
        return
    if dp_size == 1:
        return
    end_rank = (
        parallel_config.data_parallel_rank + parallel_config.data_parallel_size_local
    )
    if end_rank > dp_size:
        raise ValueError(
            "--data-parallel-start-rank + --data-parallel-size-local "
            f"({end_rank}) exceeds --data-parallel-size ({dp_size})"
        )
