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

Tensor, expert, data and pipeline parallelism are supported.

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
node. With pipeline parallelism each node additionally runs one daemon per PP
stage; the global rank is ``pp_rank * tp_size + tp_rank``.

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

Seeding a replica from another
==============================

A second replica's daemons can fill their shards from a replica that is
already loaded instead of reading the checkpoint again. Such a "mirror"
replica is just a normal preload whose daemons load with the ``peer_cache``
load format: each builds its model on the meta device (the checkpoint's
config files are needed locally, but not its safetensors), forms its world
group and finalizes its config like any other daemon, then pulls this
rank's post-processed weights from the source replica's daemons. Two movers
carry the weights directly between the daemons: ``peer_ipc`` (the default)
copies through CUDA IPC on one host, and ``rdma`` pulls through the
Mooncake TransferEngine across hosts. Because the mirror really constructs
its model, it warms up like any other daemon and adopts the source's
FlashInfer autotune table through the ordinary warmup-cache seeding from
the same source.

Same host (source replica on GPUs 0-3 already serving)::

    vllm preload \\
        --model /path/to/model --tensor-parallel-size 4 \\
        --weight-cache-socket-dir /run/vllm-b \\
        --weight-cache-seed /run/vllm-a/sock0,...,/run/vllm-a/sock3

``--weight-cache-seed`` takes one source Unix socket per local GPU, in
device order; rank *i* of the mirror is filled from rank *i* of the source,
and the cache fingerprints are compared first, so a mismatched pairing is
rejected rather than silently mirroring the wrong shard. The mirror keeps
every GPU visible (a CUDA IPC handle names its device by index, so the
source's indices must resolve) and the launcher places its ranks on the
GPUs the source does not use, inferred from the source's GPU UUIDs.

Cross host, the source exposes its daemon-to-daemon listener and the
mirror's ``host:base_port`` seed address pulls over Mooncake RDMA::

    # source host
    vllm preload \\
        --model /path/to/model --tensor-parallel-size 8 \\
        --seed-listen-host 0.0.0.0 --seed-listen-port 29700 \\
        --seed-token "$TOKEN"
    # mirror host
    vllm preload \\
        --model /path/to/model --tensor-parallel-size 8 \\
        --weight-cache-seed 10.0.0.1:29700 --seed-token "$TOKEN"

Each rank binds and dials ``base_port + its global rank``, so one flag
covers a whole replica; the draft daemon group's listeners sit past the
target group's block, so the same ``host:base_port`` seeds the draft group
too. The remote seed plane is unencrypted beyond the shared token; keep it
on a trusted network.
"""

import contextlib
import fcntl
import gc
import multiprocessing
import os
import select
import socket
import threading
from collections.abc import Callable

import torch

from vllm.config import (
    ParallelConfig,
    VllmConfig,
    replace,
    set_current_vllm_config,
)
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_world_group,
    init_distributed_environment,
)
from vllm.distributed.device_communicators.flashinfer_all_reduce import (
    destroy_fi_ar_workspace,
)
from vllm.logger import configure_logging, init_logger
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.model_loader.weight_cache.protocol import (
    TensorEntry,
    TensorKind,
    WeightCacheKey,
    WeightCacheUnavailableError,
    check_ipc_quant_support,
    ensure_private_socket_dir,
    get_current_device_uuid,
    get_socket_path,
    hash_rope_parameters,
    recv_msg,
    send_msg,
    verify_peer_is_owner,
)
from vllm.model_executor.model_loader.weight_cache.transport import (
    SeedConfig,
    SeedTransport,
)
from vllm.model_executor.model_loader.weight_cache.utils import (
    DaemonRole,
    build_weight_cache_key,
    collect_tensors,
    export_entries,
    export_model_attrs,
    exported_weight_bytes,
    format_daemon_role,
    is_draft_model_cacheable,
)
from vllm.model_executor.model_loader.weight_cache.warmup_cache import (
    FlashInferWarmupCache,
)
from vllm.model_executor.model_loader.weight_cache.weight_seed import (
    WeightShardSeed,
)
from vllm.model_executor.model_loader.weight_cache.weight_seed.loader import (
    PeerModelLoader,
)
from vllm.platforms import current_platform
from vllm.utils.mem_utils import format_gib
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.worker.workspace import init_workspace_manager

logger = init_logger("vllm.model_executor.model_loader.weight_cache.daemon")


class WeightCacheDaemon:
    """Per-GPU process that loads one TP/PP shard and serves CUDA IPC handles."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        global_rank: int,
        local_rank: int,
        distributed_init_method: str,
        socket_dir: str | None = None,
        is_draft: bool = False,
        dp_rank: int = 0,
        pp_rank: int = 0,
        seed_config: SeedConfig | None = None,
        device_index: int | None = None,
    ):
        parallel_config = vllm_config.parallel_config
        if parallel_config.enable_eplb:
            raise ValueError("The weight cache daemon does not support EPLB.")
        self.warmup_caches = [FlashInferWarmupCache()]
        self.weight_seed = WeightShardSeed(self)
        self.d2d = SeedTransport([*self.warmup_caches, self.weight_seed], seed_config)
        # Seed peers are served on their own thread; this serializes their
        # reads of the state with the local socket's requests. Reentrant
        # because the local "seed" command re-enters it through
        # WeightShardSeed.resolve.
        self._state_lock = threading.RLock()
        self.tp_size = parallel_config.tensor_parallel_size
        self.pp_size = parallel_config.pipeline_parallel_size
        self.dp_size = parallel_config.data_parallel_size
        if self.dp_size > 1:
            # The engine sets these on each DP rank's workers; the MoE/EP
            # layers read them to place their expert shards.
            if parallel_config.nnodes > 1:
                world_size = self.dp_size * self.tp_size
                local_world = world_size // parallel_config.nnodes
                start_dp_rank = parallel_config.node_rank * local_world // self.tp_size
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
        self.global_rank = global_rank
        rank_in_dp = global_rank - dp_rank * self.pp_size * self.tp_size
        # Set like the engine's workers do: per-rank state such as the
        # torch.compile cache is keyed by it, and a compiled artifact shared
        # across ranks would run every rank's kernels on rank 0's device.
        vllm_config.parallel_config.rank = rank_in_dp
        self.tp_rank = rank_in_dp % self.tp_size
        self.pp_rank = rank_in_dp // self.tp_size
        self.dp_rank = dp_rank
        self.world_size = self.dp_size * self.pp_size * self.tp_size
        self.local_rank = local_rank
        # A peer's CUDA IPC handle names its device by index, so a same-host
        # mirror keeps every device visible and places its own ranks on the
        # GPUs the source does not use; the launcher infers the placement
        # from the source's GPU UUIDs and passes it explicitly.
        self.device_index = local_rank if device_index is None else device_index
        self.distributed_init_method = distributed_init_method
        self.socket_dir = socket_dir
        self.is_draft = is_draft
        self.role = format_daemon_role(is_draft)
        self.model: torch.nn.Module | None = None
        # Engine watch connections, held open for the daemon's lifetime.
        self._watchers: list[socket.socket] = []
        # Fingerprint before loading: process_weights_after_loading may
        # mutate hf_config.quantization_config. Built by the shared helper
        # so the peer loader requests exactly the shard fingerprinted here.
        self.cache_key = build_weight_cache_key(
            vllm_config, self.model_config, is_draft=is_draft
        )

    def warmup(self) -> None:
        """Run each warmup cache's pass (adopting remote caches via D2D)."""
        for cache in self.warmup_caches:
            cache.warmup(self, self.d2d)

    def load_model(self) -> None:
        torch.accelerator.set_device_index(self.device_index)
        # Outside the config context so the daemon keeps its own rendezvous
        # instead of the engine's; model parallel groups (TP/DP/EP) are then
        # carved out of this world group like in the engine.
        init_distributed_environment(
            world_size=self.world_size,
            rank=self.global_rank,
            distributed_init_method=self.distributed_init_method,
            local_rank=self.device_index,
            backend=current_platform.dist_backend,
        )
        with set_current_vllm_config(self.vllm_config):
            ensure_model_parallel_initialized(self.tp_size, self.pp_size)
            self.model = self.get_model()
        # In-place reloads use configs already normalized by model construction.
        self.loaded_rope_parameters_hash = hash_rope_parameters(self.model_config)
        # Loading and post-processing leave freed transients in the caching
        # allocator; return them so engines sharing the GPU can use them.
        gc.collect()
        torch.accelerator.empty_cache()
        logger.info(
            "Weight cache %s daemon rank %d loaded model (%s GiB allocated, "
            "%s GiB reserved)",
            self.role,
            self.global_rank,
            format_gib(torch.accelerator.memory_allocated()),
            format_gib(torch.accelerator.memory_reserved()),
        )

    def release_distributed(self) -> None:
        """Release process groups and BLAS workspaces after loading and warmup.

        Serving only exports CUDA IPC handles, so the communicators (NCCL,
        custom all-reduce and FlashInfer workspaces) and cached BLAS scratch
        would otherwise pin GPU memory for the daemon's lifetime.
        """
        free_before, _ = torch.accelerator.get_memory_info()
        # No rank may tear down a communicator while a peer is still in a
        # warmup collective on it.
        get_world_group().barrier()
        # Created lazily by warmup's all-reduce fusion and owned by no group.
        destroy_fi_ar_workspace()
        destroy_model_parallel()
        destroy_distributed_environment()
        gc.collect()
        if current_platform.is_cuda():
            # ATen caches these separately from the model's exported storages.
            clear_workspaces = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
            if clear_workspaces is not None:
                clear_workspaces()
            else:
                logger.debug("PyTorch does not expose BLAS workspace cleanup")
        torch.accelerator.empty_cache()
        free_after, _ = torch.accelerator.get_memory_info()
        logger.info(
            "Weight cache %s daemon rank %d released startup resources (%s GiB freed)",
            self.role,
            self.global_rank,
            format_gib(free_after - free_before),
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
        if isinstance(loader, PeerModelLoader):
            # The daemon is both ends of the weights kind: it fetches this
            # rank's shard from its source over its own D2D endpoint, and
            # the loader installs the payload into a model.
            payload = self.d2d.fetch(self.weight_seed, self)
            # The weights kind answers a payload or an error, never a miss.
            assert payload is not None
            return loader.load_model(vllm_config, model_config, payload=payload)
        if getattr(loader, "loads_pre_processed_weights", False):
            # The loader fetches fully post-processed weights from a peer
            # daemon and builds the model around them itself, handling
            # dtype, device and post-processing internally.
            return loader.load_model(vllm_config, model_config)
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
        self.d2d.start_listener()
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
                        with self._state_lock:
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
            for watcher in self._watchers:
                watcher.close()
            self.d2d.close()
            with self._state_lock:
                self.weight_seed.release()
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

    @property
    def has_weights(self) -> bool:
        return self.model is not None

    def _export_state(self) -> tuple[dict[str, TensorEntry], dict[str, str], dict]:
        """Export this rank's tensors, aliases and flags for one client.

        Entries are built per request because PyTorch registers a reference
        for each IPC mapping's lifetime, so every client needs its own
        handles.
        """
        assert self.model is not None
        entries, aliases = export_entries(self.model)
        return entries, aliases, export_model_attrs(self.model)

    def _seed_state(
        self,
    ) -> tuple[dict[str, torch.Tensor], dict[str, TensorKind], dict[str, str], dict]:
        """This rank's canonical tensors, their kinds, aliases and flags."""
        assert self.model is not None
        tensors, kinds, aliases = collect_tensors(self.model)
        return tensors, kinds, aliases, export_model_attrs(self.model)

    def _handle_connection(self, conn: socket.socket) -> None:
        request = recv_msg(conn)
        cmd = request.get("cmd")
        if cmd == "get_state":
            self._handle_get_state(conn, request)
        elif cmd == "get_memory":
            send_msg(
                conn,
                {
                    "status": "ok",
                    "memory_bytes": exported_weight_bytes(self.model),
                    # The launcher maps source GPUs to local device indices
                    # when it places a same-host mirror.
                    "gpu_uuid": get_current_device_uuid(),
                },
            )
        elif cmd == "seed":
            # A same-host peer's D2D request; the pickle plane can carry the
            # peer_ipc seed's CUDA IPC handles, which JSON cannot.
            send_msg(conn, self.d2d.handle_request(request))
        elif cmd == "release":
            self._handle_release(conn)
        elif cmd == "watch":
            self._handle_watch(conn)
        else:
            send_msg(conn, {"status": "error", "message": f"Unknown command {cmd!r}"})

    def _handle_get_state(self, conn: socket.socket, request: dict) -> None:
        client_key = request.get("cache_key")
        if not isinstance(client_key, WeightCacheKey):
            send_msg(conn, {"status": "error", "message": "Missing cache_key"})
            return
        mismatched = self.cache_key.mismatched_fields(client_key)
        if (
            request.get("is_reload", False)
            and "rope_parameters_hash" in mismatched
            and client_key.rope_parameters_hash == self.loaded_rope_parameters_hash
        ):
            mismatched.remove("rope_parameters_hash")
        if mismatched:
            logger.warning("WeightCacheKey mismatch on fields: %s", mismatched)
            send_msg(conn, {"status": "mismatch", "fields": mismatched})
            return
        if not self.has_weights:
            send_msg(conn, {"status": "error", "message": "Weights were released"})
            return
        entries, aliases, attrs = self._export_state()
        send_msg(
            conn,
            {
                "status": "ok",
                "entries": entries,
                "aliases": aliases,
                "attrs": attrs,
                "gpu_uuid": get_current_device_uuid(),
            },
        )
        logger.info_once(
            "Weight cache %s daemon rank %d sent %d tensors (+%d aliases) to engine",
            self.role,
            self.global_rank,
            len(entries),
            len(aliases),
        )

    def _handle_watch(self, conn: socket.socket) -> None:
        """Hold the connection open until this daemon exits.

        Zero-copy engines keep serving from the mapped weights after the
        daemon exits; the closed connection tells them their daemon is gone.
        """
        self._prune_watchers()
        # serve_forever closes conn after the request; keep a duplicate.
        self._watchers.append(conn.dup())
        send_msg(conn, {"status": "ok"})

    def _prune_watchers(self) -> None:
        """Close the watch connections of engines that have exited."""
        alive = []
        for watcher in self._watchers:
            # Engines never write after the request, so a readable watcher
            # has been closed by its engine.
            readable, _, _ = select.select([watcher], [], [], 0)
            if readable:
                watcher.close()
            else:
                alive.append(watcher)
        self._watchers = alive

    def _handle_release(self, conn: socket.socket) -> None:
        # Unregister from the NIC before the memory goes back to the driver.
        self.weight_seed.release()
        self.model = None
        torch.accelerator.empty_cache()
        logger.info(
            "Weight cache %s daemon rank %d released cached weights",
            self.role,
            self.global_rank,
        )
        send_msg(conn, {"status": "ok"})


def _run_daemon(
    global_rank: int,
    local_rank: int,
    vllm_config: VllmConfig,
    distributed_init_method: str,
    socket_dir: str | None,
    ready_queue: "multiprocessing.Queue[tuple[DaemonRole, int]]",
    is_draft: bool = False,
    dp_rank: int = 0,
    pp_rank: int = 0,
    seed_config: SeedConfig | None = None,
    device_index: int | None = None,
) -> None:
    # Spawned daemons start with logging unconfigured, like engine workers.
    if logging_config := getattr(vllm_config, "logging_config", None):
        configure_logging(logging_config)
    daemon = WeightCacheDaemon(
        vllm_config,
        global_rank,
        local_rank,
        distributed_init_method,
        socket_dir,
        is_draft,
        dp_rank,
        pp_rank,
        seed_config=seed_config,
        device_index=device_index,
    )
    daemon.load_model()
    daemon.warmup()
    daemon.release_distributed()
    daemon.serve_forever(
        ready_callback=lambda: ready_queue.put((daemon.role, daemon.global_rank))
    )


def plan_local_ranks(
    parallel_config: ParallelConfig,
) -> list[tuple[int, int, int, int]]:
    """``(local_rank, dp_rank, pp_rank, tp_rank)`` for every local daemon.

    Global ranks enumerate DP, PP, then TP: ``global = dp_rank * pp_size *
    tp_size + pp_rank * tp_size + tp_rank``.
    With ``--nnodes`` the engine hands each node a contiguous block of global
    ranks, so node ``r`` serves ``node_rank * local + i``; without it a
    launcher's block starts at ``--data-parallel-start-rank * tp_size``.
    """
    tp_size = parallel_config.tensor_parallel_size
    pp_size = parallel_config.pipeline_parallel_size
    world_size = parallel_config.data_parallel_size * pp_size * tp_size
    if parallel_config.nnodes > 1:
        local_world_size = world_size // parallel_config.nnodes
        base = parallel_config.node_rank * local_world_size
    else:
        local_world_size = parallel_config.data_parallel_size_local * pp_size * tp_size
        base = parallel_config.data_parallel_rank * pp_size * tp_size
    placements = []
    for local_rank in range(local_world_size):
        global_rank = base + local_rank
        rank_in_dp = global_rank % (pp_size * tp_size)
        dp_rank = global_rank // (pp_size * tp_size)
        pp_rank, tp_rank = divmod(rank_in_dp, tp_size)
        placements.append((local_rank, dp_rank, pp_rank, tp_rank))
    return placements


def get_draft_daemon_config(vllm_config: VllmConfig) -> VllmConfig | None:
    """VllmConfig for the draft daemon group"""
    if not is_draft_model_cacheable(vllm_config.speculative_config):
        return None
    assert vllm_config.speculative_config is not None
    return vllm_config.speculative_config.apply_draft_overrides(vllm_config)


def _reject_unsupported_parallelism(parallel_config: ParallelConfig) -> None:
    """Reject placements the daemon cannot map."""
    dp_size = parallel_config.data_parallel_size
    tp_size = parallel_config.tensor_parallel_size
    pp_size = parallel_config.pipeline_parallel_size
    if parallel_config.nnodes > 1:
        world_size = dp_size * pp_size * tp_size
        if world_size % parallel_config.nnodes != 0:
            raise ValueError(
                f"--nnodes ({parallel_config.nnodes}) must evenly divide the "
                f"daemon world size ({world_size} = data-parallel-size "
                f"{dp_size} x pipeline-parallel-size {pp_size} x "
                f"tensor-parallel-size {tp_size})"
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
