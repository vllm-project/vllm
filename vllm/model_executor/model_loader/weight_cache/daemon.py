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
"""

import contextlib
import fcntl
import gc
import hmac
import multiprocessing
import os
import select
import socket
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

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
from vllm.model_executor.model_loader.weight_cache.artifact_cache import ArtifactStore
from vllm.model_executor.model_loader.weight_cache.protocol import (
    MAX_ARTIFACT_SIZE,
    ArtifactCacheKey,
    TensorEntry,
    WeightCacheKey,
    WeightCacheUnavailableError,
    check_ipc_quant_support,
    dataclass_to_json,
    ensure_private_socket_dir,
    get_current_device_uuid,
    get_socket_path,
    json_to_dataclass,
    recv_json,
    recv_msg,
    send_json,
    send_msg,
    verify_peer_is_owner,
    verify_socket_owner,
)
from vllm.model_executor.model_loader.weight_cache.seed import (
    PEER_IPC_SEED_SOURCE,
    RDMA_SEED_SOURCE,
    WeightCacheSeedSource,
    build_manifest,
    get_seed_source,
)
from vllm.model_executor.model_loader.weight_cache.utils import (
    export_model_attrs,
    format_daemon_role,
    is_draft_model_cacheable,
)
from vllm.platforms import current_platform
from vllm.utils.mem_utils import format_gib
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.worker.workspace import init_workspace_manager

logger = init_logger("vllm.model_executor.model_loader.weight_cache.daemon")

# Everything else mutates or exports GPU state and stays on the owner-verified
# Unix socket; these two only read what this rank already holds.
REMOTE_COMMANDS = frozenset({"fetch_manifest", "fetch_artifacts"})

_SEED_TIMEOUT_S = 300.0


@dataclass
class MirrorState:
    """This rank's tensors as received from a peer daemon.

    A mirror deliberately has no model: it never reads the checkpoint, so it
    holds the finished tensors plus the metadata a client would otherwise
    recover from the model.
    """

    tensors: dict[str, torch.Tensor]
    kinds: dict[str, str]
    """Per-tensor "param" or "buffer"."""
    aliases: dict[str, str] = field(default_factory=dict)
    attrs: dict[str, bool] = field(default_factory=dict)


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
        seed_addr: str | None = None,
        seed_backend: str = PEER_IPC_SEED_SOURCE,
        listen_addr: tuple[str, int] | None = None,
        seed_token: str | None = None,
        device_offset: int = 0,
    ):
        parallel_config = vllm_config.parallel_config
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
        self.tp_rank = rank_in_dp % self.tp_size
        self.pp_rank = rank_in_dp // self.tp_size
        self.dp_rank = dp_rank
        self.world_size = self.dp_size * self.pp_size * self.tp_size
        self.local_rank = local_rank
        # A peer's CUDA IPC handle names its device by index, so a same-host
        # mirror must see the source's GPUs at the indices the source used.
        # It therefore makes every device visible and shifts its own ranks
        # past the source's block instead of remapping with
        # CUDA_VISIBLE_DEVICES.
        self.device_index = local_rank + device_offset
        self.distributed_init_method = distributed_init_method
        self.socket_dir = socket_dir
        self.is_draft = is_draft
        self.role = format_daemon_role(is_draft)
        self.model: torch.nn.Module | None = None
        # Host-side artifacts an engine computed once and handed back, e.g.
        # the FlashInfer autotune table. They outlive every engine restart
        # like the weights do, but cost only host memory.
        self.artifacts = ArtifactStore()
        self.seed_addr = seed_addr
        self.seed_backend = seed_backend
        self.listen_addr = listen_addr
        self.seed_token = seed_token
        # Set instead of `model` when this daemon mirrors a peer rather than
        # reading the checkpoint itself.
        self.mirror: MirrorState | None = None
        self._seed_sources: dict[str, WeightCacheSeedSource] = {}
        # Fingerprint before loading: process_weights_after_loading may
        # mutate hf_config.quantization_config.
        self.cache_config = WeightCacheKey.from_model_config(
            self.model_config,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            pp_size=self.pp_size,
            pp_rank=self.pp_rank,
            dp_size=self.dp_size,
            dp_rank=dp_rank,
            is_draft=is_draft,
        )

    def load_model(self) -> None:
        torch.accelerator.set_device_index(self.device_index)
        if self.seed_addr is not None:
            # A mirror needs neither a rendezvous nor a model: it receives
            # this rank's finished tensors from a peer that already holds
            # them, which is the whole point of seeding.
            self._load_from_seed()
        else:
            # Outside the config context so the daemon keeps its own
            # rendezvous instead of the engine's; model parallel groups
            # (TP/DP/EP) are then carved out of this world group like in the
            # engine.
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
        # Loading and post-processing leave freed transients in the caching
        # allocator; return them so engines sharing the GPU can use them.
        gc.collect()
        torch.accelerator.empty_cache()
        logger.info(
            "Weight cache %s daemon rank %d %s (%s GiB allocated, %s GiB reserved)",
            self.role,
            self.global_rank,
            "loaded model"
            if self.model is not None
            else f"mirrored {len(self.mirror.tensors)} tensors"
            if self.mirror is not None
            else "cached nothing",
            format_gib(torch.accelerator.memory_allocated()),
            format_gib(torch.accelerator.memory_reserved()),
        )

    def _seed_source(self, backend: str) -> WeightCacheSeedSource:
        source = self._seed_sources.get(backend)
        if source is None:
            source = get_seed_source(backend)
            self._seed_sources[backend] = source
        return source

    def _load_from_seed(self) -> None:
        """Fill this rank from a peer daemon instead of the checkpoint."""
        assert self.seed_addr is not None
        source = self._seed_source(self.seed_backend)
        remote = not self.seed_addr.startswith(("/", "./"))
        if remote and source.is_node_local:
            raise ValueError(
                f"Seed backend {self.seed_backend!r} copies through CUDA IPC, "
                "which cannot cross hosts; seed from a Unix socket path or "
                f"pass --weight-cache-seed-backend {RDMA_SEED_SOURCE}"
            )
        response = self._request_seed(remote)
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
        manifest = response["manifest"]
        tensors = source.fill(
            manifest,
            response["seed"],
            torch.device(current_platform.device_type, self.device_index),
        )
        if set(tensors) != set(manifest):
            raise RuntimeError("Seed backend did not fill the complete tensor manifest")
        self.mirror = MirrorState(
            tensors=tensors,
            kinds={
                name: "param" if metadata["is_param"] else "buffer"
                for name, metadata in manifest.items()
            },
            aliases=dict(response.get("aliases") or {}),
            attrs=dict(response.get("attrs") or {}),
        )
        adopted = self.artifacts.merge_json(response.get("artifacts") or [])
        if adopted:
            logger.info(
                "Weight cache %s daemon rank %d adopted %d artifact(s) from %s",
                self.role,
                self.global_rank,
                adopted,
                self.seed_addr,
            )

    def _request_seed(self, remote: bool) -> dict:
        """Ask the source daemon for this rank's manifest and transfer seed.

        A same-host peer is reached over its Unix socket and speaks the
        pickle protocol, since ``peer_ipc`` ships real CUDA IPC handles. A
        remote peer speaks the JSON control plane instead.
        """
        assert self.seed_addr is not None
        request: dict[str, Any] = {
            "cmd": "fetch_manifest",
            "seed_backend": self.seed_backend,
        }
        if not remote:
            verify_socket_owner(self.seed_addr, strict_perms=False)
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            target: Any = self.seed_addr
        else:
            host, separator, port = self.seed_addr.rpartition(":")
            if not separator or not host or not port.isdigit():
                raise ValueError(
                    f"A remote seed address must be host:port, got {self.seed_addr!r}"
                )
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            target = (host, int(port))
            request["token"] = self.seed_token
        sock.settimeout(_SEED_TIMEOUT_S)
        try:
            sock.connect(target)
            if remote:
                send_json(sock, request)
                response = recv_json(sock)
            else:
                send_msg(sock, request)
                response = recv_msg(sock)
        finally:
            sock.close()
        if not isinstance(response, dict):
            raise RuntimeError(
                f"Seed daemon returned a malformed response: {response!r}"
            )
        if remote and response.get("status") == "ok":
            response["cache_config"] = json_to_dataclass(
                WeightCacheKey, response.get("cache_config")
            )
        return response

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
        remote_server = None
        if self.listen_addr is not None:
            remote_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            remote_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            remote_server.bind(self.listen_addr)
            remote_server.listen()
        listeners = [server] + ([remote_server] if remote_server is not None else [])
        logger.info(
            "Weight cache %s daemon rank %d serving on %s%s",
            self.role,
            self.global_rank,
            socket_path,
            f" and {self.listen_addr[0]}:{self.listen_addr[1]} (seed peers)"
            if self.listen_addr is not None
            else "",
        )
        if ready_callback is not None:
            ready_callback()
        try:
            while True:
                readable, _, _ = select.select(listeners, [], [], 1.0)
                for listener in readable:
                    conn, _ = listener.accept()
                    remote = listener is remote_server
                    reply = send_json if remote else send_msg
                    with conn:
                        try:
                            if remote:
                                self._handle_remote_connection(conn)
                            else:
                                verify_peer_is_owner(conn)
                                self._handle_connection(conn)
                        except (ConnectionError, EOFError):
                            logger.warning("Client disconnected mid-request")
                        except Exception as e:
                            # Report the error back instead of just closing
                            # the socket, but don't let it take the daemon
                            # down.
                            logger.exception(
                                "Error handling weight cache client; continuing"
                            )
                            with contextlib.suppress(OSError):
                                reply(conn, {"status": "error", "message": str(e)})
        finally:
            server.close()
            if remote_server is not None:
                remote_server.close()
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

    @property
    def _socket_path(self) -> str:
        return get_socket_path(
            get_current_device_uuid(),
            self.socket_dir,
            is_draft=self.is_draft,
        )

    @property
    def has_weights(self) -> bool:
        return self.model is not None or self.mirror is not None

    def _export_state(self) -> tuple[dict[str, TensorEntry], dict[str, str], dict]:
        """Export this rank's tensors, aliases and flags for one client.

        Entries are built per request because PyTorch registers a reference
        for each IPC mapping's lifetime, so every client needs its own
        handles. A mirror carries the aliases and flags it was seeded with,
        since it has no model to recover them from.
        """
        if self.model is not None:
            entries, aliases = export_entries(self.model)
            return entries, aliases, export_model_attrs(self.model)
        mirror = self.mirror
        assert mirror is not None
        entries = {
            name: TensorEntry.from_tensor(tensor, mirror.kinds[name])
            for name, tensor in mirror.tensors.items()
        }
        return entries, dict(mirror.aliases), dict(mirror.attrs)

    def _state_tensors(self) -> dict[str, torch.Tensor]:
        """This rank's tensors by name, for a mover that copies by address."""
        if self.mirror is not None:
            return self.mirror.tensors
        assert self.model is not None
        tensors: dict[str, torch.Tensor] = {}
        for name, tensor in self.model.named_parameters(remove_duplicate=False):
            tensors.setdefault(name, tensor.detach())
        for name, tensor in self.model.named_buffers(remove_duplicate=False):
            tensors.setdefault(name, tensor.detach())
        return tensors

    def _build_seed_response(self, backend: str) -> dict[str, Any]:
        """Describe this rank so a peer can mirror it."""
        source = self._seed_source(backend)
        entries, aliases, attrs = self._export_state()
        state_tensors = {
            name: tensor
            for name, tensor in self._state_tensors().items()
            if name in entries
        }
        seed = source.prepare_seed(entries, state_tensors, get_current_device_uuid())
        return {
            "cache_config": self.cache_config,
            "manifest": build_manifest(entries),
            "aliases": aliases,
            "attrs": attrs,
            "seed": seed,
            # Kilobytes next to the weights, and it saves the mirror's first
            # engine the autotune pass, so every plane carries it.
            "artifacts": self.artifacts.to_json(),
        }

    def _handle_fetch_manifest(self, conn: socket.socket, request: dict) -> None:
        if not self.has_weights:
            send_msg(conn, {"status": "error", "message": "Weights were released"})
            return
        backend = request.get("seed_backend", PEER_IPC_SEED_SOURCE)
        try:
            response = self._build_seed_response(backend)
        except Exception as error:
            send_msg(conn, {"status": "error", "message": str(error)})
            return
        send_msg(conn, {"status": "ok", **response})

    def _handle_remote_connection(self, conn: socket.socket) -> None:
        """Serve one request from a remote peer over the JSON control plane.

        The remote plane never unpickles: recv_msg would deserialize a
        peer's bytes before its token could be compared, which would make
        the listener an unauthenticated code-execution surface. JSON parsing
        is memory-safe and the length cap bounds it, so the token is checked
        against a decoded request rather than a decoded object graph.
        """
        request = recv_json(conn)
        presented = request.get("token") if isinstance(request, dict) else None
        if (
            not self.seed_token
            or not isinstance(presented, str)
            or not hmac.compare_digest(presented, self.seed_token)
        ):
            send_json(conn, {"status": "error", "message": "Unauthorized seed request"})
            return
        cmd = request.get("cmd")
        if cmd not in REMOTE_COMMANDS:
            send_json(
                conn,
                {
                    "status": "error",
                    "message": f"Command {cmd!r} is only served locally",
                },
            )
            return
        if cmd == "fetch_artifacts":
            send_json(conn, {"status": "ok", "artifacts": self.artifacts.to_json()})
            return
        if not self.has_weights:
            send_json(conn, {"status": "error", "message": "Weights were released"})
            return
        backend = request.get("seed_backend", RDMA_SEED_SOURCE)
        if not isinstance(backend, str):
            send_json(conn, {"status": "error", "message": "Malformed seed_backend"})
            return
        try:
            source = self._seed_source(backend)
            if source.is_node_local:
                raise ValueError(
                    f"Seed backend {backend!r} copies through CUDA IPC, which "
                    "cannot cross hosts; mirror it over this daemon's Unix socket"
                )
            response = self._build_seed_response(backend)
        except Exception as error:
            send_json(conn, {"status": "error", "message": str(error)})
            return
        response["cache_config"] = dataclass_to_json(response["cache_config"])
        send_json(conn, {"status": "ok", **response})

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
                    "memory_bytes": torch.accelerator.memory_allocated(),
                },
            )
        elif cmd == "fetch_manifest":
            self._handle_fetch_manifest(conn, request)
        elif cmd == "fetch_artifacts":
            send_msg(conn, {"status": "ok", "artifacts": self.artifacts.to_json()})
        elif cmd == "get_artifact":
            self._handle_get_artifact(conn, request)
        elif cmd == "put_artifact":
            self._handle_put_artifact(conn, request)
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

    def _handle_get_artifact(self, conn: socket.socket, request: dict) -> None:
        key = request.get("key")
        if not isinstance(key, ArtifactCacheKey):
            send_msg(conn, {"status": "error", "message": "Missing artifact key"})
            return
        data = self.artifacts.get(key)
        if data is None:
            send_msg(conn, {"status": "miss"})
            return
        send_msg(conn, {"status": "ok", "data": data})
        logger.info_once(
            "Weight cache %s daemon rank %d served %r artifact (%d bytes)",
            self.role,
            self.global_rank,
            key.kind,
            len(data),
        )

    def _handle_put_artifact(self, conn: socket.socket, request: dict) -> None:
        key = request.get("key")
        data = request.get("data")
        if not isinstance(key, ArtifactCacheKey) or not isinstance(data, bytes):
            send_msg(
                conn,
                {"status": "error", "message": "Missing artifact key or data"},
            )
            return
        if len(data) > MAX_ARTIFACT_SIZE:
            send_msg(
                conn,
                {
                    "status": "error",
                    "message": f"Artifact of {len(data)} bytes exceeds the "
                    f"{MAX_ARTIFACT_SIZE} byte limit",
                },
            )
            return
        self.artifacts.put(key, data)
        send_msg(conn, {"status": "ok"})
        logger.info(
            "Weight cache %s daemon rank %d cached %r artifact (%d bytes)",
            self.role,
            self.global_rank,
            key.kind,
            len(data),
        )

    def _handle_release(self, conn: socket.socket) -> None:
        self.model = None
        self.mirror = None
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
    ready_queue: "multiprocessing.Queue[tuple[str, int]]",
    is_draft: bool = False,
    dp_rank: int = 0,
    pp_rank: int = 0,
    seed_addr: str | None = None,
    seed_backend: str = PEER_IPC_SEED_SOURCE,
    listen_addr: tuple[str, int] | None = None,
    seed_token: str | None = None,
    device_offset: int = 0,
) -> None:
    daemon = WeightCacheDaemon(
        vllm_config,
        global_rank,
        local_rank,
        distributed_init_method,
        socket_dir,
        is_draft,
        dp_rank,
        pp_rank,
        seed_addr,
        seed_backend,
        listen_addr,
        seed_token,
        device_offset,
    )
    daemon.load_model()
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
