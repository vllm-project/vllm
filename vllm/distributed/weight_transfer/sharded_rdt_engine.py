# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sharded Ray Direct Transport (RDT) weight transfer engine (consumer side).

Pulls only the slice each vLLM worker consumes under tensor/expert parallelism,
not the full HF-format tensor.

Two phases. BAKE, once at ``init_transfer_engine``: drive ``model.load_weights``
against ``FakeRDTTensor`` placeholders and record, per leaf module, how each
destination slice is fetched (an op chain) and where it lands (an ``as_strided``
descriptor). REPLAY, every sync: no ``load_weights``, no ``FakeRDTTensor``
dispatch, no discovery — pull the recorded slices in packed chunks over a ring
of receive buffers, scatter them into freshly materialized params, then quant
and kernel-copy. A live name with no recorded plan fails the plan build: there
is no fallback load.

Weights arrive in checkpoint format: the engine drives layerwise reload itself,
in ``start_weight_update`` / ``finish_weight_update``.

Data flow
---------
One thing at four resolutions, over three lifetimes. ``FetchKey`` --
``(name, op_chain)``, "which slice of which trainer tensor" -- is the atom;
everything else is bookkeeping around it.

    BAKE      once at init, kept for the engine's life
      FakeRDTTensor intercepts the model's loaders; each copy_ records a
      _Scatter (sharded_rdt_fake): src FetchKey, owning layer, the destination
      as_strided region, and the produced dtype/nbytes.
        -> _name_to_plan: name -> that module's scatter list

    PLAN      once, cached, _build_call_plan
      _Chunk    one packed pull: its scatters, deduped keys, the byte-exact
                pack_layout, which producer serves it, and what to run after
                (materialize / quant / free).  One per (group, owner class).
      _CallPlan all chunks + pre_free.

    RUN       per chunk, per sync, _run_chunk_pipeline
      _Chunk -> _PendingPull   issued, not yet landed: Ray ref, buffer views,
                               ring slot.               [RPC thread]
             -> _ProcItem      chunk + results + slot.  [hand-off to the
                               background scatter thread]

The RUN pair stays split on purpose: it is the thread boundary, and only
``targets`` should outlive the get -- carrying the Ray ref and the whole-buffer
blob into the queue would keep both alive for the scatter's lifetime.

See docs/training/weight_transfer/sharded_rdt.md for the design and the measured
results behind the choices here.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from math import prod
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import torch

from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)
from vllm.distributed.weight_transfer.sharded_rdt_common import (
    RdtRouter,
    StagingDevice,
    buffer_alloc_bytes,
    check_ray_rdt_version,
    deregister_nixl_memory,
    register_nixl_memory,
    validate_sharded_rdt_settings,
)
from vllm.distributed.weight_transfer.sharded_rdt_fake import (
    BakeSink,
    FakeRDTTensor,
    FetchKey,
    _Scatter,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class HostStagingAllocationError(RuntimeError):
    """A host receive slot could not be allocated, page-locked or registered."""


def _alloc_page_locked_slot(
    nbytes: int, *, park: Callable[[torch.Tensor, str], None]
) -> torch.Tensor:
    """One exact-size host receive slot: a pageable ``torch.empty`` page-locked
    whole from its storage base with ``cudaHostRegister``, then NIXL-registered.

    Torch's pinned allocator is bypassed on purpose: it rounds every request up
    to a power of two and keeps freed blocks cached, so the pool would occupy
    more than the documented ``lanes x K x slot bytes`` and never give it back.
    A failure leaves no registration the caller has to track: a nonzero
    ``cudaHostRegister`` code (the wrapper returns codes and never raises) is
    raised with its error text, and a NIXL failure unregisters the page-locked
    range before re-raising. When that rollback fails as well (a nonzero
    ``cudaHostUnregister`` code or an exception out of the call) the range's
    page-lock state is unknown, so the buffer goes to ``park`` (the engine's
    ``_park_host_slot``) rather than being dropped, and the NIXL error is still
    the one raised.
    """
    cudart = CudaRTLibrary()
    buffer = torch.empty(nbytes, dtype=torch.uint8, device="cpu")
    storage = buffer.untyped_storage()
    base, size = storage.data_ptr(), storage.nbytes()
    code = cudart.cudaHostRegister(base, size, 0)
    if code != 0:
        cudart.cudaGetLastError()
        raise RuntimeError(
            f"cudaHostRegister({size} bytes) returned CUDA error {code}: "
            f"{cudart.cudaGetErrorString(code)}"
        )
    try:
        register_nixl_memory(buffer)
    except Exception:
        try:
            unregistered = _host_unregister(cudart, base)
        except Exception:
            logger.exception(
                "cudaHostUnregister of a half-allocated host receive slot failed"
            )
            unregistered = False
        if not unregistered:
            park(buffer, "failed cudaHostUnregister after its NIXL registration failed")
        raise
    return buffer


def _host_unregister(cudart: CudaRTLibrary, base: int) -> bool:
    """Undo ``cudaHostRegister`` on ``base``; True when it succeeded. A failure
    is logged, not raised: this runs on shutdown and inside the init rollback,
    and the caller decides the range's fate."""
    code = cudart.cudaHostUnregister(base)
    if code != 0:
        cudart.cudaGetLastError()
        logger.warning(
            "cudaHostUnregister(%#x) returned CUDA error %d: %s",
            base,
            code,
            cudart.cudaGetErrorString(code),
        )
    return code == 0


def _host_alloc_request(pool: "_InitState", nbytes: int, pool_total: int | None) -> str:
    """What the failed host allocation was for, with the pool geometry."""
    geometry = f"{pool._num_slots} slots ({pool._lanes} lane(s) x K={pool._ring_depth})"
    if pool_total is None:
        return (
            f"regrow one receive slot of its {geometry} pool to {nbytes} "
            "bytes, exactly (the other slots keep their size)"
        )
    return (
        f"build its receive pool: {geometry} x {nbytes} bytes = "
        f"{pool_total} bytes per worker, exactly"
    )


@dataclass(frozen=True)
class _ReceivePoolConfig:
    """Validated receive-pool settings recorded before the plan is known."""

    staging_device: StagingDevice
    stages_in_flight: int
    ring_depth: int
    buffer_presize: int


@dataclass
class _InitCandidate:
    """The per-init state one ``init_transfer_engine`` attempt builds.

    Its fields are the engine attributes an init (re)writes, under the same
    names, so the init helpers fill a candidate exactly as they would fill the
    engine. ``_commit_init`` copies every field onto the engine once the whole
    attempt has succeeded; a failed attempt is dropped, so the active state
    needs no restoring and a field added here cannot be missed by one.
    """

    # Driver-supplied total consumer count (init_info.num_consumers).
    _num_consumers_override: int
    _router: "RdtRouter | None" = None
    # The bake (_bake): name -> (dtype name, shape), name -> module plan, and
    # the names whose copy fired.
    _name_meta: dict[str, tuple[str, list[int]]] = field(default_factory=dict)
    _name_to_plan: "dict[str, list[_Scatter]]" = field(default_factory=dict)
    _live_names: set[str] = field(default_factory=set)
    _cached_plan: "_CallPlan | None" = None
    # The receive pool (_size_pool) and its slots (_preregister_at_init).
    _pool_geometry: tuple[StagingDevice, int, int] | None = None
    _staging_device: StagingDevice = "gpu"
    _stages_in_flight: int = 1
    _ring_depth: int = 1
    _lanes: int = 1
    _buffer_presize: int = 0
    _dest_buffers: list[dict[torch.dtype, torch.Tensor]] = field(
        default_factory=lambda: [{}]
    )
    _lane_cursors: list[int] = field(default_factory=lambda: [0])
    _targets_cache: dict[tuple[int, int], tuple[int, list[torch.Tensor]]] = field(
        default_factory=dict
    )
    # (slot, old tensor) pairs a re-init grow replaced, released on commit.
    _retired_slots: list[tuple[int, torch.Tensor]] = field(default_factory=list)
    # The post-processing worker's per-slot handshake (_ensure_proc_worker).
    _slot_read_done: list[Any] = field(default_factory=list)
    _slot_queued: list[int] = field(default_factory=list)
    _slot_done: list[int] = field(default_factory=list)
    _slot_cv: Any = None

    @property
    def _num_slots(self) -> int:
        return self._lanes * self._ring_depth


# What an init helper reads and writes: the attempt's candidate, or the
# engine's own (active) state when a caller drives the helper directly.
_InitState: TypeAlias = "_InitCandidate | ShardedRDTWeightTransferEngine"


@dataclass
class _Chunk:
    """One packed pull plus its post-processing (see Data flow).

    A module's copies span chunks when its experts sit in several owner classes,
    so ``materialize`` fires on its FIRST chunk and ``quant`` on its LAST --
    materialize-once by construction, not by a runtime counter. ``pack_layout``
    mirrors the producer's rule byte-exactly.
    """

    scatters: "list[_Scatter]"
    keys: "list[FetchKey]"
    pack_layout: "list[tuple[int, torch.dtype, int, tuple[int, ...]]]"  # off,dt,n,shape
    pack_bytes: int
    materialize: "list[Any]"
    quant: "list[Any]"
    free: "list[int]"
    # Trainer rank serving this chunk, resolved at plan time. One per chunk:
    # every name in it shares an owner class, so one producer holds them all.
    owner: int
    # This chunk's index among the chunks this worker pulls from ``owner``, in
    # issue order. It picks the producer's serve slot (``seq % ring_depth``),
    # which is what makes slot reuse safe: the pipeline drains pull i before
    # issuing i+K, so the slot seq+K takes was last used by seq, whose read is
    # finished. Choosing the slot in EXECUTION order instead -- as a per-call
    # counter on the producer would -- can hand a slot to a pull whose
    # predecessor is still being read, because Ray may start a consumer's
    # concurrent produce calls in any order. Static, since the plan is.
    seq: int = 0
    # Stage stream containing ``owner``. Chunks in one stream retain plan order.
    stream: int = 0


@dataclass
class _CallPlan:
    """The static plan for one sync (see Data flow). Pure, so it is built once
    and reused; runtime is then execution only.

    ``pre_free`` = groups with NO chunk on this worker, signaled at sync start
    (owners tolerate a signal preceding its publish). With the last-chunk signals
    this keeps the completeness invariant consumer-local: every group is signaled
    exactly once.
    """

    chunks: "list[_Chunk]"
    pre_free: "list[int]"
    # Stage streams among ``chunks``, numbered by first chunk appearance.
    num_streams: int = 0


@dataclass
class _PendingPull:
    """A dispatched pull whose blocking ``ray.get`` has not run (see Data flow).

    ``targets``/``blob`` must stay strongly referenced until it completes:
    ``set_target_for_ref`` stores WEAKREFS, so dropping them silently reroutes the
    transfer into a fallback buffer."""

    ref: "Any"
    keys: "list[FetchKey]"
    targets: "list[torch.Tensor]"
    blob: "torch.Tensor"
    slot: int


@dataclass
class _PullHooks:
    """Operations injected into the pure ``_run_lanes`` scheduler."""

    # Start a pull into the slot. The real hook waits if the slot's previous
    # scatter is still reading it; tests inject transport-free pulls.
    issue: "Callable[[_Chunk, int], Any]"
    # Whether issue into the slot would start without waiting.
    slot_free: "Callable[[int], bool]"
    # Wait for one pull to land, optionally for at most ``timeout`` seconds.
    wait_any: "Callable[[list[Any], float | None], int | None]"
    # Complete a landed pull, dispatch its scatter, and signal its groups.
    drain: "Callable[[Any, _Chunk], None]"
    # Signal a group belonging to a chunk with no keys.
    signal: "Callable[[int], None]"


@dataclass
class _ProcItem:
    """A landed pull handed to the background scatter thread (see Data flow).

    ``results`` alias the ring buffer ``slot``, held as strong refs so they
    outlive the RPC-thread frame until the scatter consumes them.
    """

    chunk: "_Chunk"
    results: "dict[FetchKey, torch.Tensor]"
    slot: int


@dataclass
class ShardedRDTWeightTransferInitInfo(WeightTransferInitInfo):
    """Initialization info for the sharded RDT backend."""

    trainer_actor_names: list[str] = field(default_factory=list)
    """Names of all trainer Ray actors exposing the producer method (set via
    ``.options(name=...)``), ordered by trainer rank. ``RdtRouter`` picks one of
    them per pull, out of the name's owner set. Every actor is bound regardless,
    because ``free_group`` fans out to every owner. Must be non-empty; a
    single-producer trainer passes a one-element list."""

    trainer_actor_namespace: str | None = None
    """Optional Ray namespace the trainer actor(s) live in."""

    produce_method_name: str = "rdt_produce_weights_batched"
    """Name of the trainer-side producer method. It and the rest of the serve
    surface (``free_group``, ``reserve_serve_buffer``) are documented where they
    are implemented, on ``_RDTProducerServer`` in ``sharded_rdt_trainer.py``."""

    names: list[str] = field(default_factory=list)
    """The trainer's complete, flat param name list. The bake drives
    ``model.load_weights`` over all of them once and keys the plan by source
    name."""

    dtype_names: list[str] = field(default_factory=list)
    """Dtype name (e.g. 'bfloat16') for each entry of ``names``."""

    shapes: list[list[int]] = field(default_factory=list)
    """Full HF shape for each entry of ``names``."""

    group_lens: list[int] = field(default_factory=list)
    """Partition of ``names`` into gather groups, in the SAME order the trainers
    gather and publish them (group-major; ``sum(group_lens) == len(names)``, and
    ``names`` must be ordered to match). Required: the engine pre-builds the whole
    static chunk/signal plan from it at init, and fires ``free_group`` at every
    owner as each group's last chunk completes."""

    owner_sets: list[list[int]] = field(default_factory=list)
    """The distinct producer sets that occur, each a sorted list of trainer ranks
    (indices into ``trainer_actor_names``). Indexed by ``name_owner_class``.
    Empty means every producer holds every name."""

    name_owner_class: list[int] = field(default_factory=list)
    """Per-name index into ``owner_sets``, parallel to ``names``: which producers
    hold each name. Empty means every producer holds every name.

    This one table expresses every layout the trainer can have — pipeline stages
    (the names of these groups have this owner set), expert parallelism (this
    expert name has this one-rank owner set), and combinations of the two — so
    the engine cuts each group's baked copies into one chunk per distinct class
    present and routes each chunk to that class's owner. Derived by the trainer
    from ``ParamMeta.owners``."""

    num_consumers: int = 0
    """Total consumer count across the fleet, for M:N routing. Authoritative when
    > 0; at 0 the engine infers it from ``parallel_config``, which is correct for
    the supported serving modes but worth setting explicitly under M:N. Each
    worker's distinct index comes from ``_global_worker_index``."""

    num_rdt_buffers: int = 2
    """K, the receive slots and in-flight pulls per active stream lane. Must
    match the producer's ring depth. The consumer pool is L x K slots, where L
    is the effective ``stages_in_flight`` clamped to the plan's stream count.
    Keep K x chunk_bytes per flow under the fabric's address-translation reach
    (~2-3 GB/flow on the reference 8xB200 RoCE cluster)."""

    buffer_presize_gb: float = 0.0
    """Pre-size each packed receive-buffer slot to this many GiB
    (0 = size to the first chunk + coarse 256MB round-up). Set it to cover the
    model's largest atomic chunk (e.g. an untied lm_head). Sizing buffers ONCE
    matters beyond perf -- see the doc's "Sizing buffers once matters beyond
    throughput"."""

    staging_device: StagingDevice = "gpu"
    """Where receive buffers live: GPU HBM or pinned host memory."""

    stages_in_flight: int = 1
    """Maximum trainer stages pulled from concurrently."""

    replica_rank: int = 0
    """This inference engine's ordinal in the fleet (0..``num_replicas``-1).

    Multi-engine deployments run several INDEPENDENT inference engines, each with
    its own self-contained parallel config, so every engine's
    ``_global_worker_index`` restarts at 0 and would collide across engines. The
    driver gives each engine a distinct ``replica_rank`` (with identical
    ``num_replicas``) so the engine offsets its consumers into a globally distinct
    range for the M:N block assignment.Default 0/1 (single
    engine) needs no override."""

    num_replicas: int = 1
    """Number of independent inference engines in the fleet. Default 1 => the
    per-replica offset is 0 and consumer identity is exactly
    ``_global_worker_index`` (preserves single-engine and single-DP-deployment
    behavior). When > 1, ``workers_per_replica = num_consumers // num_replicas``
    and this engine's consumers occupy
    ``replica_rank * workers_per_replica + _global_worker_index()``. Assumes a
    uniform fleet (every replica has the same worker count). Set alongside
    ``replica_rank``, by the driver."""


@dataclass
class ShardedRDTWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Update info for the sharded RDT backend: the sync's index, nothing else.

    The chunk/free plan is a pure function of the baked plan and the driver's
    gather-group partition, both fixed for the engine's lifetime, so it is built
    once at ``init_transfer_engine`` from ``ShardedRDTWeightTransferInitInfo``'s
    ``names`` + ``group_lens``. ONE ``update_weights`` per sync then just re-runs
    that plan.
    """

    sync_idx: int = -1
    """The trainer's count of this sync, sent with every ``free_group`` so a
    producer that has not begun the sync yet keeps the signal instead of losing
    it to its ``begin_sync``. -1 (an older trainer) sends untagged signals."""


class ShardedRDTWeightTransferEngine(
    WeightTransferEngine[
        ShardedRDTWeightTransferInitInfo,
        ShardedRDTWeightTransferUpdateInfo,
    ]
):
    """Pull-based RDT/NIXL backend that transports only the slice each worker
    consumes.

    Requires ``distributed_executor_backend="ray"``, ``nixl`` in the shared env,
    a named trainer actor exposing a ``@ray.method(tensor_transport="nixl")``
    producer, and weight loaders that stay inside ``SUPPORTED_OPS`` — anything
    needing real data (``.to``, ``.item``, arithmetic, bool-mask indexing)
    raises during the bake.

    The plan is baked once at ``init_transfer_engine`` into one scatter list
    per fully-loaded leaf module, indexed by source name; every
    ``update_weights`` replays the modules its gathered names cover.
    """

    init_info_cls = ShardedRDTWeightTransferInitInfo
    update_info_cls = ShardedRDTWeightTransferUpdateInfo
    # receive_weights pulls synchronously but defers GPU post-processing to a
    # background thread so it overlaps the next chunk's pull, so ``update_weights``
    # skips the base's device sync and ``finish_weight_update`` drains the
    # deferred work before finalize.
    defers_processing = True
    # The baked replay plan is a function of one concrete model's parameter
    # layout, so a separate draft model cannot reuse it.
    supports_draft_weight_update = False

    def __init__(
        self,
        config: WeightTransferConfig,
        vllm_config: "VllmConfig",
        device: torch.device,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(config, vllm_config, device, model)
        # This engine's data plane is Ray's: the consumer resolves each trainer's
        # serve actor with ``ray.get_actor`` and pulls over NIXL through the
        # workers' tensor transport, neither of which exists unless the workers
        # ARE Ray actors. Caught here, at construction, because the alternative
        # is an opaque failure during the first handshake. A single-worker
        # deployment defaults to "uni", so "ray" has to be asked for explicitly.
        backend = vllm_config.parallel_config.distributed_executor_backend
        # A custom Executor subclass is a deliberate override we cannot judge;
        # only reject the named backends that are definitely not Ray.
        if isinstance(backend, str) and backend != "ray":
            raise ValueError(
                f"The 'sharded_rdt' weight transfer backend requires "
                f"distributed_executor_backend='ray', got {backend!r}. Pass "
                f"--distributed-executor-backend ray so the workers are "
                f"RayExecutorV2 actors with tensor transport."
            )
        # Ownership table, routing rules, and (once bound) every producer's
        # handle — pulls go to one owner per chunk, but free_group fans out to
        # every owner of the group, so all handles are bound.
        self._router: RdtRouter | None = None
        # Driver-supplied total consumer count (init_info.num_consumers); 0 => infer
        # from parallel_config. See _num_consumers.
        self._num_consumers_override: int = 0
        # Source name -> the plan for the module consuming it. Several names of a
        # fused module share one entry; replay dedups. A live name absent here
        # fails the plan build.
        self._name_to_plan: dict[str, list[_Scatter]] = {}
        # name -> (dtype_name, shape) for every init name; the bake builds its
        # lazies from this.
        self._name_meta: dict[str, tuple[str, list[int]]] = {}
        # Names whose copy_ fired during the bake. Unbaked names not here never
        # move data (e.g. non-local EP experts), so receive_weights skips them.
        self._live_names: set[str] = set()

        # ---- Consumer-side pre-registered receive buffers ----------------------
        # One persistent 1-D buffer per dtype per ring slot, STORAGE-registered with
        # NIXL once, so every per-slice view hits Ray's registration cache and the
        # recv path does zero registration in steady state. See the doc's "Sizing
        # buffers once matters beyond throughput" for why they rarely grow.
        #
        # The pool has one K-slot block per active stage-stream lane. K is both
        # a lane's slot count and its in-flight pull limit.
        self._ring_depth = 1
        self._staging_device: StagingDevice = "gpu"
        self._stages_in_flight = 1
        self._lanes = 1
        self._pool_geometry: tuple[StagingDevice, int, int] | None = None
        self._dest_buffers: list[dict[torch.dtype, torch.Tensor]] = [{}]
        # (slot, old tensor) pairs a re-init grow replaced, released once that
        # init commits: a rollback puts them back, so they must stay registered.
        self._retired_slots: list[tuple[int, torch.Tensor]] = []
        # Host slots whose release failed or could not be proven safe (a NIXL
        # deregistration or cudaHostUnregister error, or a post-processing thread
        # that never stopped): possibly still registered or still being read, so
        # they stay page-locked and referenced for the process lifetime.
        self._parked_host_slots: list[torch.Tensor] = []
        self._slot_read_done: list[Any] = []  # one torch.Event per slot
        # Generation counters guarding the events (see _ensure_proc_worker).
        self._slot_queued: list[int] = []
        self._slot_done: list[int] = []
        self._slot_cv: Any = None
        self._lane_cursors = [0]
        # (id(chunk), slot) -> (buffer data_ptr, per-key views).
        # Lives on the engine, not on the _Chunk: the plan is static and shared
        # across syncs, so a per-slot runtime cache has no business inside it.
        self._targets_cache: dict[tuple[int, int], tuple[int, list[torch.Tensor]]] = {}
        self._buffer_presize = 0  # bytes; set from init_info
        self._pending_frees: list[Any] = []  # free_group signal refs, drained per sync
        self._sync_idx: int | None = None  # this sync's index, tagged on every free
        # The STATIC plan: built once — at init from init_info.group_lens, else
        # lazily on the first update_weights — and reused. Non-None means
        # update_weights ignores its per-sync names.
        self._cached_plan: _CallPlan | None = None
        # Completed syncs. The chunk pipeline runs SERIAL during sync 0, when both
        # sides still register buffers, and pipelines from sync 1. See the doc.
        self._completed_syncs = 0

        # ---- Background post-processing thread (pull/process pipelining) -------
        # The RPC thread hands pulled buffer views to this worker, which runs
        # materialize/scatter/quant/kernel_copy on its own stream while the next
        # pull proceeds. drain_pending() joins it before finalize.
        self._proc_queue: Any | None = None
        self._proc_thread: Any | None = None
        self._proc_stream: Any | None = None
        # Dedicated quant thread (pass 2): own queue + stream so quant never
        # sits between two items' scatters on the scatter thread (measured to
        # stall the RPC thread's slot handshake ~0.5-0.75s/iter).
        self._quant_queue: Any | None = None
        self._quant_thread: Any | None = None
        self._quant_stream: Any | None = None
        self._proc_error: BaseException | None = None

    def init_transfer_engine(self, init_info: ShardedRDTWeightTransferInitInfo) -> None:
        """Record pool settings, bind producers, bake and build the replay plan,
        size its LxK receive pool, then pre-register every NIXL buffer.

        The bake drives ``model.load_weights`` and the pre-registration blocks on
        RPCs to the producers, so this is a heavyweight one-off; every later
        ``update_weights`` is pure replay.
        """
        validate_sharded_rdt_settings(
            staging_device=init_info.staging_device,
            stages_in_flight=init_info.stages_in_flight,
            num_rdt_buffers=init_info.num_rdt_buffers,
            buffer_presize_gb=init_info.buffer_presize_gb,
            num_consumers=init_info.num_consumers,
            allow_zero_consumers=True,
        )
        # Read the attribute directly: if a vLLM bump renames it, this guard
        # must fail loudly at init rather than silently disappear.
        if self.parallel_config.enable_eplb:
            raise RuntimeError(
                "sharded_rdt does not support EPLB (enable_eplb=true): dynamic "
                "expert rearrangement invalidates the baked replay plan, which "
                "records each expert's destination slot once at init — after a "
                "rearrangement the replay would silently load weights into the "
                "wrong expert slots. Disable EPLB or use another weight-transfer "
                "backend."
            )
        # The worker is often a different install from whatever spawned it, so
        # check here too and not only where the actor options are set.
        check_ray_rdt_version(init_info.staging_device)
        pool_config = self._configure_ring(init_info)

        # Re-init prepares routing, baked state and receive buffers in a
        # candidate, off the engine: it becomes the active state only once
        # pre-registration and the worker start have succeeded, so a failed
        # attempt leaves the active plan, tensors and routing untouched and
        # releases only the slots it built itself.
        candidate = _InitCandidate(_num_consumers_override=init_info.num_consumers)
        try:
            self._resolve_producers(init_info, candidate)

            # A pure dry run: the trainer's gather cache is empty at init, so
            # nothing can (or does) get pulled -- we only record how each slice
            # is fetched and where it lands, then restore the model.
            self._bake(init_info, candidate)
            candidate._cached_plan = self._build_static_plan(init_info, candidate)
            self._size_pool(pool_config, candidate._cached_plan, candidate)
            # Register ALL NIXL memory now, while the fabric is idle: dma-buf
            # GPUDirect registration concurrent with in-flight RDMA intermittently
            # fails (ibv_reg_mr 'Bad address'), which bites under M:N fan-in. Sizes
            # come from the static plan, so this is exact.
            self._preregister_at_init(candidate)
            # Start the background post-processing worker (pull/process pipelining).
            self._ensure_proc_worker(candidate)
        except Exception:
            # Give the host memory of the slots this attempt built back. Slots a
            # re-init grow replaced are still the active pool's, so they stay
            # page-locked and NIXL-registered.
            self._release_attempt_slots(candidate)
            raise
        self._commit_init(candidate)
        # Committed: the slots a re-init grow replaced left the pool for good.
        self._release_retired_slots()

    def _commit_init(self, candidate: "_InitCandidate") -> None:
        """Make a successful attempt's candidate the active state: every field
        the dataclass lists, so nothing is copied by hand."""
        for f in fields(candidate):
            setattr(self, f.name, getattr(candidate, f.name))

    def _configure_ring(
        self, init_info: ShardedRDTWeightTransferInitInfo
    ) -> "_ReceivePoolConfig":
        """Record candidate staging/S/K settings without allocating or mutating
        the active receive pool.

        The plan determines effective lanes L, so ``_size_pool`` applies these
        settings atomically after the plan has been built.
        """
        k = init_info.num_rdt_buffers
        config = _ReceivePoolConfig(
            staging_device=init_info.staging_device,
            stages_in_flight=init_info.stages_in_flight,
            ring_depth=k,
            buffer_presize=int(init_info.buffer_presize_gb * (1 << 30)),
        )
        logger.info(
            "[RDT-RING] active_pulls=%d per lane, presize=%.2fGiB",
            k,
            config.buffer_presize / (1 << 30),
        )
        return config

    def _size_pool(
        self,
        config: "_ReceivePoolConfig",
        plan: "_CallPlan",
        state: "_InitState | None" = None,
    ) -> None:
        """Lay out an LxK receive-pool geometry, after the static plan is known,
        into ``state`` (the engine itself when omitted).

        Once the post-processing worker exists, re-init may reuse the pool only
        for the exact ``(staging_device, L, K)`` tuple. The compatibility check
        precedes every pool-field write, so rejection leaves buffers, events,
        counters, cursors, and active geometry unchanged. A matching geometry
        reuses the active slots (the same tensors, in new slot dicts, so
        pre-registration finds them registered) and the lane cursors.
        """
        into: _InitState = self if state is None else state
        lanes = max(1, min(config.stages_in_flight, plan.num_streams))
        geometry = (config.staging_device, lanes, config.ring_depth)
        worker_exists = self._proc_thread is not None or bool(self._slot_read_done)
        if (
            worker_exists
            and self._pool_geometry is not None
            and geometry != self._pool_geometry
        ):
            old_staging, old_lanes, old_k = self._pool_geometry
            new_staging, new_lanes, new_k = geometry
            raise RuntimeError(
                "Sharded RDT re-init cannot change receive-pool geometry after "
                "the post-processing worker starts: "
                f"old=(staging_device={old_staging!r}, L={old_lanes}, K={old_k}), "
                f"new=(staging_device={new_staging!r}, L={new_lanes}, K={new_k})."
            )

        num_slots = lanes * config.ring_depth
        reuse_pool = (
            self._pool_geometry == geometry and len(self._dest_buffers) == num_slots
        )
        into._staging_device = config.staging_device
        into._stages_in_flight = config.stages_in_flight
        into._ring_depth = config.ring_depth
        into._lanes = lanes
        into._buffer_presize = config.buffer_presize
        into._targets_cache = {}
        if reuse_pool:
            into._lane_cursors = list(self._lane_cursors)
            into._dest_buffers = [dict(slot) for slot in self._dest_buffers]
        else:
            into._lane_cursors = [0] * lanes
            into._dest_buffers = [{} for _ in range(num_slots)]
        into._pool_geometry = geometry

        lowered = (
            f", lowered to {lanes} (the plan's {plan.num_streams} stage stream(s))"
            if lanes < config.stages_in_flight
            else ""
        )
        logger.info(
            "[RDT-PLAN] stages in flight: %d requested%s; pool of %d slots "
            "(%d lane(s) x %d)",
            config.stages_in_flight,
            lowered,
            num_slots,
            lanes,
            config.ring_depth,
        )

    @property
    def _num_slots(self) -> int:
        return self._lanes * self._ring_depth

    def _lane_slots(self, lane: int) -> range:
        """The contiguous K-slot block owned by one lane."""
        return range(lane * self._ring_depth, (lane + 1) * self._ring_depth)

    def _resolve_producers(
        self,
        init_info: ShardedRDTWeightTransferInitInfo,
        state: "_InitState | None" = None,
    ) -> None:
        """Work out this worker's consumer identity, build the router, and bind
        EVERY producer actor; the router lands in ``state`` (the engine itself
        when omitted).

        Pull routing is M:N: each chunk goes to ONE producer holding every name
        in it (see ``RdtRouter``), so one producer serves a whole pull. All
        producers are bound regardless, because the per-group ``free_group``
        signal fans out to every owner of a group, including producers this
        worker never pulls from.
        """
        try:
            import ray
        except ImportError as e:
            raise RuntimeError(
                "Ray is required for the 'sharded_rdt' weight transfer "
                "backend. Install Ray and run workers as Ray actors "
                "(distributed_executor_backend='ray')."
            ) from e

        producer_names = list(init_info.trainer_actor_names)
        if not producer_names:
            raise RuntimeError(
                "Sharded RDT engine requires a trainer producer: set "
                "init_info.trainer_actor_names."
            )

        into: _InitState = self if state is None else state
        consumer_id = self._resolve_consumer_id(init_info, state)
        name_owner_class = list(init_info.name_owner_class or [])
        if name_owner_class and len(name_owner_class) != len(init_info.names):
            raise RuntimeError(
                f"Sharded RDT engine: {len(name_owner_class)} owner-class entries "
                f"for {len(init_info.names)} names."
            )
        router = RdtRouter(
            len(producer_names),
            self._num_consumers(state),
            list(init_info.owner_sets) or None,
            name_owner_class or None,
            list(init_info.names),
            list(init_info.group_lens),
            # Carve the block per deployment so every replica's worker w pulls
            # each name from the same producer, which is what lets that producer
            # serve all of them from one slot.
            workers_per_replica=self._workers_per_replica(init_info, state),
        )
        router.validate()

        # A fresh router per init, never appended to: every owner index is a
        # position in these lists, so a rejoining engine's second
        # init_transfer_engine must not shift them.
        actors: list[Any] = []
        methods: list[Any] = []

        for chosen_name in producer_names:
            try:
                actor = ray.get_actor(
                    chosen_name,
                    namespace=init_info.trainer_actor_namespace,
                )
            except ValueError as e:
                raise RuntimeError(
                    f"Sharded RDT engine could not find trainer actor "
                    f"{chosen_name!r} (namespace="
                    f"{init_info.trainer_actor_namespace!r})."
                ) from e
            # ``ray.get_actor`` resolves by NAME, in a process that never
            # imported the producer's class, and ``enable_tensor_transport``
            # lives in that class's metadata: the creating process passes
            # ``meta.enable_tensor_transport``, and Ray infers it from the
            # class's ``@ray.method(tensor_transport=...)`` decorators. A
            # name-resolved handle therefore always reports False and the
            # dispatch guard rejects the pull, even though the trainer set the
            # option. Inherent to resolving by name across processes, not a
            # version bug; it fails on Ray 2.56.0.
            #
            # Forcing it skips no validation. Ray's next guard,
            # ``actor_has_tensor_transport``, asks the LIVE actor whether it can
            # build a NIXL agent and still runs, and ``_spawn_server`` hardcodes
            # the actor option, so a misconfigured producer cannot reach here.
            actor._ray_enable_tensor_transport = True
            actors.append(actor)
            methods.append(getattr(actor, init_info.produce_method_name))
        router.bind(actors, methods, consumer_id)
        into._router = router
        logger.info(
            "Sharded RDT engine (consumer %d) bound to all %d producers "
            "(batched method %r, %d owner class(es))",
            consumer_id,
            len(producer_names),
            init_info.produce_method_name,
            len(init_info.owner_sets) or 1,
        )

    def _workers_per_replica(
        self,
        init_info: ShardedRDTWeightTransferInitInfo,
        state: "_InitState | None" = None,
    ) -> int:
        """Consumers per inference deployment, assuming a uniform fleet.

        Read by this worker's consumer id and by the router's block carve, which
        must not disagree, so it is derived once here.
        """
        replicas = max(1, int(init_info.num_replicas or 1))
        return max(1, self._num_consumers(state) // replicas)

    def _resolve_consumer_id(
        self,
        init_info: ShardedRDTWeightTransferInitInfo,
        state: "_InitState | None" = None,
    ) -> int:
        """This worker's DISTINCT index in 0..C-1 across the whole fleet.

        Within one engine that is ``_global_worker_index()``. But a fleet of
        INDEPENDENT engines (each with its own parallel config) restarts that
        index at 0 per engine, so each engine offsets into its own range using
        ``replica_rank``: with a uniform fleet,
        ``workers_per_replica = C // num_replicas``. ``num_replicas`` defaults to
        1 (offset 0), preserving single-engine and single-DP-deployment behaviour.
        """
        replica_rank = max(0, int(init_info.replica_rank or 0))
        return (
            replica_rank * self._workers_per_replica(init_info, state)
            + self._global_worker_index()
        )

    def _build_static_plan(
        self,
        init_info: ShardedRDTWeightTransferInitInfo,
        state: "_InitState | None" = None,
    ) -> "_CallPlan":
        """Build the chunk/free plan once, from the bake in ``state`` (the
        engine itself when omitted). It never changes across syncs, so
        ``update_weights`` needs no per-sync names."""
        if not init_info.group_lens:
            raise ValueError(
                "Sharded RDT engine requires init_info.group_lens (the gather-"
                "group partition of init_info.names)."
            )
        if sum(init_info.group_lens) != len(init_info.names):
            raise ValueError(
                f"init_info.group_lens sums to {sum(init_info.group_lens)} "
                f"but {len(init_info.names)} names were given."
            )
        plan = self._build_call_plan(init_info.names, init_info.group_lens, state)
        logger.info(
            "[RDT-PLAN] pre-built static call plan at init: %d chunks in %d "
            "stage stream(s)",
            len(plan.chunks),
            plan.num_streams,
        )
        return plan

    def _preregister_at_init(self, state: "_InitState | None" = None) -> None:
        """Register every NIXL buffer this worker will use at init, before any
        transfer runs, so nothing registers during the sync-0 RDMA churn.

        Both sides are sized from the static plan in ``state`` (the engine itself
        when omitted): receive buffers are all LxK pool slots at the largest
        chunk's ``pack_bytes``, and each bound producer is asked to pre-register
        a serve ring at the max bytes this consumer will pull from it. A no-op
        when this worker has no chunks."""
        into: _InitState = self if state is None else state
        plan = into._cached_plan
        if plan is None or not plan.chunks:
            return
        # (a) consumer receive buffers — every slot in every active lane.
        max_pack = max(c.pack_bytes for c in plan.chunks)
        alloc = buffer_alloc_bytes(max_pack, into._buffer_presize)
        for slot in range(into._num_slots):
            buffer = into._dest_buffers[slot].get(torch.uint8)
            if buffer is None or buffer.numel() < alloc:
                into._dest_buffers[slot][torch.uint8] = self._alloc_receive_buffer(
                    alloc, state=into, pool_total=into._num_slots * alloc
                )
                if buffer is not None and into._staging_device == "host":
                    into._retired_slots.append((slot, buffer))

        # (b) producer serve rings — max bytes this consumer pulls from each
        # bound producer, plus the per-producer plan digest a producer sharing
        # one ring across deployments checks them against.
        router = into._router
        assert router is not None
        serve_bytes = [0] * router.num_producers
        per_owner_keys: list[list] = [[] for _ in range(router.num_producers)]
        for c in plan.chunks:
            serve_bytes[c.owner] = max(serve_bytes[c.owner], c.pack_bytes)
            per_owner_keys[c.owner].append(c.keys)
        digests = [_plan_digest(keys) for keys in per_owner_keys]
        import ray

        refs = router.reserve_serve_buffers(serve_bytes, digests)
        if refs:
            ray.get(refs)  # block until every serve ring is registered
        logger.info(
            "[RDT-PLAN] pre-registered %d receive slots (%.0f MiB each) + serve "
            "rings on %d producer(s) %s",
            into._num_slots,
            alloc / (1 << 20),
            len(refs),
            [nb // (1 << 20) for nb in serve_bytes],
        )

    def _alloc_receive_buffer(
        self,
        nbytes: int,
        *,
        state: "_InitState | None" = None,
        pool_total: int | None = None,
    ) -> torch.Tensor:
        """Allocate and register one receive buffer in the staging mode of the
        pool in ``state`` (the engine's active pool when omitted).

        This is the sole allocation seam for init preregistration, which sizes
        every slot to ``nbytes`` and passes the exact ``pool_total`` the failure
        message names, and for the defensive grow path in ``_issue_pull``, which
        omits it because only one slot is regrown while the others keep their
        size, so the message names that slot's bytes alone.
        """
        pool: _InitState = self if state is None else state
        if pool._staging_device == "gpu":
            buffer = torch.empty(nbytes, dtype=torch.uint8, device=self.device)
            register_nixl_memory(buffer)
            return buffer

        try:
            return _alloc_page_locked_slot(nbytes, park=self._park_host_slot)
        except Exception as error:
            raise HostStagingAllocationError(
                "staging_device='host' could not "
                f"{_host_alloc_request(pool, nbytes, pool_total)}; each slot "
                "is pageable host memory page-locked whole with "
                "cudaHostRegister and registered with NIXL, with no "
                f"pinned-allocator rounding or caching: {error}"
            ) from error

    def _release_host_slot(self, buffer: torch.Tensor) -> None:
        """Give one host slot back: NIXL deregistration, then
        ``cudaHostUnregister`` on the storage base, then forget the cached dtype
        views so dropping the tensor frees the pageable storage.

        A slot whose release fails at either step is never handed back to the
        allocator: after a failed NIXL deregistration it may still be
        registered, and after a failed ``cudaHostUnregister`` its page-lock
        state is unknown, so it is parked (``_park_host_slot``) instead. One
        slot's bytes leaked loudly beats a freed range a registration still
        covers.

        Never raises: it runs on shutdown and inside the init rollback, where a
        failure must not mask the caller's state.
        """
        base = buffer.untyped_storage().data_ptr()
        try:
            deregister_nixl_memory(buffer)
        except Exception:
            self._park_host_slot(
                buffer, "its NIXL deregistration failed", exc_info=True
            )
            return
        try:
            unregistered = _host_unregister(CudaRTLibrary(), base)
        except Exception:
            logger.exception("cudaHostUnregister of a host receive slot failed")
            unregistered = False
        if not unregistered:
            self._park_host_slot(buffer, "its cudaHostUnregister failed")
            return
        self._forget_cached_targets(buffer)

    def _park_host_slot(
        self, buffer: torch.Tensor, why: str, *, exc_info: bool = False
    ) -> None:
        """Keep ``buffer`` referenced (page-locked, possibly NIXL-registered) for
        the process lifetime instead of freeing it, and say so at error level:
        one leaked slot is the safe failure, a freed range that NIXL or a
        scatter still uses is not."""
        storage = buffer.untyped_storage()
        logger.error(
            "Host receive slot %#x (%d bytes) %s; it stays page-locked and "
            "parked, never freed",
            storage.data_ptr(),
            storage.nbytes(),
            why,
            exc_info=exc_info,
        )
        self._parked_host_slots.append(buffer)
        self._forget_cached_targets(buffer)

    def _forget_cached_targets(self, buffer: torch.Tensor) -> None:
        """Drop the cached dtype views carved out of ``buffer``."""
        stale = [
            key
            for key, (ptr, _views) in self._targets_cache.items()
            if ptr == buffer.data_ptr()
        ]
        for key in stale:
            del self._targets_cache[key]

    def _release_attempt_slots(self, candidate: "_InitCandidate") -> None:
        """Release the host slots a failed init attempt built: every tensor in
        the candidate's pool that the active pool does not hold. The ones it
        does hold were reused from it and stay registered with it.
        """
        if candidate._staging_device != "host":
            return
        active = {id(buffer) for slot in self._dest_buffers for buffer in slot.values()}
        for buffers in candidate._dest_buffers:
            for buffer in buffers.values():
                if id(buffer) not in active:
                    self._release_host_slot(buffer)

    def _release_retired_slots(self) -> None:
        """Release the host slots a committed re-init grow replaced, once the
        last scatter out of each has run."""
        retired, self._retired_slots = self._retired_slots, []
        for slot, buffer in retired:
            self._wait_slot_consumed(slot)
            self._release_host_slot(buffer)

    def _wait_slot_consumed(self, slot: int) -> None:
        """Block until every scatter out of ``slot`` has run: the generation
        handshake ``_issue_pull`` performs (while the worker is alive to publish
        done counts), then the slot's read-done event."""
        if slot >= len(self._slot_read_done):
            return  # no worker yet: nothing has read the slot
        if self._proc_thread is not None:
            with self._slot_cv:
                while self._slot_done[slot] < self._slot_queued[slot]:
                    self._slot_cv.wait(timeout=1.0)
        self._slot_read_done[slot].synchronize()

    def _global_worker_index(self) -> int:
        """This worker's stable, distinct global index across the inference fleet:
        ``data_parallel_index * world_size + rank`` over the TP*PP world.

        ``data_parallel_index``, not ``data_parallel_rank``: vLLM resets the
        latter to 0 in a dense worker but keeps the former as the distinct global
        DP rank. Same formula as the sibling ``nccl_engine``, so dense-via-TP and
        MoE-via-DP+EP both yield distinct 0..C-1."""
        pc = self.parallel_config
        return pc.data_parallel_index * pc.world_size + pc.rank  # world_size = TP*PP

    def _num_consumers(self, state: "_InitState | None" = None) -> int:
        """Total inference-worker count. Prefers the driver-supplied
        ``init_info.num_consumers`` (authoritative -- the driver knows the whole
        fleet), read from ``state`` (the engine itself when omitted); else
        ``world_size_across_dp``, the same stride ``_global_worker_index``
        indexes with, so the two agree at any pp."""
        into: _InitState = self if state is None else state
        if into._num_consumers_override > 0:
            return into._num_consumers_override
        return self.parallel_config.world_size_across_dp

    def start_weight_update(self) -> None:
        """Put the model's params on meta so layerwise reload streams them in
        as each layer's slices land. Baked replay uses checkpoint format."""
        from vllm.model_executor.model_loader.reload import (
            initialize_layerwise_reload,
        )

        initialize_layerwise_reload(self.model)

    def finish_weight_update(self) -> None:
        """Drain the deferred pull/process pipeline (so every layer is fully
        loaded) before finalizing the layerwise reload."""
        from vllm.model_executor.model_loader.reload import (
            finalize_layerwise_reload,
        )

        self.drain_pending()
        finalize_layerwise_reload(self.model, self.model_config)

    def update_weights(self, update_info: dict[str, Any]) -> None:
        """Receive one update. Unlike the base, does NOT issue a per-update
        device sync: post-processing is deferred to background threads and a
        sync here would block on them and serialize the pull/process pipeline.
        Completion is guaranteed by ``drain_pending`` in
        ``finish_weight_update``."""
        self.receive_weights(self.parse_update_info(update_info))

    def receive_weights(
        self,
        update_info: ShardedRDTWeightTransferUpdateInfo,
    ) -> None:
        """Pull + replay the baked leaf modules the sync covers.

        The chunk/free plan is STATIC across syncs — a pure function of the baked
        plan and the driver's group partition — so it was built once at init and
        every sync just re-runs the pipeline over its self-describing chunks. The
        update carries only the sync index used to tag free signals.

        Assumes each baked module's source names fall within one gather group,
        which the per-layer / pre / post partition guarantees (a leaf module's
        sources all live in one decoder layer). A module that did span groups
        would be planned once per group and could pull a name whose group the
        pipeline already freed, which parks the pull until the producer's stall
        watchdog fires.
        """
        self._sync_idx = update_info.sync_idx if update_info.sync_idx >= 0 else None
        if self._router is None:
            raise RuntimeError(
                "Sharded RDT engine not initialized. Call init_transfer_engine() first."
            )
        # Surface any error the background thread hit on a prior item promptly.
        self._raise_proc_error()
        if self._cached_plan is None:
            raise RuntimeError(
                "Sharded RDT engine has no call plan: init_info.group_lens must "
                "be supplied at init_transfer_engine()."
            )
        self._run_chunk_pipeline(self._cached_plan)

    def _build_fake_weights(
        self,
        names: list[str],
        sink: "BakeSink",
        device: torch.device,
        name_meta: dict[str, tuple[str, list[int]]],
    ) -> list[tuple[str, torch.Tensor]]:
        """Zero-storage lazies for ``names``, dtype/shape from the init metadata,
        all feeding the bake's recording sink."""
        return [
            (
                name,
                FakeRDTTensor(
                    name=name,
                    shape=torch.Size(name_meta[name][1]),
                    dtype=_dtype_from_name(name_meta[name][0]),
                    device=device,
                    sink=sink,
                ),
            )
            for name in names
        ]

    def _issue_pull(self, chunk: "_Chunk", slot: int) -> "_PendingPull":
        """Reserve ``slot``, lay the targets out in its buffer, dispatch the
        produce RPC and point the transfer at the buffer — WITHOUT the blocking
        ``ray.get`` (that is ``_complete_pull``). The chunked pipeline issues
        chunk i+1 before completing chunk i, so the producer serves the next
        chunk while the in-flight RDMA streams.

        Slot-reuse guard, both stages required: a generation wait (the CUDA event
        binds only to its LAST record, so synchronizing before the background
        thread recorded this item's event passes silently — observed as
        nondeterministic weight corruption), then the event synchronize. It must
        precede ``set_target_for_ref``, not just the get: the transfer may start
        any time after the metadata push. See the doc's "slot generation
        handshake".
        """
        from ray.experimental import set_target_for_ref

        if self._slot_read_done:
            with self._slot_cv:
                while self._slot_done[slot] < self._slot_queued[slot]:
                    self._slot_cv.wait(timeout=1.0)
            self._slot_read_done[slot].synchronize()

        # Every slice is byte-packed into ONE uint8 buffer, 16B-aligned in
        # keys order, under one NIXL descriptor. ``chunk.pack_layout`` mirrors the
        # producer's rule byte-exactly; ``targets`` carves the dtype views back out.
        keys = chunk.keys
        cur = chunk.pack_bytes
        buffers = self._dest_buffers[slot]
        buffer = buffers.get(torch.uint8)
        if buffer is None or buffer.numel() < cur:
            # Size ONCE with headroom: Ray's desc cache is keyed by data_ptr and
            # outlives its tensor, so a regrowth can false-hit a recycled pointer
            # and skip registering the new extent -> NIXL_ERR_NOT_FOUND.
            replaced = buffer
            buffer = self._alloc_receive_buffer(
                buffer_alloc_bytes(cur, self._buffer_presize)
            )
            buffers[torch.uint8] = buffer
            if replaced is not None and self._staging_device == "host":
                # The read-done wait above drained every scatter out of the old
                # slot, so its host memory goes back now.
                self._release_host_slot(replaced)
        targets = self._pull_targets(chunk, slot, buffer)
        # One produce RPC serves the whole packed chunk, since the group's owner
        # holds every slice. The buffer view must stay strongly referenced through
        # the get: set_target_for_ref stores WEAKREFS, and a dropped target
        # reroutes the transfer into a fallback buffer.
        blob = buffer[:cur]
        assert self._router is not None
        ref = self._router.pull(chunk.owner, keys, chunk.seq)
        set_target_for_ref(ref, [blob])
        return _PendingPull(
            ref=ref,
            keys=keys,
            targets=targets,
            blob=blob,
            slot=slot,
        )

    def _pull_targets(
        self, chunk: "_Chunk", slot: int, buffer: torch.Tensor
    ) -> "list[torch.Tensor]":
        """Per-key dtype views into ``slot``'s buffer.

        The packed layout is static, so the views are built once per (chunk, slot)
        rather than once per pull -- rebuilding them cost ~1150 Python ops per pull
        at 235B. Keyed on the buffer pointer as well, so a regrow invalidates
        instead of handing back views into a freed buffer.
        """
        cached = self._targets_cache.get((id(chunk), slot))
        if cached is not None and cached[0] == buffer.data_ptr():
            return cached[1]
        targets = [
            buffer[off : off + n * dt.itemsize].view(dt).reshape(shape)
            for off, dt, n, shape in chunk.pack_layout
        ]
        self._targets_cache[(id(chunk), slot)] = (buffer.data_ptr(), targets)
        return targets

    def _complete_pull(self, pending: "_PendingPull") -> "dict[FetchKey, torch.Tensor]":
        """Blocking half of a pull: the NIXL read lands during this ``ray.get``."""
        import ray

        ray.get(pending.ref)
        return dict(zip(pending.keys, pending.targets))

    def _slot_free(self, slot: int) -> bool:
        """Whether a pull into ``slot`` can start without waiting."""
        if not self._slot_read_done:
            return True
        with self._slot_cv:
            if self._slot_done[slot] < self._slot_queued[slot]:
                return False
        return self._slot_read_done[slot].query()

    def _wait_any_head(
        self,
        pulls: "list[_PendingPull]",
        timeout: "float | None" = None,
    ) -> "int | None":
        """Return the index of a landed pull, or None after ``timeout``."""
        if len(pulls) == 1 and timeout is None:
            # The following complete/get blocks on this pull, as the old
            # single-window loop did; avoid an extra Ray round trip.
            return 0
        import ray

        refs = [pull.ref for pull in pulls]
        ready, _ = ray.wait(refs, num_returns=1, timeout=timeout)
        return refs.index(ready[0]) if ready else None

    # ---------------- Bake (dry run, at init) / replay ----------------

    def _bake(
        self,
        init_info: ShardedRDTWeightTransferInitInfo,
        state: "_InitState | None" = None,
    ) -> None:
        """Bake the replay plan once, as a self-driven meta dry run, into
        ``state`` (the engine itself when omitted).

        Puts the params on meta, then drives ``model.load_weights`` over
        ``init_info.names`` through the model's ORIGINAL loaders (the stamps
        bypass ``online_process_loader``, so ``_layerwise_process`` is never in
        the path). Nothing materializes or pulls; the fake's ``copy_`` records
        the source op chain and the meta destination's geometry. Afterwards one
        scatter list per fully-loaded leaf module (copied numel == loadable
        size) is indexed by source name; a partial or unrecordable module fails
        the plan build. The model is restored.

        This leans on layerwise internals a public API should expose first-class:
        a currently-loading hook instead of monkeypatched stamps, a dry-run mode
        instead of bypassing ``online_process_loader``, and an
        ``abort_layerwise_reload`` instead of ``_restore_after_dry_run``.
        """
        from vllm.model_executor.model_loader.reload.layerwise import (
            initialize_layerwise_reload,
        )
        from vllm.model_executor.model_loader.reload.utils import get_layer_size

        into: _InitState = self if state is None else state
        names, dtype_names, shapes = (
            init_info.names,
            init_info.dtype_names,
            init_info.shapes,
        )
        into._name_meta = {n: (d, s) for n, d, s in zip(names, dtype_names, shapes)}
        into._name_to_plan = {}
        into._live_names = set()
        if not names:
            return

        model = self.model
        recorder = BakeSink()

        _t0 = time.perf_counter()
        with torch.device(self.device):
            # Meta-restore params + save kernel tensors (we bypass the loader
            # wrapping it installs, below).
            initialize_layerwise_reload(model)
            # Stamp the *original* loaders (bypassing online_process_loader), so
            # the single load pass runs the loaders on meta and records via the
            # fake's copy_ — with no inline _layerwise_process, no deferral.
            self._install_recording_stamps(model, recorder)
            model.load_weights(
                self._build_fake_weights(names, recorder, self.device, into._name_meta)
            )
            # Keep only fully-loaded modules (copied numel >= loadable size, the
            # test online_process_loader uses): a partial module leaves unwritten
            # regions that finalize would init, so baking it scatters garbage.
            for module, recorded in recorder.copies_by_layer.items():
                if not recorded or any(c is None for c in recorded):
                    continue  # unrecordable copy_ -> slow path
                # Guard above guarantees every entry is a real _Scatter.
                copies = cast("list[_Scatter]", recorded)
                copied = sum(prod(c.shape) for c in copies)
                if copied < get_layer_size(module):
                    continue  # partial -> slow path
                for c in copies:
                    # Every name of the module shares ONE list, so identity
                    # dedups them back to one module at plan time.
                    into._name_to_plan[c.src[0]] = copies
            self._restore_after_dry_run(model)

        # Names whose copy_ fired during the bake. Names not in here no-op for
        # this worker (e.g. foreign-EP experts) and are skipped; live names that
        # did not bake fail the plan build.
        into._live_names = set(recorder.copied_names)

        n_groups = len({id(g) for g in into._name_to_plan.values()})
        logger.info(
            "Sharded RDT dry-run baked %d/%d names into %d leaf modules "
            "(%d live) in %.3fs",
            len(into._name_to_plan),
            len(names),
            n_groups,
            len(into._live_names),
            time.perf_counter() - _t0,
        )

    def _install_recording_stamps(
        self, model: torch.nn.Module, recorder: "BakeSink"
    ) -> None:
        """Wrap each loadable param's ``weight_loader`` to stamp
        ``recorder.current = (leaf_module, param_name)`` before delegating to the
        original loader, so the fake's ``copy_`` can attribute each recorded copy.
        ``functools.wraps`` keeps the loader's real signature (so vLLM's
        ``_layerwise_process`` ``param`` redirect still works if a stamp leaks),
        and ``_rdt_stamp_inner`` tags it so ``_restore_after_dry_run`` can unwrap it.
        """
        import functools

        from vllm.model_executor.model_loader.reload.layerwise import (
            _get_original_loader,
        )
        from vllm.model_executor.model_loader.reload.utils import get_layer_tensors
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader

        def _make_stamp(layer, name, inner, added=False):
            @functools.wraps(inner)  # keep ``inner``'s signature (incl. ``param``)
            def stamp(*args, **kwargs):
                recorder.current = (layer, name)
                try:
                    return inner(*args, **kwargs)
                finally:
                    recorder.current = None

            # Tag so _restore_after_dry_run can detect and unwrap leaked stamps,
            # and so a second bake doesn't double-wrap.
            stamp._rdt_stamp_inner = inner  # type: ignore[attr-defined]
            stamp._rdt_stamp_added = added  # type: ignore[attr-defined]
            return stamp

        for module in model.modules():
            for name, tensor in get_layer_tensors(module).items():
                if getattr(tensor, "weight_loader", None) is None:
                    # A param with NO loader (e.g. GLM's router bias, a plain
                    # nn.Parameter) is still loaded by the model's load_weights
                    # through the getattr(param, "weight_loader",
                    # default_weight_loader) fallback — unstamped, its bake copy
                    # would be unattributable, failing the module's coverage
                    # gate and the plan build. Stamp the same default loader
                    # the fallback would pick; the restore deletes it again.
                    tensor.weight_loader = _make_stamp(
                        module, name, default_weight_loader, added=True
                    )
                    continue
                # Bypass online_process_loader: stamp the *original* loader.
                original = _get_original_loader(tensor)
                tensor.weight_loader = _make_stamp(module, name, original)

    def _restore_after_dry_run(self, model: torch.nn.Module) -> None:
        """Restore each layerwise layer's saved kernel tensors without pulling
        (a real ``finalize_layerwise_reload`` would materialize/load) and reset
        its info. Also unwrap any recording ``stamp`` left on the params, since a
        leaked stamp would sit under the next sync's ``online_process_loader`` and
        silently break ``_layerwise_process``'s ``param`` redirect.
        """
        from vllm.model_executor.model_loader.reload.layerwise import (
            LAYERWISE_INFO,
            _place_kernel_tensors,
        )
        from vllm.model_executor.model_loader.reload.utils import get_layer_tensors

        for layer in model.modules():
            info = LAYERWISE_INFO.get(layer)
            if info is not None and info.can_load():
                if info.kernel_tensors is not None:
                    _place_kernel_tensors(layer, info)
                info.reset()
        # Unwrap any recording stamps left on the (now-restored) params so they
        # never leak into a later update_weights. ``_rdt_stamp_inner`` is set by
        # ``_install_recording_stamps``; unwrap repeatedly in case of nesting.
        for module in model.modules():
            for _name, tensor in get_layer_tensors(module).items():
                loader = getattr(tensor, "weight_loader", None)
                added = False
                while loader is not None and hasattr(loader, "_rdt_stamp_inner"):
                    added = added or getattr(loader, "_rdt_stamp_added", False)
                    loader = loader._rdt_stamp_inner
                    tensor.weight_loader = loader
                if added:
                    # The stamp was ATTACHED to a param that had no loader
                    # (see _install_recording_stamps); leave none behind.
                    del tensor.weight_loader
        if hasattr(model, "_original_do_torchao_reload"):
            model._do_torchao_reload = model._original_do_torchao_reload

    # ---------------- Background post-processing (pull/process pipeline) -------

    def _ensure_proc_worker(self, state: "_InitState | None" = None) -> None:
        """Lazily create the per-slot events, the background CUDA stream, the work
        queue, and the single processing thread. Idempotent: a running worker
        keeps its per-slot handshake, which ``state`` (the engine itself when
        omitted) carries over; a new worker's handshake is sized for the pool in
        ``state``."""
        into: _InitState = self if state is None else state
        if self._proc_thread is not None:
            into._slot_read_done = self._slot_read_done
            into._slot_queued = self._slot_queued
            into._slot_done = self._slot_done
            into._slot_cv = self._slot_cv
            return
        import queue
        import threading

        into._slot_read_done = [torch.Event() for _ in range(into._num_slots)]
        # Generation handshake for the events above: the RPC thread counts items
        # queued per slot, this thread counts records, and a pull may synchronize()
        # only once done[slot] has caught up with queued[slot]. Without it the
        # synchronize can bind to a stale record and pass, letting the next RDMA
        # overwrite a slot under a pending scatter. See the doc's "The slot
        # generation handshake".
        into._slot_queued = [0] * into._num_slots
        into._slot_done = [0] * into._num_slots
        into._slot_cv = threading.Condition()
        self._proc_stream = torch.cuda.Stream(device=self.device)
        self._proc_queue = queue.Queue()
        self._proc_error = None
        t = threading.Thread(
            target=self._proc_worker_loop, name="rdt-postprocess", daemon=True
        )
        self._proc_thread = t
        t.start()
        self._quant_stream = torch.cuda.Stream(device=self.device)
        self._quant_queue = queue.Queue()
        qt = threading.Thread(
            target=self._quant_worker_loop, name="rdt-quant", daemon=True
        )
        self._quant_thread = qt
        qt.start()

    def _proc_worker_loop(self) -> None:
        """Single persistent thread: run each queued item's process phase on the
        background stream. Exits on the ``None`` sentinel (shutdown). An item that
        raises is recorded in ``_proc_error`` and re-raised on the RPC thread /
        at drain, so a failed sync fails loudly rather than corrupting silently.
        """
        torch.accelerator.set_device_index(self.device.index)
        q = self._proc_queue
        assert q is not None
        while True:
            item = q.get()
            try:
                if item is None:
                    return
                # _process_item publishes the slot generation internally (in a
                # finally around its scatter pass), so an error here still
                # unblocks the RPC thread's generation wait.
                self._process_item(item)
            except BaseException as e:  # noqa: BLE001 - surfaced on the RPC thread
                self._proc_error = e
                logger.exception("RDT background post-processing failed")
            finally:
                q.task_done()

    def _mark_slot_done(self, slot: int) -> None:
        """Publish that a queued item's read-done event has been recorded (or the
        item failed) so a pull waiting to reuse ``slot`` can proceed to its
        CUDA-event synchronize."""
        with self._slot_cv:
            self._slot_done[slot] += 1
            self._slot_cv.notify_all()

    def _raise_proc_error(self) -> None:
        """Re-raise (once) any error captured by the background thread."""
        if self._proc_error is not None:
            err = self._proc_error
            self._proc_error = None
            raise RuntimeError("RDT background post-processing failed") from err

    def drain_pending(self) -> None:
        """Block until the background thread has processed every queued item and
        its stream work is complete, then re-raise any error it hit. Called from
        the worker's ``finish_weight_update`` before ``finalize_layerwise_reload``
        so every baked layer is fully loaded (and ``info.reset()``-ed) first."""
        if self._proc_queue is not None:
            self._proc_queue.join()  # every put() item task_done()'d
        # The scatter thread feeds the quant thread, so join it SECOND (all
        # completed-group batches have been put by now), then sync both streams
        # so finalize sees fully-materialized, quanted, reset layers.
        if self._quant_queue is not None:
            self._quant_queue.join()
        if self._proc_stream is not None:
            self._proc_stream.synchronize()
        if self._quant_stream is not None:
            self._quant_stream.synchronize()
        self._raise_proc_error()
        # Ensure every fired free_group signal has EXECUTED on
        # the producer before the sync ends: ``begin_sync`` resets the producer's
        # per-group signal counts, so a signal landing after the next sync
        # started would credit a group it does not belong to, over-crediting the
        # gather-lookahead backpressure (extra gather resident -> trainer OOM
        # risk).
        if self._pending_frees:
            import ray

            try:
                ray.get(self._pending_frees)
            finally:
                self._pending_frees.clear()
        # One sync iteration fully drained; buffers/registrations are at (or
        # nearer) high-water — the chunk pipeline may issue ahead from now on.
        self._completed_syncs += 1

    def _dispatch_item(self, item: "_ProcItem") -> None:
        """Hand one chunk item to the background scatter thread.

        Counts the item against its slot BEFORE dispatch: the next pull into
        that slot must wait until the background thread has processed (and
        RECORDED the read-done event for) every item ever queued on it.
        """
        with self._slot_cv:
            self._slot_queued[item.slot] += 1
        assert self._proc_queue is not None
        self._proc_queue.put(item)

    def _chunk_module_scatters(
        self,
        modules: "list[list[_Scatter]]",
        state: "_InitState | None" = None,
    ) -> "list[tuple[int, list[_Scatter]]]":
        """Cut the modules' copies into one chunk per distinct owner class
        present, ascending by class index, as ``(class_idx, scatters)`` pairs.

        A chunk is one packed pull, so every name in it must share a producer —
        which is exactly what an owner class is. The cut is a pure function of
        the bake and the ownership table (the router in ``state``, the engine's
        when omitted). Copy order within a chunk is bake order, and a module's
        copies may span chunks (materialize/quant fire on its first/last chunk;
        see ``_build_call_plan``)."""
        router = (self if state is None else state)._router
        assert router is not None
        by_class: dict[int, list[_Scatter]] = {}
        for copies in modules:
            for c in copies:
                by_class.setdefault(router.class_of(c.src[0]), []).append(c)
        return [(ci, by_class[ci]) for ci in sorted(by_class)]

    def _build_call_plan(
        self,
        names: list[str],
        group_lens: list[int],
        state: "_InitState | None" = None,
    ) -> "_CallPlan":
        """Build the STATIC plan for one whole-sync call from the bake and
        router in ``state`` (the engine's own when omitted).

        Pure — no pulls, no engine state touched — so the result is cached and
        reused every sync. Three passes:
          1. Split ``names`` into gather groups, one chunk per owner class
             present in this worker's baked copies, recording each group's last
             chunk for its ``free_group`` signal (or ``pre_free`` when this worker
             pulls nothing from it). Every group is signaled exactly once by
             construction. The stream has no per-group call boundaries, so group
             L+1's first chunk issues while L's still stream.
          2. Per leaf module, find its FIRST and LAST chunk — materialize on the
             first, quant/kernel/reset on the last, correct by construction
             instead of by runtime counters.
          3. Assemble ``_Chunk``s: dedup keys and precompute the packed layout.
        """
        baked: _InitState = self if state is None else state
        router = baked._router
        assert router is not None, "init_transfer_engine() must run before planning"

        # --- pass 1: gather groups -> per-owner-class scatter chunks -----------
        raw_chunks: list[list[_Scatter]] = []
        raw_groups: list[int] = []
        free_at: dict[int, list[int]] = {}  # chunk idx -> groups to signal after it
        pre_free: list[int] = []  # groups with no chunk here (signal at start)
        unbaked: list[str] = []
        pos = 0
        for gi, glen in enumerate(group_lens):
            gnames = names[pos : pos + glen]
            pos += glen
            modules: list[list[_Scatter]] = []
            seen: set[int] = set()
            for n in gnames:
                mod = baked._name_to_plan.get(n)
                if mod is None:
                    if n in baked._live_names:
                        unbaked.append(n)
                elif id(mod) not in seen:
                    seen.add(id(mod))
                    modules.append(mod)
            if not modules:
                # Nothing to pull for this group on this worker; its owners
                # still published it, so signal it done at sync start (they
                # tolerate a signal that arrives before the publish).
                pre_free.append(gi)
                continue
            for _ci, scatters in self._chunk_module_scatters(modules, state):
                raw_chunks.append(scatters)
                raw_groups.append(gi)
            free_at.setdefault(len(raw_chunks) - 1, []).append(gi)
        if unbaked:
            # A live name with no baked plan cannot be loaded: there is no
            # fallback, and its pull would target groups the pipeline has
            # already freed. Fail here, where the names are known.
            raise RuntimeError(
                f"Sharded RDT: {len(unbaked)} live name(s) did not bake, "
                f"first: {unbaked[:3]}. Fix the loader so these names bake "
                "(fully load their module in one pass), or exclude them from "
                "the synced set."
            )

        # --- pass 2: per-module first/last chunk -> materialize/quant ----------
        first_at: dict[int, int] = {}
        last_at: dict[int, int] = {}
        layer_by_id: dict[int, Any] = {}
        for ci, scatters in enumerate(raw_chunks):
            for sc in scatters:
                lid = id(sc.layer)
                layer_by_id[lid] = sc.layer
                first_at.setdefault(lid, ci)
                last_at[lid] = ci
        materialize_at: dict[int, list[Any]] = {}
        quant_at: dict[int, list[Any]] = {}
        for lid, ci in first_at.items():
            materialize_at.setdefault(ci, []).append(layer_by_id[lid])
        for lid, ci in last_at.items():
            quant_at.setdefault(ci, []).append(layer_by_id[lid])

        # --- pass 3: assemble _Chunks (dedup keys + precompute pack layout) ----
        from collections import defaultdict

        chunks: list[_Chunk] = []
        per_owner_seq: dict[int, int] = defaultdict(int)
        for ci, scatters in enumerate(raw_chunks):
            keys: list[FetchKey] = []
            kmeta: dict[FetchKey, tuple[torch.dtype, tuple[int, ...]]] = {}
            for sc in scatters:
                if sc.src not in kmeta:
                    kmeta[sc.src] = (sc.dtype, sc.shape)
                    keys.append(sc.src)
            pack_layout: list[tuple[int, torch.dtype, int, tuple[int, ...]]] = []
            cur = 0
            for k in keys:
                dt, shape = kmeta[k]
                numel = prod(shape) or 1  # type: ignore[arg-type]
                off = (cur + 15) & ~15
                pack_layout.append((off, dt, numel, shape))
                cur = off + numel * dt.itemsize
            owner = router.producer_for(router.consumer_id, scatters[0].src[0])
            chunks.append(
                _Chunk(
                    seq=per_owner_seq[owner],
                    scatters=scatters,
                    keys=keys,
                    pack_layout=pack_layout,
                    pack_bytes=cur,
                    materialize=materialize_at.get(ci, []),
                    quant=quant_at.get(ci, []),
                    free=free_at.get(ci, []),
                    # Every name of a chunk shares an owner class and a group,
                    # so any of them resolves the same producer.
                    owner=owner,
                )
            )
            per_owner_seq[owner] += 1

        component = _stage_components(router, names, group_lens)
        stream_of: dict[int, int] = {}
        for chunk in chunks:
            chunk.stream = stream_of.setdefault(component[chunk.owner], len(stream_of))
        _check_streams(chunks, raw_groups)
        return _CallPlan(
            chunks=chunks,
            pre_free=pre_free,
            num_streams=len(stream_of),
        )

    def _signal_group_done(self, group_idx: int) -> None:
        """Fire-and-forget ``free_group`` signal at EVERY owner of the group.

        The per-group barrier: each owner counts one signal per live consumer
        and frees the group (releasing its lookahead credit) on the last one, so
        every owner must hear from every consumer — including owners this worker
        pulls nothing from. Refs are held and drained in ``drain_pending`` so
        every signal has EXECUTED before the sync ends: ``begin_sync`` resets
        the counters, and a straggler landing in the next sync would credit a
        group it does not belong to."""
        assert self._router is not None
        self._pending_frees.extend(self._router.free_group(group_idx, self._sync_idx))

    def _run_chunk_pipeline(self, plan: "_CallPlan") -> None:
        """Pull stage streams through independent K-deep FIFO lanes.

        Slot safety rests on two arguments spelled out in the doc: the producer's
        ring is no shallower than a lane's window and FIFO drain-before-issue
        orders its reuse; the consumer's slots are held by ``_issue_pull``'s
        generation handshake.
        """
        # Gather groups with no chunk on this worker (plan.pre_free): signal
        # them done before the pipeline — their owners still published them, and
        # a signal that arrives before its publish is counted (fire-and-forget).
        for gi in plan.pre_free:
            self._signal_group_done(gi)

        def drain(pending: "_PendingPull", chunk: "_Chunk") -> None:
            results = self._complete_pull(pending)
            self._dispatch_item(
                _ProcItem(chunk=chunk, results=results, slot=pending.slot)
            )
            # Each gather group whose LAST chunk this is: its read is done, so
            # this consumer is finished with the group -> signal every owner
            # (fire-and-forget, off the critical path).
            for gi in chunk.free:
                self._signal_group_done(gi)

        # Sync 0 is serial while both endpoints finish registration. A single
        # lane always sees the whole plan, preserving the old issue/drain/free
        # order even when the plan contains several stage streams.
        if self._completed_syncs == 0:
            lanes, depth = 1, 1
        else:
            lanes, depth = self._lanes, self._ring_depth
        streams = (
            [plan.chunks]
            if lanes == 1
            else [
                [chunk for chunk in plan.chunks if chunk.stream == stream]
                for stream in range(plan.num_streams)
            ]
        )
        _run_lanes(
            streams,
            depth=depth,
            blocks=[self._lane_slots(lane) for lane in range(lanes)],
            cursors=self._lane_cursors,
            hooks=_PullHooks(
                issue=self._issue_pull,
                slot_free=self._slot_free,
                wait_any=self._wait_any_head,
                drain=drain,
                signal=self._signal_group_done,
            ),
        )

    def _process_item(self, item: "_ProcItem") -> None:
        """Scatter-thread half: materialize this chunk's first-seen modules,
        scatter its slices on the process stream, publish the slot, then hand the
        modules it COMPLETES to the quant thread.

        Mirrors ``_layerwise_process`` minus the loader replay. Once every scatter
        reading ``item.slot`` is enqueued, records the slot's read-done event so
        the RPC thread can block on it before overwriting the slot.
        """
        from vllm.model_executor.model_loader.reload.layerwise import (
            LAYERWISE_INFO,
        )
        from vllm.model_executor.model_loader.reload.meta import materialize_layer

        results = item.results
        chunk = item.chunk
        with (
            torch.accelerator.device_index(self.device.index),
            torch.cuda.stream(self._proc_stream),
            torch.device(self.device),
        ):
            # PASS 1 — slot readers: materialize the modules whose FIRST scatter
            # is in this chunk (empty HF params, once per module by construction)
            # then scatter this chunk's copies. The scatter copies are the ONLY
            # reads of the receive buffer; quant and kernel-copy operate on the
            # scattered params. Releasing the slot right after the scatters lets
            # the NEXT chunk's RDMA overwrite the buffer while quant still runs.
            try:
                for layer in chunk.materialize:
                    info = LAYERWISE_INFO.get(layer)
                    if info is None or not info.can_load():
                        raise RuntimeError(
                            f"Baked replay: layer {type(layer).__name__} "
                            "was not set up for reload this sync "
                            "(start_weight_update must run before "
                            "update_weights)."
                        )
                    materialize_layer(layer, info)
                for sc in chunk.scatters:
                    param = getattr(sc.layer, sc.param_name)
                    # Through .data, exactly like default_weight_loader: params
                    # that keep requires_grad=True (plain nn.Parameters like
                    # GLM's router bias) reject an in-place copy through an
                    # autograd view of the leaf.
                    dst = param.data.as_strided(sc.shape, sc.stride, sc.offset)
                    with torch._C.DisableTorchFunctionSubclass():
                        # Host staging needs this to enqueue rather than wait.
                        # The read-done event below still prevents overwrite
                        # until the process stream has consumed the slot.
                        dst.copy_(results[sc.src], non_blocking=True)
                # All reads of this slot's buffer are now enqueued on the process
                # stream; record + publish so the RPC thread can reuse the slot.
                self._slot_read_done[item.slot].record(self._proc_stream)
            finally:
                # Publish even on error so the RPC thread's generation wait
                # unblocks (the failure itself surfaces via _raise_proc_error).
                self._mark_slot_done(item.slot)

            # PASS 2 — param readers only: quant / kernel-copy / reset for the
            # modules whose LAST scatter is in this chunk. Handed to the
            # DEDICATED quant thread (own CUDA stream, event-chained after this
            # chunk's scatters) so it never delays this thread's next pass-1.
            # Running it in order here stalls the RPC thread's slot handshake by
            # ~0.5-0.75s/iter, since every group's quant delays the next publish.
            if chunk.quant:
                ready = torch.Event()
                ready.record(self._proc_stream)
                if self._quant_queue is None:
                    self._run_quant(chunk.quant, ready)
                else:
                    self._quant_queue.put((chunk.quant, ready))

    def _run_quant(self, layers: "list[Any]", ready: "torch.Event") -> None:
        """Quant/kernel-copy/reset the given COMPLETED leaf modules through the
        same ``_process_loaded_layer`` as _layerwise_process. Runs on the quant
        thread's own stream, ordered after the modules' scatters via ``ready``;
        touches only the scattered params (never a receive slot), so it can
        overlap subsequent chunks' RDMA and scatters. The ``info.reset()`` in
        there is what makes finalize skip the layer — drain_pending joins the
        quant queue before finalize runs."""
        from vllm.model_executor.model_loader.reload.layerwise import (
            LAYERWISE_INFO,
            _process_loaded_layer,
        )
        from vllm.model_executor.model_loader.reload.utils import get_layer_tensors

        stream = self._quant_stream or self._proc_stream
        assert stream is not None  # created in _ensure_proc_worker before use
        with (
            torch.accelerator.device_index(self.device.index),
            torch.cuda.stream(stream),
            torch.device(self.device),
        ):
            stream.wait_event(ready)
            for layer in layers:
                info = LAYERWISE_INFO.get(layer)
                assert info is not None  # completed leaf module is set up for reload
                # The layer tensors were allocated on the process stream, but
                # this stream reads and then drops them. Keep their storage live
                # until those reads complete.
                for tensor in get_layer_tensors(layer).values():
                    if tensor.is_cuda:
                        tensor.record_stream(stream)
                _process_loaded_layer(layer, info)

    def _quant_worker_loop(self) -> None:
        """Dedicated quant thread: drains (completed_modules, scatter-done event)
        batches. Errors surface via _proc_error like the scatter thread's."""
        torch.accelerator.set_device_index(self.device.index)
        q = self._quant_queue
        assert q is not None
        while True:
            batch = q.get()
            try:
                if batch is None:
                    return
                self._run_quant(*batch)
            except BaseException as e:  # noqa: BLE001 - surfaced on the RPC thread
                self._proc_error = e
                logger.exception("RDT quant thread failed")
            finally:
                q.task_done()

    def shutdown(self) -> None:
        # Stop the background post-processing thread (drain, then sentinel + join)
        # before dropping the state it touches.
        worker_alive = False
        if self._proc_thread is not None:
            try:
                self.drain_pending()
            except Exception:
                logger.exception("RDT drain during shutdown failed")
            assert self._proc_queue is not None
            self._proc_queue.put(None)  # sentinel
            self._proc_thread.join(timeout=30)
            # A worker that did not stop may still scatter out of the receive
            # slots, so they are parked below rather than freed under it.
            worker_alive = self._proc_thread.is_alive()
            if worker_alive:
                logger.error(
                    "RDT post-processing thread did not stop within 30 s; host "
                    "receive slots are parked, not freed"
                )
            self._proc_thread = None
            self._proc_queue = None
            self._proc_stream = None
        if self._quant_thread is not None:
            assert self._quant_queue is not None
            self._quant_queue.put(None)  # sentinel
            self._quant_thread.join(timeout=30)
            self._quant_thread = None
            self._quant_queue = None
            self._quant_stream = None
        # Give every host slot back (NIXL deregistration, then cudaHostUnregister)
        # before the pool is dropped below; GPU slots keep their process-lifetime
        # NIXL registration as before.
        if self._staging_device == "host":
            for slot, buffers in enumerate(self._dest_buffers):
                if worker_alive:
                    for buffer in buffers.values():
                        self._park_host_slot(
                            buffer,
                            "may still be read by the post-processing thread "
                            "that did not stop",
                        )
                    continue
                self._wait_slot_consumed(slot)
                for buffer in buffers.values():
                    self._release_host_slot(buffer)
        self._retired_slots = []
        self._slot_read_done = []
        self._router = None
        # Drop strong references to baked modules so the model can be freed.
        self._name_to_plan.clear()
        self._name_meta.clear()
        self._live_names.clear()
        # Drop the cached plan (holds _Scatter refs to the baked layers).
        self._cached_plan = None
        # Drop the receive buffers: GPU slots keep their NIXL registration for the
        # process lifetime (dropping the tensors just drops our strong refs);
        # host slots were released above.
        self._dest_buffers = [{} for _ in range(self._num_slots)]


def _stage_components(
    router: RdtRouter,
    names: list[str],
    group_lens: list[int],
) -> list[int]:
    """Return each producer's connected owner component.

    Producers are connected when they co-own names in one gather group. Pure
    pipeline stages therefore form separate components, while a cross-stage
    group conservatively merges its stages into one ordered stream.
    """
    parent = list(range(router.num_producers))

    def root(rank: int) -> int:
        while parent[rank] != rank:
            parent[rank] = parent[parent[rank]]
            rank = parent[rank]
        return rank

    pos = 0
    for glen in group_lens:
        owners = sorted(
            {owner for name in names[pos : pos + glen] for owner in router.owners(name)}
        )
        pos += glen
        for other in owners[1:]:
            parent[root(other)] = root(owners[0])
    return [root(rank) for rank in range(router.num_producers)]


def _check_streams(chunks: "list[_Chunk]", groups: list[int]) -> None:
    """Refuse streams that split an owner, gather group, or leaf module."""
    if len(chunks) != len(groups):
        raise ValueError("chunks and groups must be parallel")

    seen: dict[str, dict[Any, tuple[int, str]]] = {
        "trainer rank": {},
        "gather group": {},
        "module loading": {},
    }
    for chunk, group in zip(chunks, groups):
        members = [
            ("trainer rank", chunk.owner, str(chunk.owner)),
            ("gather group", group, str(group)),
        ]
        members.extend(
            ("module loading", id(sc.layer), repr(sc.src[0])) for sc in chunk.scatters
        )
        for kind, key, label in members:
            stream, _ = seen[kind].setdefault(key, (chunk.stream, label))
            if stream != chunk.stream:
                raise RuntimeError(
                    f"Sharded RDT plan: the {kind} {label} spans stage streams "
                    f"{stream} and {chunk.stream}, so parallel pulls would "
                    "break its order."
                )


# How long a lane waiting for a slot lets outstanding heads block before the
# scheduler rechecks that slot.
_SLOT_POLL_S = 0.005


def _run_lanes(
    streams: "list[list[_Chunk]]",
    *,
    depth: int,
    blocks: "list[range]",
    cursors: list[int],
    hooks: _PullHooks,
) -> None:
    """Run one K-deep FIFO pull window per disjoint slot block.

    A lane takes the next unstarted stream after issuing its current stream's
    last chunk, so its window spans the handoff. If a lane is waiting for slot
    reuse while other lane heads are outstanding, the head wait is bounded:
    shared producer packs can otherwise deadlock while each replica waits for a
    pull that another replica has not yet issued.
    """
    from collections import deque

    lanes = range(len(blocks))
    unstarted = deque(streams)
    todo: list[deque[_Chunk]] = [deque() for _ in lanes]
    windows: list[deque[tuple[Any, _Chunk]]] = [deque() for _ in lanes]

    def advance(lane: int) -> None:
        while True:
            if not todo[lane]:
                if not unstarted:
                    return
                todo[lane].extend(unstarted.popleft())
            elif todo[lane][0].keys:
                return
            else:
                for group in todo[lane].popleft().free:
                    hooks.signal(group)

    def next_slot(lane: int) -> int:
        return blocks[lane][cursors[lane]]

    def issue(lane: int) -> None:
        chunk = todo[lane].popleft()
        slot = next_slot(lane)
        cursors[lane] = (cursors[lane] + 1) % len(blocks[lane])
        windows[lane].append((hooks.issue(chunk, slot), chunk))
        advance(lane)

    for lane in lanes:
        advance(lane)
    while True:
        for lane in lanes:
            while (
                todo[lane]
                and len(windows[lane]) < depth
                and hooks.slot_free(next_slot(lane))
            ):
                issue(lane)
        waiting = [lane for lane in lanes if todo[lane] and len(windows[lane]) < depth]
        busy = [lane for lane in lanes if windows[lane]]
        working = [lane for lane in lanes if todo[lane] or windows[lane]]
        if waiting and (not busy or len(working) == 1):
            # No other lane can make progress, so let issue perform the slot's
            # full generation/event wait.
            issue(waiting[0])
        elif busy:
            heads = [windows[lane][0][0] for lane in busy]
            index = hooks.wait_any(heads, _SLOT_POLL_S if waiting else None)
            if index is not None:
                hooks.drain(*windows[busy[index]].popleft())
        else:
            return


def _plan_digest(keys_per_chunk: list) -> str:
    """Digest of the chunks one consumer pulls from one producer, in pull order.

    Two consumers a producer serves out of ONE shared serve ring must agree on
    this, since sharing rests on their plans being identical; the producer
    compares it at init. Over the whole (name, op-chain) list rather than the
    names, because the chains decide the bytes each pull returns.
    """
    import hashlib

    h = hashlib.sha256()
    for keys in keys_per_chunk:
        h.update(repr(keys).encode())
        h.update(b"\n")
    return h.hexdigest()[:32]


def _dtype_from_name(name: str) -> torch.dtype:
    """Resolve a string like 'bfloat16' to torch.bfloat16."""
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unknown torch dtype name: {name!r}")
    return dtype
