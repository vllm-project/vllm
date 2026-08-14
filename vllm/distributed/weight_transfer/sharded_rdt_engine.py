# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sharded Ray Direct Transport (RDT) weight transfer engine (consumer side).

Pulls only the slice each vLLM worker consumes under tensor/expert parallelism,
not the full HF-format tensor.

Two phases. BAKE, once at ``init_transfer_engine``: drive ``model.load_weights``
against ``LazyRDTTensor`` placeholders and record, per leaf module, how each
destination slice is fetched (an op chain) and where it lands (an ``as_strided``
descriptor). REPLAY, every sync: no ``load_weights``, no lazy dispatch, no
discovery — pull the recorded slices in packed chunks over a ring of receive
arenas, scatter them into freshly materialized params, then quant and
kernel-copy. A live name with no recorded plan fails the plan build: there
is no fallback load.

Only valid with ``is_checkpoint_format=True`` (layerwise reload).

Data flow
---------
One thing at four resolutions, over three lifetimes. ``FetchKey`` --
``(name, op_chain)``, "which slice of which trainer tensor" -- is the atom;
everything else is bookkeeping around it.

    BAKE      once at init, kept for the engine's life
      LazyRDTTensor intercepts the model's loaders; each copy_ records a
      _Scatter (sharded_rdt_lazy): src FetchKey, owning layer, the destination
      as_strided region, and the produced dtype/nbytes.
        -> _name_to_plan: name -> that module's scatter list

    PLAN      once, cached, _build_call_plan
      _Chunk    one packed pull: its scatters, deduped keys, the byte-exact
                pack_layout, which producer serves it, and what to run after
                (materialize / quant / free).  One per (group, owner class).
      _CallPlan all chunks + pre_free.

    RUN       per chunk, per sync, _run_chunk_pipeline
      _Chunk -> _PendingPull   issued, not yet landed: Ray ref, arena views,
                               ring slot.               [RPC thread]
             -> _ProcItem      chunk + results + slot.  [hand-off to the
                               background scatter thread]

The RUN pair stays split on purpose: it is the thread boundary, and only
``targets`` should outlive the get -- carrying the Ray ref and the whole-arena
blob into the queue would keep both alive for the scatter's lifetime.

See docs/training/weight_transfer/sharded_rdt.md for the design and the measured
results behind the choices here.
"""

import time
from dataclasses import dataclass, field
from math import prod
from typing import TYPE_CHECKING, Any, cast

import torch

from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)
from vllm.distributed.weight_transfer.sharded_rdt_common import (
    RdtRouter,
    arena_alloc_bytes,
)
from vllm.distributed.weight_transfer.sharded_rdt_lazy import (
    BakeSink,
    FetchKey,
    LazyRDTTensor,
    _Scatter,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


@dataclass
class _Chunk:
    """One packed pull plus its post-processing (see Data flow).

    A module's copies span chunks when its experts live on several producer
    coordinates, so ``materialize`` fires on its FIRST chunk and ``quant`` on its
    LAST -- materialize-once by construction, not by a runtime counter.
    ``pack_layout`` mirrors the producer's rule byte-exactly.
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
class _ProcItem:
    """A landed pull handed to the background scatter thread (see Data flow).

    ``results`` alias the ring arena ``slot``, held as strong refs so they
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
    ``.options(name=...)``), ordered by trainer rank. ``RdtRouter`` picks which
    one serves each pull: the group's owners (pipeline-stage ownership), narrowed
    to a matching expert coordinate, then rotated by group index so consumers
    spread across them rather than funneling through rank 0. Every actor is bound
    regardless, because ``free_group`` fans out to every owner. Must be
    non-empty; a single-producer trainer passes a one-element list."""

    trainer_actor_namespace: str | None = None
    """Optional Ray namespace the trainer actor(s) live in."""

    produce_method_name: str = "rdt_produce_weights_batched"
    """Name of the trainer-side producer method. It and the rest of the serve
    surface (``free_group``, ``reserve_serve_arena``) are documented where they
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
    from ``WeightSource.held_names()``."""

    num_consumers: int = 0
    """Total consumer count across the fleet, for M:N routing. Authoritative when
    > 0; at 0 the engine infers it from ``parallel_config``, which is correct for
    the supported serving modes but worth setting explicitly under M:N. Each
    worker's distinct index comes from ``_global_worker_index``."""

    num_rdt_buffers: int = 2
    """Depth of the consumer receive-arena ring. Must match the
    producer's — ``_run_chunk_pipeline``'s slot-safety argument rests on it. 2 =
    double buffer: chunk i+1's serve overlaps chunk i's RDMA, and scatter(i-1)
    overlaps RDMA(i) in the other slot. Keep depth x chunk_bytes under the
    fabric's address-translation reach (~2-3 GB/flow on the reference 8xB200 RoCE
    cluster, where K=3 measurably hurt)."""

    arena_presize_gb: float = 0.0
    """Pre-size each packed receive-arena slot to this many GiB
    (0 = size to the first chunk + coarse 256MB round-up). Set it to cover the
    model's largest atomic chunk (e.g. an untied lm_head). Sizing arenas ONCE
    matters beyond perf -- see the doc's "Sizing arenas once matters beyond
    throughput"."""

    replica_rank: int = 0
    """This inference engine's ordinal in the fleet (0..``num_replicas``-1).

    Multi-engine deployments run several INDEPENDENT inference engines, each with
    its own self-contained parallel config, so every engine's
    ``_global_worker_index`` restarts at 0 and would collide across engines. The
    driver assigns each engine a distinct ``replica_rank`` (with identical
    ``num_replicas``) so the engine offsets its consumers into a globally distinct
    range for the M:N block assignment. Default 0 (single engine) is unchanged."""

    num_replicas: int = 1
    """Number of independent inference engines in the fleet. Default 1 => the
    per-replica offset is 0 and consumer identity is exactly
    ``_global_worker_index`` (preserves single-engine and single-DP-deployment
    behavior). When > 1, ``workers_per_replica = num_consumers // num_replicas``
    and this engine's consumers occupy
    ``replica_rank * workers_per_replica + _global_worker_index()``. Assumes a
    uniform fleet (every replica has the same worker count)."""


@dataclass
class ShardedRDTWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Update info for the sharded RDT backend: intentionally EMPTY.

    The chunk/free plan is a pure function of the baked plan and the driver's
    gather-group partition, both fixed for the engine's lifetime, so it is built
    once at ``init_transfer_engine`` from ``ShardedRDTWeightTransferInitInfo``'s
    ``names`` + ``group_lens``. ONE ``update_weights`` per sync then just re-runs
    that plan; there is nothing per-sync to carry.
    """


class ShardedRDTWeightTransferEngine(
    WeightTransferEngine[
        ShardedRDTWeightTransferInitInfo,
        ShardedRDTWeightTransferUpdateInfo,
    ]
):
    """Pull-based RDT/NIXL backend that transports only the slice each worker
    consumes.

    Requires ``distributed_executor_backend="ray"``, ``nixl`` in the shared env,
    ``is_checkpoint_format=True``, a named trainer actor exposing a
    ``@ray.method(tensor_transport="nixl")`` producer, and weight loaders that
    stay inside ``SUPPORTED_OPS`` — anything needing real data (``.to``,
    ``.item``, arithmetic, bool-mask indexing) raises during the bake.

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

        # ---- Consumer-side pre-registered receive arenas -----------------------
        # One persistent 1-D arena per dtype per ring slot, STORAGE-registered with
        # NIXL once, so every per-slice view hits Ray's registration cache and the
        # recv path does zero registration in steady state. See the doc's "Sizing
        # arenas once matters beyond throughput" for why they rarely grow.
        #
        # Ring depth is both the slot count and the in-flight pull count, one
        # quantity from ``num_rdt_buffers``. Decoupling them with a spare slot
        # measurably slows the sync; see the doc before trying it.
        self._ring_depth = 1
        self._dest_arenas: list[dict[torch.dtype, torch.Tensor]] = [{}]
        self._slot_read_done: list[Any] = []  # one torch.cuda.Event per slot
        # Generation counters guarding the events (see _ensure_proc_worker).
        self._slot_queued: list[int] = []
        self._slot_done: list[int] = []
        self._slot_cv: Any = None
        self._pull_slot = 0
        # (id(chunk), slot) -> (arena data_ptr, per-key views).
        # Lives on the engine, not on the _Chunk: the plan is static and shared
        # across syncs, so a per-slot runtime cache has no business inside it.
        self._targets_cache: dict[tuple[int, int], tuple[int, list[torch.Tensor]]] = {}
        self._arena_presize = 0  # bytes; set from init_info
        self._pending_frees: list[Any] = []  # free_group signal refs, drained per sync
        # The STATIC plan: built once — at init from init_info.group_lens, else
        # lazily on the first update_weights — and reused. Non-None means
        # update_weights ignores its per-sync names.
        self._cached_plan: _CallPlan | None = None
        # Completed syncs. The chunk pipeline runs SERIAL during sync 0, when both
        # sides still register arenas, and pipelines from sync 1. See the doc.
        self._completed_syncs = 0

        # ---- Background post-processing thread (pull/process pipelining) -------
        # The RPC thread hands pulled arena views to this worker, which runs
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
        """Configure the ring, bind the producers, bake the replay plan, and
        pre-register every NIXL buffer -- in that order, because each step depends
        on the previous one.

        The bake drives ``model.load_weights`` and the pre-registration blocks on
        RPCs to the producers, so this is a heavyweight one-off; every later
        ``update_weights`` is pure replay.
        """
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
        self._configure_ring(init_info)
        self._resolve_producers(init_info)

        # A pure dry run: the trainer's gather cache is empty at init, so nothing
        # can (or does) get pulled -- we only record how each slice is fetched and
        # where it lands, then restore the model.
        self._bake(init_info)
        self._build_static_plan(init_info)
        # Register ALL NIXL memory now, while the fabric is idle: dma-buf GPUDirect
        # registration concurrent with in-flight RDMA intermittently fails
        # (ibv_reg_mr 'Bad address'), which bites under M:N fan-in. Sizes come from
        # the static plan, so this is exact.
        self._preregister_at_init()
        # Start the background post-processing worker (pull/process pipelining).
        self._ensure_proc_worker()

    def _configure_ring(self, init_info: ShardedRDTWeightTransferInitInfo) -> None:
        """Ring depth K.

        Must run before ``_ensure_proc_worker`` creates the per-slot events and
        counters, and before any arena is grown (both happen on the first pull).
        """
        self._num_consumers_override = int(init_info.num_consumers or 0)
        k = max(1, int(init_info.num_rdt_buffers))
        self._ring_depth = k
        self._dest_arenas = [{} for _ in range(k)]
        self._targets_cache = {}
        self._arena_presize = int(float(init_info.arena_presize_gb) * (1 << 30))
        logger.info(
            "[RDT-RING] active_pulls=%d slots=%d presize=%.2fGiB",
            k,
            k,
            self._arena_presize / (1 << 30),
        )

    def _resolve_producers(self, init_info: ShardedRDTWeightTransferInitInfo) -> None:
        """Work out this worker's consumer identity, build the router, and bind
        EVERY producer actor.

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

        consumer_id = self._resolve_consumer_id(init_info)
        name_owner_class = list(init_info.name_owner_class or [])
        if name_owner_class and len(name_owner_class) != len(init_info.names):
            raise RuntimeError(
                f"Sharded RDT engine: {len(name_owner_class)} owner-class entries "
                f"for {len(init_info.names)} names."
            )
        router = RdtRouter(
            len(producer_names),
            self._num_consumers(),
            list(init_info.owner_sets) or None,
            name_owner_class or None,
            list(init_info.names),
            list(init_info.group_lens),
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
            # Ray 2.51.1: handles from ray.get_actor lose
            # _ray_enable_tensor_transport, so the NIXL dispatch guard rejects the
            # call even when the trainer enabled it. Force it back on.
            actor._ray_enable_tensor_transport = True
            actors.append(actor)
            methods.append(getattr(actor, init_info.produce_method_name))
        router.bind(actors, methods, consumer_id)
        self._router = router
        logger.info(
            "Sharded RDT engine (consumer %d) bound to all %d producers "
            "(batched method %r, %d owner class(es))",
            consumer_id,
            len(producer_names),
            init_info.produce_method_name,
            len(init_info.owner_sets) or 1,
        )

    def _resolve_consumer_id(self, init_info: ShardedRDTWeightTransferInitInfo) -> int:
        """This worker's DISTINCT index in 0..C-1 across the whole fleet.

        Within one engine that is ``_global_worker_index()``. But a fleet of
        INDEPENDENT engines (each with its own parallel config) restarts that
        index at 0 per engine, so each engine offsets into its own range using
        ``replica_rank``: with a uniform fleet,
        ``workers_per_replica = C // num_replicas``. ``num_replicas`` defaults to
        1 (offset 0), preserving single-engine and single-DP-deployment behaviour.
        """
        num_replicas = max(1, int(init_info.num_replicas or 1))
        replica_rank = max(0, int(init_info.replica_rank or 0))
        workers_per_replica = self._num_consumers() // num_replicas
        return replica_rank * workers_per_replica + self._global_worker_index()

    def _build_static_plan(self, init_info: ShardedRDTWeightTransferInitInfo) -> None:
        """Build the chunk/free plan once. It never changes across syncs, so
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
        self._cached_plan = self._build_call_plan(init_info.names, init_info.group_lens)
        logger.info(
            "[RDT-PLAN] pre-built static call plan at init: %d chunks",
            len(self._cached_plan.chunks),
        )

    def _preregister_at_init(self) -> None:
        """Register every NIXL buffer this worker will use at init, before any
        transfer runs, so nothing registers during the sync-0 RDMA churn.

        Both sides are sized from the static plan: receive arenas are
        ``ring_depth`` slots at the largest chunk's ``pack_bytes``, and each bound
        producer is asked to pre-register a serve ring at the max bytes this
        consumer will pull from it. A no-op when this worker has no chunks."""
        plan = self._cached_plan
        if plan is None or not plan.chunks:
            return
        from ray.experimental import register_nixl_memory

        # (a) consumer receive arenas — one per ring slot, at the largest chunk.
        max_pack = max(c.pack_bytes for c in plan.chunks)
        alloc = arena_alloc_bytes(max_pack, self._arena_presize)
        for slot in range(self._ring_depth):
            arena = self._dest_arenas[slot].get(torch.uint8)
            if arena is None or arena.numel() < alloc:
                arena = torch.empty(alloc, dtype=torch.uint8, device=self.device)
                register_nixl_memory(arena)
                self._dest_arenas[slot][torch.uint8] = arena

        # (b) producer serve rings — max bytes this consumer pulls from each
        # bound producer.
        assert self._router is not None
        serve_bytes = [0] * self._router.num_producers
        for c in plan.chunks:
            serve_bytes[c.owner] = max(serve_bytes[c.owner], c.pack_bytes)
        import ray

        refs = self._router.reserve_serve_arenas(serve_bytes)
        if refs:
            ray.get(refs)  # block until every serve ring is registered
        logger.info(
            "[RDT-PLAN] pre-registered %d receive slots (%.0f MiB each) + serve "
            "rings on %d producer(s) %s",
            self._ring_depth,
            alloc / (1 << 20),
            len(refs),
            [nb // (1 << 20) for nb in serve_bytes],
        )

    def _global_worker_index(self) -> int:
        """This worker's stable, distinct global index across the inference fleet:
        ``data_parallel_index * world_size + rank`` over the TP*PP world.

        ``data_parallel_index``, not ``data_parallel_rank``: vLLM resets the
        latter to 0 in a dense worker but keeps the former as the distinct global
        DP rank. Same formula as the sibling ``nccl_engine``, so dense-via-TP and
        MoE-via-DP+EP both yield distinct 0..C-1."""
        pc = self.parallel_config
        return pc.data_parallel_index * pc.world_size + pc.rank  # world_size = TP*PP

    def _num_consumers(self) -> int:
        """Total inference-worker count. Prefers the driver-supplied
        ``init_info.num_consumers`` (authoritative -- the driver knows the whole
        fleet); else infers ``data_parallel_size * tensor_parallel_size``, correct
        for the supported serving modes (dense via TP, MoE via DP+EP) and wrong
        only for DP-over-dense, which vLLM itself rejects."""
        if self._num_consumers_override > 0:
            return self._num_consumers_override
        pc = self.parallel_config
        return pc.data_parallel_size * pc.tensor_parallel_size

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
        every sync just re-runs the pipeline over its self-describing chunks, with
        no per-sync bookkeeping and an empty ``update_info``. Residual names with
        no baked plan — attention scales, padded/partial layers — take the plain
        per-slice load; ``load_weights`` is used only by that path.

        Assumes each baked module's source names fall within one gather group
        (true for the per-layer / pre / post partition); if not, the pull
        fails loudly on the missing slice rather than loading wrong data.
        """
        del update_info  # the plan is static; nothing arrives per sync
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

    def _build_lazy_weights(
        self,
        names: list[str],
        sink: "BakeSink",
        device: torch.device,
    ) -> list[tuple[str, torch.Tensor]]:
        """Zero-storage lazies for ``names``, dtype/shape from the init metadata,
        all feeding the bake's recording sink."""
        return [
            (
                name,
                LazyRDTTensor(
                    name=name,
                    shape=torch.Size(self._name_meta[name][1]),
                    dtype=_dtype_from_name(self._name_meta[name][0]),
                    device=device,
                    sink=sink,
                ),
            )
            for name in names
        ]

    def _issue_pull(self, chunk: "_Chunk", slot: int) -> "_PendingPull":
        """Reserve ``slot``, lay the targets out in its arena, dispatch the
        produce RPC and point the transfer at the arena — WITHOUT the blocking
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
        from ray.experimental import register_nixl_memory, set_target_for_ref

        if self._slot_read_done:
            with self._slot_cv:
                while self._slot_done[slot] < self._slot_queued[slot]:
                    self._slot_cv.wait(timeout=1.0)
            self._slot_read_done[slot].synchronize()

        # Every slice is byte-packed into ONE uint8 arena, 16B-aligned in
        # keys order, under one NIXL descriptor. ``chunk.pack_layout`` mirrors the
        # producer's rule byte-exactly; ``targets`` carves the dtype views back out.
        keys = chunk.keys
        cur = chunk.pack_bytes
        arenas = self._dest_arenas[slot]
        arena = arenas.get(torch.uint8)
        if arena is None or arena.numel() < cur:
            # Size ONCE with headroom: Ray's desc cache is keyed by data_ptr and
            # outlives its tensor, so a regrowth can false-hit a recycled pointer
            # and skip registering the new extent -> NIXL_ERR_NOT_FOUND.
            alloc = arena_alloc_bytes(cur, self._arena_presize)
            arena = torch.empty(alloc, dtype=torch.uint8, device=self.device)
            register_nixl_memory(arena)
            arenas[torch.uint8] = arena
        targets = self._pull_targets(chunk, slot, arena)
        # One produce RPC serves the whole packed chunk, since the group's owner
        # holds every slice. The arena view must stay strongly referenced through
        # the get: set_target_for_ref stores WEAKREFS, and a dropped target
        # reroutes the transfer into a fallback buffer.
        blob = arena[:cur]
        assert self._router is not None
        ref = self._router.pull(chunk.owner, keys)
        set_target_for_ref(ref, [blob])
        return _PendingPull(
            ref=ref,
            keys=keys,
            targets=targets,
            blob=blob,
            slot=slot,
        )

    def _pull_targets(
        self, chunk: "_Chunk", slot: int, arena: torch.Tensor
    ) -> "list[torch.Tensor]":
        """Per-key dtype views into ``slot``'s arena.

        The packed layout is static, so the views are built once per (chunk, slot)
        rather than once per pull -- rebuilding them cost ~1150 Python ops per pull
        at 235B. Keyed on the arena pointer as well, so a regrow invalidates
        instead of handing back views into a freed buffer.
        """
        cached = self._targets_cache.get((id(chunk), slot))
        if cached is not None and cached[0] == arena.data_ptr():
            return cached[1]
        targets = [
            arena[off : off + n * dt.itemsize].view(dt).reshape(shape)
            for off, dt, n, shape in chunk.pack_layout
        ]
        self._targets_cache[(id(chunk), slot)] = (arena.data_ptr(), targets)
        return targets

    def _complete_pull(self, pending: "_PendingPull") -> "dict[FetchKey, torch.Tensor]":
        """Blocking half of a pull: the NIXL read lands during this ``ray.get``."""
        import ray

        ray.get(pending.ref)
        return dict(zip(pending.keys, pending.targets))

    # ---------------- Bake (dry run, at init) / replay ----------------

    def _bake(self, init_info: ShardedRDTWeightTransferInitInfo) -> None:
        """Bake the replay plan once, as a self-driven meta dry run.

        Puts the params on meta, then drives ``model.load_weights`` over
        ``init_info.names`` through the model's ORIGINAL loaders (the stamps
        bypass ``online_process_loader``, so ``_layerwise_process`` is never in
        the path). Nothing materializes or pulls; the lazy's ``copy_`` records
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

        names, dtype_names, shapes = (
            init_info.names,
            init_info.dtype_names,
            init_info.shapes,
        )
        self._name_meta = {n: (d, s) for n, d, s in zip(names, dtype_names, shapes)}
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
            # lazy's copy_ — with no inline _layerwise_process, no deferral.
            self._install_recording_stamps(model, recorder)
            model.load_weights(self._build_lazy_weights(names, recorder, self.device))
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
                    self._name_to_plan[c.src[0]] = copies
            self._restore_after_dry_run(model)

        # Names whose copy_ fired during the bake. Names not in here no-op for
        # this worker (e.g. foreign-EP experts) and are skipped; live names that
        # did not bake fail the plan build.
        self._live_names = set(recorder.copied_names)

        n_groups = len({id(g) for g in self._name_to_plan.values()})
        logger.info(
            "Sharded RDT dry-run baked %d/%d names into %d leaf modules "
            "(%d live) in %.3fs",
            len(self._name_to_plan),
            len(names),
            n_groups,
            len(self._live_names),
            time.perf_counter() - _t0,
        )

    def _install_recording_stamps(
        self, model: torch.nn.Module, recorder: "BakeSink"
    ) -> None:
        """Wrap each loadable param's ``weight_loader`` to stamp
        ``recorder.current = (leaf_module, param_name)`` before delegating to the
        original loader, so the lazy's ``copy_`` can attribute each recorded copy.
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

    def _ensure_proc_worker(self) -> None:
        """Lazily create the per-slot events, the background CUDA stream, the work
        queue, and the single processing thread. Idempotent."""
        if self._proc_thread is not None:
            return
        import queue
        import threading

        self._slot_read_done = [torch.cuda.Event() for _ in range(self._ring_depth)]
        # Generation handshake for the events above: the RPC thread counts items
        # queued per slot, this thread counts records, and a pull may synchronize()
        # only once done[slot] has caught up with queued[slot]. Without it the
        # synchronize can bind to a stale record and pass, letting the next RDMA
        # overwrite a slot under a pending scatter. See the doc's "The slot
        # generation handshake".
        self._slot_queued = [0] * self._ring_depth
        self._slot_done = [0] * self._ring_depth
        self._slot_cv = threading.Condition()
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
        torch.cuda.set_device(self.device)
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
        # One sync iteration fully drained; arenas/registrations are at (or
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
        self, modules: "list[list[_Scatter]]"
    ) -> "list[tuple[int, list[_Scatter]]]":
        """Cut the modules' copies into one chunk per distinct owner class
        present, ascending by class index, as ``(class_idx, scatters)`` pairs.

        A chunk is one packed pull, so every name in it must share a producer —
        which is exactly what an owner class is. The cut is a pure function of
        the bake and the ownership table. Copy order within a chunk is bake
        order, and a module's copies may span chunks (materialize/quant fire on
        its first/last chunk; see ``_build_call_plan``)."""
        assert self._router is not None
        by_class: dict[int, list[_Scatter]] = {}
        for copies in modules:
            for c in copies:
                by_class.setdefault(self._router.class_of(c.src[0]), []).append(c)
        return [(ci, by_class[ci]) for ci in sorted(by_class)]

    def _build_call_plan(self, names: list[str], group_lens: list[int]) -> "_CallPlan":
        """Build the STATIC plan for one whole-sync call.

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
        router = self._router
        assert router is not None, "init_transfer_engine() must run before planning"

        # --- pass 1: gather groups -> per-owner-class scatter chunks -----------
        raw_chunks: list[list[_Scatter]] = []
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
                mod = self._name_to_plan.get(n)
                if mod is None:
                    if n in self._live_names:
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
            for _ci, scatters in self._chunk_module_scatters(modules):
                raw_chunks.append(scatters)
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
        chunks: list[_Chunk] = []
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
            chunks.append(
                _Chunk(
                    scatters=scatters,
                    keys=keys,
                    pack_layout=pack_layout,
                    pack_bytes=cur,
                    materialize=materialize_at.get(ci, []),
                    quant=quant_at.get(ci, []),
                    free=free_at.get(ci, []),
                    # Every name of a chunk shares an owner class and a group,
                    # so any of them resolves the same producer.
                    owner=router.producer_for(router.consumer_id, scatters[0].src[0]),
                )
            )
        return _CallPlan(chunks=chunks, pre_free=pre_free)

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
        self._pending_frees.extend(self._router.free_group(group_idx))

    def _run_chunk_pipeline(self, plan: "_CallPlan") -> None:
        """Pipelined chunk pulls over the ring of receive slots.

        Issues up to ``ring_depth`` produce RPCs ahead of the blocking gets, so
        while chunk i's RDMA streams the producer serves i+1 into its own ring
        slot and the background thread scatters i-1 out of another. Reads stay
        serialized on the shared NIC — the bandwidth floor, not a loss.

        Slot safety rests on two arguments spelled out in the doc: the producer's
        ring is no shallower than this one and drain-before-issue orders its
        reuse; the consumer's slots are held by ``_issue_pull``'s generation
        handshake.
        """
        from collections import deque

        inflight: deque[tuple[_PendingPull, _Chunk]] = deque()

        # Gather groups with no chunk on this worker (plan.pre_free): signal
        # them done before the pipeline — their owners still published them, and
        # a signal that arrives before its publish is counted (fire-and-forget).
        for gi in plan.pre_free:
            self._signal_group_done(gi)

        def drain_one() -> None:
            pending, chunk = inflight.popleft()
            results = self._complete_pull(pending)
            self._dispatch_item(
                _ProcItem(chunk=chunk, results=results, slot=pending.slot)
            )
            # Each gather group whose LAST chunk this is: its read is done, so
            # this consumer is finished with the group -> signal every owner
            # (fire-and-forget, off the critical path).
            for gi in chunk.free:
                self._signal_group_done(gi)

        for chunk in plan.chunks:
            if not chunk.keys:
                # _chunk_module_scatters never emits empty chunks; keep the
                # signals safe anyway if this ever changes.
                for gi in chunk.free:
                    self._signal_group_done(gi)
                continue
            # Drain BEFORE issue once the ring is full: frees this chunk's slot
            # (generation-wise) and guarantees the producer-slot invariant above.
            # Sync 0 runs SERIAL (max 1 in flight): both sides still grow and
            # register arenas, and a producer registration mid-flight churns its
            # agent-metadata version under an in-flight pull (see __init__).
            depth = 1 if self._completed_syncs == 0 else self._ring_depth
            if len(inflight) >= depth:
                drain_one()
            slot = self._pull_slot
            self._pull_slot = (slot + 1) % self._ring_depth
            inflight.append((self._issue_pull(chunk, slot), chunk))
        while inflight:
            drain_one()

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
            torch.cuda.device(self.device),
            torch.cuda.stream(self._proc_stream),
            torch.device(self.device),
        ):
            # PASS 1 — slot readers: materialize the modules whose FIRST scatter
            # is in this chunk (empty HF params, once per module by construction)
            # then scatter this chunk's copies. The scatter copies are the ONLY
            # reads of the receive arena; quant and kernel-copy operate on the
            # scattered params. Releasing the slot right after the scatters lets
            # the NEXT chunk's RDMA overwrite the arena while quant still runs.
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
                        dst.copy_(results[sc.src])
                # All reads of this slot's arena are now enqueued on the process
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
                ready = torch.cuda.Event()
                ready.record(self._proc_stream)
                if self._quant_queue is None:
                    self._run_quant(chunk.quant, ready)
                else:
                    self._quant_queue.put((chunk.quant, ready))

    def _run_quant(self, layers: "list[Any]", ready: "torch.cuda.Event") -> None:
        """Quant/kernel-copy/reset the given COMPLETED leaf modules, exactly as
        _layerwise_process. Runs on the quant thread's own stream, ordered after
        the modules' scatters via ``ready``; touches only the scattered params
        (never a receive slot), so it can overlap subsequent chunks' RDMA and
        scatters. ``info.reset()`` is what makes finalize skip the layer —
        drain_pending joins the quant queue before finalize runs."""
        from vllm.model_executor.layers.quantization.base_config import (
            QuantizeMethodBase,
        )
        from vllm.model_executor.model_loader.reload.layerwise import (
            LAYERWISE_INFO,
            _copy_and_restore_kernel_tensors,
        )

        stream = self._quant_stream or self._proc_stream
        assert stream is not None  # created in _ensure_proc_worker before use
        with (
            torch.cuda.device(self.device),
            torch.cuda.stream(stream),
            torch.device(self.device),
        ):
            stream.wait_event(ready)
            for layer in layers:
                info = LAYERWISE_INFO.get(layer)
                assert info is not None  # completed leaf module is set up for reload
                quant_method = getattr(layer, "quant_method", None)
                if isinstance(quant_method, QuantizeMethodBase):
                    if hasattr(layer, "_already_called_process_weights_after_loading"):
                        delattr(layer, "_already_called_process_weights_after_loading")
                    quant_method.process_weights_after_loading(layer)
                # Copy into persistent kernel storage (preserves cudagraph refs).
                if info.kernel_tensors is not None:
                    _copy_and_restore_kernel_tensors(layer, info)
                # Reset so finalize_layerwise_reload skips this (loaded) layer.
                info.reset()

    def _quant_worker_loop(self) -> None:
        """Dedicated quant thread: drains (completed_modules, scatter-done event)
        batches. Errors surface via _proc_error like the scatter thread's."""
        torch.cuda.set_device(self.device)
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
        if self._proc_thread is not None:
            try:
                self.drain_pending()
            except Exception:
                logger.exception("RDT drain during shutdown failed")
            assert self._proc_queue is not None
            self._proc_queue.put(None)  # sentinel
            self._proc_thread.join(timeout=30)
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
        self._slot_read_done = []
        self._router = None
        # Drop strong references to baked modules so the model can be freed.
        self._name_to_plan.clear()
        self._name_meta.clear()
        self._live_names.clear()
        # Drop the cached plan (holds _Scatter refs to the baked layers).
        self._cached_plan = None
        # Release the receive arenas (their NIXL registration is pinned for the
        # process lifetime; freeing the tensors just drops our strong refs).
        self._dest_arenas = [{} for _ in range(self._ring_depth)]


def _dtype_from_name(name: str) -> torch.dtype:
    """Resolve a string like 'bfloat16' to torch.bfloat16."""
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unknown torch dtype name: {name!r}")
    return dtype
