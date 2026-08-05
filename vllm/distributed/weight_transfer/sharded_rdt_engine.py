# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sharded Ray Direct Transport (RDT) weight transfer engine.

This backend pulls only the *slice* that each vLLM worker actually consumes
(under tensor/expert parallelism), not the full HF-format tensor.

It works in two phases, keyed per ``update_weights`` name set:

  * **Bake** (first sync for a name set): drive ``model.load_weights`` with
    ``LazyRDTTensor`` placeholders that defer materialization. The
    placeholders intercept a whitelisted set of view/slice ops into a single
    ordered op chain; the whole payload's slices are fetched in one batched
    RPC, the trainer replays each chain on its live parameter and ships only
    the resulting slice. While replaying the loaders, we *record* a plan: for
    each leaf module, how each destination slice is fetched (source op-chain)
    and where it lands (an ``as_strided`` descriptor into a real param).

  * **Replay** (every later sync): no ``model.load_weights``, no lazy-tensor
    dispatch, no per-loader discovery. One batched pull, then scatter each
    recorded slice directly into freshly materialized params, run
    ``process_weights_after_loading``, and copy into kernel storage.

Within a single ``update_weights`` call we replay the baked groups its names
cover (one batched pull) and route only the residual names — those with no
recorded plan (attention/partial-layer finalize path, or experts owned by
another EP rank that no-op in their loader) — to the plain per-slice load.

Only valid with ``is_checkpoint_format=True`` (layerwise reload). See
``sharded_weight_loader_rdt.md`` and ``baked_rdt_replay.md`` for the design,
and the spike in ``nixl_slice_spike.py`` confirming NIXL is view-aware.
"""

import time
from collections.abc import Iterator
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

# The op allowlist, M:N binding, arena sizing and the chunk split all live in
# sharded_rdt_common so the producer (trainer) side agrees with the consumer
# here.
from vllm.distributed.weight_transfer.sharded_rdt_common import (
    RdtRouter,
    arena_alloc_bytes,
    greedy_run_starts,
)

# Op-chain recording: the lazy tensor the model's loaders see, and the two sinks
# its ``copy_`` can mean (record during the bake / fetch on the plain load).
from vllm.distributed.weight_transfer.sharded_rdt_lazy import (
    BakeSink,
    FetchKey,
    LazyRDTTensor,
    PullSink,
    _BakedCopy,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


@dataclass
class _BakedModule:
    """A baked leaf module and the recorded copies that fill its params.

    Named "module", not "group": a *group* in this file is always one of the
    driver's gather groups (see ``layerwise_groups``), and one gather group covers
    many baked modules.

    ``layer`` is a strong reference to the module, held for the engine's
    lifetime and cleared in ``shutdown``. The module persists across syncs (the
    model is not rebuilt), and its ``LayerReloadingInfo`` — with the meta
    ``restore_metadata`` and per-sync ``kernel_tensors`` — is re-established by
    ``initialize_layerwise_reload`` at the start of every update. Whether the
    layer needs ``process_weights_after_loading`` is decided at replay time
    (same ``quant_method`` check the stock path uses), so it isn't stored here.
    """

    layer: Any
    copies: list[_BakedCopy]


@dataclass
class _Scatter:
    """One self-contained scatter: pull ``src`` and copy the received slice into
    ``layer``'s ``param_name`` at the recorded strided region.

    Enriched form of ``_BakedCopy`` for the runtime plan — it carries its own
    produced ``dtype`` / ``nbytes`` (so the pack layout and byte-balancing need
    no side-table lookup) and a strong ref to its leaf ``layer``. The param is
    resolved at RUNTIME (``getattr(layer, param_name)``), not baked: every sync
    re-materializes fresh param tensors, so a param handle captured at plan time
    would be stale.
    """

    layer: Any
    param_name: str
    src: FetchKey
    offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    nbytes: int


@dataclass
class _Chunk:
    """One packed pull + its post-processing, fully described at plan time.

    ``scatters`` is the flat list of copies this chunk pulls (byte-balanced cut
    of the gather group's copies; a module's copies may span chunks).
    ``keys``/``pack_layout``/``pack_bytes`` are the deduped source keys and the
    byte-exact packed arena layout (16B-aligned, keys order) mirroring the
    producer — precomputed so the pull path does no per-call arithmetic.
    ``materialize`` = leaf modules whose FIRST scatter is in this chunk (empty
    HF params allocated before the scatter loop, once per module by
    construction). ``quant`` = modules whose LAST scatter is in this chunk (run
    ``process_weights_after_loading`` / kernel-copy / ``info.reset()`` after the
    scatter). ``free`` = ``(group_idx, names)`` of the gather groups whose last
    chunk this is (fire ``free_gather`` after the pull returns).
    """

    scatters: "list[_Scatter]"
    keys: "list[FetchKey]"
    pack_layout: "list[tuple[int, torch.dtype, int, tuple[int, ...]]]"  # off,dt,n,shape
    pack_bytes: int
    materialize: "list[Any]"
    quant: "list[Any]"
    free: "list[tuple[int, list[str]]]"
    # Local index (into ``_produce_methods``) of the producer that owns this
    # chunk's gather group and therefore serves the whole pull. One producer per
    # chunk: an owner holds every slice of its group, and splitting a pull across
    # producers would only multiply produce calls since the consumer's own NIC
    # bounds it either way.
    owner: int


@dataclass
class _CallPlan:
    """The STATIC plan for one sync — a pure function of the baked plan
    (``_name_to_module`` / ``_live_names`` / ``_name_meta``) and the driver's
    group partition, both fixed for the engine's lifetime. Built ONCE (at init
    when the driver passes ``group_lens`` on the init info, else lazily on the
    first ``update_weights``) and reused every sync: each self-describing
    ``_Chunk`` carries its own scatter/pack/materialize/quant/free actions, so
    runtime is pure execution — no per-sync counters or side-tables.

    ``pre_free`` = ``(group_idx, names)`` of gather groups with NO chunk on this
    worker but which the trainer still gathered (freed before the pipeline; the
    owner completes such a free when it publishes, since it can arrive first).
    ``residual`` = live-but-unbaked names for the plain-load fallback.
    ``name_group_idx`` maps every name to its gather group, which is what selects
    the owning producer for a residual name's on-demand pull.
    """

    chunks: "list[_Chunk]"
    pre_free: "list[tuple[int, list[str]]]"
    residual: "list[str]"
    name_group_idx: "dict[str, int]"


@dataclass
class _PendingPull:
    """An issued-but-not-completed pull: the produce RPC is dispatched and the
    transfer pointed at ring-slot arena views, but the blocking ``ray.get`` has
    not run.

    ``targets``/``blob`` hold the arena views strongly referenced until
    completion -- ``set_target_for_ref`` stores WEAKREFS, so dropping them would
    silently reroute the transfer into a fallback buffer. ``targets`` are the
    per-key dtype views the scatter reads; ``blob`` is the whole-arena uint8 view
    handed to ``set_target_for_ref``."""

    ref: "Any"
    keys: "list[FetchKey]"
    targets: "list[torch.Tensor]"
    blob: "torch.Tensor"
    slot: int
    # [RDT-SLOT-WAIT diagnostic] time the RPC thread spent blocked reusing the
    # slot: generation wait (bg thread reaching the item's record) + CUDA event
    # synchronize (the scatters actually finishing). Quantifies how much of the
    # background thread's in-order pass-2 (quant) work leaks onto the pull path.
    slot_wait_seconds: float = 0.0
    slot_sync_seconds: float = 0.0


@dataclass
class _ProcItem:
    """One chunk of deferred post-processing handed from the RPC thread (which
    did the synchronous pull) to the background process thread.

    ``chunk`` is the self-describing ``_Chunk`` (scatters + materialize/quant
    module lists). ``results`` are views aliasing the ring arena ``slot``; they
    are held as strong refs here so they outlive the RPC-thread frame until the
    background scatter consumes them. The timing fields were measured on the RPC
    thread during the pull and are logged (together with the process-phase
    split) by the background thread after it finishes the item.
    """

    chunk: "_Chunk"
    results: "dict[FetchKey, torch.Tensor]"
    slot: int
    t_recv: float
    pull_seconds: float
    nixl_delta: dict
    pull_bytes: int


@dataclass
class ShardedRDTWeightTransferInitInfo(WeightTransferInitInfo):
    """Initialization info for the sharded RDT backend."""

    trainer_actor_names: list[str] = field(default_factory=list)
    """Names of all trainer Ray actors that expose the producer method (set via
    ``.options(name=...)``), ordered by trainer rank. Every rank that gathers a
    layer can serve NIXL pulls for it, so workers spread their pulls across this
    list to parallelize the trainer-side clone + NIC egress instead of funneling
    through rank 0. ``RdtRouter`` decides which of them serves each gather group
    for this worker (see ``group_owners`` and ``_select_producer_indices``).
    Must be non-empty; a single-producer trainer passes a one-element list."""

    trainer_actor_namespace: str | None = None
    """Optional Ray namespace the trainer actor(s) live in."""

    produce_method_name: str = "rdt_produce_weights_batched"
    """Name of the trainer-side producer method (implemented by the serve actor
    the trainer engine spawns; see
    vllm/distributed/weight_transfer/sharded_rdt_trainer.py). Must be decorated
    with
    ``@ray.method(tensor_transport="nixl")``. Contract: given a batched specs
    list ``[(name, [(op_name, args, kwargs_items), ...]), ...]``, replay each
    chain on the named tensor and return ONE contiguous uint8 blob with every
    slice byte-packed at 16B-aligned offsets in specs order (the engine
    computes the identical layout and carves dtype views back out; a one-spec
    request is that slice at offset 0). The trainer must also expose
    ``free_gather(names)`` (may be a no-op when it has no gather plan)."""

    names: list[str] = field(default_factory=list)
    """The full, flat list of parameters to transfer (the trainer's complete
    param name list). The engine bakes a replay plan once at
    ``init_transfer_engine`` by driving ``model.load_weights`` over all of these
    against meta params, then keys the plan by source name. ``update_weights``
    later passes the subset of these names it gathered for that call."""

    dtype_names: list[str] = field(default_factory=list)
    """Dtype name (e.g. 'bfloat16') for each entry of ``names``."""

    shapes: list[list[int]] = field(default_factory=list)
    """Full HF shape for each entry of ``names``."""

    group_lens: list[int] = field(default_factory=list)
    """Partition of ``names`` into gather groups, in the SAME order the trainers
    gather and publish them (group-major; ``sum(group_lens) == len(names)``, and
    ``names`` must be ordered to match). Required: the engine pre-builds the whole
    static chunk/free plan from it at init, and fires ``free_gather`` on the bound
    producer as each group's last chunk completes."""

    group_owners: list[list[int]] = field(default_factory=list)
    """Optional per-group ownership: ``group_owners[g]`` lists the trainer ranks
    (indices into ``trainer_actor_names``) that gather and publish group ``g``.
    Set when producers hold disjoint parts of the model — pipeline-parallel
    producers that gather within a stage instead of to all ranks — so each pull
    must go to an owner of that group. Empty means every producer owns every
    group (gather-to-all). Aligned with ``group_lens``."""

    num_consumers: int = 0
    """Total inference-worker (consumer) count across the whole fleet, for the M:N
    producer/consumer routing (see ``RdtRouter``). The
    driver knows it (``tensor_parallel_size * data_parallel_size``). Authoritative
    when > 0; when 0 the engine infers it from ``parallel_config`` (correct for the
    supported serving modes — dense→TP, MoE→DP+EP). Set it explicitly for M:N so
    the count is never guessed. Each worker's DISTINCT index comes from
    ``data_parallel_index * world_size + rank`` (see ``_global_worker_index``)."""

    num_rdt_buffers: int = 2
    """[RDT-RING] Depth of the consumer receive-arena ring; the producer mirrors
    it from ``ShardedRDTTrainerInitInfo.num_rdt_buffers``, and the two MUST
    agree (the producer-ring safety argument in ``_run_chunk_pipeline`` rests on
    it). 2 = double buffer: chunk i+1's
    produce/serve overlaps chunk i's RDMA read, and scatter(i-1) overlaps
    RDMA(i) in the other slot. Tune with ``layerwise_split`` so
    num_rdt_buffers x chunk_bytes stays under the fabric's address-translation
    reach (~2-3 GB/flow on the reference 8xB200 RoCE cluster; K=3 measurably
    HURT there) or the transfer drops out of the fast regime."""

    layerwise_split: int = 1
    """[RDT-SPLIT] Split each gather group's copies into this many byte-balanced
    chunk pulls (1 = whole group per pull, 3 = thirds). Chunks are cut at
    individual-tensor granularity (copies atomic; a module's copies may span
    chunks — quant defers to its last copy). Tune together with
    num_rdt_buffers for the working-set/reach budget."""

    arena_presize_gb: float = 0.0
    """[RDT-RING] Pre-size each packed receive-arena slot to this many GiB
    (0 = size to the first chunk + coarse 256MB round-up). Set it to cover the
    model's largest atomic chunk (e.g. an untied lm_head). Sizing arenas ONCE
    matters beyond perf: Ray's NIXL desc cache is keyed by data_ptr and entries
    outlive their tensors, so repeated small regrowths can false-hit a recycled
    pointer and silently skip registering the new extent (NIXL_ERR_NOT_FOUND at
    initialize_xfer, or worse a stale-MR write)."""

    pack_check: bool = False
    """[RDT-PACK-CHECK diagnostic] After every pull, checksum the received
    packed blob and append {pid, bytes, sum} to
    /tmp/rdt_profile/packcheck_cons.jsonl; the producer logs the matching sum
    when ``ShardedRDTTrainerInitInfo.pack_check`` is set. Diffing the streams
    localizes any producer/consumer packed-layout divergence (the core invariant
    of the packed contract)."""

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
    """Pull-based RDT/NIXL backend that transports only the slice each
    worker consumes.

    Requires:
      - ``distributed_executor_backend="ray"`` so workers are Ray actors.
      - The trainer actor is created with ``.options(name=...)`` and exposes
        a method decorated with ``@ray.method(tensor_transport="nixl")``
        that takes a list of ``(name, op_chain)`` specs and returns a list
        of slice tensors. The chain is replayed on the trainer's live
        parameter via ``getattr(tensor, op_name)(*args, **kwargs)``.
      - ``nixl`` is installed in the env shared by trainer and workers.
      - ``is_checkpoint_format=True`` (layerwise reload).
      - Weight loaders that only use the supported op set (narrow, view,
        reshape, transpose, t, permute, __getitem__ with int/slice/tuple,
        unsqueeze, squeeze, flatten, contiguous, chunk, copy_). Loaders
        that need .to(), .float(), .item(), .data, bool-mask indexing, or
        arithmetic on the loaded weight land in ``__torch_dispatch__`` during
        the bake and raise (not supported by this backend).

    The plan is baked once at ``init_transfer_engine`` (a meta dry run over
    ``init_info.names``) into one ``_BakedModule`` per fully-loaded leaf module,
    indexed by source name. Every ``update_weights`` *replays* the leaf modules
    its gathered names cover. See the module docstring and ``baked_rdt_replay.md``.
    """

    init_info_cls = ShardedRDTWeightTransferInitInfo
    update_info_cls = ShardedRDTWeightTransferUpdateInfo
    # receive_weights pulls synchronously but defers the GPU post-processing
    # (materialize/scatter/quant/kernel-copy) to a background thread so it
    # overlaps the next chunk's pull. See _run_chunk_pipeline / _process_item.
    # ``update_weights`` therefore skips the base's per-update device sync, and
    # ``finish_weight_update`` drains the deferred work before finalize.
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
        # M:N binding: this consumer worker binds a LIST of producers (the owners
        # it routes to; see _select_producer_indices). ``_producer_actors[i]`` is
        # the Ray actor handle and ``_produce_methods[i]`` its bound producer
        # method, in the consumer's local producer order;
        # ``_local_of_producer`` maps a global trainer rank to that local index.
        # ``_consumer_id`` is this worker's stable global index, passed on every
        # produce call so a producer serving multiple consumers keeps a per-consumer
        # serve ring (the C>P fan-in regime).
        self._producer_actors: list[Any] = []
        self._produce_methods: list[Any] = []
        self._local_of_producer: dict[int, int] = {}
        self._router: RdtRouter | None = None
        self._consumer_id: int = 0
        # Driver-supplied total consumer count (init_info.num_consumers); 0 => infer
        # from parallel_config. See _num_consumers.
        self._num_consumers_override: int = 0
        # Baked plan: source name -> the _BakedModule (leaf module) that consumes
        # it. Several names of one fused module map to the same group; replay
        # dedups. A name absent here isn't baked (attention scale / padded /
        # partial) and takes the plain load.
        self._name_to_module: dict[str, _BakedModule] = {}
        # name -> (dtype_name, shape) for every init name, so the plain-load
        # fallback can rebuild lazies from just the gathered names.
        self._name_meta: dict[str, tuple[str, list[int]]] = {}
        # Names whose copy_ fired during the bake (live). Residual (unbaked) names
        # NOT in here never move data (e.g. non-local EP experts), so
        # receive_weights skips them instead of paying the per-sync _load_unbaked.
        self._live_names: set[str] = set()

        # ---- Consumer-side pre-registered receive arena (no per-pull register) --
        # (Produced slice shape/dtype per copy now lives on each ``_Scatter`` in
        # the cached plan; no separate FetchKey side-table is needed.)
        # Persistent receive arenas, ONE PER DTYPE (1-D), DOUBLE-BUFFERED. We
        # register each arena's STORAGE once with NIXL (register_nixl_memory) and
        # carve a contiguous view per slice as the set_target_for_ref target.
        # Because the registration cache (_add_tensor_descs) is keyed by
        # untyped_storage().data_ptr() and registers the full storage, every view
        # into an arena is a cache hit -> the recv path never re-registers or
        # deregisters. Grown (and re-registered) only if a pull needs more than it
        # currently holds; steady state does zero registration. Per-dtype (not one
        # arena) so mixed-dtype groups (e.g. Kimi fp8 weights + fp32
        # weight_scale_inv + bf16 norms) still use the arena.
        #
        # Receive slots are the RDMA landing zones AND the recycling unit, and
        # today there are exactly as many as in-flight pulls. That couples them:
        # chunk j takes the slot of chunk j-K, which drain_one() completed on the
        # line before, so the guard below waits on a scatter dispatched microseconds
        # earlier and every chunk pays it (measured 7.2ms/pull at 235B).
        # MEASURED TWICE (235B, 2 stages, 96 groups): a SPARE slot
        # (one more slot than in-flight pulls) does remove that wait entirely, and both
        # times the WALL GOT WORSE — 3.39s vs 3.10s with the old per-name pack,
        # then 3.44s vs 2.70s after the pack was made cheap. It fails because a
        # deeper receive pipeline pulls demand forward past what the trainer can
        # supply under gather_lookahead=2: the serve RPC then arrives before the
        # group is published (producer wait 3.6 -> 9.3ms/call) and the extra
        # concurrency inflates the pack (8.3 -> 21.4ms/call, GIL-bound in the
        # sidecar). So the binding constraint is gather SUPPLY, not the slot
        # structure. Revisit the spare slot only after the gather can run further
        # ahead — e.g. freeing a gathered group once it is PACKED rather than once
        # its RDMA completes. A per-slot CUDA event records when the background
        # scatter finished reading a slot; the RPC thread blocks on it before
        # overwriting the slot.
        # Ring depth: the number of receive slots AND the number of pulls in
        # flight. They are one quantity, both from ``num_rdt_buffers`` -- see the
        # spare-slot experiment above for why decoupling them was reverted, and
        # ``_run_chunk_pipeline`` for the producer-ring safety argument that rests
        # on the producer using the same depth.
        self._ring_depth = 1
        self._dest_arenas: list[dict[torch.dtype, torch.Tensor]] = [{}]
        self._slot_read_done: list[Any] = []  # one torch.cuda.Event per slot
        # Generation counters guarding the events (see _ensure_proc_worker).
        self._slot_queued: list[int] = []
        self._slot_done: list[int] = []
        self._slot_cv: Any = None
        self._pull_slot = 0
        # [RDT-PULL-TARGETS] (id(chunk), slot) -> (arena data_ptr, per-key views).
        # Lives on the engine, not on the _Chunk: the plan is static and shared
        # across syncs, so a per-slot runtime cache has no business inside it.
        self._targets_cache: dict[tuple[int, int], tuple[int, list[torch.Tensor]]] = {}
        self._pack_check = False  # [RDT-PACK-CHECK diagnostic] set from init_info
        self._split = 1  # [RDT-SPLIT] chunk pulls per group; set from init_info
        self._arena_presize = 0  # [RDT-RING] bytes; set from init_info
        self._pending_frees: list[Any] = []  # free_gather refs, drained per sync
        # The STATIC plan (see _CallPlan): built once — at init from
        # init_info.group_lens, else lazily on the first update_weights — and
        # reused every sync (each _Chunk self-describes its work; nothing is
        # rebuilt per sync). Non-None means update_weights ignores its (empty or
        # redundant) per-sync names.
        self._cached_plan: _CallPlan | None = None
        # Completed sync iterations (bumped in drain_pending). The FIRST sync
        # still grows/registers arenas on both sides; a producer-side
        # registration churns its NIXL agent-metadata version, and with pulls
        # in flight the consumer's remote-agent cache can go stale for one of
        # them (createXferReq: "no backend had the required registrations").
        # So the chunk pipeline runs SERIAL during sync 0 and pipelines from
        # sync 1, when registrations are at high-water (steady state = 0 regs).
        self._completed_syncs = 0

        # ---- Background post-processing thread (pull/process pipelining) -------
        # receive_weights pulls synchronously on the RPC thread, then hands the
        # pulled arena views to this single worker thread, which runs
        # materialize/scatter/quant/kernel_copy on its own CUDA stream while the
        # next group's pull proceeds. Drained by drain_pending() (called from the
        # worker's finish_weight_update) before finalize_layerwise_reload runs.
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
        self._configure_ring(init_info)
        self._resolve_producers(init_info)

        # Hardcoded profiling: patch Ray's NIXL transport so this worker's
        # register/transfer/deregister calls accumulate into per-process counters
        # we can snapshot around each pull. Fail-soft.
        from vllm.distributed.weight_transfer._nixl_profile import install_nixl_timing

        install_nixl_timing()

        # A pure dry run: the trainer's gather cache is empty at init, so nothing
        # can (or does) get pulled -- we only record how each slice is fetched and
        # where it lands, then restore the model.
        self._bake(init_info)
        self._build_static_plan(init_info)
        # Register ALL NIXL memory now, while the fabric is idle. Concurrent
        # dma-buf GPUDirect registration that coincides with in-flight RDMA
        # intermittently fails (ibv_reg_mr 'Bad address' / NIXL_ERR_BACKEND); this
        # only bites under M:N fan-in, where the per-consumer producer serve rings
        # add registrations into the sync-0 churn window. Sizes come from the
        # static plan, so this is exact (no guessing).
        self._preregister_at_init()
        # Start the background post-processing worker (pull/process pipelining).
        self._ensure_proc_worker()

    def _configure_ring(self, init_info: ShardedRDTWeightTransferInitInfo) -> None:
        """[RDT-RING] Ring depth K and chunk split S.

        Must run before ``_ensure_proc_worker`` creates the per-slot events and
        counters, and before any arena is grown (both happen on the first pull).
        """
        self._num_consumers_override = int(init_info.num_consumers or 0)
        self._pack_check = bool(init_info.pack_check)
        k = max(1, int(init_info.num_rdt_buffers))
        self._split = max(1, int(init_info.layerwise_split))
        self._ring_depth = k
        self._dest_arenas = [{} for _ in range(k)]
        self._targets_cache = {}
        self._arena_presize = int(float(init_info.arena_presize_gb) * (1 << 30))
        logger.info(
            "[RDT-RING] active_pulls=%d slots=%d layerwise_split=%d presize=%.2fGiB",
            k,
            k,
            self._split,
            self._arena_presize / (1 << 30),
        )

    def _resolve_producers(self, init_info: ShardedRDTWeightTransferInitInfo) -> None:
        """Work out this worker's consumer identity, build the router, and bind
        the producer actors it will pull from.

        M:N routing: each worker pulls every gather group from ONE producer that
        owns it (see ``RdtRouter``). Workers pull disjoint slice sets (their
        EP-local experts / TP shard), and an owner holds every slice of its group,
        so one producer can serve a whole pull.
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

        self._consumer_id = self._resolve_consumer_id(init_info)
        group_owners = list(init_info.group_owners) or None
        if group_owners is not None and len(group_owners) != len(init_info.group_lens):
            raise RuntimeError(
                f"Sharded RDT engine: {len(group_owners)} ownership rows for "
                f"{len(init_info.group_lens)} gather groups."
            )
        self._router = RdtRouter(
            len(producer_names),
            self._num_consumers(),
            group_owners,
            len(init_info.group_lens),
        )
        producer_indices = self._select_producer_indices(len(producer_names))
        # Chunk owners and frees index producers LOCALLY (into _produce_methods).
        self._local_of_producer = {p: i for i, p in enumerate(producer_indices)}

        for producer_idx in producer_indices:
            chosen_name = producer_names[producer_idx]
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
            # Ray 2.51.1 workaround: actor handles reconstructed via
            # ray.get_actor lose the actor-level _ray_enable_tensor_transport
            # flag, so the NIXL dispatch guard at ray/actor.py rejects the
            # method call even when the trainer was created with
            # enable_tensor_transport=True. Force it back on.
            actor._ray_enable_tensor_transport = True
            self._producer_actors.append(actor)
            self._produce_methods.append(getattr(actor, init_info.produce_method_name))
        logger.info(
            "Sharded RDT engine (consumer %d) bound to %d/%d producers %r "
            "(batched method %r)",
            self._consumer_id,
            len(producer_indices),
            len(producer_names),
            [producer_names[i] for i in producer_indices],
            init_info.produce_method_name,
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
            "[RDT-PLAN] pre-built static call plan at init: %d chunks, "
            "%d residual name(s)",
            len(self._cached_plan.chunks),
            len(self._cached_plan.residual),
        )

    def _preregister_at_init(self) -> None:
        """Register every NIXL buffer this worker will use, at init, before any
        transfer runs — so nothing registers during the sync-0 RDMA churn.

        Both sides are sized from the (static) cached plan:
          * consumer receive arenas: ``ring_depth`` uint8 ring slots, each sized to
            the largest chunk's packed bytes (``pack_bytes``);
          * producer serve rings: each bound producer is asked (``reserve_serve_arena``)
            to pre-register a ring at the max bytes THIS consumer will pull from
            it (the max over the chunks routed to it — a whole chunk each, since
            a pull is served by one producer).
        A no-op when the plan has no chunks for this worker."""
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
        # bound producer (index into self._produce_methods == producer_local).
        serve_bytes = [0] * len(self._produce_methods)
        for c in plan.chunks:
            serve_bytes[c.owner] = max(serve_bytes[c.owner], c.pack_bytes)
        import ray

        refs = [
            self._producer_actors[p].reserve_serve_arena.remote(self._consumer_id, nb)
            for p, nb in enumerate(serve_bytes)
            if nb > 0
        ]
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
        """This inference worker's stable, DISTINCT global index across the whole
        inference fleet: ``data_parallel_index * world_size + rank`` where
        ``world_size`` is TP*PP and ``rank`` is the rank within the TP*PP world.

        Uses ``data_parallel_index`` (NOT ``data_parallel_rank``): vLLM resets
        ``data_parallel_rank`` to 0 in a dense (non-MoE) worker — each dense DP
        replica is an independent engine — but keeps ``data_parallel_index`` as the
        distinct global DP rank ("not overridden for dense models"). This is the
        same worker-rank formula the sibling ``nccl_engine`` uses, so it is correct
        with EP on or off, ray or mp: dense served via TP (index = tp_rank) and MoE
        served via DP+EP (index = dp rank) both yield distinct 0..C-1."""
        pc = self.parallel_config
        return pc.data_parallel_index * pc.world_size + pc.rank  # world_size = TP*PP

    def _num_consumers(self) -> int:
        """Total inference-worker count. Prefer the driver-supplied
        ``init_info.num_consumers`` (authoritative — the driver knows the full
        fleet); else infer ``data_parallel_size * tensor_parallel_size``, which is
        correct for the SUPPORTED serving modes (dense→TP keeps tensor_parallel_size;
        MoE→DP+EP keeps data_parallel_size). It is only wrong for DP-over-dense,
        which vLLM itself rejects as "not supported/useful for dense models"."""
        if self._num_consumers_override > 0:
            return self._num_consumers_override
        pc = self.parallel_config
        return pc.data_parallel_size * pc.tensor_parallel_size

    def _select_producer_indices(self, num_producers: int) -> list[int]:
        """The producers this worker pulls from, over all gather groups.

        Delegates to ``RdtRouter``: the union of the per-group owners it routes
        this consumer to. With P==C that is one producer per worker; with P>C the
        worker rotates between a block of them by group; with C>P several workers
        share one producer, which ref-counts frees. Isolated so the policy can
        change without touching the pull path.
        """
        if num_producers <= 1:
            return [0]
        assert self._router is not None
        return self._router.bound_producers(self._consumer_id)

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
        if not self._produce_methods:
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
        residual = self._cached_plan.residual
        if residual:
            # Rare/absent path (0% once unbaked-skip prunes dead names); runs
            # inline after the pipeline. It touches only non-baked layers, so it
            # does not race the background threads' baked layers.
            self._load_unbaked(residual)

    def _build_lazy_weights(
        self,
        names: list[str],
        sinks: "list[BakeSink | PullSink]",
        device: torch.device,
    ) -> list[tuple[str, torch.Tensor]]:
        """Zero-storage lazies for ``names``, dtype/shape from the init metadata.

        One sink per name: the same ``BakeSink`` for all of them during the dry
        run, a per-name ``PullSink`` on the plain-load path (names in different
        gather groups are served by different producers). Building them upfront is
        just a few object allocations.
        """
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
            for name, sink in zip(names, sinks)
        ]

    def _issue_pull(self, chunk: "_Chunk", slot: int) -> "_PendingPull":
        """Reserve ``slot``, lay the targets out in its arena, dispatch the
        produce RPC and point the transfer at the arena — WITHOUT the blocking
        ``ray.get`` (that is ``_complete_pull``). The chunked pipeline issues
        chunk i+1 before completing chunk i, so the producer serves the next
        chunk while the in-flight RDMA streams.

        Slot-reuse guard, TWO stages, both required: (1) generation wait — the
        CUDA event only binds to its LAST record(), so first wait for the
        background thread to have RECORDED the event for every item ever queued
        on this slot (else the synchronize binds to a stale record and passes
        silently — observed as nondeterministic weight corruption);
        (2) event synchronize — waits for the recorded scatters to finish on
        the GPU. Note the transfer may start any time after set_target_for_ref
        (metadata push), so the guard must precede it, not just the get.
        """
        from ray.experimental import register_nixl_memory, set_target_for_ref

        _slot_wait = _slot_sync = 0.0
        if self._slot_read_done:
            _t = time.perf_counter()
            with self._slot_cv:
                while self._slot_done[slot] < self._slot_queued[slot]:
                    self._slot_cv.wait(timeout=1.0)
            _t2 = time.perf_counter()
            self._slot_read_done[slot].synchronize()
            _slot_sync = time.perf_counter() - _t2
            _slot_wait = _t2 - _t

        # [RDT-PACK] Byte-pack every slice into ONE uint8 arena (16B-aligned,
        # keys order); the M:N split hands each bound producer a contiguous run
        # (one NIXL descriptor each, all into disjoint sub-ranges of this arena —
        # a single producer => one descriptor over the whole arena). The packed
        # layout was precomputed at plan time (``chunk.pack_layout``, byte-exact
        # mirror of the producer's rule); ``targets`` carves the dtype views back
        # out for the scatter.
        keys = chunk.keys
        cur = chunk.pack_bytes
        arenas = self._dest_arenas[slot]
        arena = arenas.get(torch.uint8)
        if arena is None or arena.numel() < cur:
            # Size ONCE with headroom (presize + coarse round-up): repeated
            # small regrowths alloc/free near-identical blocks, and Ray's
            # desc cache (keyed by data_ptr, entries outlive their tensor)
            # can false-hit a recycled pointer and skip registering the new
            # extent -> NIXL_ERR_NOT_FOUND (see arena_presize_gb docstring).
            alloc = arena_alloc_bytes(cur, self._arena_presize)
            arena = torch.empty(alloc, dtype=torch.uint8, device=self.device)
            register_nixl_memory(arena)
            arenas[torch.uint8] = arena
        targets = self._pull_targets(chunk, slot, arena)
        # Stamp pull-start so the NIXL patch can cleave this pull into
        # produce_wait vs recv_wall. With chunks in flight the cleave is
        # approximate (issues overlap completions) but the transfer sums stand.
        from vllm.distributed.weight_transfer import _nixl_profile

        _nixl_profile.mark_pull_start()
        # The owner of this chunk's gather group holds every slice of it, so ONE
        # produce RPC serves the whole packed chunk into this slot's arena. The
        # arena view stays strongly referenced through the get --
        # set_target_for_ref stores WEAKREFS, and a dropped target reroutes the
        # transfer into a fallback buffer. ``consumer_id`` lets a producer serving
        # several consumers keep a per-consumer serve ring (the C>P fan-in case).
        blob = arena[:cur]
        ref = self._produce_methods[chunk.owner].remote(
            keys, consumer_id=self._consumer_id
        )
        set_target_for_ref(ref, [blob])
        return _PendingPull(
            ref=ref,
            keys=keys,
            targets=targets,
            blob=blob,
            slot=slot,
            slot_wait_seconds=_slot_wait,
            slot_sync_seconds=_slot_sync,
        )

    def _pull_targets(
        self, chunk: "_Chunk", slot: int, arena: torch.Tensor
    ) -> "list[torch.Tensor]":
        """[RDT-PULL-TARGETS] Per-key dtype views into ``slot``'s arena.

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

        from vllm.distributed.weight_transfer import _nixl_profile

        # [RDT-META-WAIT diagnostic] cleave produce_wait into in-get metadata
        # wait vs issue-side work (see _nixl_profile.mark_get_entry).
        _nixl_profile.mark_get_entry()
        ray.get(pending.ref)
        if self._pack_check:
            # [RDT-PACK-CHECK] checksum each received sub-blob; the producer logs
            # one matching sum PER produce call (i.e. per sub-blob), so we log one
            # record per sub-pull too and the two streams line up under the M:N
            # split (one sub-blob == the pre-M:N whole-arena case). Chunked sums:
            # .sum(dtype=int64) upcasts its INPUT to int64, so a whole-blob sum
            # materializes 8x the blob (OOM on the consumer).
            import json as _json
            import os as _os

            _os.makedirs("/tmp/rdt_profile", exist_ok=True)
            _w = 32 << 20
            with open("/tmp/rdt_profile/packcheck_cons.jsonl", "a") as f:
                for sub in pending.blob:
                    n = sub.numel()
                    s = 0
                    for _i in range(0, n, _w):
                        s += int(sub[_i : min(_i + _w, n)].sum(dtype=torch.int64))
                    f.write(
                        _json.dumps({"pid": _os.getpid(), "bytes": n, "sum": s}) + "\n"
                    )
        return dict(zip(pending.keys, pending.targets))

    # ---------------- Bake (dry run, at init) / replay ----------------

    def _load_unbaked(
        self,
        names: list[str],
    ) -> None:
        """Plain load for a call whose names aren't all baked: rebuild lazies
        for ``names`` (dtype/shape from the init metadata) and run vLLM's stock
        inline layerwise reload — the worker's ``initialize_layerwise_reload`` is
        active for the sync, so each layer is processed as it completes and the
        lazy's Pass-2 ``copy_`` pulls its slice on demand. No recording, no
        batching; runs every sync for the call (the rare, unbaked case)."""
        device = torch.empty(0).device
        _t = time.perf_counter()
        self.model.load_weights(
            # A producer is bound on this path: ``receive_weights`` raises before
            # calling ``_load_unbaked`` if none is. Each name pulls from the
            # producer that owns its gather group, so the load order (and the
            # layerwise-reload completion order) is unaffected by routing.
            self._build_lazy_weights(
                names, [self._pull_sink_for(n) for n in names], device
            )
        )
        self._log_timing("unbaked", time.perf_counter() - _t, 0.0, 0, 0.0)

    def _pull_sink_for(self, name: str) -> "PullSink":
        """A one-slice-at-a-time pull sink for ``name``, bound to the producer
        that owns its gather group AND to this worker's ``consumer_id``.

        The consumer id is what keeps the producer's per-consumer serve rings
        apart. Without it every worker's residual pull is served out of consumer
        0's ring, and concurrent pulls overwrite each other's packed blob.
        """
        assert self._cached_plan is not None  # residuals come from the plan
        group_idx = self._cached_plan.name_group_idx.get(name, 0)
        method = self._produce_methods[self._local_owner_of(group_idx)]
        consumer_id = self._consumer_id

        def _pull(keys: "list[FetchKey]") -> torch.Tensor:
            import ray

            # The producer always answers with ONE byte-packed blob, so a
            # single-key request is that slice at offset 0.
            return ray.get(method.remote(keys, consumer_id=consumer_id))[0]

        return PullSink(_pull)

    def _bake(self, init_info: ShardedRDTWeightTransferInitInfo) -> None:
        """Bake the replay plan once, as a self-driven meta dry run.

        We put the model's params on meta (via ``initialize_layerwise_reload``)
        and then drive ``model.load_weights`` over all of ``init_info.names``
        **through the model's original loaders** — `_install_recording_stamps`
        wraps the *original* loader, bypassing ``online_process_loader`` entirely
        — so ``_layerwise_process`` is never in the path. Nothing materializes,
        pulls, or kernel-copies; the lazy's ``copy_`` just records, per leaf
        module, the source op chain + the meta destination's ``param_name`` and
        ``offset/shape/stride``. Afterwards we build one ``_BakedModule`` per
        **fully-loaded** leaf module (copied numel == the module's loadable
        size) and index it by source name; partial / attention / unrecordable
        modules are left out and take the plain load. The model is restored.

        FUTURE / cleanliness: this still reaches into layerwise internals that a
        richer, public layerwise API should expose first-class — and which a
        downstream RL framework porting this engine (and unable to patch vLLM)
        must replicate. **vLLM's layerwise reload should grow proper support for
        these trace-only flows**, e.g.:
          1. A "currently-loading (module, param_name)" hook so the lazy can
             attribute each ``copy_`` without us monkeypatching loaders
             (``_install_recording_stamps``).
          2. A trace/dry-run mode that drives the loaders against meta without
             materializing or processing — so we don't have to bypass
             ``online_process_loader`` by hand to keep ``_layerwise_process``
             from firing.
          3. A public ``abort_layerwise_reload`` to restore without
             materializing (we hand-roll ``_place_kernel_tensors`` + ``reset``
             in ``_restore_after_dry_run`` because ``finalize_layerwise_reload``
             would materialize real params, defeating the meta/memory win).
        Until then we lean on existing symbols (``initialize_layerwise_reload``,
        ``_get_original_loader``, ``get_layer_size``, ``_place_kernel_tensors``).
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
            model.load_weights(
                self._build_lazy_weights(names, [recorder] * len(names), self.device)
            )
            # Build the plan from what was recorded, keeping only modules that
            # fully loaded — a partial module would leave unwritten regions that
            # the standard finalize path inits, so baking it would scatter
            # garbage. "Fully loaded" = copied numel >= the module's loadable
            # size (the same test online_process_loader uses). The dict is keyed
            # by module, so a FusedMoE's entry already holds every expert's copy.
            for module, recorded in recorder.copies_by_layer.items():
                if not recorded or any(c is None for c in recorded):
                    continue  # unrecordable copy_ -> slow path
                # Guard above guarantees every entry is a real _BakedCopy.
                copies = cast("list[_BakedCopy]", recorded)
                copied = sum(prod(c.shape) for c in copies)
                if copied < get_layer_size(module):
                    continue  # partial -> slow path
                group = _BakedModule(layer=module, copies=copies)
                for c in copies:
                    self._name_to_module[c.src[0]] = group
            self._restore_after_dry_run(model)

        # Names whose copy_ fired during the bake (baked + unbaked-but-live).
        # Residual names not in here no-op for this worker and are skipped.
        self._live_names = set(recorder.copied_names)

        n_groups = len({id(g) for g in self._name_to_module.values()})
        logger.info(
            "Sharded RDT dry-run baked %d/%d names into %d leaf modules "
            "(%d live) in %.3fs",
            len(self._name_to_module),
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

        def _make_stamp(layer, name, inner):
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
            return stamp

        for module in model.modules():
            for name, tensor in get_layer_tensors(module).items():
                if getattr(tensor, "weight_loader", None) is None:
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
                while loader is not None and hasattr(loader, "_rdt_stamp_inner"):
                    loader = loader._rdt_stamp_inner
                    tensor.weight_loader = loader
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
        # Generation handshake for the events above. A torch.cuda.Event only
        # waits on its LAST record(); if the RPC thread reaches synchronize()
        # for slot s before the background thread has RECORDED item N's event
        # (it is still in Python before the scatter), the wait silently binds to
        # item N-1's record and the next RDMA overwrites the slot under the
        # pending scatter — subtle nondeterministic weight corruption, seen live
        # with a single slot. So: the RPC thread counts items queued per slot, the
        # background thread counts records; a pull may synchronize() only after
        # done[slot] has caught up with queued[slot].
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
        # [RDT-SINGLE-CALL] Ensure every fired free_gather has EXECUTED on the
        # producer before the sync ends: ``begin_sync`` resets the producer's
        # per-group free counts, so a free landing after the next sync started
        # would credit a group it does not belong to, over-crediting the
        # gather-lookahead backpressure (extra ~17GiB gather resident ->
        # trainer OOM risk).
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

    def _scatter_of(self, layer: Any, c: "_BakedCopy") -> "_Scatter":
        """Build a self-contained ``_Scatter`` from a bake-time ``_BakedCopy``,
        folding in the produced slice's dtype/nbytes (dtype from the source
        name's metadata; produced shape == the destination region ``c.shape``)."""
        dtype = _dtype_from_name(self._name_meta[c.src[0]][0])
        return _Scatter(
            layer=layer,
            param_name=c.param_name,
            src=c.src,
            offset=c.offset,
            shape=tuple(c.shape),
            stride=tuple(c.stride),
            dtype=dtype,
            nbytes=prod(c.shape) * dtype.itemsize,
        )

    def _chunk_module_scatters(
        self, modules: "list[_BakedModule]"
    ) -> "list[list[_Scatter]]":
        """Cut the modules' copies (in order, order-stable) into at most
        ``layerwise_split`` byte-balanced chunks of flat ``_Scatter``s. Copies
        are atomic — a single huge copy (e.g. lm_head) becomes its own oversized
        chunk — and a module's copies may span chunks (materialize/quant fire on
        its first/last chunk; see ``_build_call_plan``)."""
        s = self._split
        entries = [(m.layer, c) for m in modules for c in m.copies]
        if s <= 1 or len(entries) <= 1:
            return [[self._scatter_of(layer, c) for layer, c in entries]]
        # numel is a fine byte proxy for balance; greedy contiguous cut.
        starts = greedy_run_starts([prod(c.shape) for _layer, c in entries], s)
        scatters = [self._scatter_of(layer, c) for layer, c in entries]
        return [
            scatters[st : (starts[r + 1] if r + 1 < len(starts) else len(scatters))]
            for r, st in enumerate(starts)
        ]

    def _build_call_plan(self, names: list[str], group_lens: list[int]) -> "_CallPlan":
        """[RDT-SINGLE-CALL] Build the STATIC plan for one whole-sync call.

        PURE — no pulls, no engine state touched — so the result is cached and
        reused every sync (see ``_CallPlan``). Three passes:
          1. Split ``names`` into the driver's gather groups; chunk-plan EACH
             group into flat ``_Scatter`` chunks (layerwise_split chunks/group,
             same working-set/reach math); record each group's last chunk index
             for ``free_gather`` (or ``pre_free`` for groups this worker doesn't
             pull). The concatenated stream has no per-group call boundaries, so
             group L+1's first chunk issues while group L's chunks still stream.
          2. For each leaf module, find its FIRST and LAST chunk (materialize on
             the first, quant/kernel/reset on the last — replaces the runtime
             remaining-copy counters, correct materialize-once by construction).
          3. Assemble ``_Chunk``s: dedup keys + precompute the packed byte layout
             (16B-aligned, keys order) so the pull path does no per-call work.
        """
        # --- pass 1: gather groups -> flat scatter chunks + free timing --------
        # Frees and pulls both carry the group INDEX: it selects the owning
        # producer to talk to (see _fire_free_gather / pass 3).
        raw_chunks: list[list[_Scatter]] = []
        chunk_group: list[int] = []  # chunk idx -> its gather group index
        free_at: dict[int, list[tuple[int, list[str]]]] = {}
        pre_free: list[tuple[int, list[str]]] = []  # groups with no chunk here
        residual: list[str] = []
        name_group_idx: dict[str, int] = {}
        pos = 0
        for gi, glen in enumerate(group_lens):
            gnames = names[pos : pos + glen]
            pos += glen
            for n in gnames:
                name_group_idx[n] = gi
            modules: list[_BakedModule] = []
            seen: set[int] = set()
            for n in gnames:
                mod = self._name_to_module.get(n)
                if mod is None:
                    if n in self._live_names:
                        residual.append(n)
                elif id(mod) not in seen:
                    seen.add(id(mod))
                    modules.append(mod)
            if not modules:
                # Nothing to pull for this group on this worker; still free its
                # gather — after the current last chunk, or before the pipeline.
                if raw_chunks:
                    free_at.setdefault(len(raw_chunks) - 1, []).append((gi, gnames))
                else:
                    pre_free.append((gi, gnames))
                continue
            group_chunks = self._chunk_module_scatters(modules)
            raw_chunks.extend(group_chunks)
            chunk_group.extend([gi] * len(group_chunks))
            free_at.setdefault(len(raw_chunks) - 1, []).append((gi, gnames))

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
                    owner=self._local_owner_of(chunk_group[ci]),
                )
            )
        return _CallPlan(
            chunks=chunks,
            pre_free=pre_free,
            residual=residual,
            name_group_idx=name_group_idx,
        )

    def _local_owner_of(self, group_idx: int) -> int:
        """Local producer index this worker pulls ``group_idx`` from."""
        assert self._router is not None
        if len(self._produce_methods) <= 1:
            return 0
        return self._local_of_producer[
            self._router.producer_for(self._consumer_id, group_idx)
        ]

    def _fire_free_gather(self, group_idx: int, gnames: list[str]) -> None:
        """Fire-and-forget free of one gather group on the producer that serves it.

        Exactly the producer this consumer pulls the group from, whether or not
        this worker had anything to pull: that producer ref-counts the frees of
        every consumer routed to it for the group, and releases the group's
        backpressure slot on the last one. Firing at other producers instead
        would credit a group they never served. Refs are held and drained in
        drain_pending so every free has executed before the sync ends."""
        local = self._local_owner_of(group_idx)
        self._pending_frees.append(
            self._producer_actors[local].free_gather.remote(list(gnames))
        )

    def _run_chunk_pipeline(self, plan: "_CallPlan") -> None:
        """[RDT-RING] Pipelined chunk pulls over the ring of receive slots.

        Issues up to ``ring_depth`` produce RPCs ahead of the blocking gets, so
        while chunk i's RDMA streams (inside its ray.get): the producer serves
        chunk i+1 into ITS ring slot, and the background thread scatters chunk
        i-1 out of another receive slot. Reads themselves stay serialized (they
        share the flow's NIC — that is the bandwidth floor, not a loss).

        Producer-ring safety needs no coordination, even though a producer now
        serves only the chunks of the groups it owns — a SUBSEQUENCE of this
        stream. Its ring (depth K == this ring depth) rotates once per call from
        this consumer, so its call n+K belongs to a chunk at least K later than
        call n's, and drain-before-issue below means that chunk is issued only
        after call n's get returned. So read #n completes before its slot is
        reused. This rests on the producer ring being no shallower than this
        one; both come from ``num_rdt_buffers``. The CONSUMER's
        receive slots are the same depth (one slot per in-flight pull), which is
        why the slot a chunk lands in is always the one drain_one() just read —
        see __init__ for why the spare slot that would decouple them is not
        worth its cost yet.
        Consumer-slot safety is _issue_pull's generation handshake; the drain of
        chunk i-K (which queues its scatter and bumps queued[slot]) strictly
        precedes the issue of chunk i on the same thread.
        """
        from collections import deque

        from vllm.distributed.weight_transfer import _nixl_profile

        inflight: deque[tuple[_PendingPull, _Chunk]] = deque()

        # Gather groups with no chunk on this worker (plan.pre_free): free before
        # the pipeline — the trainer gathered them under the lockstep plan but
        # there is no chunk of theirs to hang the free on (fire-and-forget).
        for gi, gnames in plan.pre_free:
            self._fire_free_gather(gi, gnames)

        def drain_one() -> None:
            pending, chunk = inflight.popleft()
            _t_recv = time.perf_counter()
            _before = _nixl_profile.snapshot()
            _t0 = time.perf_counter()
            results = self._complete_pull(pending)
            pull_seconds = time.perf_counter() - _t0
            delta = _nixl_profile.delta(_before, _nixl_profile.snapshot())
            # [RDT-SLOT-WAIT diagnostic] surface the issue-side slot waits in the
            # jsonl (per-pid sums via the driver's aggregation).
            delta["slot_wait_seconds"] = pending.slot_wait_seconds
            delta["slot_sync_seconds"] = pending.slot_sync_seconds
            self._dispatch_item(
                _ProcItem(
                    chunk=chunk,
                    results=results,
                    slot=pending.slot,
                    t_recv=_t_recv,
                    pull_seconds=pull_seconds,
                    nixl_delta=delta,
                    pull_bytes=sum(sc.nbytes for sc in chunk.scatters),
                )
            )
            # Each gather group whose LAST chunk this is: its read is done, so
            # every serve of that group is done -> the producer can drop the
            # gather buffers (fire-and-forget, off the critical path).
            for gi, gnames in chunk.free:
                self._fire_free_gather(gi, gnames)

        for chunk in plan.chunks:
            if not chunk.keys:
                # _chunk_module_scatters never emits empty chunks; keep the frees
                # safe anyway if this ever changes.
                for gi, gnames in chunk.free:
                    self._fire_free_gather(gi, gnames)
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
        """Scatter-thread half: materialize this chunk's first-seen modules +
        scatter its slices on the process stream, publish the slot, then hand
        the modules this chunk COMPLETES to the quant thread (see _run_quant).

        Mirrors ``_layerwise_process`` minus the loader replay; the quant /
        kernel-copy / ``info.reset()`` tail runs on the quant thread, ordered
        after this chunk's scatters via a recorded event.

        After all scatters that read ``item.slot``'s arena are enqueued on the
        process stream, record the slot's read-done event so the RPC thread can
        block on it before overwriting the slot with a later pull.
        """
        from vllm.distributed.weight_transfer._nixl_profile import PhaseTimer
        from vllm.model_executor.model_loader.reload.layerwise import (
            LAYERWISE_INFO,
        )
        from vllm.model_executor.model_loader.reload.meta import materialize_layer

        results = item.results
        chunk = item.chunk
        ph = PhaseTimer(self._proc_stream)  # stream-scoped syncs, not global
        _t_proc = time.perf_counter()
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
                if chunk.materialize:
                    with ph.phase("materialize_seconds"):
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
                if chunk.scatters:
                    with ph.phase("scatter_seconds"):
                        for sc in chunk.scatters:
                            param = getattr(sc.layer, sc.param_name)
                            dst = param.as_strided(sc.shape, sc.stride, sc.offset)
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
            # chunk's scatters) so it never delays this thread's next pass-1 — an
            # in-order pass 2 here was measured to stall the RPC thread's slot
            # handshake ~0.5-0.75s/iter (every group's quant postponed the next
            # item's publication).
            if chunk.quant:
                ready = torch.cuda.Event()
                ready.record(self._proc_stream)
                if self._quant_queue is None:
                    self._run_quant(chunk.quant, ready)
                else:
                    self._quant_queue.put((chunk.quant, ready))
        process_seconds = time.perf_counter() - _t_proc
        self._log_timing(
            "replay",
            time.perf_counter() - item.t_recv,
            item.pull_seconds,
            1,
            process_seconds,
            item.nixl_delta,
            dict(ph.t),
            item.pull_bytes,
        )

    def _run_quant(self, layers: "list[Any]", ready: "torch.cuda.Event") -> None:
        """Quant/kernel-copy/reset the given COMPLETED leaf modules, exactly as
        _layerwise_process. Runs on the quant thread's own stream, ordered after
        the modules' scatters via ``ready``; touches only the scattered params
        (never a receive slot), so it can overlap subsequent chunks' RDMA and
        scatters. ``info.reset()`` is what makes finalize skip the layer —
        drain_pending joins the quant queue before finalize runs."""
        from vllm.distributed.weight_transfer._nixl_profile import PhaseTimer
        from vllm.model_executor.layers.quantization.base_config import (
            QuantizeMethodBase,
        )
        from vllm.model_executor.model_loader.reload.layerwise import (
            LAYERWISE_INFO,
            _copy_and_restore_kernel_tensors,
        )

        stream = self._quant_stream or self._proc_stream
        assert stream is not None  # created in _ensure_proc_worker before use
        ph = PhaseTimer(stream)
        _t0 = time.perf_counter()
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
                    with ph.phase("quant_seconds"):
                        quant_method.process_weights_after_loading(layer)
                # Copy into persistent kernel storage (preserves cudagraph refs).
                if info.kernel_tensors is not None:
                    with ph.phase("kernel_copy_seconds"):
                        _copy_and_restore_kernel_tensors(layer, info)
                # Reset so finalize_layerwise_reload skips this (loaded) layer.
                info.reset()
        t = time.perf_counter() - _t0
        # Separate jsonl record; the driver aggregation SUMS numeric fields per
        # pid, so per-worker process/quant totals stay correct.
        self._log_timing("replay", t, 0.0, 0, t, None, dict(ph.t), 0)

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

    # Hardcoded profiling sink: vLLM workers run in an EngineCore subprocess
    # whose logs are not streamed to the driver, so each receive_weights call
    # appends one JSON line here. The driver truncates it at sync-loop start and
    # aggregates it at the end. Single-node benchmark scaffolding only.
    _CONSUMER_TIMING_FILE = "/tmp/rdt_profile/consumer.jsonl"

    @staticmethod
    def _log_timing(
        mode: str,
        total_seconds: float,
        pull_seconds: float,
        pull_calls: int,
        process_seconds: float,
        nixl_delta: dict | None = None,
        phase_split: dict | None = None,
        pull_bytes: int = 0,
    ) -> None:
        """Log a one-line timing summary for one ``receive_weights`` call.

        ``mode`` is ``replay`` or ``unbaked``. ``pull_seconds`` is the full
        ``ray.get`` round trip; ``nixl_delta`` (when present) splits that into the
        consumer-side registration / transfer / deregistration AND the
        produce_wait / recv_wall cleave measured by the NIXL patch.
        ``process_seconds`` is the scatter/materialize/quantize/kernel-copy work
        after the pull; ``phase_split`` (when present) breaks it into its
        per-phase ``*_seconds`` (from the engine's PhaseTimer). ``pull_bytes`` is
        the bytes THIS worker pulled this call, logged as ``bytes`` so the driver
        can compute true per-worker bandwidth (bytes/transfer) and distinguish an
        EP-imbalance straggler (more bytes) from a transport straggler (equal
        bytes, lower GB/s).
        """
        nixl_delta = nixl_delta or {}
        logger.info(
            "[RDT-TIMING] receive_weights mode=%s total=%.4fs nixl_pull=%.4fs "
            "(%d pull%s) process=%.4fs | nixl_transfer=%.4fs register=%.4fs "
            "(%d) dereg=%.4fs",
            mode,
            total_seconds,
            pull_seconds,
            pull_calls,
            "" if pull_calls == 1 else "s",
            process_seconds,
            nixl_delta.get("transfer_seconds", 0.0),
            nixl_delta.get("register_seconds", 0.0),
            nixl_delta.get("register_calls", 0),
            nixl_delta.get("deregister_seconds", 0.0),
        )

        import json
        import os

        os.makedirs(
            os.path.dirname(ShardedRDTWeightTransferEngine._CONSUMER_TIMING_FILE),
            exist_ok=True,
        )
        record = {
            "pid": os.getpid(),
            "mode": mode,
            "total": total_seconds,
            "pull": pull_seconds,
            "process": process_seconds,
            "bytes": pull_bytes,
            **nixl_delta,
            **(phase_split or {}),
        }
        with open(ShardedRDTWeightTransferEngine._CONSUMER_TIMING_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")

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
        self._producer_actors = []
        self._produce_methods = []
        # Drop strong references to baked modules so the model can be freed.
        self._name_to_module.clear()
        self._name_meta.clear()
        self._live_names.clear()
        # Drop the cached plan (holds _Scatter refs to the baked layers).
        self._cached_plan = None
        # Release the receive arenas (their NIXL registration is pinned for the
        # process lifetime; freeing the tensors just drops our strong refs).
        self._dest_arenas = [{} for _ in range(self._ring_depth)]

    @staticmethod
    def trainer_send_weights(
        iterator: Iterator[tuple[str, torch.Tensor]],
        trainer_args: dict[str, Any] | Any,
    ) -> None:
        """No-op for the pull-based sharded RDT backend.

        Workers initiate the transfer themselves via the trainer's
        ``@ray.method(tensor_transport="nixl")`` batched accessor.
        Retained to satisfy the abstract base class.
        """
        del iterator, trainer_args


def _dtype_from_name(name: str) -> torch.dtype:
    """Resolve a string like 'bfloat16' to torch.bfloat16."""
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unknown torch dtype name: {name!r}")
    return dtype
