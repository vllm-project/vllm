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
from collections import defaultdict
from collections.abc import Callable, Iterator
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
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

# A single recorded op: ("op_name", positional_args_tuple, sorted_kwargs_items).
# All entries must be hashable so the chain itself is hashable for use as
# a FetchKey. Slices, ints, tuples, Ellipsis, None, and memory_format enums
# are all hashable on Python 3.12+.
OpSpec = tuple[str, tuple[Any, ...], tuple[tuple[str, Any], ...]]
OpChain = tuple[OpSpec, ...]
FetchKey = tuple[str, OpChain]

# Allowlist: torch.Tensor methods that map to pure-view / shape-only / byte-
# bounding operations. Every entry maps `torch.Tensor.fn` to the string name
# the trainer uses to reach the method via getattr. Anything that escapes
# this set (arithmetic, .to/.float/.cpu, .item, .data, bool-mask indexing,
# .copy_ as source) lands in __torch_dispatch__ and raises with a clear
# message — we'd rather fail loud than silently transfer the wrong bytes.
_SUPPORTED_OPS: dict[Callable, str] = {
    torch.Tensor.narrow: "narrow",
    torch.Tensor.view: "view",
    torch.Tensor.reshape: "reshape",
    torch.Tensor.__getitem__: "__getitem__",
    torch.Tensor.unsqueeze: "unsqueeze",
    torch.Tensor.squeeze: "squeeze",
    torch.Tensor.transpose: "transpose",
    torch.Tensor.t: "t",
    torch.Tensor.permute: "permute",
    torch.Tensor.flatten: "flatten",
    torch.Tensor.contiguous: "contiguous",
    torch.Tensor.chunk: "chunk",
}


def _freeze_kwargs(kwargs: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Sort kwargs into a tuple of items for hashable storage in OpSpec."""
    return tuple(sorted(kwargs.items()))


@dataclass
class _BakedCopy:
    """One recorded scatter: pull ``src`` from the trainer and copy it into
    ``param_name`` at the recorded strided region.

    Captured once during the bake's dry run — the lazy source carries the op
    chain (``src``); the loader binds the destination param (``param_name``),
    whose **meta** view yields ``offset/shape/stride`` (valid on meta; no real
    storage needed). On every later sync the destination is reconstructed as
    ``param.as_strided(shape, stride, offset)`` and filled by ``copy_`` — no
    loader, no lazy tensor, no discovery.
    """

    src: FetchKey
    param_name: str
    offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]


@dataclass
class _BakedGroup:
    """A baked leaf module and the destination scatters that fill its params.

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
    scatter). ``free`` = gather-group name lists whose last chunk this is (fire
    ``free_gather`` after the pull returns).
    """

    scatters: "list[_Scatter]"
    keys: "list[FetchKey]"
    pack_layout: "list[tuple[int, torch.dtype, int, tuple[int, ...]]]"  # off,dt,n,shape
    pack_bytes: int
    materialize: "list[Any]"
    quant: "list[Any]"
    free: "list[list[str]]"


@dataclass
class _CallPlan:
    """The STATIC plan for one sync — a pure function of the baked plan
    (``_name_to_group`` / ``_live_names`` / ``_name_meta``) and the driver's
    group partition, both fixed for the engine's lifetime. Built ONCE (at init
    when the driver passes ``group_lens`` on the init info, else lazily on the
    first ``update_weights``) and reused every sync: each self-describing
    ``_Chunk`` carries its own scatter/pack/materialize/quant/free actions, so
    runtime is pure execution — no per-sync counters or side-tables.

    ``pre_free`` = gather groups with NO chunk on this worker but which the
    trainer still gathered under the lockstep plan (freed before the pipeline).
    ``residual`` = live-but-unbaked names for the plain-load fallback.
    """

    chunks: "list[_Chunk]"
    pre_free: "list[list[str]]"
    residual: "list[str]"


@dataclass
class _PendingPull:
    """An issued-but-not-completed pull: the produce RPC is dispatched and the
    transfer is pointed at ring-slot arena views, but the blocking ``ray.get``
    has not run. ``targets``/``blob`` hold the arena views strongly referenced
    until completion — ``set_target_for_ref`` stores weakrefs, so dropping them
    would silently reroute the transfer into a fallback buffer."""

    ref: Any
    keys: "list[FetchKey]"
    targets: "list[torch.Tensor]"
    blob: "list[torch.Tensor]"
    slot: int
    pack_arena: "torch.Tensor | None"  # packed uint8 arena (for pack_check)
    pack_span: int  # packed bytes this pull (for pack_check)
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
class _BakeRecorder:
    """Shared recording context for the dry-run bake.

    During the single dry-run ``load_weights`` pass, the engine stamps
    ``current = (leaf_module, param_name)`` around each param's loader (see
    ``_install_recording_stamps``). The lazy's ``copy_`` then reads ``current``
    to attribute the copy to its destination param, appending a ``_BakedCopy``
    (op chain from the source; ``offset/shape/stride`` from the meta dest view)
    into ``copies_by_layer[leaf_module]``. ``None`` marks a copy_ we couldn't
    attribute (its group then falls back to a plain load). The dict is keyed by
    the module object, so iterating it after the pass yields each leaf module
    once. No real storage, no transfer — everything is meta.
    """

    copies_by_layer: "dict[Any, list[_BakedCopy | None]]" = field(
        default_factory=lambda: defaultdict(list)
    )
    current: "tuple[Any, str] | None" = None
    # Source names whose ``copy_`` actually fired during the bake. Names NOT here
    # never moved data (e.g. experts owned by another EP rank, whose loader
    # no-ops) -> ``receive_weights`` can skip them entirely instead of paying the
    # per-name ``_load_unbaked`` lazy-build + load_weights cost every sync.
    copied_names: "set[str]" = field(default_factory=set)


class _UnsupportedLazyOp(NotImplementedError):
    """Raised when a weight loader calls an op we don't support on a LazyRDTTensor.

    Surfaced as NotImplementedError so callers can distinguish "this backend
    can't handle this loader" from genuine bugs.
    """


class LazyRDTTensor(torch.Tensor):
    """Zero-storage tensor that records how to fetch a weight slice.

    Built via ``_make_wrapper_subclass`` so ``.shape``/``.dtype``/``.device``/
    ``.size()``/``.dim()`` work without allocating storage. Every supported op
    (narrow/view/reshape/transpose/__getitem__/...) returns a new
    ``LazyRDTTensor`` with the spec appended to its chain; ``copy_`` is the data
    sink. Its behaviour depends on the ``_ctx`` the engine installed:

    - ``_BakeRecorder`` (dry-run bake): record a ``_BakedCopy`` (the op chain
      plus the bound ``param_name`` and the meta destination's
      offset/shape/stride) and fire a meta ``copy_``. No data moves.
    - the trainer's producer method (slow path): a meta destination is a no-op
      meta ``copy_``; a real destination pulls this one slice via a single
      ``produce_method`` RPC and copies it in.

    Any op outside the allowlist (arithmetic, .item, .to, .float, .data,
    bool-mask indexing, etc.) raises ``_UnsupportedLazyOp`` in
    ``__torch_dispatch__`` so failures are loud rather than silently fetching
    the wrong bytes.
    """

    # Declared at class scope so mypy can infer attribute types on
    # instances built via ``_make_wrapper_subclass`` (where ``__new__``
    # returns a tensor it can't annotate as ``self``).
    _name: str
    _ops: OpChain
    # The collaborator that handles this lazy's copy_: a ``_BakeRecorder`` (bake
    # — record the scatter, no data) or the trainer's producer method (slow path
    # — pull this slice on demand). ``None`` only on bare construction.
    _ctx: "_BakeRecorder | Callable | None"
    _materialized: "torch.Tensor | None"

    @staticmethod
    def __new__(
        cls,
        name: str,
        shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        ops: OpChain = (),
        ctx: "_BakeRecorder | Callable | None" = None,
    ) -> "LazyRDTTensor":
        t = torch.Tensor._make_wrapper_subclass(
            cls,
            shape,
            dtype=dtype,
            device=device,
            requires_grad=False,
        )
        t._name = name
        t._ops = tuple(ops)
        t._ctx = ctx
        t._materialized = None
        return t

    def _key(self) -> FetchKey:
        return (self._name, self._ops)

    def _make_child(
        self,
        new_shape: torch.Size,
        new_dtype: torch.dtype,
        *new_ops: OpSpec,
    ) -> "LazyRDTTensor":
        """Append one or more ops to the chain and return a fresh child.

        Variadic so multi-return ops (e.g. chunk) can append both the base
        op and an indexing op in a single call.
        """
        return LazyRDTTensor(
            name=self._name,
            shape=new_shape,
            dtype=new_dtype,
            device=self.device,
            ops=self._ops + new_ops,
            ctx=self._ctx,
        )

    def _meta(self) -> torch.Tensor:
        """A zero-storage meta tensor of this lazy's current shape/dtype.

        Used to compute the post-op shape/dtype via PyTorch itself, which is
        more reliable than reimplementing shape inference per op. The result
        is never used for data — only its metadata.
        """
        return torch.empty(self.shape, dtype=self.dtype, device="meta")

    def _materialize(self) -> torch.Tensor:
        # Only the slow (unbaked) path materializes; the bake records and never
        # pulls. So ``_ctx`` here is the trainer's producer method — pull this one
        # slice on demand (no batching) via a single-tensor RPC.
        if self._materialized is not None:
            return self._materialized
        assert self._ctx is not None and not isinstance(self._ctx, _BakeRecorder)
        import ray

        # ``_ctx`` here is the Ray actor producer method; ``.remote`` is
        # injected by Ray and invisible to the type checker.
        tensor = ray.get(self._ctx.remote([self._key()]))[0]  # type: ignore[attr-defined]
        self._materialized = tensor
        return tensor

    @classmethod
    def __torch_function__(
        cls,
        func,
        types,
        args=(),
        kwargs=None,
    ):
        kwargs = kwargs or {}

        # copy_: this is the data sink. Handled before the generic allowlist
        # because (a) we don't want to record copy_ in the chain, and (b)
        # the meta-vs-device branching has special semantics.
        if func is torch.Tensor.copy_:
            dest = args[0]
            src = args[1] if len(args) > 1 else kwargs.get("src")
            if isinstance(src, cls):
                ctx = src._ctx
                if isinstance(ctx, _BakeRecorder):
                    # Dry-run bake: dest is a meta view of the param the loader
                    # is bound to. The engine's loader stamp set ctx.current =
                    # (leaf_module, param_name) for this call, so we attribute the
                    # copy to that param: record how to fetch the source slice
                    # (the op chain) and where it lands (the meta view's
                    # offset/shape/stride — valid on meta), then fire a meta
                    # copy_ so the layer's numel still counts. No pull, no real
                    # storage. A copy_ with no stamp (ctx.current is None) can't
                    # be attributed — left unrecorded, so its group fails the
                    # coverage gate and takes the plain load.
                    # Mark this source name as "live" (its copy_ fired), so
                    # receive_weights can skip names that never copy (no-ops).
                    ctx.copied_names.add(src._name)
                    if ctx.current is not None:
                        layer, param_name = ctx.current
                        ctx.copies_by_layer[layer].append(
                            _BakedCopy(
                                src._key(),
                                param_name,
                                dest.storage_offset(),
                                tuple(dest.shape),
                                tuple(dest.stride()),
                            )
                        )
                    meta_src = torch.empty(src.shape, dtype=src.dtype, device="meta")
                    with torch._C.DisableTorchFunctionSubclass():
                        return dest.copy_(meta_src)
                # Slow path (stock inline reload); ``ctx`` is the producer method.
                # Pass 1 over a meta-restored param: fire a meta-backed copy_ so
                # layerwise's `CopyCounter` still counts the numel (otherwise
                # `load_numel` stays 0 and `_layerwise_process` never triggers).
                if dest.device.type == "meta":
                    meta_src = torch.empty(src.shape, dtype=src.dtype, device="meta")
                    with torch._C.DisableTorchFunctionSubclass():
                        return dest.copy_(meta_src)
                # Pass 2 onto the materialized param: pull this slice on demand,
                # copy it in, free immediately.
                mat = src._materialize()
                with torch._C.DisableTorchFunctionSubclass():
                    result = dest.copy_(mat)
                src._materialized = None
                return result

        # Allowlisted slice/view/shape ops: append to chain and return child.
        op_name = _SUPPORTED_OPS.get(func)
        if op_name is not None:
            self_ = args[0]
            if isinstance(self_, cls) and self_._materialized is None:
                rest = tuple(args[1:])
                return cls._intercept(self_, func, op_name, rest, kwargs)

        # Fallthrough: anything else routes through the underlying op. Pure
        # metadata reads (.shape, .size(), .dim(), .numel(), .dtype, .device)
        # don't reach __torch_dispatch__ because the wrapper subclass stored
        # those at construction. Ops that actually need data DO reach
        # __torch_dispatch__, where we raise.
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def _intercept(
        cls,
        self_: "LazyRDTTensor",
        func: Callable,
        op_name: str,
        args: tuple,
        kwargs: dict,
    ):
        """Append the op to ``self_._ops`` and return a child (or tuple of
        children for chunk-like multi-return ops).

        Shape/dtype of each child come from running the op on a meta tensor —
        PyTorch already knows the semantics, no need to reimplement them.
        """
        meta = self_._meta()
        with torch._C.DisableTorchFunctionSubclass():
            meta_result = func(meta, *args, **kwargs)

        base_op: OpSpec = (op_name, tuple(args), _freeze_kwargs(kwargs))

        # Single-tensor result: one child carrying base_op alone.
        if isinstance(meta_result, torch.Tensor):
            return self_._make_child(meta_result.shape, meta_result.dtype, base_op)

        # Multi-return result (chunk, split, ...): one child per output,
        # each carrying base_op followed by ("__getitem__", (i,), ()) so the
        # trainer replay can index back into the multi-return result.
        if isinstance(meta_result, (tuple, list)):
            return tuple(
                self_._make_child(
                    m.shape,
                    m.dtype,
                    base_op,
                    ("__getitem__", (i,), ()),
                )
                for i, m in enumerate(meta_result)
            )

        # Op produced something that isn't a tensor (e.g. .item() snuck into
        # the allowlist). Bail loudly.
        raise _UnsupportedLazyOp(
            f"LazyRDTTensor: {op_name!r} returned a non-tensor "
            f"({type(meta_result).__name__}); cannot defer."
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        # If any lazy arg made it down to the aten level un-materialized, the
        # loader called an op we don't support. Raise loudly with the op and
        # the recorded chain so the user can identify which loader/weight is
        # at fault. We deliberately do NOT silently materialize here — that
        # would mask correctness bugs.
        for a in args:
            if isinstance(a, cls) and a._materialized is None:
                raise _UnsupportedLazyOp(
                    f"LazyRDTTensor: unsupported op {func} reached "
                    f"__torch_dispatch__ on lazy {a._name!r} (chain={a._ops}). "
                    "Supported ops are: "
                    f"{sorted(_SUPPORTED_OPS.values())}, plus copy_. "
                    "Loaders that need .to(), .float(), .item(), arithmetic, "
                    "bool-mask indexing, or .data access are not supported by "
                    "the sharded RDT backend."
                )
        for v in kwargs.values():
            if isinstance(v, cls) and v._materialized is None:
                raise _UnsupportedLazyOp(
                    f"LazyRDTTensor: unsupported op {func} reached "
                    f"__torch_dispatch__ on lazy {v._name!r} (chain={v._ops}); "
                    "this loader is not supported by the sharded RDT backend."
                )
        return func(*args, **kwargs)


@dataclass
class ShardedRDTWeightTransferInitInfo(WeightTransferInitInfo):
    """Initialization info for the sharded RDT backend."""

    trainer_actor_name: str | None = None
    """Name of a single trainer Ray actor (set via ``.options(name=...)``).

    Back-compat single-producer form. Prefer ``trainer_actor_names`` when the
    trainer exposes more than one NIXL producer (see below). Exactly one of
    ``trainer_actor_name`` / ``trainer_actor_names`` must be non-empty."""

    trainer_actor_names: list[str] = field(default_factory=list)
    """Names of all trainer Ray actors that expose the producer method, ordered
    by trainer rank. When the trainer all-gathers each layer to *every* rank
    (not just rank 0), all of them can serve NIXL pulls, so inference workers
    spread their pulls across this list to parallelize the trainer-side clone +
    NIC egress instead of funneling everything through rank 0. Each inference
    worker statically binds **one** producer from this list, chosen by its own
    global worker index modulo ``len(trainer_actor_names)`` (see
    ``_select_producer_index``). If empty, ``trainer_actor_name`` is used."""

    trainer_actor_namespace: str | None = None
    """Optional Ray namespace the trainer actor(s) live in."""

    produce_method_name: str = "rdt_produce_weights_batched"
    """Name of the trainer-side producer method (see examples/rl/rdt_producer.py
    for the reference implementation). Must be decorated with
    ``@ray.method(tensor_transport="nixl")``. Contract: given a batched specs
    list ``[(name, [(op_name, args, kwargs_items), ...]), ...]``, replay each
    chain on the named tensor and return ONE contiguous uint8 blob with every
    slice byte-packed at 16B-aligned offsets in specs order (the engine
    computes the identical layout and carves dtype views back out). With
    ``pack=False`` (the engine's rare residual/unbaked path), return one slice
    tensor per spec instead. The trainer must also expose ``free_gather(names)``
    (may be a no-op when it has no gather plan)."""

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
    """Optional partition of ``names`` into gather groups (same meaning as
    ``ShardedRDTWeightTransferUpdateInfo.group_lens``). When set, ``names`` must
    be in group-major order matching this partition, and the engine PRE-BUILDS
    the whole static chunk/free plan once at init (it never changes across
    syncs) so ``update_weights`` can be called with an EMPTY update info. When
    empty, the plan is instead built lazily from the first ``update_weights``
    call and cached (back-compat: the driver keeps passing names+group_lens)."""

    num_rdt_buffers: int = 2
    """[RDT-RING] Depth of the consumer receive-arena ring (the producer mirrors
    it from the NUM_RDT_BUFFERS env var). 2 = double buffer: chunk i+1's
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
    when RDT_PACK_CHECK=1. Diffing the streams localizes any producer/consumer
    packed-layout divergence (the core invariant of the packed contract)."""


@dataclass
class ShardedRDTWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Update info for the sharded RDT backend (single-call design).

    ONE ``update_weights`` per sync carries ALL of the sync's names in
    gather-group order; the engine chunk-plans each group and pipelines the
    packed pulls over the receive ring (see ``receive_weights``).

    Both fields are optional: when the driver supplied ``group_lens`` on the
    INIT info, the engine pre-built the (static) plan and this update info can
    be left empty."""

    names: list[str] = field(default_factory=list)

    group_lens: list[int] = field(default_factory=list)
    """Partition of ``names`` into gather groups (group-major;
    ``sum(group_lens) == len(names)``), in the SAME order the driver sent to
    the trainers' ``run_gather_plan``. The engine fires ``free_gather`` on the
    bound producer as each group's last chunk completes. Empty = treat all of
    ``names`` as one group (e.g. a gather-free trainer serving live params)."""


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
    ``init_info.names``) into one ``_BakedGroup`` per fully-loaded leaf module,
    indexed by source name. Every ``update_weights`` *replays* the leaf modules
    its gathered names cover. See the module docstring and ``baked_rdt_replay.md``.
    """

    init_info_cls = ShardedRDTWeightTransferInitInfo
    update_info_cls = ShardedRDTWeightTransferUpdateInfo
    # receive_weights pulls synchronously but defers the GPU post-processing
    # (materialize/scatter/quant/kernel-copy) to a background thread so it
    # overlaps the next chunk's pull. See _run_chunk_pipeline / _process_item.
    defers_processing = True

    def __init__(
        self,
        config: WeightTransferConfig,
        vllm_config: "VllmConfig",
        device: torch.device,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(config, vllm_config, device, model)
        self._trainer_actor: Any | None = None
        self._produce_method: Any | None = None
        # Baked plan: source name -> the _BakedGroup (leaf module) that consumes
        # it. Several names of one fused module map to the same group; replay
        # dedups. A name absent here isn't baked (attention scale / padded /
        # partial) and takes the plain load.
        self._name_to_group: dict[str, _BakedGroup] = {}
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
        # Two slots (fixed depth 2): pull(N+1) writes slot (N+1)%2 while the
        # background thread is still scattering out of slot N%2. A per-slot CUDA
        # event records when the background scatter finished reading a slot; the
        # RPC thread blocks on it before overwriting that slot with a later pull
        # (replaces the old end-of-replay global torch.cuda.synchronize()).
        self._NSLOTS = 2
        self._dest_arenas: list[dict[torch.dtype, torch.Tensor]] = [
            {} for _ in range(self._NSLOTS)
        ]
        self._slot_read_done: list[Any] = []  # one torch.cuda.Event per slot
        # Generation counters guarding the events (see _ensure_proc_worker).
        self._slot_queued: list[int] = []
        self._slot_done: list[int] = []
        self._slot_cv: Any = None
        self._pull_slot = 0
        self._pack_check = False  # [RDT-PACK-CHECK diagnostic] set from init_info
        self._split = 1  # [RDT-SPLIT] chunk pulls per group; set from init_info
        self._arena_presize = 0  # [RDT-RING] bytes; set from init_info
        self._pending_frees: list[Any] = []  # free_gather refs, drained per sync
        # The STATIC plan (see _CallPlan): built once — at init from
        # init_info.group_lens, else lazily on the first update_weights — and
        # reused every sync (each _Chunk self-describes its work; nothing is
        # rebuilt per sync). Non-None means update_weights ignores its (empty or
        # redundant) per-sync names.
        self._cached_plan: "_CallPlan | None" = None
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
        """Resolve the trainer actor and bind its batched producer method."""
        self._pack_check = bool(init_info.pack_check)
        # [RDT-RING] Ring depth K + chunk split S. Must be set before
        # _ensure_proc_worker creates the per-slot events/counters and before
        # any arena is grown (both happen on first pull, after init).
        k = max(1, int(init_info.num_rdt_buffers))
        self._split = max(1, int(init_info.layerwise_split))
        self._NSLOTS = k
        self._dest_arenas = [{} for _ in range(k)]
        self._arena_presize = int(float(init_info.arena_presize_gb) * (1 << 30))
        logger.info(
            "[RDT-RING] num_rdt_buffers=%d layerwise_split=%d presize=%.2fGiB",
            k, self._split, self._arena_presize / (1 << 30),
        )
        try:
            import ray
        except ImportError as e:
            raise RuntimeError(
                "Ray is required for the 'sharded_rdt' weight transfer "
                "backend. Install Ray and run workers as Ray actors "
                "(distributed_executor_backend='ray')."
            ) from e

        producer_names = list(init_info.trainer_actor_names)
        if not producer_names and init_info.trainer_actor_name is not None:
            producer_names = [init_info.trainer_actor_name]
        if not producer_names:
            raise RuntimeError(
                "Sharded RDT engine requires a trainer producer: set "
                "init_info.trainer_actor_names (preferred) or "
                "trainer_actor_name."
            )

        # Static 1:1 load balancing: each inference worker binds exactly ONE
        # producer, chosen by its global worker index modulo the producer count.
        # Workers pull disjoint slice sets (their EP-local experts / TP shard),
        # so spreading them across the trainer ranks parallelizes both the
        # trainer-side clone and the NIXL egress. One stable (consumer, producer)
        # pair per worker keeps the NIXL agent cache simple.
        producer_idx = self._select_producer_index(len(producer_names))
        chosen_name = producer_names[producer_idx]

        try:
            self._trainer_actor = ray.get_actor(
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
        self._trainer_actor._ray_enable_tensor_transport = True

        self._produce_method = getattr(
            self._trainer_actor, init_info.produce_method_name
        )
        logger.info(
            "Sharded RDT engine bound to trainer actor %r (%d/%d producers, "
            "batched method %r)",
            chosen_name,
            producer_idx,
            len(producer_names),
            init_info.produce_method_name,
        )

        # Hardcoded profiling: patch Ray's NIXL transport so this worker's
        # register/transfer/deregister calls accumulate into per-process
        # counters we can snapshot around each pull. Fail-soft.
        from vllm.distributed.weight_transfer._nixl_profile import install_nixl_timing

        install_nixl_timing()

        # Bake the replay plan now. This is a pure dry run: the trainer's gather
        # cache is empty at init, so nothing can (or does) get pulled — we only
        # record how each slice is fetched and where it lands, then restore the
        # model. Every later update_weights is a replay.
        self._bake(init_info)

        # If the driver supplied the gather-group partition at init, PRE-BUILD
        # the (static) chunk/free plan now — it never changes across syncs — so
        # update_weights needs no per-sync names. init_info.names must then be
        # in group-major order matching group_lens (the same flat list the
        # driver would otherwise send every sync).
        if init_info.group_lens:
            if sum(init_info.group_lens) != len(init_info.names):
                raise ValueError(
                    f"init_info.group_lens sums to {sum(init_info.group_lens)} "
                    f"but {len(init_info.names)} names were given."
                )
            self._cached_plan = self._build_call_plan(
                init_info.names, init_info.group_lens
            )
            logger.info(
                "[RDT-PLAN] pre-built static call plan at init: %d chunks, "
                "%d residual name(s)",
                len(self._cached_plan.chunks),
                len(self._cached_plan.residual),
            )

        # Start the background post-processing worker (pull/process pipelining).
        self._ensure_proc_worker()

    def _select_producer_index(self, num_producers: int) -> int:
        """Pick which trainer producer this worker pulls from (static 1:1).

        Returns ``global_worker_index % num_producers`` where the global index
        uniquely identifies this inference worker across the cluster:
        ``data_parallel_rank * tensor_parallel_size + tp_rank``. With DP=4/TP=1
        and 4 producers this is the identity map (worker i -> trainer rank i),
        evenly spreading the disjoint per-worker pull volume across the trainer
        ranks.

        Isolated so a future policy (per-layer rotation, per-pull split) can
        replace this single decision without touching the pull path.
        """
        if num_producers <= 1:
            return 0
        pc = self.parallel_config
        tp_size = getattr(pc, "tensor_parallel_size", 1) or 1
        tp_rank = getattr(pc, "rank", 0) or 0
        dp_rank = getattr(pc, "data_parallel_rank", 0) or 0
        global_worker_index = dp_rank * tp_size + tp_rank
        return global_worker_index % num_producers

    def receive_weights(
        self,
        update_info: ShardedRDTWeightTransferUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        """Pull + replay the baked leaf modules the sync covers.

        The chunk/free plan is STATIC across syncs (a pure function of the baked
        plan + the driver's group partition), so it is built once and cached:
        from ``init_info.group_lens`` at init if the driver supplied it — in
        which case this ``update_info`` may be EMPTY — else lazily from the
        first non-empty ``update_info`` (the driver keeps passing names +
        group_lens). Either way every sync just re-runs the pipeline over the
        self-describing chunks — no per-sync bookkeeping. Residual names with no
        baked plan — attention scales, padded/partial layers — take the plain
        per-slice load; ``load_weights`` is used only by that path.

        Assumes each baked module's source names fall within one gather group
        (true for the per-layer / pre / post partition); if not, the pull
        fails loudly on the missing slice rather than loading wrong data.
        """
        if self._produce_method is None:
            raise RuntimeError(
                "Sharded RDT engine not initialized. Call init_transfer_engine() first."
            )
        # Surface any error the background thread hit on a prior item promptly.
        self._raise_proc_error()
        if self._cached_plan is None:
            # First call and no init-time plan: build from this call's names +
            # group_lens (group_lens absent => one group), then cache and reuse.
            names = update_info.names
            group_lens = list(update_info.group_lens) or [len(names)]
            if sum(group_lens) != len(names):
                raise ValueError(
                    f"group_lens sums to {sum(group_lens)} but "
                    f"{len(names)} names were passed."
                )
            self._cached_plan = self._build_call_plan(names, group_lens)
        residual = self._run_call_plan(self._cached_plan)
        if residual:
            # Rare/absent path (0% once unbaked-skip prunes dead names); runs
            # inline after the pipeline. It touches only non-baked layers, so it
            # does not race the background threads' baked layers.
            self._load_unbaked(residual, load_weights)

    def _build_lazy_weights(
        self,
        names: list[str],
        dtype_names: list[str],
        shapes: list[list[int]],
        ctx: "_BakeRecorder | Callable",
        device: torch.device,
    ) -> list[tuple[str, torch.Tensor]]:
        # LazyRDTTensors are zero-storage, so building them upfront is just a
        # few object allocations. ``ctx`` is the bake recorder (dry run) or the
        # trainer's producer method (slow path, for on-demand per-copy pulls).
        return [
            (
                name,
                LazyRDTTensor(
                    name=name,
                    shape=torch.Size(shape),
                    dtype=_dtype_from_name(dtype_name),
                    device=device,
                    ctx=ctx,
                ),
            )
            for name, dtype_name, shape in zip(names, dtype_names, shapes)
        ]

    def _pull(self, keys) -> dict[FetchKey, torch.Tensor]:
        """The single NIXL pull site: one batched RPC for ``keys`` (a set/list of
        ``(name, op_chain)``), returning ``{key: slice tensor}``."""
        keys = list(keys)
        if not keys:
            return {}
        import ray

        # ``.remote`` is Ray-injected; the producer is bound (non-None) before
        # any pull, guarded by ``_replay``'s init check.
        # pack=False: the rare per-name slow path (no packed layout, no
        # set_target) — the producer returns one slice tensor per spec.
        tensors = ray.get(
            self._produce_method.remote(keys, pack=False)  # type: ignore[union-attr]
        )
        if len(tensors) != len(keys):
            raise RuntimeError(
                f"Trainer returned {len(tensors)} tensors for {len(keys)} keys."
            )
        return dict(zip(keys, tensors))

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
        # keys order) so the pull is ONE contiguous NIXL descriptor. The packed
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
            alloc = max(
                cur,
                self._arena_presize,
                -(-cur // (256 << 20)) * (256 << 20),
            )
            arena = torch.empty(alloc, dtype=torch.uint8, device=self.device)
            register_nixl_memory(arena)
            arenas[torch.uint8] = arena
        targets = [
            arena[off : off + n * dt.itemsize].view(dt).reshape(shape)
            for off, dt, n, shape in chunk.pack_layout
        ]
        blob = [arena[:cur]]
        pack_arena, pack_span = arena, cur
        # Stamp pull-start so the NIXL patch can cleave this pull into
        # produce_wait vs recv_wall. With chunks in flight the cleave is
        # approximate (issues overlap completions) but the transfer sums stand.
        from vllm.distributed.weight_transfer import _nixl_profile
        _nixl_profile.mark_pull_start()
        ref = self._produce_method.remote(keys)  # type: ignore[union-attr]
        # ``blob``/``targets`` stay strongly referenced in the returned
        # _PendingPull through the get — set_target_for_ref stores WEAKREFS, and
        # a dropped target reroutes the transfer into a fallback buffer.
        set_target_for_ref(ref, blob)
        return _PendingPull(
            ref=ref,
            keys=keys,
            targets=targets,
            blob=blob,
            slot=slot,
            pack_arena=pack_arena,
            pack_span=pack_span,
            slot_wait_seconds=_slot_wait,
            slot_sync_seconds=_slot_sync,
        )

    def _complete_pull(
        self, pending: "_PendingPull"
    ) -> "dict[FetchKey, torch.Tensor]":
        """Blocking half of a pull: the NIXL read lands during this ``ray.get``."""
        import ray

        from vllm.distributed.weight_transfer import _nixl_profile

        # [RDT-META-WAIT diagnostic] cleave produce_wait into in-get metadata
        # wait vs issue-side work (see _nixl_profile.mark_get_entry).
        _nixl_profile.mark_get_entry()
        ray.get(pending.ref)
        if self._pack_check and pending.pack_arena is not None:
            # [RDT-PACK-CHECK] checksum the received blob; producer logs the
            # matching sum of what it served (compare offline per pull order).
            # Chunked sums: .sum(dtype=int64) upcasts its INPUT to int64, so a
            # whole-blob sum materializes 8x the blob (OOM on the consumer).
            s = 0
            _w = 32 << 20
            cur = pending.pack_span
            for _i in range(0, cur, _w):
                s += int(
                    pending.pack_arena[_i : min(_i + _w, cur)].sum(
                        dtype=torch.int64
                    )
                )
            import json as _json
            import os as _os
            _os.makedirs("/tmp/rdt_profile", exist_ok=True)
            with open("/tmp/rdt_profile/packcheck_cons.jsonl", "a") as f:
                f.write(_json.dumps(
                    {"pid": _os.getpid(), "bytes": cur, "sum": s}) + "\n")
        return dict(zip(pending.keys, pending.targets))

    # ---------------- Bake (dry run, at init) / replay ----------------

    def _load_unbaked(
        self,
        names: list[str],
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        """Plain load for a call whose names aren't all baked: rebuild lazies
        for ``names`` (dtype/shape from the init metadata) and run vLLM's stock
        inline layerwise reload — the worker's ``initialize_layerwise_reload`` is
        active for the sync, so each layer is processed as it completes and the
        lazy's Pass-2 ``copy_`` pulls its slice on demand. No recording, no
        batching; runs every sync for the call (the rare, unbaked case)."""
        dtype_names = [self._name_meta[n][0] for n in names]
        shapes = [self._name_meta[n][1] for n in names]
        device = torch.empty(0).device
        _t = time.perf_counter()
        load_weights(
            self._build_lazy_weights(
                # Producer is bound (non-None) on this path: ``_replay`` raises
                # before calling ``_load_unbaked`` if it isn't.
                names,
                dtype_names,
                shapes,
                self._produce_method,  # type: ignore[arg-type]
                device,
            )
        )
        self._log_timing("unbaked", time.perf_counter() - _t, 0.0, 0, 0.0)

    def _bake(self, init_info: ShardedRDTWeightTransferInitInfo) -> None:
        """Bake the replay plan once, as a self-driven meta dry run.

        We put the model's params on meta (via ``initialize_layerwise_reload``)
        and then drive ``model.load_weights`` over all of ``init_info.names``
        **through the model's original loaders** — `_install_recording_stamps`
        wraps the *original* loader, bypassing ``online_process_loader`` entirely
        — so ``_layerwise_process`` is never in the path. Nothing materializes,
        pulls, or kernel-copies; the lazy's ``copy_`` just records, per leaf
        module, the source op chain + the meta destination's ``param_name`` and
        ``offset/shape/stride``. Afterwards we build one ``_BakedGroup`` per
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
        recorder = _BakeRecorder()

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
                self._build_lazy_weights(
                    names, dtype_names, shapes, recorder, self.device
                )
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
                group = _BakedGroup(layer=module, copies=copies)
                for c in copies:
                    self._name_to_group[c.src[0]] = group
            self._restore_after_dry_run(model)

        # Names whose copy_ fired during the bake (baked + unbaked-but-live).
        # Residual names not in here no-op for this worker and are skipped.
        self._live_names = set(recorder.copied_names)

        n_groups = len({id(g) for g in self._name_to_group.values()})
        logger.info(
            "Sharded RDT dry-run baked %d/%d names into %d leaf modules "
            "(%d live) in %.3fs",
            len(self._name_to_group),
            len(names),
            n_groups,
            len(self._live_names),
            time.perf_counter() - _t0,
        )

    def _install_recording_stamps(
        self, model: torch.nn.Module, recorder: "_BakeRecorder"
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

        self._slot_read_done = [torch.cuda.Event() for _ in range(self._NSLOTS)]
        # Generation handshake for the events above. A torch.cuda.Event only
        # waits on its LAST record(); if the RPC thread reaches synchronize()
        # for slot s before the background thread has RECORDED item N's event
        # (it is still in Python before the scatter), the wait silently binds to
        # item N-1's record and the next RDMA overwrites the slot under the
        # pending scatter — subtle nondeterministic weight corruption, seen live
        # with _NSLOTS=1. So: the RPC thread counts items queued per slot, the
        # background thread counts records; a pull may synchronize() only after
        # done[slot] has caught up with queued[slot].
        self._slot_queued = [0] * self._NSLOTS
        self._slot_done = [0] * self._NSLOTS
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
        # producer before the sync ends: the producer recreates its 2-deep
        # gather-lookahead semaphore per run_gather_plan, so a free landing
        # after the next sync's plan started would over-credit it (extra ~17GiB
        # gather resident -> trainer OOM risk).
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

    def _chunk_group_scatters(
        self, groups: "list[_BakedGroup]"
    ) -> "list[list[_Scatter]]":
        """Cut the groups' copies (group-major, order-stable) into at most
        ``layerwise_split`` byte-balanced chunks of flat ``_Scatter``s. Copies
        are atomic — a single huge copy (e.g. lm_head) becomes its own oversized
        chunk — and a module's copies may span chunks (materialize/quant fire on
        its first/last chunk; see ``_build_call_plan``)."""
        s = self._split
        entries = [(g.layer, c) for g in groups for c in g.copies]
        if s <= 1 or len(entries) <= 1:
            return [[self._scatter_of(layer, c) for layer, c in entries]]
        weighted = [(layer, c, prod(c.shape)) for layer, c in entries]
        total = sum(b for _, _, b in weighted)
        target = -(-total // s)  # ceil (numel is a fine byte proxy for balance)
        cuts: list[list[_Scatter]] = []
        cur: list[_Scatter] = []
        cur_b = 0
        for layer, c, b in weighted:
            if cur and cur_b + b > target and len(cuts) < s - 1:
                cuts.append(cur)
                cur, cur_b = [], 0
            cur.append(self._scatter_of(layer, c))
            cur_b += b
        if cur:
            cuts.append(cur)
        return cuts

    def _build_call_plan(
        self, names: list[str], group_lens: list[int]
    ) -> "_CallPlan":
        """[RDT-SINGLE-CALL] Build the STATIC plan for one whole-sync call.

        PURE — no pulls, no side effects — so the result is cached and reused
        every sync (see ``_run_call_plan`` / ``_CallPlan``). Three passes:
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
        raw_chunks: list[list[_Scatter]] = []
        free_at: dict[int, list[list[str]]] = {}  # last-chunk idx -> groups to free
        pre_free: list[list[str]] = []  # groups with no chunk on this worker
        residual: list[str] = []
        pos = 0
        for glen in group_lens:
            gnames = names[pos : pos + glen]
            pos += glen
            groups: list[_BakedGroup] = []
            seen: set[int] = set()
            for n in gnames:
                g = self._name_to_group.get(n)
                if g is None:
                    if n in self._live_names:
                        residual.append(n)
                elif id(g) not in seen:
                    seen.add(id(g))
                    groups.append(g)
            if not groups:
                # Nothing to pull for this group on this worker; still free its
                # gather — after the current last chunk, or before the pipeline.
                if raw_chunks:
                    free_at.setdefault(len(raw_chunks) - 1, []).append(gnames)
                else:
                    pre_free.append(gnames)
                continue
            group_chunks = self._chunk_group_scatters(groups)
            raw_chunks.extend(group_chunks)
            free_at.setdefault(len(raw_chunks) - 1, []).append(gnames)

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
                n = prod(shape) or 1
                off = (cur + 15) & ~15
                pack_layout.append((off, dt, n, shape))
                cur = off + n * dt.itemsize
            chunks.append(_Chunk(
                scatters=scatters,
                keys=keys,
                pack_layout=pack_layout,
                pack_bytes=cur,
                materialize=materialize_at.get(ci, []),
                quant=quant_at.get(ci, []),
                free=free_at.get(ci, []),
            ))
        return _CallPlan(chunks=chunks, pre_free=pre_free, residual=residual)

    def _run_call_plan(self, plan: "_CallPlan") -> list[str]:
        """Execute a (cached) call plan: run the chunk pipeline (each chunk
        self-describes its scatter/materialize/quant/free work), return the
        residual names for the plain-load fallback."""
        self._run_chunk_pipeline(plan)
        return plan.residual

    def _fire_free_gather(self, gnames: list[str]) -> None:
        """Fire-and-forget free of one gather group on the bound producer (the
        engine-side replacement for the driver's free_group in single-call
        mode). Refs are held and drained in drain_pending so every free has
        executed before the sync ends."""
        if self._trainer_actor is None:
            return
        self._pending_frees.append(
            self._trainer_actor.free_gather.remote(list(gnames))
        )

    def _run_chunk_pipeline(self, plan: "_CallPlan") -> None:
        """[RDT-RING] Pipelined chunk pulls over the ring of receive slots.

        Issues up to ``_NSLOTS`` produce RPCs ahead of the blocking gets, so
        while chunk i's RDMA streams (inside its ray.get): the producer serves
        chunk i+1 into ITS ring slot, and the background thread scatters chunk
        i-1 out of another receive slot. Reads themselves stay serialized (they
        share the flow's NIC — that is the bandwidth floor, not a loss).

        Producer-ring safety needs no coordination: chunk i+K is issued only
        after chunk i's get returned (drain-before-issue below), so by the time
        produce call #(i+K) reuses producer slot (i mod K), read #i is done.
        Consumer-slot safety is _issue_pull's generation handshake; the drain of
        chunk i-K (which queues its scatter and bumps queued[slot]) strictly
        precedes the issue of chunk i on the same thread.
        """
        from collections import deque

        from vllm.distributed.weight_transfer import _nixl_profile

        inflight: "deque[tuple[_PendingPull, _Chunk]]" = deque()

        # Gather groups with no chunk on this worker (plan.pre_free): free before
        # the pipeline — the trainer gathered them under the lockstep plan but
        # there is no chunk of theirs to hang the free on (fire-and-forget).
        for gnames in plan.pre_free:
            self._fire_free_gather(gnames)

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
            self._dispatch_item(_ProcItem(
                chunk=chunk,
                results=results,
                slot=pending.slot,
                t_recv=_t_recv,
                pull_seconds=pull_seconds,
                nixl_delta=delta,
                pull_bytes=sum(sc.nbytes for sc in chunk.scatters),
            ))
            # Each gather group whose LAST chunk this is: its read is done, so
            # every serve of that group is done -> the producer can drop the
            # gather buffers (fire-and-forget, off the critical path).
            for gnames in chunk.free:
                self._fire_free_gather(gnames)

        for chunk in plan.chunks:
            if not chunk.keys:
                # _chunk_group_scatters never emits empty chunks; keep the frees
                # safe anyway if this ever changes.
                for gnames in chunk.free:
                    self._fire_free_gather(gnames)
                continue
            # Drain BEFORE issue once the ring is full: frees this chunk's slot
            # (generation-wise) and guarantees the producer-slot invariant above.
            # Sync 0 runs SERIAL (max 1 in flight): both sides still grow and
            # register arenas, and a producer registration mid-flight churns its
            # agent-metadata version under an in-flight pull (see __init__).
            max_inflight = 1 if self._completed_syncs == 0 else self._NSLOTS
            if len(inflight) >= max_inflight:
                drain_one()
            slot = self._pull_slot
            self._pull_slot = (slot + 1) % self._NSLOTS
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
                quant_method = getattr(layer, "quant_method", None)
                if isinstance(quant_method, QuantizeMethodBase):
                    if hasattr(
                        layer, "_already_called_process_weights_after_loading"
                    ):
                        delattr(
                            layer, "_already_called_process_weights_after_loading"
                        )
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

        os.makedirs(os.path.dirname(ShardedRDTWeightTransferEngine._CONSUMER_TIMING_FILE), exist_ok=True)
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
        self._trainer_actor = None
        self._produce_method = None
        # Drop strong references to baked modules so the model can be freed.
        self._name_to_group.clear()
        self._name_meta.clear()
        self._live_names.clear()
        # Drop the cached plan (holds _Scatter refs to the baked layers).
        self._cached_plan = None
        # Release the receive arenas (their NIXL registration is pinned for the
        # process lifetime; freeing the tensors just drops our strong refs).
        self._dest_arenas = [{} for _ in range(self._NSLOTS)]

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
