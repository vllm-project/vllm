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
class _ProcItem:
    """One unit of deferred post-processing handed from the RPC thread (which did
    the synchronous pull) to the background process thread.

    ``results`` are views aliasing the double-buffer arena ``slot``; they are held
    as strong refs here so they outlive the RPC-thread frame until the background
    scatter consumes them. The timing fields were measured on the RPC thread
    during the pull and are logged (together with the process-phase split) by the
    background thread after it finishes the item.
    """

    groups: "list[_BakedGroup]"
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
    """Name of the trainer-side method that takes a batched specs list and
    returns a list of slice tensors. Must be decorated with
    ``@ray.method(tensor_transport="nixl")``. Each spec is
    ``(name, [(op_name, args, kwargs_items), ...])`` and the trainer applies
    the chain in order on its live parameter before cloning and returning.
    """

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

    warmup_method_name: str | None = None
    """Optional trainer-side method that returns a tiny tensor over NIXL,
    used to prime the worker->trainer NIXL connection during
    ``init_transfer_engine`` so the first real ``update_weights`` doesn't pay
    the one-time agent/connection-setup latency. Must be decorated with
    ``@ray.method(tensor_transport="nixl")`` and take no required args; its
    result is fetched and discarded. If None (default), no warmup is done.
    The warmup must run on the worker (the NIXL consumer) since the connection
    is per consumer/producer pair."""

    inline_process: bool = False
    """[RDT-INLINE diagnostic] If set, receive_weights runs _process_item INLINE
    (synchronously, before returning) instead of deferring it to the background
    thread. This serializes process so no consumer quant/scatter overlaps any NIXL
    read -- used to test whether concurrent-compute HBM contention depresses the
    transfer BW (expect transfer to climb toward the microbench 44.7 if it does).
    Costs the pull/process pipeline; diagnostic only."""

    one_slot: bool = False
    """[RDT-ONE-SLOT diagnostic] If set, use a SINGLE receive-arena slot instead of
    the depth-2 double buffer. Every pull then writes the same arena region, so the
    consumer's RDMA-write working set is ~group bytes instead of ~2x group bytes.
    Purpose: validate in-engine that the ~34-vs-42 GB/s transfer gap is the NIC/
    IOMMU address-translation-cache reach (~2-3 GB/flow, measured in nixl_envtax.py):
    with one slot the working set fits the reach and transfer should climb to ~42.
    Safe (the per-slot read-done event already serializes pull(N+1) against
    process(N)) but costs the pull/process pipeline; diagnostic only."""

    pack_single_blob: bool = False
    """[RDT-PACK] Extend coalescing ACROSS dtypes (implies coalesce): producer and
    consumer byte-pack every slice (16B-aligned, keys order) into ONE uint8 arena,
    so each pull is ONE contiguous NIXL descriptor instead of one per dtype
    (Kimi: fp8 ~96% + fp32 ~2% + bf16 ~2%). The consumer carves dtype views back
    out of the packed arena at the same offsets for the scatter. Both sides
    compute the layout independently from the same (keys order, dtype, shape,
    16B-align) rule, so the bytes land identically. Set from RDT_COALESCE=2.
    Bench (nixl_sustained.py): 3 dtype blobs -> 1 blob = 42.3 -> 44.4 GB/s."""

    pack_check: bool = False
    """[RDT-PACK-CHECK diagnostic] With pack_single_blob: after every pull, checksum
    the received packed blob (int64 sum of the uint8 arena) and append
    {pid, bytes, sum} to /tmp/rdt_profile/packcheck_cons.jsonl. The producer logs
    the matching checksum of its served blob (RDT_PACK_CHECK=1). Diffing the two
    streams localizes nondeterministic corruption to transfer vs serve vs scatter."""

    coalesce_dtype_blobs: bool = False
    """[RDT-COALESCE] If set, the producer returns ONE contiguous tensor per dtype
    (its packed per-dtype serve arena) instead of one view per source name, and the
    consumer points ``set_target_for_ref`` at its matching per-dtype receive-arena
    regions. Both sides already lay slices out with the IDENTICAL per-dtype,
    8-elem-aligned, keys-order layout, so this transfers the same bytes as ~2 large
    NIXL descriptors instead of ~hundreds of small ones (approaches contiguous-buffer
    speed-of-light). Carried here (not via env) because vLLM only copies a filtered
    env allowlist to its worker procs; the driver sets it from RDT_COALESCE and the
    producer reads RDT_COALESCE directly."""


@dataclass
class ShardedRDTWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Update info for the sharded RDT backend.

    The subset of ``init_info.names`` the driver gathered for this call (e.g.
    one decoder layer's params). The engine replays the baked leaf modules
    those names cover; any name without a baked plan (attention scales, padded
    layers) takes the plain load. Dtypes/shapes were supplied once at init, so
    no new name is ever encountered mid-run."""

    names: list[str]


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
    # overlaps the next group's pull. See _pull_groups / _process_item.
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
        # The baked source slice metadata, so _replay can size receive views:
        #   src FetchKey -> produced slice shape / torch dtype.
        self._src_shapes: dict[FetchKey, tuple[int, ...]] = {}
        self._src_dtypes: dict[FetchKey, torch.dtype] = {}
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
        # [RDT-COALESCE] Set from init_info in init_transfer_engine: transfer one
        # contiguous tensor per dtype instead of one per source name (see the flag's
        # docstring). Off by default.
        self._coalesce = False
        self._pack = False  # [RDT-PACK] single cross-dtype blob; set from init_info
        self._pack_check = False  # [RDT-PACK-CHECK diagnostic] set from init_info
        self._inline_process = False  # [RDT-INLINE diagnostic] set from init_info

        # ---- Background post-processing thread (pull/process pipelining) -------
        # receive_weights pulls synchronously on the RPC thread, then hands the
        # pulled arena views to this single worker thread, which runs
        # materialize/scatter/quant/kernel_copy on its own CUDA stream while the
        # next group's pull proceeds. Drained by drain_pending() (called from the
        # worker's finish_weight_update) before finalize_layerwise_reload runs.
        self._proc_queue: Any | None = None
        self._proc_thread: Any | None = None
        self._proc_stream: Any | None = None
        self._proc_error: BaseException | None = None

    def init_transfer_engine(self, init_info: ShardedRDTWeightTransferInitInfo) -> None:
        """Resolve the trainer actor and bind its batched producer method."""
        self._coalesce = bool(init_info.coalesce_dtype_blobs)
        self._pack = bool(init_info.pack_single_blob)
        self._pack_check = bool(init_info.pack_check)
        self._inline_process = bool(init_info.inline_process)
        if init_info.one_slot:
            # [RDT-ONE-SLOT diagnostic] single receive slot; must be set before
            # _ensure_proc_thread creates the per-slot events and before any
            # arena is grown (both happen on first pull, after init).
            self._NSLOTS = 1
            self._dest_arenas = [{}]
            logger.info("[RDT-ONE-SLOT] single receive-arena slot (diagnostic)")
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
        # pair per worker keeps the NIXL agent cache + warmup simple.
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

        # Optional NIXL warmup: prime the worker->trainer NIXL connection now
        # so the first real update_weights doesn't eat the one-time
        # agent/connection-setup latency (~seconds on the first transfer).
        # The pull MUST be initiated here, on the worker, because the NIXL
        # connection is established per consumer (this worker) / producer
        # (trainer) pair -- a driver-side call would not warm up this path.
        if init_info.warmup_method_name is not None:
            warmup_method = getattr(self._trainer_actor, init_info.warmup_method_name)
            t0 = time.perf_counter()
            _ = ray.get(warmup_method.remote())
            del _
            logger.info(
                "Sharded RDT engine warmed up NIXL connection via %r in %.3fs",
                init_info.warmup_method_name,
                time.perf_counter() - t0,
            )

        # Hardcoded profiling: patch Ray's NIXL transport so this worker's
        # register/transfer/deregister calls accumulate into per-process
        # counters we can snapshot around each pull. Fail-soft.
        from vllm.distributed.weight_transfer._nixl_profile import install_nixl_timing

        install_nixl_timing()

        # Measure the per-RPC fixed-cost floor from THIS worker to its trainer,
        # on the same path produce takes: a bare Ray actor call (no nixl) and a
        # tiny nixl pull (warmup). Their difference isolates the nixl
        # control-plane overhead; the bare call is the pure Ray dispatch RTT.
        self._measure_rpc_baseline()

        # Bake the replay plan now. This is a pure dry run: the trainer's gather
        # cache is empty at init, so nothing can (or does) get pulled — we only
        # record how each slice is fetched and where it lands, then restore the
        # model. Every later update_weights is a replay.
        self._bake(init_info)

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

    def _measure_rpc_baseline(self, n: int = 30) -> None:
        """Time the bare Ray actor-call RTT and the tiny-nixl-pull RTT.

        Hardcoded benchmark probe. ``ping`` is a no-op method (pure Ray
        dispatch, no nixl); ``rdt_warmup`` returns a 1-element tensor over nixl
        (Ray dispatch + full nixl control plane + negligible data). The
        difference is the per-RPC nixl control-plane fixed cost; ``ping`` alone
        is the Ray dispatch floor. Results go to the consumer timing file as a
        ``mode=baseline`` record so the driver can read them.
        """
        import json
        import os

        import ray

        actor = self._trainer_actor
        ping = getattr(actor, "ping", None)
        warmup = getattr(actor, "rdt_warmup", None)
        bare_ms = nixl_ms = None
        try:
            if ping is not None:
                ray.get(ping.remote())  # prime
                t0 = time.perf_counter()
                for _ in range(n):
                    ray.get(ping.remote())
                bare_ms = (time.perf_counter() - t0) / n * 1e3
            if warmup is not None:
                ray.get(warmup.remote())  # prime (also warms the nixl connection)
                t0 = time.perf_counter()
                for _ in range(n):
                    ray.get(warmup.remote())
                nixl_ms = (time.perf_counter() - t0) / n * 1e3
        except Exception as e:
            logger.warning("RPC baseline probe failed: %r", e)
            return

        logger.info(
            "[RDT-TIMING] rpc baseline: bare_ray=%.3fms nixl_ping=%.3fms "
            "(nixl control-plane ~= %.3fms)",
            bare_ms or 0.0,
            nixl_ms or 0.0,
            (nixl_ms - bare_ms) if (bare_ms and nixl_ms) else 0.0,
        )
        os.makedirs(
            os.path.dirname(ShardedRDTWeightTransferEngine._CONSUMER_TIMING_FILE),
            exist_ok=True,
        )
        with open(ShardedRDTWeightTransferEngine._CONSUMER_TIMING_FILE, "a") as f:
            f.write(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "mode": "baseline",
                        "bare_ray_ms": bare_ms,
                        "nixl_ping_ms": nixl_ms,
                    }
                )
                + "\n"
            )

    def receive_weights(
        self,
        update_info: ShardedRDTWeightTransferUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        """Replay the baked leaf modules the call's gathered names cover.

        ``update_info.names`` is the subset of the init names the driver
        gathered for this call. We replay every baked ``_BakedGroup`` those
        names reach (deduped) in one batched pull, and route only the
        **residual** names with no baked plan — attention scales, padded/partial
        layers, and experts owned by another EP rank (which no-op in their
        loader) — to the plain per-slice load. ``load_weights`` is used only by that residual path.

        Assumes each baked module's source names are gathered within a single
        call (true for the per-layer / pre / post partition); if not, ``_pull``
        would fail loudly on the missing slice rather than load wrong data.
        """
        if self._produce_method is None:
            raise RuntimeError(
                "Sharded RDT engine not initialized. Call init_transfer_engine() first."
            )
        # Surface any error the background thread hit on a prior item promptly.
        self._raise_proc_error()
        names = update_info.names
        groups: list[_BakedGroup] = []
        seen: set[int] = set()
        residual: list[str] = []
        for n in names:
            g = self._name_to_group.get(n)
            if g is None:
                # Skip residual names that never copied during the bake -- they
                # no-op for this worker (e.g. experts owned by another EP rank),
                # so running _load_unbaked on them just burns CPU building lazies
                # and re-running load_weights' name-matching every sync.
                if n in self._live_names:
                    residual.append(n)
            elif id(g) not in seen:
                seen.add(id(g))
                groups.append(g)
        if groups:
            # Pull synchronously on this RPC thread (keeps the pull on the driver's
            # critical path and preserves the producer's double-buffer safety),
            # then hand the pulled slices to the background thread for
            # scatter/quant/kernel-copy and return -- so the NEXT group's pull
            # overlaps this group's processing.
            item = self._pull_groups(groups)
            # Count the item against its slot BEFORE dispatch: the next pull into
            # this slot must wait until the background thread has processed (and
            # RECORDED the read-done event for) every item ever queued on it.
            with self._slot_cv:
                self._slot_queued[item.slot] += 1
            if self._inline_process:
                # [RDT-INLINE diagnostic] process synchronously so no quant overlaps
                # the next group's transfer (tests HBM contention on the read path).
                # (_process_item publishes the slot generation internally.)
                self._process_item(item)
            else:
                assert self._proc_queue is not None
                self._proc_queue.put(item)
        if residual:
            # Rare/absent path (0% for Kimi after unbaked-skip); run inline. It
            # touches only non-baked layers, so it does not race the background
            # thread's baked layers, and it completes before this call returns.
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
        tensors = ray.get(self._produce_method.remote(keys))  # type: ignore[union-attr]
        if len(tensors) != len(keys):
            raise RuntimeError(
                f"Trainer returned {len(tensors)} tensors for {len(keys)} keys."
            )
        return dict(zip(keys, tensors))

    def _pull_into_registered(
        self, keys: list[FetchKey], slot: int
    ) -> dict[FetchKey, torch.Tensor]:
        """Batched NIXL pull that reads each slice straight into a persistent,
        pre-registered receive arena -- eliminating the per-pull dest allocation
        and registration that the default recv path does.

        How it avoids registration: ``register_nixl_memory(arena)`` registers the
        arena's full storage once (the registration cache ``_add_tensor_descs`` is
        keyed by ``untyped_storage().data_ptr()`` and pins the refcount so it is
        never deregistered). Each per-slice TARGET is a view INTO that arena, so it
        shares the same storage data_ptr -> every recv is a cache hit and does no
        register/deregister. ``set_target_for_ref`` routes the transfer into our
        views instead of letting Ray allocate fresh (unregistered) buffers.

        Double-buffered: ``slot`` selects one of ``_NSLOTS`` per-dtype arena sets,
        so this pull can write into slot (N+1)%2 while the background thread is
        still scattering out of slot N%2. Each set grows (and re-registers) only if
        a call needs more than it currently holds; steady state is zero
        registration. The returned views alias ``slot``'s arena and MUST be
        consumed (scatter-copied into params) before this slot is pulled into
        again -- guaranteed by the per-slot read-done event this method waits on
        below (recorded by ``_process_item`` after its scatter).
        """
        if not keys:
            return {}
        import ray
        from ray.experimental import register_nixl_memory, set_target_for_ref

        # Block until the background thread's scatter out of this slot (from the
        # last pull that used it) has completed on the GPU, so the NIXL read
        # below does not overwrite data still being read. TWO stages, both
        # required: (1) generation wait — the CUDA event only binds to its LAST
        # record(), so we must first wait for the background thread to have
        # RECORDED the event for every item ever queued on this slot (else the
        # synchronize binds to a stale record and the guard silently passes —
        # observed as nondeterministic weight corruption with _NSLOTS=1);
        # (2) event synchronize — waits for the recorded stream work (the
        # scatters) to actually finish on the GPU.
        if self._slot_read_done:
            with self._slot_cv:
                while self._slot_done[slot] < self._slot_queued[slot]:
                    self._slot_cv.wait(timeout=1.0)
            self._slot_read_done[slot].synchronize()

        arenas = self._dest_arenas[slot]

        if self._pack:
            # [RDT-PACK] Byte-pack every slice into ONE uint8 arena (16B-aligned,
            # keys order) so the pull is ONE contiguous NIXL descriptor. The
            # producer computes the identical layout from the same rule, so the
            # bytes land at these exact offsets; ``targets`` carves the dtype
            # views back out for the scatter (transfer granularity changes, the
            # returned per-name views do not).
            playout: list[tuple[int, torch.dtype, int, Any]] = []  # off,dt,n,shape
            cur = 0
            for k in keys:
                dt = self._src_dtypes[k]
                shape = self._src_shapes[k]
                n = prod(shape) or 1
                off = (cur + 15) & ~15
                playout.append((off, dt, n, shape))
                cur = off + n * dt.itemsize
            arena = arenas.get(torch.uint8)
            if arena is None or arena.numel() < cur:
                arena = torch.empty(cur, dtype=torch.uint8, device=self.device)
                register_nixl_memory(arena)
                arenas[torch.uint8] = arena
            targets = [
                arena[off : off + n * dt.itemsize].view(dt).reshape(shape)
                for off, dt, n, shape in playout
            ]
            from vllm.distributed.weight_transfer import _nixl_profile
            _nixl_profile.mark_pull_start()
            ref = self._produce_method.remote(keys)  # type: ignore[union-attr]
            # Strong ref through the ray.get — set_target_for_ref stores WEAKREFS
            # to the target tensors, so a temporary view here would be collected
            # before the NIXL read and the bytes would land in a fallback buffer,
            # never reaching the arena the ``targets`` views alias.
            blob = [arena[:cur]]
            set_target_for_ref(ref, blob)
            ray.get(ref)  # ONE NIXL read straight into the packed arena
            if self._pack_check:
                # [RDT-PACK-CHECK] checksum the received blob; producer logs the
                # matching sum of what it served (compare offline per pull order).
                # Chunked: .sum(dtype=int64) upcasts its INPUT to int64, so a
                # whole-blob sum materializes 8x the blob (OOM on the consumer).
                s = 0
                _w = 32 << 20
                for _i in range(0, cur, _w):
                    s += int(arena[_i : min(_i + _w, cur)].sum(dtype=torch.int64))
                import json as _json
                import os as _os
                _os.makedirs("/tmp/rdt_profile", exist_ok=True)
                with open("/tmp/rdt_profile/packcheck_cons.jsonl", "a") as f:
                    f.write(_json.dumps(
                        {"pid": _os.getpid(), "bytes": cur, "sum": s}) + "\n")
            return dict(zip(keys, targets))

        # Lay each slice into the persistent arena FOR ITS DTYPE, so mixed-dtype
        # groups (Kimi: fp8 weights + fp32 weight_scale_inv + bf16 norms) all use
        # registered arenas. Per-dtype element offsets, lightly aligned for RDMA.
        layout: dict[FetchKey, tuple[torch.dtype, int]] = {}
        totals: dict[torch.dtype, int] = {}
        for k in keys:
            dt = self._src_dtypes[k]
            off = totals.get(dt, 0)
            layout[k] = (dt, off)
            n = prod(self._src_shapes[k]) or 1
            totals[dt] = off + ((n + 7) & ~7)  # 8-element alignment

        for dt, total in totals.items():
            arena = arenas.get(dt)
            if arena is None or arena.numel() < total:
                arena = torch.empty(total, dtype=dt, device=self.device)
                # One-time (per growth) registration of the whole storage; pinned
                # for the process lifetime so views never register/deregister.
                register_nixl_memory(arena)
                arenas[dt] = arena

        # Carve a contiguous view per slice (keys order). Held in ``targets``
        # (strong refs) through the ray.get -- set_target_for_ref stores weakrefs.
        targets: list[torch.Tensor] = []
        for k in keys:
            dt, off = layout[k]
            shape = self._src_shapes[k]
            n = prod(shape) or 1
            targets.append(arenas[dt][off : off + n].reshape(shape))

        # Stamp pull-start so the NIXL patch can cleave this pull into
        # produce_wait (blocked on the producer: serve + Ray dispatch + meta
        # cuda.sync) vs recv_wall (the actual RDMA read). Without this the split
        # stays dead and only the coarse pull/transfer numbers are available.
        from vllm.distributed.weight_transfer import _nixl_profile
        _nixl_profile.mark_pull_start()
        ref = self._produce_method.remote(keys)  # type: ignore[union-attr]
        if self._coalesce:
            # [RDT-COALESCE] The producer returns ONE tensor per dtype (its whole
            # packed arena region, sorted-dtype order); point each at our matching
            # per-dtype receive-arena region. Same layout as ``targets`` above, so the
            # bytes land identically -- but as ~2 big NIXL descriptors, not ~hundreds.
            # ``targets`` (per-name views into the same arenas) is still what we return
            # for the scatter; only the transfer granularity changes.
            blob_targets = [arenas[dt][: totals[dt]] for dt in sorted(totals, key=str)]
            set_target_for_ref(ref, blob_targets)
        else:
            set_target_for_ref(ref, targets)
        ray.get(ref)  # NIXL reads each slice directly into its arena view
        return dict(zip(keys, targets))

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
                    # Record the produced slice's shape/dtype so _replay can size
                    # pre-registered receive views. The produced slice matches the
                    # destination region (c.shape); its dtype is the source name's.
                    self._src_shapes[c.src] = tuple(c.shape)
                    self._src_dtypes[c.src] = _dtype_from_name(
                        self._name_meta[c.src[0]][0]
                    )
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
        if self._proc_stream is not None:
            self._proc_stream.synchronize()
        self._raise_proc_error()

    def _pull_groups(self, groups: "list[_BakedGroup]") -> "_ProcItem":
        """RPC-thread half of the pipeline: one batched NIXL pull for every unique
        source slice the groups need, into the next double-buffer slot. Returns a
        work item for the background thread; performs NO post-processing so this
        call returns as soon as the pull's ``ray.get`` completes and the next
        group's pull can begin.

        Snapshot the consumer-side NIXL counters around the pull so the delta is
        exactly this pull's registration / transfer split (logged later by the
        background thread together with the process-phase split).
        """
        from vllm.distributed.weight_transfer import _nixl_profile

        _t_recv = time.perf_counter()
        _nixl_before = _nixl_profile.snapshot()
        _t_pull = time.perf_counter()
        # De-duplicated, ORDER-STABLE source keys: the producer returns slices in
        # this order, and set_target_for_ref requires the target list to match.
        keys: list[FetchKey] = []
        seen_keys: set[FetchKey] = set()
        for g in groups:
            for c in g.copies:
                if c.src not in seen_keys:
                    seen_keys.add(c.src)
                    keys.append(c.src)
        slot = self._pull_slot
        self._pull_slot = (slot + 1) % self._NSLOTS
        results = self._pull_into_registered(keys, slot)
        pull_seconds = time.perf_counter() - _t_pull
        _nixl_delta = _nixl_profile.delta(_nixl_before, _nixl_profile.snapshot())
        # Bytes this worker actually pulled (for true per-worker BW / straggler
        # diagnosis: slow worker + more bytes = imbalance; slow + equal bytes =
        # transport straggler).
        pull_bytes = sum(
            prod(self._src_shapes[k]) * self._src_dtypes[k].itemsize for k in keys
        )
        return _ProcItem(
            groups=groups,
            results=results,
            slot=slot,
            t_recv=_t_recv,
            pull_seconds=pull_seconds,
            nixl_delta=_nixl_delta,
            pull_bytes=pull_bytes,
        )

    def _process_item(self, item: "_ProcItem") -> None:
        """Background-thread half: scatter each baked slice into its destination
        param, run ``process_weights_after_loading``, and copy into persistent
        kernel storage -- all on the dedicated process stream so it overlaps the
        next group's pull on the RPC thread.

        Mirrors ``_layerwise_process`` minus the loader replay: materialize ->
        scatter via ``as_strided().copy_()`` -> quant -> kernel-copy ->
        ``info.reset()`` (the reset is what makes ``finalize_layerwise_reload``
        skip the layer instead of clobbering it with the old kernel tensors).

        After all scatters that read ``item.slot``'s arena are enqueued on the
        process stream, record the slot's read-done event so the RPC thread can
        block on it before overwriting the slot with a later pull.
        """
        from vllm.distributed.weight_transfer._nixl_profile import PhaseTimer
        from vllm.model_executor.layers.quantization.base_config import (
            QuantizeMethodBase,
        )
        from vllm.model_executor.model_loader.reload.layerwise import (
            LAYERWISE_INFO,
            _copy_and_restore_kernel_tensors,
        )
        from vllm.model_executor.model_loader.reload.meta import materialize_layer

        results = item.results
        ph = PhaseTimer(self._proc_stream)  # stream-scoped syncs, not global
        _t_proc = time.perf_counter()
        with (
            torch.cuda.device(self.device),
            torch.cuda.stream(self._proc_stream),
            torch.device(self.device),
        ):
            # PASS 1 — slot readers: materialize + scatter every group. The
            # scatter copies are the ONLY reads of the receive arena; quant and
            # kernel-copy operate on the scattered params. Releasing the slot
            # right after the scatters lets the NEXT pull's RDMA overwrite the
            # arena while this item's quant still runs — quant is hidden behind
            # the next transfer even at _NSLOTS=1.
            try:
                for g in item.groups:
                    layer = g.layer
                    info = LAYERWISE_INFO.get(layer)
                    if info is None or not info.can_load():
                        raise RuntimeError(
                            f"Baked replay: layer {type(layer).__name__} was not "
                            "set up for reload this sync (start_weight_update "
                            "must run before update_weights)."
                        )
                    with ph.phase("materialize_seconds"):
                        materialize_layer(layer, info)  # cheap empty HF params
                    with ph.phase("scatter_seconds"):
                        for c in g.copies:
                            param = getattr(layer, c.param_name)
                            dst = param.as_strided(c.shape, c.stride, c.offset)
                            with torch._C.DisableTorchFunctionSubclass():
                                dst.copy_(results[c.src])
                # All reads of this slot's arena are now enqueued on the process
                # stream; record + publish so the RPC thread can reuse the slot.
                self._slot_read_done[item.slot].record(self._proc_stream)
            finally:
                # Publish even on error so the RPC thread's generation wait
                # unblocks (the failure itself surfaces via _raise_proc_error).
                self._mark_slot_done(item.slot)

            # PASS 2 — param readers only: quant / kernel-copy / reset, exactly
            # as _layerwise_process. May overlap the next pull's RDMA.
            for g in item.groups:
                layer = g.layer
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
        self._slot_read_done = []
        self._trainer_actor = None
        self._produce_method = None
        # Drop strong references to baked modules so the model can be freed.
        self._name_to_group.clear()
        self._name_meta.clear()
        self._src_shapes.clear()
        self._src_dtypes.clear()
        self._live_names.clear()
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
