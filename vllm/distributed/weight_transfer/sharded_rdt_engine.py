# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sharded Ray Direct Transport (RDT) weight transfer engine.

Unlike the unsharded RDT engine, this backend pulls only the *slice* that
each vLLM worker actually consumes, not the full HF-format tensor.

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

Groups whose loaders don't reduce to a pure view + ``copy_`` (or whose
requested names aren't fully covered by recorded copies — e.g. weights that
load via the attention/partial-layer finalize path) fail the coverage gate
and fall back to a plain batched load every sync.

Only valid with ``is_checkpoint_format=True`` (layerwise reload). See
``sharded_weight_loader_rdt.md`` and ``baked_rdt_replay.md`` for the design,
and the spike in ``nixl_slice_spike.py`` confirming NIXL is view-aware.
"""

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

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
    """A baked leaf module: its destination scatters plus the post-load
    processing flag.

    ``layer`` is a strong reference to the module, held for the engine's
    lifetime and cleared in ``shutdown``. The module persists across syncs (the
    model is not rebuilt), and its ``LayerReloadingInfo`` — with the meta
    ``restore_metadata`` and per-sync ``kernel_tensors`` — is re-established by
    ``initialize_layerwise_reload`` at the start of every update.
    """

    layer: Any
    copies: list[_BakedCopy]
    needs_pwal: bool


@dataclass
class _BakeRecorder:
    """Per-leaf-module recording context for the dry-run bake.

    The dry run replays each buffered loader against the layer's **meta** params
    (no real storage, no transfer). Before each loader call the engine sets
    ``param_name`` to the param the loader is bound to; the lazy's ``copy_``
    then appends a ``_BakedCopy`` reading the op chain from the source and
    ``offset/shape/stride`` from the (meta) destination view. ``None`` marks a
    copy_ we couldn't record (the group falls back to a plain load).
    """

    sink: "list[_BakedCopy | None]" = field(default_factory=list)
    param_name: str = ""


@dataclass
class _FetchPlan:
    """Per-load state for the plain (unbaked) slow path.

    Pass 1 (``online_process_loader`` -> ``get_numel_loaded``) records every
    ``(name, op_chain)`` it would copy_ onto a meta destination into ``needed``.
    The engine pulls them all in one batched RPC (``_pull``) into ``results``;
    Pass 2 (replay inside ``_layerwise_process``) pops from ``results`` on the
    real copy_.
    """

    needed: set[FetchKey] = field(default_factory=set)
    results: dict[FetchKey, torch.Tensor] = field(default_factory=dict)


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
    - ``_FetchPlan`` (slow path): Pass 1 over a meta destination records the
      chain into ``needed``; the engine then pulls all chains in one batched
      RPC into ``results``; Pass 2 over the real destination pops the slice and
      copies it in.

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
    _ctx: "_FetchPlan | _BakeRecorder | None"
    _materialized: "torch.Tensor | None"

    @staticmethod
    def __new__(
        cls,
        name: str,
        shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        ops: OpChain = (),
        ctx: "_FetchPlan | _BakeRecorder | None" = None,
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
        # pulls. So the ctx here is always a _FetchPlan, and every chain Pass 1
        # recorded was pulled into ``results`` before this replay.
        if self._materialized is not None:
            return self._materialized
        assert isinstance(self._ctx, _FetchPlan)
        tensor = self._ctx.results.pop(self._key(), None)
        if tensor is None:
            raise RuntimeError(
                f"LazyRDTTensor {self._name!r} (chain={self._ops}) has no "
                "prefetched slice; the batched pull missed this key, which "
                "means the loader was non-deterministic between record and "
                "replay."
            )
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
                    # is bound to (ctx.param_name). Record how to fetch the
                    # source slice (the op chain) and where it lands (the meta
                    # view's offset/shape/stride — valid on meta), then let a
                    # meta copy_ fire so the layer's numel still counts. No pull,
                    # no real storage.
                    if ctx.param_name:
                        ctx.sink.append(
                            _BakedCopy(
                                src._key(),
                                ctx.param_name,
                                dest.storage_offset(),
                                tuple(dest.shape),
                                tuple(dest.stride()),
                            )
                        )
                    else:
                        ctx.sink.append(None)
                    meta_src = torch.empty(src.shape, dtype=src.dtype, device="meta")
                    with torch._C.DisableTorchFunctionSubclass():
                        return dest.copy_(meta_src)
                # Slow path. Pass 1 over a meta-restored param: record the chain
                # into the plan, then let a meta-backed copy_ fire so layerwise's
                # `CopyCounter` still counts the numel (otherwise `load_numel`
                # stays 0 and `_layerwise_process` never triggers).
                assert isinstance(ctx, _FetchPlan)
                if dest.device.type == "meta":
                    ctx.needed.add(src._key())
                    meta_src = torch.empty(src.shape, dtype=src.dtype, device="meta")
                    with torch._C.DisableTorchFunctionSubclass():
                        return dest.copy_(meta_src)
                # Pass 2 replay onto the materialized param: pop the prefetched
                # tensor, copy it in, free immediately.
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
                    "bool-mask indexing, or .data access are not compatible "
                    "with the sharded RDT backend; use backend='rdt' instead."
                )
        for v in kwargs.values():
            if isinstance(v, cls) and v._materialized is None:
                raise _UnsupportedLazyOp(
                    f"LazyRDTTensor: unsupported op {func} reached "
                    f"__torch_dispatch__ on lazy {v._name!r} (chain={v._ops}). "
                    "Use backend='rdt' for loaders that require materialization."
                )
        return func(*args, **kwargs)


@dataclass
class ShardedRDTWeightTransferInitInfo(WeightTransferInitInfo):
    """Initialization info for the sharded RDT backend."""

    trainer_actor_name: str
    """Name of the trainer Ray actor (set via ``.options(name=...)``)."""

    trainer_actor_namespace: str | None = None
    """Optional Ray namespace the trainer actor lives in."""

    produce_method_name: str = "rdt_produce_weights_batched"
    """Name of the trainer-side method that takes a batched specs list and
    returns a list of slice tensors. Must be decorated with
    ``@ray.method(tensor_transport="nixl")``. Each spec is
    ``(name, [(op_name, args, kwargs_items), ...])`` and the trainer applies
    the chain in order on its live parameter before cloning and returning.
    """

    group_names: list[list[str]] = field(default_factory=list)
    """The full set of parameters to transfer, partitioned into groups (one
    inner list per group), in the trainer's group-index order. The engine bakes
    a replay plan for every group once, at ``init_transfer_engine`` time, and
    ``update_weights`` then refers to a group by its index. Group ``i`` is the
    set of names the driver gathers + replays together (typically one decoder
    layer, plus a "pre" group for embeddings and a "post" group for the head).
    """

    group_dtype_names: list[list[str]] = field(default_factory=list)
    """Per-group dtype names (e.g. 'bfloat16'), parallel to ``group_names``."""

    group_shapes: list[list[list[int]]] = field(default_factory=list)
    """Per-group full HF shapes, parallel to ``group_names``."""

    warmup_method_name: str | None = None
    """Optional trainer-side method that returns a tiny tensor over NIXL,
    used to prime the worker->trainer NIXL connection during
    ``init_transfer_engine`` so the first real ``update_weights`` doesn't pay
    the one-time agent/connection-setup latency. Must be decorated with
    ``@ray.method(tensor_transport="nixl")`` and take no required args; its
    result is fetched and discarded. If None (default), no warmup is done.
    The warmup must run on the worker (the NIXL consumer) since the connection
    is per consumer/producer pair."""


@dataclass
class ShardedRDTWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Update info for the sharded RDT backend.

    A single integer: which baked group (from ``init_info.group_names``) to
    apply this call. Names/dtypes/shapes were supplied once at init, so we never
    encounter a new name mid-run."""

    group_index: int


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
      - ``is_checkpoint_format=True`` (layerwise reload). ``_bake_all_groups``
        raises at init if no layerwise infos are active.
      - Weight loaders that only use the supported op set (narrow, view,
        reshape, transpose, t, permute, __getitem__ with int/slice/tuple,
        unsqueeze, squeeze, flatten, contiguous, chunk, copy_). Loaders
        that need .to(), .float(), .item(), .data, bool-mask indexing, or
        arithmetic on the loaded weight will land in ``__torch_dispatch__``
        during the bake (raising) and should fall back to backend='rdt'.

    Plans are baked once at ``init_transfer_engine`` (a dry run over
    ``init_info.group_names``); every ``update_weights`` *replays* its group by
    index. See the module docstring and ``baked_rdt_replay.md``.
    """

    init_info_cls = ShardedRDTWeightTransferInitInfo
    update_info_cls = ShardedRDTWeightTransferUpdateInfo

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
        # One entry per trainer-provided group, in group-index order: the baked
        # leaf-module list to replay for that group, or None if the group was
        # found unbakeable (then it takes the plain load every sync).
        self._plans: list[list[_BakedGroup] | None] = []
        # Per-group (names, dtype_names, shapes), so an unbakeable group can
        # reconstruct its load on the slow path from just the group index.
        self._group_meta: list[tuple[list[str], list[str], list[list[int]]]] = []

    def init_transfer_engine(self, init_info: ShardedRDTWeightTransferInitInfo) -> None:
        """Resolve the trainer actor and bind its batched producer method."""
        try:
            import ray
        except ImportError as e:
            raise RuntimeError(
                "Ray is required for the 'sharded_rdt' weight transfer "
                "backend. Install Ray and run workers as Ray actors "
                "(distributed_executor_backend='ray')."
            ) from e

        try:
            self._trainer_actor = ray.get_actor(
                init_info.trainer_actor_name,
                namespace=init_info.trainer_actor_namespace,
            )
        except ValueError as e:
            raise RuntimeError(
                f"Sharded RDT engine could not find trainer actor "
                f"{init_info.trainer_actor_name!r} (namespace="
                f"{init_info.trainer_actor_namespace!r})."
            ) from e

        # Ray 2.51.1 workaround: actor handles reconstructed via
        # ray.get_actor lose the actor-level _ray_enable_tensor_transport
        # flag, so the NIXL dispatch guard at ray/actor.py rejects the
        # method call even when the trainer was created with
        # enable_tensor_transport=True. Force it back on. See rdt_engine.py
        # for the matching workaround on the unsharded engine.
        self._trainer_actor._ray_enable_tensor_transport = True

        self._produce_method = getattr(
            self._trainer_actor, init_info.produce_method_name
        )
        logger.info(
            "Sharded RDT engine bound to trainer actor %r (batched method %r)",
            init_info.trainer_actor_name,
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

        # Bake a replay plan for every group now. This is a pure dry run: the
        # trainer's gather cache is empty at init, so nothing can (or does) get
        # pulled — we only record how each slice is fetched and where it lands,
        # then restore the model. Every later update_weights is a replay.
        self._bake_all_groups(init_info)

    def receive_weights(
        self,
        update_info: ShardedRDTWeightTransferUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        """Apply one group by index: replay its baked plan, or — if the group
        was found unbakeable at init — fall back to a plain batched load.

        Plans are baked once in ``init_transfer_engine`` and indexed by group,
        so every name was seen up front; ``load_weights`` is used only by the
        slow path.
        """
        if self._produce_method is None:
            raise RuntimeError(
                "Sharded RDT engine not initialized. Call init_transfer_engine() first."
            )
        idx = update_info.group_index
        if not 0 <= idx < len(self._plans):
            raise RuntimeError(
                f"group_index {idx} out of range [0, {len(self._plans)}); the "
                "init_info.group_names baked at init defines the valid indices."
            )
        plan = self._plans[idx]
        if plan is None:
            self._load_unbaked(idx, load_weights)
        else:
            self._replay(plan)

    def _build_lazy_weights(
        self,
        names: list[str],
        dtype_names: list[str],
        shapes: list[list[int]],
        ctx: "_FetchPlan | _BakeRecorder",
        device: torch.device,
    ) -> list[tuple[str, torch.Tensor]]:
        # LazyRDTTensors are zero-storage, so building them upfront is just a
        # few object allocations. ``ctx`` is the bake recorder (dry run) or the
        # slow-path fetch plan.
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

        tensors = ray.get(self._produce_method.remote(keys))
        if len(tensors) != len(keys):
            raise RuntimeError(
                f"Trainer returned {len(tensors)} tensors for {len(keys)} keys."
            )
        return dict(zip(keys, tensors))

    # ---------------- Bake (dry run, at init) / replay ----------------

    def _load_unbaked(
        self,
        group_index: int,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        """Plain batched load for a group that failed the bake coverage gate:
        rebuild its lazies from the stored names, one batched pull, then the
        deferred layer processing. No recording; runs every sync."""
        from vllm.model_executor.model_loader.reload.layerwise import (
            deferred_layerwise_processing,
            run_deferred_layer_processing,
        )

        names, dtype_names, shapes = self._group_meta[group_index]
        plan = _FetchPlan()
        device = torch.empty(0).device
        _t_recv = time.perf_counter()
        with deferred_layerwise_processing() as deferred:
            load_weights(self._build_lazy_weights(names, dtype_names, shapes, plan, device))
            _t_pull = time.perf_counter()
            plan.results = self._pull(plan.needed)
            pull_seconds = time.perf_counter() - _t_pull
            _t_proc = time.perf_counter()
            run_deferred_layer_processing(deferred)
            process_seconds = time.perf_counter() - _t_proc
        self._log_timing(
            "unbaked", time.perf_counter() - _t_recv, pull_seconds, 1, process_seconds
        )

    def _bake_all_groups(
        self, init_info: ShardedRDTWeightTransferInitInfo
    ) -> None:
        """Bake every group's replay plan once, as a pure dry run.

        Drives layerwise reload on the local model, runs ``load_weights`` over
        all groups' names against **meta** params (Pass 1: buffer + count
        numel, no data), then re-runs each completed leaf module's buffered
        loaders with a ``_BakeRecorder`` installed — capturing, per copy_, the
        source op chain and the (meta) destination's ``offset/shape/stride``
        plus the bound ``param_name``. No pull, no scatter, no kernel copy. The
        model is restored to its pre-bake weights. Results land in
        ``self._plans`` indexed by group; a group is ``None`` (slow path) if any
        of its names never flowed into a recorded copy (e.g. weights applied via
        the attention/partial-layer finalize path).
        """
        from collections import defaultdict

        from vllm.model_executor.layers.quantization.base_config import (
            QuantizeMethodBase,
        )
        from vllm.model_executor.model_loader.reload.layerwise import (
            deferred_layerwise_processing,
            initialize_layerwise_reload,
        )

        self._group_meta = list(
            zip(
                init_info.group_names,
                init_info.group_dtype_names,
                init_info.group_shapes,
            )
        )
        group_names = init_info.group_names
        if not group_names:
            self._plans = []
            return

        name_to_idx = {n: i for i, names in enumerate(group_names) for n in names}
        flat_names = [n for g in group_names for n in g]
        flat_dtypes = [d for g in init_info.group_dtype_names for d in g]
        flat_shapes = [s for g in init_info.group_shapes for s in g]

        model = self.model
        recorder = _BakeRecorder()
        groups_by_idx: dict[int, list[_BakedGroup]] = defaultdict(list)
        unbakeable: set[int] = set()

        _t0 = time.perf_counter()
        with torch.device(self.device):
            initialize_layerwise_reload(model)
            with deferred_layerwise_processing() as deferred:
                # Pass 1: route names -> modules, buffer bound_args, count numel
                # (copy_ onto meta). The recorder's param_name is unset here, so
                # nothing useful is recorded yet — _record_layer re-runs the
                # buffered loaders below with it set.
                model.load_weights(
                    self._build_lazy_weights(
                        flat_names, flat_dtypes, flat_shapes, recorder, self.device
                    )
                )
                # Record each completed leaf module exactly once. (A FusedMoE
                # re-enqueues itself per expert in deferred mode — see the
                # "excessive loading" note in online_process_loader.)
                seen: set[int] = set()
                for layer, info in deferred:
                    if id(layer) in seen:
                        continue
                    seen.add(id(layer))
                    copies = self._record_layer(layer, info, recorder)
                    idxs = {
                        name_to_idx[c.src[0]]
                        for c in copies
                        if c is not None and c.src[0] in name_to_idx
                    }
                    if any(c is None for c in copies) or len(idxs) != 1:
                        # A copy_ we couldn't record, or copies spanning >1
                        # group: don't replay any group this module touched.
                        unbakeable |= idxs
                        continue
                    needs_pwal = isinstance(
                        getattr(layer, "quant_method", None), QuantizeMethodBase
                    )
                    groups_by_idx[next(iter(idxs))].append(
                        _BakedGroup(layer=layer, copies=copies, needs_pwal=needs_pwal)
                    )
            self._restore_after_dry_run(model)

        # Finalize plans per group index. A group is replayable iff every one of
        # its names was covered by a recorded copy and no module flagged it.
        self._plans = []
        for i, names in enumerate(group_names):
            grps = groups_by_idx.get(i, [])
            covered = {c.src[0] for g in grps for c in g.copies}
            if i in unbakeable or (set(names) - covered):
                self._plans.append(None)
            else:
                self._plans.append(grps)
        n_baked = sum(1 for p in self._plans if p is not None)
        logger.info(
            "Sharded RDT dry-run baked %d/%d groups (%d -> slow path) in %.3fs",
            n_baked,
            len(group_names),
            len(group_names) - n_baked,
            time.perf_counter() - _t0,
        )

    def _record_layer(
        self, layer: Any, info: Any, recorder: "_BakeRecorder"
    ) -> "list[_BakedCopy | None]":
        """Re-run a completed leaf module's buffered loaders against its **meta**
        params with ``recorder`` installed, returning the recorded copies.

        Each loader is bound to its destination param (so the recorder knows
        ``param_name``) and run via the original (unwrapped) loader; the lazy's
        ``copy_`` appends a ``_BakedCopy`` per slice. No data moves.
        """
        from vllm.model_executor.model_loader.reload.layerwise import (
            _get_original_loader,
        )

        recorder.sink = []
        for name, bargs in info.loaded_weights:
            recorder.param_name = name
            bargs.arguments["param"] = getattr(layer, name)
            original_loader = _get_original_loader(getattr(layer, name))
            original_loader(*bargs.args, **bargs.kwargs)
        return recorder.sink

    def _restore_after_dry_run(self, model: torch.nn.Module) -> None:
        """Restore every layerwise layer to its pre-bake weights WITHOUT pulling.

        The dry run left params on meta and moved no data; calling
        ``finalize_layerwise_reload`` here would try to load attention/partial
        layers (lazy loaders -> a pull on the empty gather cache), so we place
        the saved kernel tensors back directly and reset each info.
        """
        from vllm.model_executor.model_loader.reload.layerwise import (
            LAYERWISE_INFO,
            _place_kernel_tensors,
        )

        for layer in model.modules():
            info = LAYERWISE_INFO.get(layer)
            if info is not None and info.can_load():
                if info.kernel_tensors is not None:
                    _place_kernel_tensors(layer, info)
                info.reset()
        if hasattr(model, "_original_do_torchao_reload"):
            model._do_torchao_reload = model._original_do_torchao_reload

    def _replay(self, groups: "list[_BakedGroup]") -> None:
        """Fast path: one batched pull, then scatter each baked group directly
        into freshly materialized params — no ``load_weights``, no discovery,
        no lazy-tensor dispatch.

        Mirrors ``_layerwise_process`` minus the loader replay: materialize ->
        scatter via ``as_strided`` -> ``process_weights_after_loading`` -> copy
        into persistent kernel storage -> ``info.reset()`` (the reset is what
        makes ``finalize_layerwise_reload`` skip the layer instead of clobbering
        it with the old kernel tensors).
        """
        from vllm.model_executor.layers.quantization.base_config import (
            QuantizeMethodBase,
        )
        from vllm.model_executor.model_loader.reload.layerwise import (
            LAYERWISE_INFO,
            _copy_and_restore_kernel_tensors,
        )
        from vllm.model_executor.model_loader.reload.meta import materialize_layer

        _t_recv = time.perf_counter()

        # One batched pull for every unique source slice this group needs.
        _t_pull = time.perf_counter()
        results = self._pull({c.src for g in groups for c in g.copies})
        pull_seconds = time.perf_counter() - _t_pull

        _t_proc = time.perf_counter()
        for g in groups:
            layer = g.layer
            info = LAYERWISE_INFO.get(layer)
            if info is None or not info.can_load():
                raise RuntimeError(
                    f"Baked replay: layer {type(layer).__name__} was not set up "
                    "for reload this sync (start_weight_update must run before "
                    "update_weights)."
                )
            # Materialize HF params for this leaf module (cheap empty alloc).
            materialize_layer(layer, info)
            # Scatter each recorded slice into its destination region.
            for c in g.copies:
                param = getattr(layer, c.param_name)
                dst = param.as_strided(c.shape, c.stride, c.offset)
                with torch._C.DisableTorchFunctionSubclass():
                    dst.copy_(results[c.src])
            # Quantization / repack, matching _layerwise_process.
            if g.needs_pwal:
                quant_method = getattr(layer, "quant_method", None)
                if isinstance(quant_method, QuantizeMethodBase):
                    if hasattr(layer, "_already_called_process_weights_after_loading"):
                        delattr(layer, "_already_called_process_weights_after_loading")
                    quant_method.process_weights_after_loading(layer)
            # Copy into persistent kernel storage (preserves cudagraph refs).
            if info.kernel_tensors is not None:
                _copy_and_restore_kernel_tensors(layer, info)
            # Reset so finalize_layerwise_reload skips this (already-loaded) layer.
            info.reset()
        process_seconds = time.perf_counter() - _t_proc
        self._log_timing(
            "replay", time.perf_counter() - _t_recv, pull_seconds, 1, process_seconds
        )

    @staticmethod
    def _log_timing(
        mode: str,
        total_seconds: float,
        pull_seconds: float,
        pull_calls: int,
        process_seconds: float,
    ) -> None:
        """Log a one-line timing summary for one ``receive_weights`` call.

        ``mode`` is ``replay`` or ``unbaked``. ``pull_seconds`` isolates the
        NIXL transfer; ``process_seconds`` is the scatter/materialize/quantize/
        kernel-copy work after the pull.
        """
        logger.info(
            "[RDT-TIMING] receive_weights mode=%s total=%.4fs nixl_pull=%.4fs "
            "(%d pull%s) process=%.4fs",
            mode,
            total_seconds,
            pull_seconds,
            pull_calls,
            "" if pull_calls == 1 else "s",
            process_seconds,
        )
        # vLLM workers run in an EngineCore subprocess whose logs are not
        # streamed to the driver; optionally append the timing to a file so
        # benchmarks can read the pull/process split deterministically.
        import os

        timing_file = os.environ.get("VLLM_RDT_TIMING_FILE")
        if timing_file:
            with open(timing_file, "a") as f:
                f.write(
                    f"mode={mode} total={total_seconds:.4f} "
                    f"nixl_pull={pull_seconds:.4f} pull_calls={pull_calls} "
                    f"process={process_seconds:.4f}\n"
                )

    def shutdown(self) -> None:
        self._trainer_actor = None
        self._produce_method = None
        # Drop strong references to baked modules so the model can be freed.
        self._plans.clear()
        self._group_meta.clear()

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
