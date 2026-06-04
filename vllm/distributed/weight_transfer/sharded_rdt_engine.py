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
      - ``is_checkpoint_format=True`` (layerwise reload). ``_bake_all_groups``
        raises at init if no layerwise infos are active.
      - Weight loaders that only use the supported op set (narrow, view,
        reshape, transpose, t, permute, __getitem__ with int/slice/tuple,
        unsqueeze, squeeze, flatten, contiguous, chunk, copy_). Loaders
        that need .to(), .float(), .item(), .data, bool-mask indexing, or
        arithmetic on the loaded weight will land in ``__torch_dispatch__``
        during the bake (raising) and should fall back to backend='rdt'.

    The plan is baked once at ``init_transfer_engine`` (a meta dry run over
    ``init_info.names``) into one ``_BakedGroup`` per fully-loaded leaf module,
    indexed by source name. Every ``update_weights`` *replays* the leaf modules
    its gathered names cover. See the module docstring and ``baked_rdt_replay.md``.
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
        # Baked plan: source name -> the _BakedGroup (leaf module) that consumes
        # it. Several names of one fused module map to the same group; replay
        # dedups. A name absent here isn't baked (attention scale / padded /
        # partial) and takes the plain load.
        self._name_to_group: dict[str, _BakedGroup] = {}
        # name -> (dtype_name, shape) for every init name, so the plain-load
        # fallback can rebuild lazies from just the gathered names.
        self._name_meta: dict[str, tuple[str, list[int]]] = {}

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

        # Bake the replay plan now. This is a pure dry run: the trainer's gather
        # cache is empty at init, so nothing can (or does) get pulled — we only
        # record how each slice is fetched and where it lands, then restore the
        # model. Every later update_weights is a replay.
        self._bake(init_info)

    def receive_weights(
        self,
        update_info: ShardedRDTWeightTransferUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        """Replay the baked leaf modules the call's gathered names cover.

        ``update_info.names`` is the subset of the init names the driver
        gathered for this call. We replay every baked ``_BakedGroup`` those
        names reach (deduped). If any name has no baked plan — attention scales,
        padded/partial layers — we fall back to a plain batched load of the
        whole call (correct, just not accelerated). ``load_weights`` is used
        only by that slow path.

        Assumes each baked module's source names are gathered within a single
        call (true for the per-layer / pre / post partition); if not, ``_pull``
        would fail loudly on the missing slice rather than load wrong data.
        """
        if self._produce_method is None:
            raise RuntimeError(
                "Sharded RDT engine not initialized. Call init_transfer_engine() first."
            )
        names = update_info.names
        groups: list[_BakedGroup] = []
        seen: set[int] = set()
        all_covered = True
        for n in names:
            g = self._name_to_group.get(n)
            if g is None:
                all_covered = False
            elif id(g) not in seen:
                seen.add(id(g))
                groups.append(g)
        if all_covered:
            self._replay(groups)
        else:
            self._load_unbaked(names, load_weights)

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

        n_groups = len({id(g) for g in self._name_to_group.values()})
        logger.info(
            "Sharded RDT dry-run baked %d/%d names into %d leaf modules in %.3fs",
            len(self._name_to_group),
            len(names),
            n_groups,
            time.perf_counter() - _t0,
        )

    def _install_recording_stamps(
        self, model: torch.nn.Module, recorder: "_BakeRecorder"
    ) -> None:
        """Wrap each loadable param's ``weight_loader`` so it stamps
        ``recorder.current = (leaf_module, param_name)`` before delegating.

        Engine-side monkeypatch (no vLLM edit): ``initialize_layerwise_reload``
        set each param's loader to ``online_process_loader``; we wrap the
        **original** loader underneath it (via ``_get_original_loader``) so the
        single load pass runs the original loaders directly — bypassing
        ``online_process_loader`` and thus its inline ``_layerwise_process`` —
        while the lazy's ``copy_`` reads the destination param from
        ``recorder.current``. The stamps live on the meta params and are dropped
        when ``_restore_after_dry_run`` re-registers the kernel tensors, so no
        cleanup is needed. See the FUTURE note in ``_bake`` for the clean
        alternative (a public "currently-loading param" layerwise hook).
        """
        from vllm.model_executor.model_loader.reload.layerwise import (
            _get_original_loader,
        )
        from vllm.model_executor.model_loader.reload.utils import get_layer_tensors

        def _make_stamp(layer, name, inner):
            def stamp(*args, **kwargs):
                recorder.current = (layer, name)
                try:
                    return inner(*args, **kwargs)
                finally:
                    recorder.current = None

            return stamp

        for module in model.modules():
            for name, tensor in get_layer_tensors(module).items():
                if getattr(tensor, "weight_loader", None) is None:
                    continue
                # Bypass online_process_loader: stamp the *original* loader.
                original = _get_original_loader(tensor)
                tensor.weight_loader = _make_stamp(module, name, original)

    def _restore_after_dry_run(self, model: torch.nn.Module) -> None:
        """Restore every layerwise layer to its pre-bake weights WITHOUT pulling.

        The dry run left params on meta and moved no data; calling
        ``finalize_layerwise_reload`` here would try to load attention/partial
        layers (lazy loaders -> a pull on the empty gather cache) and would
        materialize real params, so we place the saved kernel tensors back
        directly and reset each info. See the FUTURE note in ``_bake`` — a
        public ``abort_layerwise_reload`` would replace this.
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
        """Fast path: one batched pull, then scatter each baked slice into its
        destination param — no ``load_weights``, no discovery, no lazy-tensor
        dispatch.

        Mirrors ``_layerwise_process`` minus the loader replay: materialize ->
        scatter via ``as_strided().copy_()`` -> ``process_weights_after_loading``
        -> copy into persistent kernel storage -> ``info.reset()`` (the reset is
        what makes ``finalize_layerwise_reload`` skip the layer instead of
        clobbering it with the old kernel tensors).

        FUTURE: today we pull into Ray-allocated buffers and scatter with
        ``copy_`` because Ray's NIXL receiver allocates the destination. On Ray
        >= 2.55.1, ``ray.experimental.set_target_for_ref(ref, [dst_views])`` lets
        NIXL read each slice **straight into** the pre-materialized destination
        view — dropping the intermediate buffer and the scatter copy. Wire that
        in once we're on a Ray version that has it.
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
            materialize_layer(layer, info)  # cheap empty HF params
            for c in g.copies:
                param = getattr(layer, c.param_name)
                dst = param.as_strided(c.shape, c.stride, c.offset)
                with torch._C.DisableTorchFunctionSubclass():
                    dst.copy_(results[c.src])
            # Quantization / repack, exactly as _layerwise_process: run it iff
            # the layer has a QuantizeMethodBase quant method (a no-op for
            # unquantized layers, real work for quantized ones).
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
        self._name_to_group.clear()
        self._name_meta.clear()

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
