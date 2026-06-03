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
from typing import Any

import torch

from vllm.config.parallel import ParallelConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)
from vllm.logger import init_logger

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
class _FetchPlan:
    """Per-bake state shared by every LazyRDTTensor in one bake load.

    Pass 1 (``online_process_loader`` -> ``get_numel_loaded``) records every
    ``(name, op_chain)`` it would copy_ onto a meta destination into
    ``needed``. ``drain`` issues one batched RPC for every chain in ``needed``
    and stores the results in ``results``. Pass 2 (replay inside
    ``_layerwise_process``) pops from ``results`` on the real copy_, and — when
    ``trace_sink`` is set — records the destination descriptor for replay.
    """

    produce_method: Any = None
    needed: set[FetchKey] = field(default_factory=set)
    results: dict[FetchKey, torch.Tensor] = field(default_factory=dict)

    # Trace mode (bake): when ``trace_sink`` is set, the Pass-2 ``copy_`` branch
    # appends a ``_BakedCopy`` (source key + destination descriptor) for every
    # materialized copy_ so the scatter can be replayed via ``param.as_strided``
    # on later syncs without re-running the loader. ``trace_layer`` is the module
    # currently being processed, used to match the copy_ destination's storage
    # back to one of its params. A ``None`` entry marks an un-matchable copy_
    # (the layer is then not bakeable and falls back to the slow path).
    trace_sink: "list[_BakedCopy | None] | None" = None
    trace_layer: Any = None

    # Timing instrumentation: cumulative wall-clock seconds spent in the
    # NIXL pull (the trainer RPC + transfer) and the number of pulls issued.
    pull_seconds: float = 0.0
    pull_calls: int = 0

    def drain(self) -> None:
        """Single batched RPC for every chain Pass 1 recorded into ``needed``."""
        if not self.needed:
            return
        needed = list(self.needed)
        import ray

        _t0 = time.perf_counter()
        tensors = ray.get(self.produce_method.remote(needed))
        self.pull_seconds += time.perf_counter() - _t0
        self.pull_calls += 1
        if len(tensors) != len(needed):
            raise RuntimeError(
                f"Trainer returned {len(tensors)} tensors for {len(needed)} "
                f"requested specs."
            )
        for key, tensor in zip(needed, tensors):
            self.results[key] = tensor
        self.needed.clear()


@dataclass
class _BakedCopy:
    """One recorded scatter: pull ``src`` from the trainer and copy it into
    ``param_name`` at the recorded strided region.

    Captured once during the bake from the Pass-2 ``copy_`` (which holds both
    the lazy source's op-chain and the materialized destination view). On every
    later sync the destination is reconstructed as
    ``param.as_strided(shape, stride, offset)`` and filled by ``copy_`` — no
    loader, no lazy tensor, no Pass-1 discovery.
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


def _match_dst_descriptor(
    dest: torch.Tensor, layer: Any
) -> tuple[str, int, tuple[int, ...], tuple[int, ...]] | None:
    """Match a ``copy_`` destination view back to one of ``layer``'s tensors.

    Returns ``(param_name, storage_offset, shape, stride)`` such that
    ``getattr(layer, param_name).as_strided(shape, stride, offset)`` reproduces
    ``dest`` exactly, or ``None`` if ``dest`` does not alias any of the layer's
    param/buffer storages — in which case the layer is not bakeable (the loader
    built its destination by some means other than a view into a registered
    param) and the engine falls back to the slow path for that group.
    """
    from vllm.model_executor.model_loader.reload.utils import get_layer_tensors

    try:
        dest_ptr = dest.untyped_storage().data_ptr()
    except (RuntimeError, ValueError):
        return None
    for name, tensor in get_layer_tensors(layer).items():
        try:
            if tensor.untyped_storage().data_ptr() == dest_ptr:
                return (
                    name,
                    dest.storage_offset(),
                    tuple(dest.shape),
                    tuple(dest.stride()),
                )
        except (RuntimeError, ValueError):
            continue
    return None


class _UnsupportedLazyOp(NotImplementedError):
    """Raised when a weight loader calls an op we don't support on a LazyRDTTensor.

    Surfaced as NotImplementedError so callers can distinguish "this backend
    can't handle this loader" from genuine bugs.
    """


class LazyRDTTensor(torch.Tensor):
    """Zero-storage tensor that defers slice fetching during a bake load.

    Built via ``_make_wrapper_subclass`` so ``.shape``/``.dtype``/``.device``/
    ``.size()``/``.dim()`` work without allocating storage. The two layerwise
    passes interact with us as follows:

    Pass 1 (buffering): every supported op (narrow/view/reshape/transpose/
    __getitem__/...) returns a new ``LazyRDTTensor`` with the spec appended
    to its chain. ``copy_`` onto a meta destination records ``(name, chain)``
    into the fetch plan and returns; no data moves. After load_weights
    returns, the engine calls ``plan.drain()`` once to fetch every recorded
    chain in a single batched RPC, populating ``plan.results``.

    Pass 2 (replay): the loader rebuilds the same chain (loader is
    deterministic over identical bound_args). ``copy_`` onto a real
    destination pops the prefetched tensor from ``plan.results``, copies it
    into the destination, then frees the materialized buffer immediately.

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
    _fetch_plan: "_FetchPlan | None"
    _materialized: "torch.Tensor | None"

    @staticmethod
    def __new__(
        cls,
        name: str,
        shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        ops: OpChain = (),
        fetch_plan: _FetchPlan | None = None,
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
        t._fetch_plan = fetch_plan
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
            fetch_plan=self._fetch_plan,
        )

    def _meta(self) -> torch.Tensor:
        """A zero-storage meta tensor of this lazy's current shape/dtype.

        Used to compute the post-op shape/dtype via PyTorch itself, which is
        more reliable than reimplementing shape inference per op. The result
        is never used for data — only its metadata.
        """
        return torch.empty(self.shape, dtype=self.dtype, device="meta")

    def _materialize(self) -> torch.Tensor:
        if self._materialized is not None:
            return self._materialized
        assert self._fetch_plan is not None
        key = self._key()
        tensor = self._fetch_plan.results.pop(key, None)
        if tensor is None:
            # The batched drain should have prefetched every recorded chain;
            # a miss means this lazy materialized outside the traced replay.
            # Fall back to a single-tensor RPC and warn so it surfaces.
            logger.warning(
                "LazyRDTTensor %r falling back to single-tensor RPC "
                "(chain=%s); batched drain missed this key.",
                self._name,
                self._ops,
            )
            import ray

            tensor = ray.get(
                self._fetch_plan.produce_method.remote([(self._name, self._ops)])
            )[0]
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
                if dest.device.type == "meta":
                    # Pass 1 over a meta-restored param. Record the chain into
                    # the plan, then let a meta-backed copy_ fire so layerwise's
                    # `CopyCounter` (a TorchDispatchMode) still counts the numel
                    # -- otherwise `info.load_numel` stays at 0 and
                    # `_layerwise_process` is never triggered, so the slice is
                    # never fetched.
                    assert src._fetch_plan is not None
                    src._fetch_plan.needed.add(src._key())
                    meta_src = torch.empty(src.shape, dtype=src.dtype, device="meta")
                    with torch._C.DisableTorchFunctionSubclass():
                        return dest.copy_(meta_src)
                # Pass 2 replay onto the materialized param. Pop the
                # prefetched tensor, copy it in, free immediately.
                mat = src._materialize()
                with torch._C.DisableTorchFunctionSubclass():
                    result = dest.copy_(mat)
                # Bake: record (source key -> destination descriptor) so later
                # syncs replay this scatter directly via param.as_strided,
                # skipping the loader and Pass-1 discovery entirely.
                plan = src._fetch_plan
                if plan is not None and plan.trace_sink is not None:
                    desc = _match_dst_descriptor(dest, plan.trace_layer)
                    if desc is None:
                        plan.trace_sink.append(None)
                    else:
                        name, offset, shape, stride = desc
                        plan.trace_sink.append(
                            _BakedCopy(src._key(), name, offset, shape, stride)
                        )
                # Release the prefetched buffer as soon as the data lives in
                # the destination param. See "Memory contract" in the design.
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
    """Update info for the sharded RDT backend."""

    names: list[str]
    """Parameter names to fetch, in iteration order."""

    dtype_names: list[str]
    """Dtype for each parameter (e.g. 'bfloat16'). Used to build the lazy
    placeholders before fetch."""

    shapes: list[list[int]]
    """Full HF-shape for each parameter. Used to build the lazy placeholders."""

    def __post_init__(self) -> None:
        n = len(self.names)
        if len(self.dtype_names) != n:
            raise ValueError(
                f"`dtype_names` should be of the same size as `names`: "
                f"got {len(self.dtype_names)} and {n}"
            )
        if len(self.shapes) != n:
            raise ValueError(
                f"`shapes` should be of the same size as `names`: "
                f"got {len(self.shapes)} and {n}"
            )


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
      - ``is_checkpoint_format=True`` (layerwise reload). Raises in
        ``receive_weights`` if the kernel-format path was selected.
      - Weight loaders that only use the supported op set (narrow, view,
        reshape, transpose, t, permute, __getitem__ with int/slice/tuple,
        unsqueeze, squeeze, flatten, contiguous, chunk, copy_). Loaders
        that need .to(), .float(), .item(), .data, bool-mask indexing, or
        arithmetic on the loaded weight will raise from
        ``__torch_dispatch__`` and should fall back to backend='rdt'.

    The first ``update_weights`` for a given name set *bakes* a replay plan
    (and loads correctly); every later one *replays* it. See the module
    docstring and ``baked_rdt_replay.md``.
    """

    init_info_cls = ShardedRDTWeightTransferInitInfo
    update_info_cls = ShardedRDTWeightTransferUpdateInfo

    def __init__(
        self, config: WeightTransferConfig, parallel_config: ParallelConfig
    ) -> None:
        super().__init__(config, parallel_config)
        self._trainer_actor: Any | None = None
        self._produce_method: Any | None = None
        # Keyed by tuple(update_info.names). Value is the baked group list for
        # that name set, or None once the set is found unbakeable (slow path).
        self._baked_plans: dict[tuple[str, ...], "list[_BakedGroup] | None"] = {}

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

    def receive_weights(
        self,
        update_info: ShardedRDTWeightTransferUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        """Bake on first sight of a name set, replay after.

        Keyed by ``tuple(update_info.names)`` so the (driver-stable) per-layer
        call boundary maps each call to its own plan, order-independently. A
        name set found unbakeable stores ``None`` and takes the plain batched
        load every sync.
        """
        if self._produce_method is None:
            raise RuntimeError(
                "Sharded RDT engine not initialized. Call init_transfer_engine() first."
            )

        key = tuple(update_info.names)
        if key not in self._baked_plans:
            # First sight: bake (a correct, real load plus recording). Stores
            # None if the group turns out not to be replayable.
            self._baked_plans[key] = self._do_bake(update_info, load_weights)
            return
        plan = self._baked_plans[key]
        if plan is None:
            self._load_unbaked(update_info, load_weights)
        else:
            self._replay(update_info, plan)

    def _build_lazy_weights(
        self,
        update_info: ShardedRDTWeightTransferUpdateInfo,
        plan: "_FetchPlan",
        device: torch.device,
    ) -> list[tuple[str, torch.Tensor]]:
        # The base ``load_weights`` callable is typed as taking a ``list``,
        # and LazyRDTTensors are zero-storage so building them upfront is
        # just a few object allocations.
        return [
            (
                name,
                LazyRDTTensor(
                    name=name,
                    shape=torch.Size(shape),
                    dtype=_dtype_from_name(dtype_name),
                    device=device,
                    fetch_plan=plan,
                ),
            )
            for name, dtype_name, shape in zip(
                update_info.names,
                update_info.dtype_names,
                update_info.shapes,
            )
        ]

    # ---------------- Bake / replay ----------------

    def _load_unbaked(
        self,
        update_info: ShardedRDTWeightTransferUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        """Plain batched load for name sets that failed the bake coverage gate:
        one batched pull for the whole call, then the deferred layer processing.
        No recording, runs every sync."""
        from vllm.model_executor.model_loader.reload.layerwise import (
            deferred_layerwise_processing,
            run_deferred_layer_processing,
        )

        plan = _FetchPlan(produce_method=self._produce_method)
        device = torch.empty(0).device
        _t_recv = time.perf_counter()
        with deferred_layerwise_processing() as deferred:
            load_weights(self._build_lazy_weights(update_info, plan, device))
            plan.drain()
            _t_proc = time.perf_counter()
            run_deferred_layer_processing(deferred)
            process_seconds = time.perf_counter() - _t_proc
        self._log_timing(
            "unbaked", plan, time.perf_counter() - _t_recv, process_seconds
        )

    def _do_bake(
        self,
        update_info: ShardedRDTWeightTransferUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> "list[_BakedGroup] | None":
        """Run one real (payload-batch) load while recording a replay plan.

        Produces correct weights *and* returns the per-leaf-module
        ``_BakedGroup`` list — or ``None`` if the group is not safely
        replayable (an un-matchable copy_ destination, or requested names not
        fully covered by recorded copies, e.g. weights that flow through the
        attention/partial-layer finalize path).
        """
        from vllm.model_executor.layers.quantization.base_config import (
            QuantizeMethodBase,
        )
        from vllm.model_executor.model_loader.reload.layerwise import (
            LAYERWISE_INFO,
            _layerwise_process,
            deferred_layerwise_processing,
        )

        if not any(info.can_load() for info in LAYERWISE_INFO.values()):
            raise RuntimeError(
                "ShardedRDTWeightTransferEngine requires layerwise mode "
                "(is_checkpoint_format=True). For the kernel-format path, "
                "use backend='rdt' instead."
            )

        plan = _FetchPlan(produce_method=self._produce_method)
        device = torch.empty(0).device

        groups: list[_BakedGroup] = []
        bakeable = True
        _t_recv = time.perf_counter()
        with deferred_layerwise_processing() as deferred:
            load_weights(self._build_lazy_weights(update_info, plan, device))
            plan.drain()
            _t_proc = time.perf_counter()
            # A layer can appear in the deferred queue more than once: in
            # deferred mode a module whose load crosses load_numel_total early
            # (FusedMoE, some quant configs — see online_process_loader's
            # "excessive loading" note) re-enqueues itself on every subsequent
            # loader call. The first _layerwise_process replays *all* of its
            # accumulated loaded_weights and resets the info, so process each
            # layer exactly once; later duplicates would be no-ops (and would
            # double-reset on replay).
            seen: set[int] = set()
            for layer, info in deferred:
                if id(layer) in seen:
                    continue
                seen.add(id(layer))
                sink: list[_BakedCopy | None] = []
                plan.trace_sink = sink
                plan.trace_layer = layer
                try:
                    # Pass 2: replays the buffered loaders onto the materialized
                    # params; each copy_ appends its (src, dst) to ``sink``.
                    _layerwise_process(layer, info)
                finally:
                    plan.trace_sink = None
                    plan.trace_layer = None
                if any(c is None for c in sink):
                    # A copy_ whose destination didn't alias a registered param
                    # — this layer can't be replayed by as_strided.
                    bakeable = False
                    continue
                needs_pwal = isinstance(
                    getattr(layer, "quant_method", None), QuantizeMethodBase
                )
                groups.append(
                    _BakedGroup(
                        layer=layer,
                        copies=[c for c in sink if c is not None],
                        needs_pwal=needs_pwal,
                    )
                )
            process_seconds = time.perf_counter() - _t_proc
        self._log_timing("bake", plan, time.perf_counter() - _t_recv, process_seconds)

        # Coverage gate: every requested name must flow into a recorded copy.
        # Uncovered names load via the finalize path (attention scales / padded
        # layers) which replay does not reproduce, so we fall back fail-closed.
        covered = {c.src[0] for g in groups for c in g.copies}
        uncovered = set(update_info.names) - covered
        if uncovered or not bakeable:
            logger.warning(
                "Sharded RDT bake: group not replayable "
                "(uncovered_names=%d, unmatched_copy=%s); using slow path. "
                "Example uncovered: %s",
                len(uncovered),
                not bakeable,
                sorted(uncovered)[:3],
            )
            return None
        logger.info(
            "Sharded RDT baked %d names -> %d leaf groups, %d copies (replay armed)",
            len(update_info.names),
            len(groups),
            sum(len(g.copies) for g in groups),
        )
        return groups

    def _replay(
        self,
        update_info: ShardedRDTWeightTransferUpdateInfo,
        groups: "list[_BakedGroup]",
    ) -> None:
        """Fast path: one batched pull, then scatter each baked group directly
        into freshly materialized params — no ``load_weights``, no Pass-1
        discovery, no lazy-tensor dispatch.

        Mirrors ``_layerwise_process`` minus the loader replay: materialize ->
        scatter via ``as_strided`` -> ``process_weights_after_loading`` ->
        copy into persistent kernel storage -> ``info.reset()`` (the reset is
        what makes ``finalize_layerwise_reload`` skip the layer instead of
        clobbering it with the old kernel tensors).
        """
        import ray

        from vllm.model_executor.layers.quantization.base_config import (
            QuantizeMethodBase,
        )
        from vllm.model_executor.model_loader.reload.layerwise import (
            LAYERWISE_INFO,
            _copy_and_restore_kernel_tensors,
        )
        from vllm.model_executor.model_loader.reload.meta import materialize_layer

        _t_recv = time.perf_counter()

        # One batched pull for every unique source slice this call needs.
        keys = sorted({c.src for g in groups for c in g.copies})
        _t_pull = time.perf_counter()
        tensors = ray.get(self._produce_method.remote(keys))
        pull_seconds = time.perf_counter() - _t_pull
        if len(tensors) != len(keys):
            raise RuntimeError(
                f"Trainer returned {len(tensors)} tensors for {len(keys)} keys."
            )
        results = dict(zip(keys, tensors))

        _t_proc = time.perf_counter()
        for g in groups:
            layer = g.layer
            info = LAYERWISE_INFO.get(layer)
            if info is None or not info.can_load():
                raise RuntimeError(
                    f"Baked replay: layer {type(layer).__name__} was not set up "
                    "for reload this sync (start_weight_update must run before "
                    "update_weights, and each baked layer must be processed at "
                    "most once per replay)."
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

        # Report timing through the same channel as the slow path. Synthesize a
        # plan-shaped record so _log_timing can read pull stats uniformly.
        plan = _FetchPlan(produce_method=self._produce_method)
        plan.pull_seconds = pull_seconds
        plan.pull_calls = 1
        self._log_timing(
            "replay", plan, time.perf_counter() - _t_recv, process_seconds
        )

    @staticmethod
    def _log_timing(
        mode: str, plan: "_FetchPlan", total_seconds: float, process_seconds: float
    ) -> None:
        """Log a one-line timing summary for one ``receive_weights`` call.

        ``mode`` is one of ``bake`` / ``replay`` / ``unbaked``. ``pull_seconds``
        isolates the NIXL transfer; ``process_seconds`` is the
        scatter/materialize/quantize/kernel-copy work after the pull.
        """
        logger.info(
            "[RDT-TIMING] receive_weights mode=%s total=%.4fs nixl_pull=%.4fs "
            "(%d pull%s) deferred_process=%.4fs",
            mode,
            total_seconds,
            plan.pull_seconds,
            plan.pull_calls,
            "" if plan.pull_calls == 1 else "s",
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
                    f"nixl_pull={plan.pull_seconds:.4f} pull_calls={plan.pull_calls} "
                    f"deferred_process={process_seconds:.4f}\n"
                )

    def shutdown(self) -> None:
        self._trainer_actor = None
        self._produce_method = None
        # Drop strong references to baked modules so the model can be freed.
        self._baked_plans.clear()

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
