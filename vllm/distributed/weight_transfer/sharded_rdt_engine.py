# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sharded Ray Direct Transport (RDT) weight transfer engine.

Unlike the unsharded RDT engine, this backend pulls only the *slice* that
each vLLM worker actually consumes, not the full HF-format tensor. It does
this by handing ``model.load_weights`` ``LazyRDTTensor`` placeholders that
defer materialization. The placeholders intercept a whitelisted set of
view/slice ops into a single ordered op chain; the actual transfer happens
in one batched RPC per layer via ``LayerReloadingInfo.pre_replay_hook`` —
the trainer replays the chain on its live parameter and ships only the
resulting slice.

Only valid with ``is_checkpoint_format=True`` (layerwise reload). See
``sharded_weight_loader_rdt.md`` for the full design and the spike in
``nixl_slice_spike.py`` confirming NIXL is view-aware.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from vllm.config.parallel import ParallelConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.reload.types import LayerReloadingInfo

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
    """Per-receive_weights state shared by every LazyRDTTensor in the call.

    When ``layerwise_batch`` is True (default):
        Pass 1 (``online_process_loader`` -> ``get_numel_loaded``) records every
        ``(name, op_chain)`` it would copy_ onto a meta destination into
        ``needed``. ``drain`` (bound as the layer's pre-replay hook) issues one
        batched RPC for every chain in ``needed`` and stores the results in
        ``results``. Pass 2 (replay inside ``_layerwise_process``) pops from
        ``results`` on the real copy_.

    When ``layerwise_batch`` is False:
        ``needed``/``results`` stay empty and no hook is installed. Pass 2's
        ``copy_`` materializes each lazy by issuing one single-tensor RPC
        right then. Simpler control flow at the cost of one round-trip per
        slice instead of one per layer.
    """

    produce_method: Any = None
    needed: set[FetchKey] = field(default_factory=set)
    results: dict[FetchKey, torch.Tensor] = field(default_factory=dict)
    layerwise_batch: bool = True

    def drain(self, info: "LayerReloadingInfo | None" = None) -> None:
        """Single batched RPC for every chain pass 1 recorded.

        Bound as ``LayerReloadingInfo.pre_replay_hook`` so the layerwise
        replay path triggers it before materializing the layer. ``info``
        matches the hook signature but is unused — the chains we need are
        already in ``self.needed``.
        """
        del info
        if not self.needed:
            return
        needed = list(self.needed)
        import ray

        tensors = ray.get(self.produce_method.remote(needed))
        if len(tensors) != len(needed):
            raise RuntimeError(
                f"Trainer returned {len(tensors)} tensors for {len(needed)} "
                f"requested specs."
            )
        for key, tensor in zip(needed, tensors):
            self.results[key] = tensor
        self.needed.clear()


class _UnsupportedLazyOp(NotImplementedError):
    """Raised when a weight loader calls an op we don't support on a LazyRDTTensor.

    Surfaced as NotImplementedError so callers can distinguish "this backend
    can't handle this loader" from genuine bugs.
    """


class LazyRDTTensor(torch.Tensor):
    """Zero-storage tensor that defers slice fetching to a layer-boundary RPC.

    Built via ``_make_wrapper_subclass`` so ``.shape``/``.dtype``/``.device``/
    ``.size()``/``.dim()`` work without allocating storage. The two layerwise
    passes interact with us as follows:

    Pass 1 (buffering): every supported op (narrow/view/reshape/transpose/
    __getitem__/...) returns a new ``LazyRDTTensor`` with the spec appended
    to its chain. ``copy_`` onto a meta destination records ``(name, chain)``
    into the fetch plan and returns; no data moves.

    Pre-replay hook: batches every chain seen this layer into one RPC,
    populates ``plan.results``.

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
            # No prefetched slice. Two cases:
            #   1. layerwise_batch=True but the hook missed this key (e.g.
            #      lazy materialized outside the layerwise replay). Warn so
            #      the miss surfaces in tests.
            #   2. layerwise_batch=False — the per-copy RPC IS the design.
            #      Silent.
            if self._fetch_plan.layerwise_batch:
                logger.warning(
                    "LazyRDTTensor %r falling back to single-tensor RPC "
                    "(chain=%s); pre_replay_hook missed this key.",
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
                    # Pass 1 over a meta-restored param. Record the chain
                    # into the plan when batching, then let a meta-backed
                    # copy_ fire so layerwise's `CopyCounter` (a
                    # TorchDispatchMode) still counts the numel -- otherwise
                    # `info.load_numel` stays at 0 and `_layerwise_process`
                    # is never triggered, so the hook never fires and no
                    # weights are fetched. When batching is disabled, skip
                    # the record: there's no drain that would consume it.
                    assert src._fetch_plan is not None
                    if src._fetch_plan.layerwise_batch:
                        src._fetch_plan.needed.add(src._key())
                    meta_src = torch.empty(src.shape, dtype=src.dtype, device="meta")
                    with torch._C.DisableTorchFunctionSubclass():
                        return dest.copy_(meta_src)
                # Pass 2 replay onto the materialized param. Pop the
                # prefetched tensor, copy it in, free immediately.
                mat = src._materialize()
                with torch._C.DisableTorchFunctionSubclass():
                    result = dest.copy_(mat)
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

    layerwise_batch: bool = True
    """Coalesce all of a layer's slice fetches into one RPC at the layer
    boundary (default). When False, each ``copy_`` issues its own
    single-tensor RPC — simpler control flow at the cost of one round-trip
    per slice instead of one per layer. The trainer-side producer signature
    is identical in both modes (a list of specs); batched mode just sends
    larger lists less often."""


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

    Set ``init_info.layerwise_batch=False`` to disable the layer-boundary
    batching and issue one RPC per slice.
    """

    init_info_cls = ShardedRDTWeightTransferInitInfo
    update_info_cls = ShardedRDTWeightTransferUpdateInfo

    def __init__(
        self, config: WeightTransferConfig, parallel_config: ParallelConfig
    ) -> None:
        super().__init__(config, parallel_config)
        self._trainer_actor: Any | None = None
        self._produce_method: Any | None = None
        self._layerwise_batch: bool = True

    def init_transfer_engine(self, init_info: ShardedRDTWeightTransferInitInfo) -> None:
        """Resolve the trainer actor and bind its batched producer method."""
        self._layerwise_batch = init_info.layerwise_batch
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

    def receive_weights(
        self,
        update_info: ShardedRDTWeightTransferUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        """Drive ``model.load_weights`` with lazy placeholders, batching
        each layer's slice fetches into one RPC via the pre-replay hook."""
        if self._produce_method is None:
            raise RuntimeError(
                "Sharded RDT engine not initialized. Call init_transfer_engine() first."
            )

        # Deferred import: pulling LAYERWISE_INFO from reload.layerwise
        # transitively imports attention layers, which need the compiled
        # flash-attn extensions. Keep the import inside receive_weights so
        # the engine module itself stays importable in lightweight envs
        # (matches the lazy-loading the factory already does).
        from vllm.model_executor.model_loader.reload.layerwise import (
            LAYERWISE_INFO,
        )

        layerwise_infos = [info for info in LAYERWISE_INFO.values() if info.can_load()]
        if not layerwise_infos:
            raise RuntimeError(
                "ShardedRDTWeightTransferEngine requires layerwise mode "
                "(is_checkpoint_format=True). For the kernel-format path, "
                "use backend='rdt' instead."
            )

        plan = _FetchPlan(
            produce_method=self._produce_method,
            layerwise_batch=self._layerwise_batch,
        )

        # Save/restore so we don't permanently mutate global LAYERWISE_INFO
        # state if a caller later reuses these infos with a different engine.
        # In non-batching mode we still walk the infos (and save empties) so
        # the finally-clause is symmetric — installing nothing, restoring
        # nothing.
        saved_hooks: list[tuple[Any, Any]] = []
        if self._layerwise_batch:
            for info in layerwise_infos:
                saved_hooks.append((info, info.pre_replay_hook))
                info.pre_replay_hook = plan.drain

        # The worker has already entered `with torch.device(self.device):`
        # before calling receive_weights, so torch.empty's device is the
        # correct GPU for this worker.
        device = torch.empty(0).device

        try:
            # The base ``load_weights`` callable is typed as taking a
            # ``list``, and LazyRDTTensors are zero-storage so the upfront
            # materialization here is just a few object allocations.
            lazy_weights: list[tuple[str, torch.Tensor]] = [
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
            load_weights(lazy_weights)
        finally:
            for info, prior in saved_hooks:
                info.pre_replay_hook = prior

    def shutdown(self) -> None:
        self._trainer_actor = None
        self._produce_method = None

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
