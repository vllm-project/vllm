# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sharded Ray Direct Transport (RDT) weight transfer engine.

Unlike the unsharded RDT engine, this backend pulls only the *slice* that
each vLLM worker actually consumes, not the full HF-format tensor. It does
this by handing ``model.load_weights`` ``LazyRDTTensor`` placeholders that
defer materialization. The placeholders intercept ``narrow``/view ops to
record the slice the loader will request, and the actual transfer happens
in a single batched RPC per layer via ``LayerReloadingInfo.pre_replay_hook``.

Only valid with ``is_checkpoint_format=True`` (layerwise reload). See
``sharded_weight_loader_rdt.md`` for the full design and the spike in
``nixl_slice_spike.py`` confirming NIXL is view-aware.
"""

import functools
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

NarrowSpec = tuple[int, int, int]  # (dim, start, size)
NarrowChain = tuple[NarrowSpec, ...]
FetchKey = tuple[str, NarrowChain]


@dataclass
class _FetchPlan:
    """Per-receive_weights state shared by every LazyRDTTensor in the call.

    Pass 1 (``online_process_loader`` -> ``get_numel_loaded``) records every
    ``(name, narrow_chain)`` it would copy_ onto a meta destination into
    ``needed``. The pre-replay hook drains ``needed`` into a single batched
    RPC and stores the results in ``results``. Pass 2 (replay inside
    ``_layerwise_process``) pops from ``results`` on the real copy_.
    """

    produce_method: Any = None
    needed: set[FetchKey] = field(default_factory=set)
    results: dict[FetchKey, torch.Tensor] = field(default_factory=dict)


class LazyRDTTensor(torch.Tensor):
    """Zero-storage tensor that defers slice fetching to a layer-boundary RPC.

    Built via ``_make_wrapper_subclass`` so ``.shape``/``.dtype``/``.device``/
    ``.size()``/``.dim()`` work without allocating storage. The two layerwise
    passes interact with us as follows:

    Pass 1 (buffering): ``narrow`` returns a new ``LazyRDTTensor`` with the
    spec appended to its chain. ``copy_`` onto a meta destination records
    ``(name, chain)`` into the fetch plan and returns; no data moves.

    Pre-replay hook: batches every chain seen this layer into one RPC,
    populates ``plan.results``.

    Pass 2 (replay): ``narrow`` rebuilds the same chain (loader is
    deterministic over identical bound_args). ``copy_`` onto a real
    destination pops the prefetched tensor from ``plan.results``, copies it
    into the destination, then frees the materialized buffer immediately.
    """

    # Declared at class scope so mypy can infer attribute types on
    # instances built via ``_make_wrapper_subclass`` (where ``__new__``
    # returns a tensor it can't annotate as ``self``).
    _name: str
    _narrow_specs: tuple[NarrowSpec, ...]
    _post_view_ops: tuple[tuple[Callable, tuple, dict], ...]
    _fetch_plan: "_FetchPlan | None"
    _materialized: "torch.Tensor | None"

    @staticmethod
    def __new__(
        cls,
        name: str,
        shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        narrow_specs: tuple[NarrowSpec, ...] = (),
        post_view_ops: tuple[tuple[Callable, tuple, dict], ...] = (),
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
        t._narrow_specs = tuple(narrow_specs)
        t._post_view_ops = tuple(post_view_ops)
        t._fetch_plan = fetch_plan
        t._materialized = None
        return t

    def _key(self) -> FetchKey:
        return (self._name, self._narrow_specs)

    def _make_child(
        self,
        new_shape: torch.Size,
        extra_narrow: NarrowSpec | None = None,
        extra_view: tuple[Callable, tuple, dict] | None = None,
    ) -> "LazyRDTTensor":
        return LazyRDTTensor(
            name=self._name,
            shape=new_shape,
            dtype=self.dtype,
            device=self.device,
            narrow_specs=(
                self._narrow_specs + (extra_narrow,)
                if extra_narrow is not None
                else self._narrow_specs
            ),
            post_view_ops=(
                self._post_view_ops + (extra_view,)
                if extra_view is not None
                else self._post_view_ops
            ),
            fetch_plan=self._fetch_plan,
        )

    def _materialize(self) -> torch.Tensor:
        if self._materialized is not None:
            return self._materialized
        assert self._fetch_plan is not None
        key = self._key()
        tensor = self._fetch_plan.results.pop(key, None)
        if tensor is None:
            # Fallback path: pre-replay hook didn't pre-fetch this chain
            # (e.g. the lazy is being materialized outside the layerwise
            # replay, or the hook was never installed). Slow single-tensor
            # RPC. Should be rare; logged so it surfaces in test runs.
            logger.warning(
                "LazyRDTTensor %r falling back to single-tensor RPC "
                "(chain=%s); pre_replay_hook missed this key.",
                self._name,
                self._narrow_specs,
            )
            import ray

            spec_list = [list(s) for s in self._narrow_specs]
            tensor = ray.get(
                self._fetch_plan.produce_method.remote([(self._name, spec_list)])
            )[0]
        for op, op_args, op_kwargs in self._post_view_ops:
            tensor = op(tensor, *op_args, **op_kwargs)
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

        # narrow: defer, extend the chain.
        if func is torch.Tensor.narrow:
            self_, dim, start, size = args
            if isinstance(self_, cls) and self_._materialized is None:
                new_shape = list(self_.shape)
                new_shape[dim] = size
                return self_._make_child(
                    torch.Size(new_shape),
                    extra_narrow=(dim, start, size),
                )

        # view / reshape: defer, apply locally after materialize.
        if func in (torch.Tensor.view, torch.Tensor.reshape):
            self_ = args[0]
            if isinstance(self_, cls) and self_._materialized is None:
                rest = args[1:]
                if len(rest) == 1 and isinstance(rest[0], (tuple, list, torch.Size)):
                    new_shape = torch.Size(rest[0])
                else:
                    new_shape = torch.Size(rest)
                # Resolve a single -1 against numel.
                if any(d == -1 for d in new_shape):
                    known = 1
                    for d in new_shape:
                        if d != -1:
                            known *= d
                    inferred = self_.numel() // known
                    new_shape = torch.Size(
                        inferred if d == -1 else d for d in new_shape
                    )
                return self_._make_child(
                    new_shape,
                    extra_view=(func, rest, kwargs),
                )

        # __getitem__ with a single integer key: defer (drops leading dim).
        if func is torch.Tensor.__getitem__:
            self_, key = args
            if (
                isinstance(self_, cls)
                and self_._materialized is None
                and isinstance(key, int)
            ):
                new_shape = self_.shape[1:]
                return self_._make_child(
                    new_shape,
                    extra_view=(func, (key,), kwargs),
                )

        # copy_: this is the data sink. Two cases:
        #  - dest on meta: pass 1 over a meta-restored param. Record the
        #    chain into the plan, then let a meta-backed copy_ fire so
        #    layerwise's `CopyCounter` (a TorchDispatchMode) still counts
        #    the numel -- otherwise `info.load_numel` stays at 0 and
        #    `_layerwise_process` is never triggered, so the hook never
        #    fires and no weights are fetched.
        #  - dest on device: pass 2 replay onto the materialized param. Pop
        #    the prefetched tensor, copy it in, free immediately.
        if func is torch.Tensor.copy_:
            dest = args[0]
            src = args[1] if len(args) > 1 else kwargs.get("src")
            if isinstance(src, cls):
                if dest.device.type == "meta":
                    assert src._fetch_plan is not None
                    src._fetch_plan.needed.add(src._key())
                    meta_src = torch.empty(src.shape, dtype=src.dtype, device="meta")
                    with torch._C.DisableTorchFunctionSubclass():
                        return dest.copy_(meta_src)
                mat = src._materialize()
                with torch._C.DisableTorchFunctionSubclass():
                    result = dest.copy_(mat)
                # Release the prefetched buffer as soon as the data lives in
                # the destination param. See "Memory contract" in the design.
                src._materialized = None
                return result

        # Fallthrough: for anything else (metadata reads, arithmetic, item,
        # etc.) drop the subclass dispatch so the C++ op runs directly.
        # Metadata reads work because _make_wrapper_subclass already stored
        # the right sizes/dtype/device. Ops that actually need data fall
        # through to __torch_dispatch__ below, which materializes lazy args.
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        def _mat(x):
            return x._materialize() if isinstance(x, cls) else x

        new_args = tuple(_mat(a) for a in args)
        new_kwargs = {k: _mat(v) for k, v in kwargs.items()}
        return func(*new_args, **new_kwargs)


def _pre_replay_hook(info: "LayerReloadingInfo", plan: _FetchPlan) -> None:
    """Single batched RPC per layer for every chain pass 1 recorded.

    Pass 1 records into ``plan.needed`` while buffering. By the time
    ``_layerwise_process`` calls us, ``plan.needed`` holds exactly the
    chains this layer's replay will copy_. We submit them all at once;
    pass 2's copy_ pops from ``plan.results``.
    """
    if not plan.needed:
        return
    needed = list(plan.needed)
    specs = [(name, list(chain)) for name, chain in needed]
    import ray

    tensors = ray.get(plan.produce_method.remote(specs))
    if len(tensors) != len(needed):
        raise RuntimeError(
            f"Trainer returned {len(tensors)} tensors for {len(needed)} "
            f"requested specs."
        )
    for key, tensor in zip(needed, tensors):
        plan.results[key] = tensor
    plan.needed.clear()


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
    ``@ray.method(tensor_transport="nixl")``. Signature:
    ``(specs: list[tuple[str, list[tuple[int, int, int]]]]) -> list[Tensor]``.
    """


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
        that takes a list of ``(name, narrow_chain)`` specs and returns a
        list of slice tensors.
      - ``nixl`` is installed in the env shared by trainer and workers.
      - ``is_checkpoint_format=True`` (layerwise reload). Raises in
        ``receive_weights`` if the kernel-format path was selected.
    """

    init_info_cls = ShardedRDTWeightTransferInitInfo
    update_info_cls = ShardedRDTWeightTransferUpdateInfo

    def __init__(
        self, config: WeightTransferConfig, parallel_config: ParallelConfig
    ) -> None:
        super().__init__(config, parallel_config)
        self._trainer_actor: Any | None = None
        self._produce_method: Any | None = None

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

        plan = _FetchPlan(produce_method=self._produce_method)
        hook = functools.partial(_pre_replay_hook, plan=plan)

        # Save/restore so we don't permanently mutate global LAYERWISE_INFO
        # state if a caller later reuses these infos with a different engine.
        saved_hooks: list[tuple[Any, Any]] = []
        for info in layerwise_infos:
            saved_hooks.append((info, info.pre_replay_hook))
            info.pre_replay_hook = hook

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
