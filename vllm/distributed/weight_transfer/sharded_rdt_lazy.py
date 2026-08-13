# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Op-chain recording for the sharded-RDT backend.

The consumer never asks the trainer for a whole parameter; it asks for the exact
*slice* a given vLLM worker consumes, described as an **op chain** replayed on
the trainer's live tensor. This module is where that chain is built and where the
two things that can consume it live:

  * `LazyRDTTensor` -- a zero-storage tensor handed to the model's own weight
    loaders. Every allowlisted view/slice/shape op returns a new lazy with the op
    appended; anything that needs real data raises. `copy_` is the data sink.
  * `BakeSink` / `PullSink` -- the two things a `copy_` can mean. During the
    init-time dry run the bake RECORDS how the slice would be fetched and where
    it would land (`_BakedCopy`); on the plain-load fallback the pull sink
    actually FETCHES it.

Nothing here touches the engine, Ray, or a GPU: chains are built against meta
tensors, so the whole module is exercised on CPU.
"""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from math import prod
from typing import Any

import torch

from vllm.distributed.weight_transfer.sharded_rdt_common import SUPPORTED_OPS

# A single recorded op: ("op_name", positional_args_tuple, sorted_kwargs_items).
# All entries must be hashable so the chain itself is hashable for use as
# a FetchKey. Slices, ints, tuples, Ellipsis, None, and memory_format enums
# are all hashable on Python 3.12+.
OpSpec = tuple[str, tuple[Any, ...], tuple[tuple[str, Any], ...]]
OpChain = tuple[OpSpec, ...]
FetchKey = tuple[str, OpChain]


def _freeze_kwargs(kwargs: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Sort kwargs into a tuple of items for hashable storage in OpSpec."""
    return tuple(sorted(kwargs.items()))


# ---------------- M:N producer/consumer routing ----------------
# The consumer and producer fleets can differ in size, and a producer may gather
# only part of the model (pipeline-parallel producers gather within their stage).
# ``RdtRouter`` in ``sharded_rdt_common`` — built identically here and on the
# trainer from the ownership table shipped in the init info — answers both
# questions with one rule: which producer serves each gather group for this
# consumer, and how many consumers each producer must be freed by per group.
# Every group is served by exactly ONE producer per consumer.


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


def _meta_copy_(dest: torch.Tensor, src: "LazyRDTTensor") -> torch.Tensor:
    """Fire ``dest.copy_`` from a zero-storage meta source of ``src``'s geometry.

    Moves no data, but still counts against the layer's loaded numel -- vLLM's
    layerwise ``CopyCounter`` drives ``_layerwise_process`` off that count, so a
    skipped ``copy_`` would leave the layer looking unloaded forever.

    A ``dest`` that is NOT on meta is one the layerwise reload never moved
    there (the ``SKIP_LOAD_TENSORS`` set — e.g. GLM's router
    ``e_score_correction_bias``, a plain ``nn.Parameter``). Torch forbids
    ``real.copy_(meta)``, and skipping the counter copy is consistent:
    ``get_layer_size`` excludes the same set, so these never count toward a
    layer's total either. The copy is still RECORDED by the caller, so the
    sync-time replay writes the real param.
    """
    if not dest.is_meta:
        return dest
    meta_src = torch.empty(src.shape, dtype=src.dtype, device="meta")
    with torch._C.DisableTorchFunctionSubclass():
        return dest.copy_(meta_src)


@dataclass
class BakeSink:
    """``copy_`` sink for the dry-run bake: record how each slice WOULD be
    fetched and where it would land, and move nothing.

    During the single dry-run ``load_weights`` pass the engine stamps
    ``current = (leaf_module, param_name)`` around each param's loader (see
    ``_install_recording_stamps``), so ``accept_copy`` can attribute the copy to
    its destination param: the op chain comes from the source lazy, the
    ``offset/shape/stride`` from the meta destination view (valid on meta; no real
    storage needed). A ``copy_`` with no stamp cannot be attributed and is left
    unrecorded, so its module fails the coverage gate and takes the plain load.
    ``copies_by_layer`` is keyed by the module object, so iterating it after the
    pass yields each leaf module once.
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

    def accept_copy(self, dest: torch.Tensor, src: "LazyRDTTensor") -> torch.Tensor:
        self.copied_names.add(src._name)
        if self.current is not None:
            layer, param_name = self.current
            self.copies_by_layer[layer].append(
                _BakedCopy(
                    src._key(),
                    param_name,
                    dest.storage_offset(),
                    tuple(dest.shape),
                    tuple(dest.stride()),
                )
            )
        return _meta_copy_(dest, src)


class PullSink:
    """``copy_`` sink for the plain (unbaked) load: fetch this one slice on
    demand and copy it in.

    Used only by ``_load_unbaked``, for names with no recorded plan (attention
    scales, partial layers). ``pull`` takes a one-element key list and returns the
    packed uint8 blob; the engine binds it per name to the producer that owns that
    name's gather group AND to this worker's ``consumer_id``, so the producer
    serves it from the right per-consumer ring.

    Layerwise reload drives each param twice: pass 1 against a meta-restored param
    (nothing to copy yet, so a meta no-op that still counts the numel) and pass 2
    against the materialized param, which is the real fetch.
    """

    def __init__(self, pull: "Callable[[list[FetchKey]], torch.Tensor]") -> None:
        self._pull = pull

    def fetch(
        self, key: "FetchKey", shape: torch.Size, dtype: torch.dtype
    ) -> torch.Tensor:
        blob = self._pull([key])
        nbytes = prod(shape) * dtype.itemsize
        return blob[:nbytes].view(dtype).reshape(shape)

    def accept_copy(self, dest: torch.Tensor, src: "LazyRDTTensor") -> torch.Tensor:
        if dest.device.type == "meta":
            return _meta_copy_(dest, src)
        mat = self.fetch(src._key(), src.shape, src.dtype)
        with torch._C.DisableTorchFunctionSubclass():
            return dest.copy_(mat)


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
    sink, delegated whole to the ``sink`` the engine installed -- ``BakeSink``
    during the dry run, ``PullSink`` on the plain-load path.

    Any op outside ``SUPPORTED_OPS`` (arithmetic, .item, .to, .float, .data,
    bool-mask indexing, etc.) raises ``_UnsupportedLazyOp`` in
    ``__torch_dispatch__`` so failures are loud rather than silently fetching
    the wrong bytes.
    """

    # Declared at class scope so mypy can infer attribute types on
    # instances built via ``_make_wrapper_subclass`` (where ``__new__``
    # returns a tensor it can't annotate as ``self``).
    _name: str
    _ops: OpChain
    # Handles this lazy's copy_. ``None`` only on bare construction.
    _sink: "BakeSink | PullSink | None"

    @staticmethod
    def __new__(
        cls,
        name: str,
        shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        ops: OpChain = (),
        sink: "BakeSink | PullSink | None" = None,
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
        t._sink = sink
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
            sink=self._sink,
        )

    def _meta(self) -> torch.Tensor:
        """A zero-storage meta tensor of this lazy's current shape/dtype.

        Used to compute the post-op shape/dtype via PyTorch itself, which is
        more reliable than reimplementing shape inference per op. The result
        is never used for data — only its metadata.
        """
        return torch.empty(self.shape, dtype=self.dtype, device="meta")

    @classmethod
    def __torch_function__(
        cls,
        func,
        types,
        args=(),
        kwargs=None,
    ):
        kwargs = kwargs or {}

        # copy_ is the data sink, and the only op whose handling depends on which
        # sink is installed. Checked before the allowlist because copy_ must not
        # be recorded into the chain.
        if func is torch.Tensor.copy_:
            dest = args[0]
            src = args[1] if len(args) > 1 else kwargs.get("src")
            if isinstance(src, cls) and src._sink is not None:
                return src._sink.accept_copy(dest, src)

        # Allowlisted slice/view/shape ops: append to chain and return child.
        op_name = SUPPORTED_OPS.get(func)
        if op_name is not None:
            self_ = args[0]
            if isinstance(self_, cls):
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

        # A lazy reaching the aten level means the loader called an op we do not
        # support. Raise loudly with the op and the recorded chain so the user can
        # identify which loader/weight is at fault. We deliberately do NOT
        # silently materialize here -- that would mask correctness bugs.
        for arg in (*args, *kwargs.values()):
            if isinstance(arg, cls):
                raise _UnsupportedLazyOp(
                    f"LazyRDTTensor: unsupported op {func} reached "
                    f"__torch_dispatch__ on lazy {arg._name!r} "
                    f"(chain={arg._ops}). Supported ops are: "
                    f"{sorted(SUPPORTED_OPS.values())}, plus copy_. "
                    "Loaders that need .to(), .float(), .item(), arithmetic, "
                    "bool-mask indexing, or .data access are not supported by "
                    "the sharded RDT backend."
                )
        return func(*args, **kwargs)
