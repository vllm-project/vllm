# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Op-chain recording for the sharded-RDT backend.

The consumer asks the trainer for the exact slice a worker consumes, described as
an op chain replayed on the trainer's live tensor. ``LazyRDTTensor`` builds the
chain by intercepting the model's own weight loaders; ``copy_`` is its data sink,
meaning either ``BakeSink`` (dry run: record how the slice would be fetched and
where it lands) or ``PullSink`` (plain-load fallback: actually fetch it).

Chains are built against meta tensors, so nothing here touches Ray or a GPU.
"""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from math import prod
from typing import Any

import torch

from vllm.distributed.weight_transfer.sharded_rdt_common import SUPPORTED_OPS

# ("op_name", positional_args, sorted_kwargs_items). Entries must be hashable so
# the chain can serve as a FetchKey.
OpSpec = tuple[str, tuple[Any, ...], tuple[tuple[str, Any], ...]]
OpChain = tuple[OpSpec, ...]
FetchKey = tuple[str, OpChain]


def _freeze_kwargs(kwargs: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Sort kwargs into a tuple of items for hashable storage in OpSpec."""
    return tuple(sorted(kwargs.items()))


@dataclass
class _Scatter:
    """One recorded scatter: pull ``src``, copy it into ``layer``'s
    ``param_name`` at the recorded strided region. The bake's output unit; see
    the engine module's Data flow.

    Self-contained so replay needs no lookups. Destination geometry is read off
    the meta view and rebuilt each sync as ``as_strided(shape, stride, offset)``
    -- ``layer`` is resolved to a param at replay time, never baked, since every
    sync re-materializes fresh tensors.

    ``dtype``/``nbytes`` describe the slice ON THE WIRE and feed the packed
    layout, so ``dtype`` is the PRODUCED dtype (the lazy's after its op chain),
    not the source name's: ``view(dtype)`` is allowlisted, and taking the
    source's would size the slice with the wrong itemsize and carve the packed
    blob differently on the two sides.
    """

    layer: Any
    param_name: str
    src: FetchKey
    offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    nbytes: int


def _meta_copy_(dest: torch.Tensor, src: "LazyRDTTensor") -> torch.Tensor:
    """Fire ``dest.copy_`` from a zero-storage meta source of ``src``'s geometry.

    Moves no data but still counts against the layer's loaded numel, which drives
    ``_layerwise_process``; skipping it would leave the layer looking unloaded
    forever.

    A non-meta ``dest`` is one the layerwise reload never moved there (the
    ``SKIP_LOAD_TENSORS`` set, e.g. GLM's router bias). Torch forbids
    ``real.copy_(meta)``, and skipping the count is consistent -- ``get_layer_size``
    excludes the same set. The caller still RECORDS the copy, so the replay writes
    the real param.
    """
    if not dest.is_meta:
        return dest
    meta_src = torch.empty(src.shape, dtype=src.dtype, device="meta")
    with torch._C.DisableTorchFunctionSubclass():
        return dest.copy_(meta_src)


@dataclass
class BakeSink:
    """``copy_`` sink for the dry-run bake: record how each slice would be fetched
    and where it would land, and move nothing.

    ``_install_recording_stamps`` sets ``current = (leaf_module, param_name)``
    around each loader so ``accept_copy`` can attribute the copy. An unstamped
    ``copy_`` cannot be attributed and stays unrecorded, so its module fails the
    coverage gate and takes the plain load. ``copies_by_layer`` is keyed by module
    object, so iterating it yields each leaf module once.
    """

    copies_by_layer: "dict[Any, list[_Scatter | None]]" = field(
        default_factory=lambda: defaultdict(list)
    )
    current: "tuple[Any, str] | None" = None
    # Source names whose ``copy_`` fired during the bake. Names not here never
    # moved data (e.g. experts owned by another EP rank), so ``receive_weights``
    # skips them instead of paying ``_load_unbaked`` every sync.
    copied_names: "set[str]" = field(default_factory=set)

    def accept_copy(self, dest: torch.Tensor, src: "LazyRDTTensor") -> torch.Tensor:
        self.copied_names.add(src._name)
        if self.current is not None:
            layer, param_name = self.current
            self.copies_by_layer[layer].append(
                _Scatter(
                    layer=layer,
                    param_name=param_name,
                    src=src._key(),
                    offset=dest.storage_offset(),
                    shape=tuple(dest.shape),
                    stride=tuple(dest.stride()),
                    dtype=src.dtype,
                    nbytes=src.numel() * src.dtype.itemsize,
                )
            )
        return _meta_copy_(dest, src)


class PullSink:
    """``copy_`` sink for the plain (unbaked) load: fetch one slice on demand and
    copy it in.

    Used only by ``_load_unbaked``, for names with no recorded plan. The engine
    binds ``pull`` per name to the producer owning that name's gather group and to
    this worker's ``consumer_id``, so the producer serves from the right
    per-consumer ring.

    Layerwise reload drives each param twice: pass 1 against a meta-restored param
    (a no-op that still counts the numel), pass 2 the real fetch.
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

    ``_make_wrapper_subclass`` gives it shape/dtype/device without storage. Every
    op in ``SUPPORTED_OPS`` returns a child with the spec appended; ``copy_``
    delegates to the installed sink. Anything else reaches ``__torch_dispatch__``
    and raises ``_UnsupportedLazyOp``, so failures are loud rather than silently
    fetching the wrong bytes.
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
        """A zero-storage meta tensor of this lazy's shape/dtype, so PyTorch
        itself computes post-op geometry rather than us reimplementing it."""
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

        # Checked before the allowlist: copy_ must not be recorded into the chain.
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

        # Metadata reads are answered by the wrapper subclass and stop here; ops
        # that need data fall through to __torch_dispatch__, which raises.
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
        """Append the op and return a child, or a tuple of children for
        multi-return ops. Each child's geometry comes from running the op on a
        meta tensor."""
        meta = self_._meta()
        with torch._C.DisableTorchFunctionSubclass():
            meta_result = func(meta, *args, **kwargs)

        base_op: OpSpec = (op_name, tuple(args), _freeze_kwargs(kwargs))

        # Single-tensor result: one child carrying base_op alone.
        if isinstance(meta_result, torch.Tensor):
            return self_._make_child(meta_result.shape, meta_result.dtype, base_op)

        # Multi-return: one child per output, each carrying base_op plus a
        # trailing __getitem__(i) so the replay can index the result.
        if isinstance(meta_result, tuple | list):
            return tuple(
                self_._make_child(
                    m.shape,
                    m.dtype,
                    base_op,
                    ("__getitem__", (i,), ()),
                )
                for i, m in enumerate(meta_result)
            )

        # Non-tensor result (e.g. .item() snuck into the allowlist).
        raise _UnsupportedLazyOp(
            f"LazyRDTTensor: {op_name!r} returned a non-tensor "
            f"({type(meta_result).__name__}); cannot defer."
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        # A lazy at the aten level means an unsupported loader op. Name the op and
        # the chain; materializing instead would mask a correctness bug.
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
