# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tracing a model's weight loaders on zero-storage checkpoint tensors.

A ``TraceTensor`` stands in for one checkpoint tensor while ``load_weights``
runs. View ops (``VIEW_OPS``) return a child that carries the op chain and,
as a meta view over the checkpoint tensor, where each of its elements sits in
that tensor. ``copy_`` from a trace is handed to the trace's sink; nothing
else moves data. Ops outside ``VIEW_OPS`` and value reads go to the sink too,
whose default raises ``UnsupportedTraceOp``.

Used by the load plan (``plan.py``) and by the sharded RDT dry run.
"""

from collections.abc import Callable
from typing import Any

import torch
from torch.utils import _pytree as pytree

__all__ = [
    "OpChain",
    "OpSpec",
    "TraceSink",
    "TraceTensor",
    "UnsupportedTraceOp",
    "VIEW_OPS",
]

# ("op_name", positional_args, sorted_kwargs_items). Entries are hashable so a
# chain can serve as a key.
OpSpec = tuple[str, tuple[Any, ...], tuple[tuple[str, Any], ...]]
OpChain = tuple[OpSpec, ...]

# Ops a trace records into its chain. Every entry must be a pure view on its
# source; ``contiguous`` copies but keeps each element's value. Sharded RDT
# replays chains on the trainer against ``sharded_rdt_common.SUPPORTED_OPS``,
# which a test keeps equal to this table: the trainer cannot import this
# module, since importing the reload package pulls in the model loaders.
VIEW_OPS: dict[Callable, str] = {
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
    # Multi-return like chunk: one child per output, trailing __getitem__(i).
    torch.Tensor.unbind: "unbind",
}

# Calls that need the tensor's values on the host.
_VALUE_READS = frozenset(
    {
        torch.Tensor.item,
        torch.Tensor.tolist,
        torch.Tensor.__bool__,
        torch.Tensor.__int__,
        torch.Tensor.__float__,
        torch.Tensor.__index__,
    }
)


def _freeze_kwargs(kwargs: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Sort kwargs into a tuple of items for hashable storage in OpSpec."""
    return tuple(sorted(kwargs.items()))


class UnsupportedTraceOp(NotImplementedError):
    """A weight loader did something to a ``TraceTensor`` that its sink does
    not support (the default for anything but view ops and ``copy_``).

    A NotImplementedError so callers can tell "this loader can't be traced"
    from genuine bugs."""


class TraceSink:
    """Receives what a trace cannot answer by itself."""

    def accept_copy(self, dest: torch.Tensor, src: "TraceTensor") -> torch.Tensor:
        """``dest.copy_(src)``; returns what ``copy_`` returns."""
        raise NotImplementedError

    def unsupported(
        self, func: Callable, args: tuple, kwargs: dict, traces: "list[TraceTensor]"
    ) -> Any:
        """An op outside ``VIEW_OPS`` on ``traces``; returns the op's result."""
        raise UnsupportedTraceOp(_unsupported_message(func, traces[0]))

    def value_read(self, func: Callable, trace: "TraceTensor") -> Any:
        """``.item()``, ``bool()`` and the like on ``trace``."""
        return self.unsupported(func, (trace,), {}, [trace])


def _unsupported_message(func: Callable, trace: "TraceTensor") -> str:
    return (
        f"TraceTensor: unsupported op {func} on {trace._name!r} "
        f"(chain={trace._ops}). Supported ops are: "
        f"{sorted(VIEW_OPS.values())}, plus copy_ and .to() without a dtype "
        "or device change. Loaders that need .float(), .item(), arithmetic, "
        "bool-mask indexing, or .data access cannot be traced."
    )


def _same_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.untyped_storage()._cdata == b.untyped_storage()._cdata


class TraceTensor(torch.Tensor):
    """Zero-storage tensor that records how a loader slices a checkpoint tensor.

    ``_make_wrapper_subclass`` gives it shape/dtype/device without storage.
    Every op in ``VIEW_OPS`` returns a child with the spec appended to
    ``_ops``, and with ``_geom`` advanced: a meta view over a contiguous meta
    tensor of the checkpoint shape, whose offset and strides say which
    checkpoint element each of the child's elements is. ``_geom`` is ``None``
    (and ``_lost`` says why) once the child is no longer a view of the
    checkpoint tensor (a ``reshape`` that copies, advanced indexing, a
    ``view(dtype)`` to another element size).

    ``copy_`` from a trace goes to ``_sink.accept_copy``; other ops and value
    reads to ``_sink.unsupported`` / ``_sink.value_read``.

    TODO(E1): drop the op chain (``_ops``, ``_key``, the trailing
    ``__getitem__`` of multi-return ops) once sharded RDT runs on the plan.
    """

    # Declared at class scope so mypy can infer attribute types on instances
    # built via ``_make_wrapper_subclass``.
    _name: str
    _ops: OpChain
    _sink: "TraceSink | None"  # None only on bare construction
    _geom: torch.Tensor | None
    _lost: str | None

    @staticmethod
    def __new__(
        cls,
        name: str,
        shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        ops: OpChain = (),
        sink: "TraceSink | None" = None,
        geom: torch.Tensor | None = None,
        lost: str | None = None,
    ) -> "TraceTensor":
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
        if geom is None and lost is None and not ops:  # the checkpoint tensor
            geom = torch.empty(shape, dtype=dtype, device="meta")
        t._geom = geom
        t._lost = lost
        return t

    def _key(self) -> tuple[str, OpChain]:
        return (self._name, self._ops)

    def _make_child(
        self,
        new_shape: torch.Size,
        new_dtype: torch.dtype,
        geom: torch.Tensor | None,
        lost: str | None,
        *new_ops: OpSpec,
    ) -> "TraceTensor":
        """Append one or more ops to the chain and return a fresh child.

        Variadic so multi-return ops (e.g. chunk) can append both the base op
        and an indexing op in a single call."""
        return TraceTensor(
            name=self._name,
            shape=new_shape,
            dtype=new_dtype,
            device=self.device,
            ops=self._ops + new_ops,
            sink=self._sink,
            geom=geom,
            lost=lost,
        )

    def _meta(self) -> torch.Tensor:
        """A zero-storage meta tensor of this trace's shape/dtype, so PyTorch
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

        self_ = args[0] if args else None
        if isinstance(self_, cls):
            # Allowlisted view ops: append to chain and return child.
            op_name = VIEW_OPS.get(func)
            if op_name is not None:
                return cls._intercept(self_, func, op_name, tuple(args[1:]), kwargs)
            if func is torch.Tensor.to and self_._is_noop_to(args[1:], kwargs):
                return self_
            if func in _VALUE_READS and self_._sink is not None:
                return self_._sink.value_read(func, self_)

        # Metadata reads are answered by the wrapper subclass and stop here; ops
        # that need data fall through to __torch_dispatch__.
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    def _is_noop_to(self, args: tuple, kwargs: dict) -> bool:
        """``.to()`` that changes neither dtype nor device."""
        kwargs = {k: v for k, v in kwargs.items() if k != "copy"}
        try:
            device, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        except (TypeError, RuntimeError):
            return False
        return (device is None or torch.device(device) == self.device) and (
            dtype is None or dtype == self.dtype
        )

    @classmethod
    def _intercept(
        cls,
        self_: "TraceTensor",
        func: Callable,
        op_name: str,
        args: tuple,
        kwargs: dict,
    ):
        """Append the op and return a child, or a tuple of children for
        multi-return ops. Each child's shape comes from running the op on a
        meta tensor, its ``_geom`` from running it on the parent's."""
        meta = self_._meta()
        with torch._C.DisableTorchFunctionSubclass():
            meta_result = func(meta, *args, **kwargs)
            geoms, lost = self_._child_geoms(func, op_name, args, kwargs)

        base_op: OpSpec = (op_name, tuple(args), _freeze_kwargs(kwargs))

        # Single-tensor result: one child carrying base_op alone.
        if isinstance(meta_result, torch.Tensor):
            geom = geoms[0] if geoms else None
            return self_._make_child(
                meta_result.shape, meta_result.dtype, geom, lost, base_op
            )

        # Multi-return: one child per output, each carrying base_op plus a
        # trailing __getitem__(i) so the replay can index the result.
        if isinstance(meta_result, tuple | list):
            return tuple(
                self_._make_child(
                    m.shape,
                    m.dtype,
                    geoms[i] if geoms else None,
                    lost,
                    base_op,
                    ("__getitem__", (i,), ()),
                )
                for i, m in enumerate(meta_result)
            )

        # Non-tensor result (e.g. .item() snuck into the allowlist).
        raise UnsupportedTraceOp(
            f"TraceTensor: {op_name!r} returned a non-tensor "
            f"({type(meta_result).__name__}); cannot defer."
        )

    def _child_geoms(
        self, func: Callable, op_name: str, args: tuple, kwargs: dict
    ) -> tuple[list[torch.Tensor] | None, str | None]:
        """The op's outputs on ``_geom``, or ``None`` and why they are no
        longer views of the checkpoint tensor."""
        geom = self._geom
        if geom is None:
            return None, self._lost
        if op_name == "contiguous":  # a copy with the same element order
            return [geom], None
        try:
            out = func(geom, *args, **kwargs)
        except Exception:
            return None, f"{op_name} that is not a view of the checkpoint tensor"
        outs = [out] if isinstance(out, torch.Tensor) else list(out)
        for o in outs:
            if not _same_storage(o, geom) or o.dtype.itemsize != geom.dtype.itemsize:
                return None, f"{op_name} that copies or reinterprets elements"
        return outs, None

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        # A trace at the aten level means an op outside VIEW_OPS.
        traces = [
            a for a in pytree.arg_tree_leaves(*args, **kwargs) if isinstance(a, cls)
        ]
        sink = next((t._sink for t in traces if t._sink is not None), None)
        if sink is None:
            raise UnsupportedTraceOp(_unsupported_message(func, traces[0]))
        return sink.unsupported(func, args, kwargs, traces)
