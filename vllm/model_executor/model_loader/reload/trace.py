# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tracing a model's weight loaders on zero-storage checkpoint tensors.

A `TraceTensor` stands in for one checkpoint tensor while `load_weights` runs. View ops
(`VIEW_OPS`) return a child that knows, as a meta view over the checkpoint tensor, where
each of its elements sits in that tensor. What a trace cannot settle by itself (`copy_`
from it, ops outside `VIEW_OPS`, value reads) it hands to the bake's `_PlanSink`
(`bake.py`), which decides what each means for the plan; nothing moves data.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from torch.utils import _pytree as pytree

if TYPE_CHECKING:
    from .bake import _PlanSink

__all__ = ["TraceTensor", "VIEW_OPS"]

# Pure views of their source; ``contiguous`` copies but keeps each element.
VIEW_OPS = frozenset(
    {
        torch.Tensor.narrow,
        torch.Tensor.view,
        torch.Tensor.reshape,
        torch.Tensor.__getitem__,
        torch.Tensor.unsqueeze,
        torch.Tensor.squeeze,
        torch.Tensor.transpose,
        torch.Tensor.t,
        torch.Tensor.permute,
        torch.Tensor.flatten,
        torch.Tensor.unflatten,
        torch.Tensor.contiguous,
        torch.Tensor.chunk,
        torch.Tensor.split,
        torch.Tensor.unbind,
    }
)

# Calls that need the tensor's values on the host.
_VALUE_READS = frozenset(
    {
        torch.Tensor.item,
        torch.Tensor.tolist,
        torch.Tensor.numpy,
        torch.Tensor.__bool__,
        torch.Tensor.__int__,
        torch.Tensor.__float__,
        torch.Tensor.__index__,
    }
)


class TraceTensor(torch.Tensor):
    """Zero-storage meta tensor that records how a loader slices a checkpoint
    tensor.

    `_geom` is a meta view over a contiguous meta tensor of the checkpoint
    shape: its offset and strides say which checkpoint element each element of this
    tensor is. It is `None` (and `_lost` says why) once this tensor is no longer a view
    of the checkpoint tensor. `_sink` is the bake that owns the trace.
    """

    # Declared at class scope so mypy can infer attribute types on instances
    # built via ``_make_wrapper_subclass``.
    _name: str
    _sink: "_PlanSink"
    _geom: torch.Tensor | None
    _lost: str | None

    @staticmethod
    def __new__(
        cls,
        name: str,
        shape: torch.Size,
        dtype: torch.dtype,
        sink: "_PlanSink",
        geom: torch.Tensor | None,
        lost: str | None = None,
    ) -> "TraceTensor":
        t = torch.Tensor._make_wrapper_subclass(
            cls, shape, dtype=dtype, device="meta", requires_grad=False
        )
        t._name = name
        t._sink = sink
        t._geom = geom
        t._lost = lost
        return t

    @classmethod
    def checkpoint(
        cls, name: str, shape: torch.Size, dtype: torch.dtype, sink: "_PlanSink"
    ) -> "TraceTensor":
        """The whole checkpoint tensor `name`."""
        return cls(
            name, shape, dtype, sink, torch.empty(shape, dtype=dtype, device="meta")
        )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.Tensor.copy_:
            src = args[1] if len(args) > 1 else kwargs.get("src")
            if isinstance(src, cls) and not isinstance(args[0], cls):
                return src._sink.accept_copy(args[0], src)
        self_ = args[0] if args else None
        if isinstance(self_, cls):
            if func in VIEW_OPS:
                return self_._view(func, args[1:], kwargs)
            if func is torch.Tensor.to and self_._is_noop_to(args[1:], kwargs):
                return self_
            if func in _VALUE_READS:
                return self_._sink.value_read(self_)
        # Metadata reads are answered by the wrapper subclass and stop here; ops
        # that need data fall through to __torch_dispatch__.
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    def _is_noop_to(self, args: tuple, kwargs: dict) -> bool:
        """`.to()` that keeps the dtype, such as `.to(param.device)`: a device move
        keeps every byte, and a trace has no device to move from."""
        kwargs = {k: v for k, v in kwargs.items() if k != "copy"}
        try:
            _, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        except (TypeError, RuntimeError):
            return False
        return dtype is None or dtype == self.dtype

    def _view(self, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Run the view op on `_geom` for the children's shapes and
        geometry; on a plain meta tensor when that is lost or fails."""
        name = func.__name__
        with torch._C.DisableTorchFunctionSubclass():
            if self._geom is not None and name == "contiguous":
                out: Any = self._geom
            elif self._geom is not None:
                try:
                    out = func(self._geom, *args, **kwargs)
                except Exception:
                    out = None
            else:
                out = None
            lost = self._lost
            if out is None:
                lost = lost or f"{name} that is not a view of the checkpoint tensor"
                meta = torch.empty(self.shape, dtype=self.dtype, device="meta")
                out = func(meta, *args, **kwargs)

        base = self._geom

        def child(o: torch.Tensor) -> "TraceTensor":
            if (
                lost is None
                and base is not None
                and (
                    o.untyped_storage()._cdata != base.untyped_storage()._cdata
                    or o.dtype.itemsize != base.dtype.itemsize
                )
            ):
                reason = f"{name} that copies or reinterprets elements"
                return TraceTensor(
                    self._name, o.shape, o.dtype, self._sink, None, reason
                )
            geom = o if lost is None else None
            return TraceTensor(self._name, o.shape, o.dtype, self._sink, geom, lost)

        if isinstance(out, torch.Tensor):
            return child(out)
        return type(out)(child(o) for o in out)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # A trace at the aten level means an op outside VIEW_OPS.
        kwargs = kwargs or {}
        leaves = pytree.arg_tree_leaves(*args, **kwargs)
        traces = [a for a in leaves if isinstance(a, cls)]
        return traces[0]._sink.unsupported(func, args, kwargs, traces)
