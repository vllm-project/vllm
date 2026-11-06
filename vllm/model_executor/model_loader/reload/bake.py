# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The load plan's bake: run ``load_weights`` once over ``TraceTensor``s, on
the model's real tensors as ``begin_direct_reload`` left them, and record each
``copy_`` from a trace and every reason a name cannot be written directly.

A trace has no data, so nothing it feeds reaches memory: its ``copy_`` is
recorded, and any other op on it runs on meta copies of its arguments. Ops on
real tensors alone run for real: ``_TargetGuard`` makes a loader that reads a
checkpoint tensor a fallback, and constant writes into one (``fill_``,
``zero_``) happen, mid-update, and stay out of the plan (post-load processing
owns padding)."""

from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NoReturn

import torch
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode

from vllm.logger import init_logger

from .direct import loadable_tensors
from .trace import TraceTensor

if TYPE_CHECKING:
    from vllm.distributed.weight_transfer.base import ParamMeta

logger = init_logger(__name__)

_META = torch.device("meta")


# ---- targets


@dataclass
class Target:
    """A checkpoint tensor, as a write destination."""

    module: str
    tensor: str
    real: torch.Tensor
    byte_offset: int  # in its storage, as `begin_direct_reload` viewed it


class Targets:
    """Every checkpoint tensor in the layout ``begin_direct_reload`` applied,
    by storage and by name. Writes name their tensor, for readable plans and
    tests; ``by_name`` looks it up."""

    def __init__(self, model: torch.nn.Module):
        self.by_storage: dict[int, Target] = {}
        self.by_name: dict[tuple[str, str], Target] = {}
        for name, t in loadable_tensors(model):
            if hasattr(t, "_ckpt"):
                module, _, key = name.rpartition(".")
                target = Target(module, key, t, t.storage_offset() * t.dtype.itemsize)
                # writes into a storage are labelled with its first target
                self.by_storage.setdefault(t.untyped_storage()._cdata, target)
                self.by_name[(module, key)] = target


def _storage_key(t: torch.Tensor) -> int | None:
    return None if isinstance(t, TraceTensor) else t.untyped_storage()._cdata


# ---- the sink


class _BakeStopped(Exception):
    """A loader needs a traced tensor's values, or an op on it fails on meta:
    the loader cannot continue. ``name`` is the trace's name."""

    def __init__(self, name: str):
        super().__init__(name)
        self.name = name


class _SinkClosed(RuntimeError):
    """A trace was used after its bake ended."""


_VALUE_READ = "its loader reads tensor values (.item(), bool(), ...)"


@dataclass
class Copy:
    """One ``dest.copy_(src)`` from a trace."""

    src: TraceTensor
    storage: int  # destination storage key
    byte_offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype


@dataclass
class _PlanSink:
    """One bake's record. Traces call ``accept_copy`` for a ``copy_`` from
    them, ``unsupported`` for any other op outside ``VIEW_OPS`` and
    ``value_read`` for ``.item()`` and the like; it turns these into copies,
    fallbacks and bake stops."""

    targets: Targets
    copies: dict[str, list[Copy]] = field(default_factory=lambda: defaultdict(list))
    fallback: dict[str, str] = field(default_factory=dict)
    # Names that stopped their loader, and why. Kept even though these also
    # raise: a loader may catch the exception and carry on, and the name must
    # still fall back.
    stops: dict[str, str] = field(default_factory=dict)
    wrote: set[str] = field(default_factory=set)  # copied anywhere
    last_writer: dict[int, str] = field(default_factory=dict)  # per storage
    # The name last handed to load_weights, or last copied from. A loader that
    # reads ahead (as AutoWeightsLoader's groupby does) makes this the next
    # name; it is used only to blame a destination read that no copy_ preceded.
    current: str | None = None
    # Set when the bake ends: a trace a loader kept on the module must not be
    # used again (by a later load_weights), or its writes would go nowhere.
    closed: bool = False

    def mark(self, name: str, reason: str) -> None:
        self.fallback.setdefault(name, reason)

    def _check_open(self, trace: TraceTensor) -> None:
        if self.closed:
            raise _SinkClosed(
                f"load plan: a planning trace of {trace._name!r} was used after "
                "planning; a loader keeps an incoming tensor across load_weights "
                "calls, which the load plan does not support"
            )

    def accept_copy(self, dest: torch.Tensor, src: TraceTensor) -> torch.Tensor:
        self._check_open(src)
        name = src._name
        self.current = name
        self.wrote.add(name)
        key = dest.untyped_storage()._cdata  # dest is never a trace
        self.last_writer[key] = name
        self.copies[name].append(
            Copy(
                src=src,
                storage=key,
                byte_offset=dest.storage_offset() * dest.dtype.itemsize,
                shape=tuple(dest.shape),
                stride=tuple(dest.stride()),
                dtype=dest.dtype,
            )
        )
        return dest

    def value_read(self, trace: TraceTensor) -> NoReturn:
        self._check_open(trace)
        self.stops.setdefault(trace._name, _VALUE_READ)
        raise _BakeStopped(trace._name)

    def unsupported(
        self, func: Callable, args: tuple, kwargs: dict, traces: list[TraceTensor]
    ) -> Any:
        """Mark the names fallbacks and run ``func`` on meta copies of its
        arguments, so it writes no real tensor; tensor results become traces
        of the first name, with no geometry."""
        self._check_open(traces[0])
        if _writes_a_real_tensor(func, args, kwargs):  # in place, or out=
            reason = f"writes it without Tensor.copy_ ({func.__name__})"
            self.wrote.update(t._name for t in traces)
        else:
            reason = f"applies {func.__name__} to it"
        for t in traces:
            self.mark(t._name, reason)

        def to_meta(a: Any) -> Any:
            if isinstance(a, TraceTensor):
                return torch.empty_strided(
                    a.shape, a.stride(), dtype=a.dtype, device=_META
                )
            return a.to(_META) if isinstance(a, torch.Tensor) else a

        first = traces[0]
        flat, spec = pytree.tree_flatten((args, kwargs))
        meta_flat = [to_meta(a) for a in flat]
        meta_args, meta_kwargs = pytree.tree_unflatten(meta_flat, spec)
        try:
            out = func(*meta_args, **meta_kwargs)
        except Exception as e:  # no meta kernel, a shape error, ...
            # the error stays out of the reason, which groups the report
            logger.debug("load plan: %s on %s fails on meta: %r", func, first._name, e)
            self.stops.setdefault(first._name, f"{func.__name__} fails on meta")
            raise _BakeStopped(first._name) from e
        originals = {id(m): a for m, a in zip(meta_flat, flat)}

        def wrap(o: Any) -> Any:
            if not isinstance(o, torch.Tensor):
                return o
            if id(o) in originals:  # in place: return the original argument
                return originals[id(o)]
            return TraceTensor(first._name, o.shape, o.dtype, self, None, reason)

        return pytree.tree_map(wrap, out)


def _writes_a_real_tensor(func: Any, args: tuple, kwargs: dict) -> bool:
    """Whether ``func`` writes an argument that is not a trace (in place or
    through ``out=``)."""
    schema = getattr(func, "_schema", None)
    for i, arg in enumerate(getattr(schema, "arguments", ())):
        if arg.alias_info is None or not arg.alias_info.is_write:
            continue
        value = args[i] if i < len(args) else kwargs.get(arg.name)
        leaves = pytree.tree_leaves(value)
        if any(isinstance(t, torch.Tensor) for t in leaves) and not any(
            isinstance(t, TraceTensor) for t in leaves
        ):
            return True
    return False


# Ops on a target that read only its metadata.
_METADATA_OPS = {
    torch.ops.aten.empty_like,
    torch.ops.aten.zeros_like,
    torch.ops.aten.ones_like,
    torch.ops.aten.full_like,
    torch.ops.aten.new_empty,
    torch.ops.aten.new_empty_strided,
    torch.ops.aten.new_zeros,
    torch.ops.aten.new_ones,
    torch.ops.aten.new_full,
}
# Writes into a target that do not read it: constants, or copies of tensors
# that are not checkpoint tensors. They run, and stay out of the plan:
# post-load processing owns padding.
_CONSTANT_WRITES = {
    torch.ops.aten.copy_,
    torch.ops.aten.fill_,
    torch.ops.aten.zero_,
    torch.ops.aten.index_put_,
}


class _TargetGuard(TorchDispatchMode):
    """Ops on real tensors alone run for real. One that reads a checkpoint
    tensor (not a view, not only its metadata) makes the last name written
    into that storage (or the name being loaded) a fallback: the direct path
    would skip it. For example ``composed_weight_loader``'s
    ``param.copy_(fn(param))``, a check such as ``torch.equal(param, x)``, or
    a mirror copied from ``param``. Reads that bypass the dispatcher
    (``tolist()`` or ``numpy()`` of a CPU tensor, a raw kernel launch) go
    unseen."""

    def __init__(self, sink: _PlanSink):
        super().__init__()
        self.sink = sink

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        packet = func.overloadpacket
        leaves = [
            a
            for a in pytree.arg_tree_leaves(*args, **kwargs)
            if isinstance(a, torch.Tensor)
        ]
        if any(isinstance(a, TraceTensor) for a in leaves):
            return func(*args, **kwargs)  # the trace's own handler
        if not func.is_view and packet not in _METADATA_OPS:
            # copy_(target, x) and friends write the target but read only x
            read = args[1:] if packet in _CONSTANT_WRITES else args
            self._check_reads(packet, pytree.arg_tree_leaves(*read, **kwargs))
        return func(*args, **kwargs)

    def _check_reads(self, packet: Any, leaves: list) -> None:
        for a in leaves:
            key = _storage_key(a) if isinstance(a, torch.Tensor) else None
            if key is not None and key in self.sink.targets.by_storage:
                name = self.sink.last_writer.get(key, self.sink.current)
                if name is not None:
                    self.sink.mark(name, f"its loader reads the destination ({packet})")
                return


# ---- the bake


def _bake(
    model: torch.nn.Module, metas: dict[str, "ParamMeta"], targets: Targets
) -> _PlanSink:
    """Trace ``load_weights`` over every name, in one pass. A name that stops
    the loader (a value read, an op that fails on meta) unwinds the model's
    ``load_weights``; it falls back, and ``load_weights`` is called again
    with the names after it, on the same sink. Loaders handle each name as it
    arrives, so nothing is replayed.
    Module state a loader sets stays: it is what that loader would leave on
    the ``load_weights`` path."""
    sink = _PlanSink(targets)
    names = list(metas)

    def weights(rest: list[str]) -> Iterator[tuple[str, torch.Tensor]]:
        for name in rest:
            meta = metas[name]
            sink.current = name
            shape = torch.Size(meta.shape)
            yield name, TraceTensor.checkpoint(name, shape, meta.dtype, sink)

    start = 0
    try:
        with torch.no_grad(), _TargetGuard(sink):
            while start < len(names):
                try:
                    model.load_weights(weights(names[start:]))
                    break
                except _BakeStopped as stop:
                    # Resume after the name that stopped (not after the last
                    # name yielded: a loader may have read ahead).
                    at = names.index(stop.name)
                    if at < start:
                        raise RuntimeError(
                            f"load plan: a loader reads the values of "
                            f"{stop.name!r} after later names arrived (it keeps "
                            "an incoming tensor across names)"
                        ) from stop
                    start = at + 1
    finally:
        sink.closed = True
    return sink


@dataclass
class Traced:
    copies: dict[str, list[Copy]]  # per name
    reasons: dict[str, str]  # fallback names and why
    lonely: list[str]  # fallbacks that wrote nothing


def trace_loads(
    model: torch.nn.Module, metas: dict[str, "ParamMeta"], targets: Targets
) -> Traced:
    """Bake all names (see ``_bake``). A name that stopped the loader falls
    back; what it recorded before stopping is dropped."""
    try:
        sink = _bake(model, metas, targets)
    except Exception as e:
        raise RuntimeError(f"load plan: tracing load_weights failed: {e!r}") from e
    reasons = dict(sink.stops)
    for name in reasons:
        sink.copies.pop(name, None)
    lonely = [n for n in sink.fallback if n not in sink.wrote and n not in reasons]
    for name, reason in sink.fallback.items():
        reasons.setdefault(name, reason)
    return Traced(sink.copies, reasons, lonely)
