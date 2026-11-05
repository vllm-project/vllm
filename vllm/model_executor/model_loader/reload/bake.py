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

import itertools
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode

from vllm.logger import init_logger

from .direct import loadable_tensors
from .trace import TraceSink, TraceTensor

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
    """Every checkpoint tensor in the layout ``begin_direct_reload`` applied
    (its single layout source; for example, a tensor ``set_`` into a fused
    buffer keeps its new offset), by storage and by name."""

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
    """A loader needs a traced tensor's values, or an op on it fails on meta;
    the bake cannot continue."""


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
class _PlanSink(TraceSink):
    targets: Targets
    copies: dict[str, list[Copy]] = field(default_factory=lambda: defaultdict(list))
    fallback: dict[str, str] = field(default_factory=dict)
    # Names that stopped the bake, and why. Kept even though these also raise:
    # a loader may catch the exception and carry on, and the name must still
    # be rerun without.
    stops: dict[str, str] = field(default_factory=dict)
    wrote: set[str] = field(default_factory=set)  # copied anywhere
    last_writer: dict[int, str] = field(default_factory=dict)  # per storage
    # The name last handed to load_weights. A loader that reads ahead (as
    # AutoWeightsLoader's groupby does) makes this the next name; it is used
    # only to blame a destination read that no copy_ preceded.
    current: str | None = None

    def mark(self, name: str, reason: str) -> None:
        self.fallback.setdefault(name, reason)

    def accept_copy(self, dest: torch.Tensor, src: TraceTensor) -> torch.Tensor:
        name = src._name
        self.current = name
        self.wrote.add(name)
        key = _storage_key(dest)
        if key is None:
            self.mark(name, "copies into another checkpoint tensor")
            return dest
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

    def value_read(self, func: Callable, trace: TraceTensor) -> Any:
        self.stops.setdefault(trace._name, _VALUE_READ)
        raise _BakeStopped

    def unsupported(
        self, func: Callable, args: tuple, kwargs: dict, traces: list[TraceTensor]
    ) -> Any:
        """Mark the names fallbacks and run ``func`` on meta copies of its
        arguments, so it writes no real tensor; tensor results become traces
        of the first name, with no geometry."""
        if _writes_a_real_tensor(func, args, kwargs):  # in place, or out=
            reason = f"writes it without Tensor.copy_ ({func.__name__})"
            self.wrote.update(t._name for t in traces)
        else:
            reason = f"applies {func.__name__} to it"
        for t in traces:
            self.mark(t._name, reason)
        first = traces[0]

        def trace(o: torch.Tensor) -> TraceTensor:
            return TraceTensor(first._name, o.shape, o.dtype, self, None, reason)

        return self.on_meta(func, args, kwargs, first._name, trace)

    def on_meta(
        self,
        func: Any,
        args: tuple,
        kwargs: dict,
        name: str,
        wrap: Callable[[torch.Tensor], torch.Tensor] = lambda o: o,
    ) -> Any:
        """Run ``func`` on meta copies of its tensor arguments, so it writes
        no real tensor. An in-place result is the original argument; other
        tensor results go through ``wrap``. If ``func`` fails on meta, ``name``
        stops the bake."""

        def to_meta(a: Any) -> Any:
            if isinstance(a, TraceTensor):
                return torch.empty_strided(
                    a.shape, a.stride(), dtype=a.dtype, device=_META
                )
            return a.to(_META) if isinstance(a, torch.Tensor) else a

        flat, spec = pytree.tree_flatten((args, kwargs))
        meta_flat = [to_meta(a) for a in flat]
        meta_args, meta_kwargs = pytree.tree_unflatten(meta_flat, spec)
        try:
            out = func(*meta_args, **meta_kwargs)
        except Exception as e:  # no meta kernel, a shape error, ...
            # the error stays out of the reason, which groups the report
            logger.debug("load plan: %s on %s fails on meta: %r", func, name, e)
            self.stops.setdefault(name, f"{func.__name__} fails on meta")
            raise _BakeStopped from e
        originals = {id(m): a for m, a in zip(meta_flat, flat)}

        def back(o: Any) -> Any:
            if not isinstance(o, torch.Tensor):
                return o
            return originals[id(o)] if id(o) in originals else wrap(o)

        return pytree.tree_map(back, out)


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
    unseen.

    An op mixing a meta temporary (one a loader made on the incoming
    tensor's device) with a real tensor runs on meta instead and makes the
    name that last wrote the temporary (or the name being loaded) a
    fallback."""

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
        meta = [a for a in leaves if a.is_meta]
        if meta and len(meta) < len(leaves):
            sink = self.sink
            key = meta[0].untyped_storage()._cdata
            name = sink.last_writer.get(key) or sink.current
            if name is not None:
                sink.mark(name, f"mixes a meta temporary with a real tensor ({packet})")
                return sink.on_meta(func, args, kwargs, name)
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


# ---- module state


class _Snapshot:
    """Module, parameter and persistent buffer attributes, CPU tensors among
    them, and where each parameter and persistent buffer points (and whether
    it requires grad), compared and restored after a bake: a loader that
    changes such state will not run on the direct path, and planning must
    leave none behind. Target bytes are not saved; the update rewrites
    them."""

    def __init__(self, model: torch.nn.Module):
        self.dicts: list[tuple[str, dict, dict[str, _Value]]] = []
        for module_name, module in model.named_modules():
            self._add(module_name, module.__dict__)
        self.layouts: list[tuple[str, torch.Tensor, torch.Tensor, tuple]] = []
        for name, t in loadable_tensors(model):
            self._add(name, t.__dict__)
            self.layouts.append((name, t, t.data, _where(t)))

    def _add(self, label: str, d: dict) -> None:
        self.dicts.append((label, d, {k: _Value(v) for k, v in d.items()}))

    def restore(self) -> list[str]:
        """Undo every change; returns what changed."""
        changed = []
        for label, d, saved in self.dicts:
            for k in d.keys() - saved.keys():
                changed.append(f"{label}.{k}")
                del d[k]
            for k, value in saved.items():
                if k not in d or not value.holds(d[k]):
                    changed.append(f"{label}.{k}")
                    d[k] = value.restore()
        for name, t, data, where in self.layouts:  # e.g. `param.data = ...`
            if _where(t) != where:
                changed.append(name)
                t.data = data
                t.requires_grad_(where[-1])
        return changed


def _where(t: torch.Tensor) -> tuple:
    storage = t.untyped_storage()._cdata
    return storage, t.storage_offset(), t.shape, t.stride(), t.dtype, t.requires_grad


class _Value:
    """An attribute's value, with a shallow copy of a container or a clone of
    a CPU tensor so in-place changes are seen too."""

    def __init__(self, v: Any):
        self.obj = v
        self.copy: Any = None
        if isinstance(v, torch.Tensor):
            if v.device.type == "cpu" and not isinstance(v, TraceTensor):
                self.copy = v.detach().clone()
        elif isinstance(v, list | dict | set):
            self.copy = type(v)(v)

    def holds(self, v: Any) -> bool:
        if v is not self.obj:  # equal scalars count as unchanged
            scalar = type(v) in (bool, int, float, str) and type(v) is type(self.obj)
            return scalar and v == self.obj
        c = self.copy
        if isinstance(c, torch.Tensor):
            return torch.equal(v, c)
        if isinstance(c, list):
            return len(v) == len(c) and all(a is b for a, b in zip(v, c))
        if isinstance(c, dict):
            return v.keys() == c.keys() and all(v[k] is c[k] for k in c)
        return c is None or v == c

    def restore(self) -> Any:
        v, c = self.obj, self.copy
        if isinstance(c, torch.Tensor):
            v.copy_(c)
        elif isinstance(c, list):
            v[:] = c
        elif isinstance(c, dict | set):
            v.clear()
            v.update(c)
        return v


# ---- the bake


def _bake(
    model: torch.nn.Module,
    metas: dict[str, "ParamMeta"],
    names: list[str],
    targets: Targets,
) -> tuple[_PlanSink, list[str]]:
    """One traced ``load_weights`` over ``names``; returns the sink and the
    state it changed (restored)."""
    sink = _PlanSink(targets)

    def weights() -> Iterator[tuple[str, torch.Tensor]]:
        for name in names:
            meta = metas[name]
            sink.current = name
            shape = torch.Size(meta.shape)
            yield name, TraceTensor.checkpoint(name, shape, meta.dtype, sink)

    snapshot = _Snapshot(model)
    try:
        with torch.no_grad(), _TargetGuard(sink):
            model.load_weights(weights())
    except _BakeStopped:
        pass  # in sink.stops; the caller reruns without the name
    finally:
        changed = snapshot.restore()
    return sink, changed


@dataclass
class Traced:
    copies: dict[str, list[Copy]]  # per name, from the full bake
    reasons: dict[str, str]  # fallback names and why
    lonely: list[str]  # fallbacks that write nothing when loaded alone
    rebakes: int  # full bakes rerun after a name stopped one


def trace_loads(
    model: torch.nn.Module, metas: dict[str, "ParamMeta"], targets: Targets
) -> Traced:
    """Bake all names. Each name whose loader reads tensor values (or runs an
    op that fails on meta) reruns the whole bake without it (at most one full
    bake per such name). Names that change module state are then found by
    bisecting, only if the full bake did."""
    reasons: dict[str, str] = {}
    names = list(metas)
    for rebakes in itertools.count():
        try:
            sink, changed = _bake(model, metas, names, targets)
        except Exception as e:
            raise RuntimeError(f"load plan: tracing load_weights failed: {e!r}") from e
        if not sink.stops:
            break
        for name, reason in sink.stops.items():
            reasons.setdefault(name, reason)
        names = [n for n in names if n not in reasons]

    lonely = [n for n in sink.fallback if n not in sink.wrote]
    if changed:
        _bisect_state(model, metas, names, changed, sink, targets, reasons, lonely)
    for name, reason in sink.fallback.items():
        reasons.setdefault(name, reason)
    return Traced(sink.copies, reasons, lonely, rebakes)


def _bisect_state(
    model: torch.nn.Module,
    metas: dict[str, "ParamMeta"],
    names: list[str],
    changed: list[str],
    sink: _PlanSink,
    targets: Targets,
    reasons: dict[str, str],
    lonely: list[str],
) -> None:
    """Given that baking ``names`` changed state, find the names whose loader
    changes it when run alone. A name whose loader fails or writes nothing
    alone is lonely."""
    if len(names) == 1:
        (name,) = names
        reasons[name] = f"its loader changes module state ({', '.join(changed)})"
        if name not in sink.wrote:
            lonely.append(name)
        return
    half = len(names) // 2
    for part in (names[:half], names[half:]):
        try:
            part_sink, part_changed = _bake(model, metas, part, targets)
        except Exception as e:
            if len(part) > 1:  # find the name that fails
                _bisect_state(model, metas, part, [], sink, targets, reasons, lonely)
            else:
                reasons[part[0]] = f"its loader fails when loaded alone ({e!r})"
                lonely.append(part[0])
            continue
        if part_changed:
            _bisect_state(
                model, metas, part, part_changed, part_sink, targets, reasons, lonely
            )
