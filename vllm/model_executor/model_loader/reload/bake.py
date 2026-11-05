# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The load plan's bake: run ``load_weights`` once over ``TraceTensor``s with
every parameter and persistent buffer replaced by a meta stand-in of itself as
``begin_direct_reload`` left it, and record each ``copy_`` and every reason a
name cannot be written directly."""

from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple

import torch
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode

from vllm.model_executor.utils import storage_view

from .direct import loadable_slots
from .trace import TraceSink, TraceTensor

if TYPE_CHECKING:
    from vllm.distributed.weight_transfer.base import ParamMeta

_META = torch.device("meta")


# ---- stand-ins


@dataclass
class Target:
    """A checkpoint tensor, as a write destination."""

    module: str
    tensor: str
    real: torch.Tensor
    byte_offset: int  # in its storage, as `begin_direct_reload` viewed it


class _Slot(NamedTuple):
    tensors: dict[str, torch.Tensor]  # a module's _parameters or _buffers
    key: str
    original: torch.Tensor
    stand_in: torch.Tensor


class StandIns:
    """Every parameter and persistent buffer as it is now, on meta: one meta
    storage per real storage, so tensors sharing storage (tied weights) alias
    exactly as in the model.

    Built after ``begin_direct_reload``, so each tensor's current layout and
    loader attributes are the ones ``begin`` applied: the stand-ins copy them
    rather than re-deriving the checkpoint layout from the cold-start record
    (``begin`` uses the current storage offset of a tensor whose storage
    moved, for example by a ``set_`` into a fused buffer)."""

    def __init__(self, model: torch.nn.Module):
        self.slots: list[_Slot] = []
        self.by_storage: dict[int, Target] = {}  # meta storage key
        self.by_name: dict[tuple[str, str], Target] = {}
        stand_ins: dict[int, torch.Tensor] = {}
        meta_storages: dict[int, torch.UntypedStorage] = {}
        for module_name, tensors, key, t in loadable_slots(model):
            if id(t) not in stand_ins:
                storage = t.untyped_storage()
                if storage._cdata not in meta_storages:
                    meta_storages[storage._cdata] = torch.UntypedStorage(
                        storage.nbytes(), device=_META
                    )
                meta = meta_storages[storage._cdata]
                stand_ins[id(t)] = self._stand_in(module_name, key, t, meta)
            self.slots.append(_Slot(tensors, key, t, stand_ins[id(t)]))

    def _stand_in(
        self, module_name: str, key: str, t: torch.Tensor, meta: torch.UntypedStorage
    ) -> torch.Tensor:
        offset = t.storage_offset()
        stand_in = storage_view(meta, offset, t.shape, t.stride(), t.dtype)
        stand_in.__class__ = t.__class__
        stand_in.__dict__ = t.__dict__.copy()
        if hasattr(t, "_ckpt"):  # a checkpoint tensor: a write target
            target = Target(module_name, key, t, offset * t.dtype.itemsize)
            # writes into this storage are labelled with its first target
            self.by_storage.setdefault(meta._cdata, target)
            self.by_name[(module_name, key)] = target
        return stand_in

    @contextmanager
    def swapped(self) -> Iterator[None]:
        for slot in self.slots:
            slot.tensors[slot.key] = slot.stand_in
        try:
            yield
        finally:
            for slot in self.slots:
                slot.tensors[slot.key] = slot.original


def _storage_key(t: torch.Tensor) -> int | None:
    return None if isinstance(t, TraceTensor) else t.untyped_storage()._cdata


# ---- the sink


class _ValueRead(Exception):
    """A loader needs a traced tensor's values; the bake cannot continue."""


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
    stand_ins: StandIns
    copies: dict[str, list[Copy]] = field(default_factory=lambda: defaultdict(list))
    fallback: dict[str, str] = field(default_factory=dict)
    # Kept even though reads also raise: a loader may catch the exception and
    # carry on, and the name must still be rerun without.
    value_reads: list[str] = field(default_factory=list)
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
        self.value_reads.append(trace._name)
        raise _ValueRead

    def unsupported(
        self, func: Callable, args: tuple, kwargs: dict, traces: list[TraceTensor]
    ) -> Any:
        """Mark the names fallbacks and run ``func`` on meta stand-ins; tensor
        results become traces of the first name, with no geometry."""
        if _mutates_first_arg(func) and not isinstance(args[0], TraceTensor):
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

        flat, spec = pytree.tree_flatten((args, kwargs))
        meta_flat = [to_meta(a) for a in flat]
        meta_args, meta_kwargs = pytree.tree_unflatten(meta_flat, spec)
        try:
            out = func(*meta_args, **meta_kwargs)
        except Exception as e:  # data-dependent: needs values
            self.value_reads.append(traces[0]._name)
            raise _ValueRead from e
        originals = {id(m): a for m, a in zip(meta_flat, flat)}
        first = traces[0]

        def wrap(o: Any) -> Any:
            if not isinstance(o, torch.Tensor):
                return o
            if id(o) in originals:  # in place: return the original argument
                return originals[id(o)]
            return TraceTensor(first._name, o.shape, o.dtype, self, None, reason)

        return pytree.tree_map(wrap, out)


def _mutates_first_arg(func: Any) -> bool:
    schema = getattr(func, "_schema", None)
    if schema is None or not schema.arguments:
        return False
    alias = schema.arguments[0].alias_info
    return alias is not None and alias.is_write


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
# that are not checkpoint tensors. Post-load processing owns padding.
_CONSTANT_WRITES = {
    torch.ops.aten.copy_,
    torch.ops.aten.fill_,
    torch.ops.aten.zero_,
    torch.ops.aten.index_put_,
}


class _TargetGuard(TorchDispatchMode):
    """Catches loaders that read their destination, such as
    ``composed_weight_loader``'s ``param.copy_(fn(param))``: the direct path
    would skip ``fn``. The last name written into that storage (or the name
    being loaded) becomes a fallback."""

    def __init__(self, sink: _PlanSink):
        super().__init__()
        self.sink = sink

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        packet = func.overloadpacket
        if not func.is_view and packet not in _METADATA_OPS:
            # copy_(target, x) and friends write the target but read only x
            read = args[1:] if packet in _CONSTANT_WRITES else args
            self._check_reads(packet, pytree.arg_tree_leaves(*read, **kwargs))
        return func(*args, **kwargs)

    def _check_reads(self, packet: Any, leaves: list) -> None:
        if any(isinstance(a, TraceTensor) for a in leaves):
            return  # the trace's own handler marks its name
        for a in leaves:
            key = _storage_key(a) if isinstance(a, torch.Tensor) else None
            if key is not None and key in self.sink.stand_ins.by_storage:
                name = self.sink.last_writer.get(key, self.sink.current)
                if name is not None:
                    self.sink.mark(name, f"its loader reads the destination ({packet})")
                return


# ---- module state


class _Snapshot:
    """Module and parameter attributes and CPU tensors, compared and restored
    after a bake: a loader that sets such state will not run on the direct
    path."""

    def __init__(self, model: torch.nn.Module):
        self.dicts: list[tuple[str, dict, dict[str, _Value]]] = []
        for module_name, module in model.named_modules():
            self._add(module_name, module.__dict__)
            for p_name, p in module._parameters.items():
                if p is not None:
                    self._add(f"{module_name}.{p_name}", p.__dict__)

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
        return changed


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
    stand_ins: StandIns,
) -> tuple[_PlanSink, list[str]]:
    """One traced ``load_weights`` over ``names``; returns the sink and the
    state it changed (restored)."""
    sink = _PlanSink(stand_ins)

    def weights() -> Iterator[tuple[str, torch.Tensor]]:
        for name in names:
            meta = metas[name]
            sink.current = name
            shape = torch.Size(meta.shape)
            yield name, TraceTensor.checkpoint(name, shape, meta.dtype, sink)

    with stand_ins.swapped():
        snapshot = _Snapshot(model)
        try:
            with torch.no_grad(), _TargetGuard(sink):
                model.load_weights(weights())
        except _ValueRead:
            pass  # in sink.value_reads; the caller reruns without the name
        finally:
            changed = snapshot.restore()
    return sink, changed


@dataclass
class Traced:
    copies: dict[str, list[Copy]]  # per name, from the full bake
    reasons: dict[str, str]  # fallback names and why
    lonely: list[str]  # fallbacks that write nothing when loaded alone


def trace_loads(
    model: torch.nn.Module, metas: dict[str, "ParamMeta"], stand_ins: StandIns
) -> Traced:
    """Bake all names. Each name whose loader reads tensor values reruns the
    whole bake without it (one full bake per such name). Names that change
    module state are then found by bisecting, only if the full bake did."""
    reasons: dict[str, str] = {}
    names = list(metas)
    while True:
        try:
            sink, changed = _bake(model, metas, names, stand_ins)
        except Exception as e:
            raise RuntimeError(f"load plan: tracing load_weights failed: {e!r}") from e
        if not sink.value_reads:
            break
        for name in sink.value_reads:
            reasons.setdefault(
                name, "its loader reads tensor values (.item(), bool(), ...)"
            )
        names = [n for n in names if n not in reasons]

    lonely = [n for n in sink.fallback if n not in sink.wrote]
    if changed:
        _bisect_state(model, metas, names, changed, sink, stand_ins, reasons, lonely)
    for name, reason in sink.fallback.items():
        reasons.setdefault(name, reason)
    return Traced(sink.copies, reasons, lonely)


def _bisect_state(
    model: torch.nn.Module,
    metas: dict[str, "ParamMeta"],
    names: list[str],
    changed: list[str],
    sink: _PlanSink,
    stand_ins: StandIns,
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
            part_sink, part_changed = _bake(model, metas, part, stand_ins)
        except Exception as e:
            if len(part) > 1:  # find the name that fails
                _bisect_state(model, metas, part, [], sink, stand_ins, reasons, lonely)
            else:
                reasons[part[0]] = f"its loader fails when loaded alone ({e!r})"
                lonely.append(part[0])
            continue
        if part_changed:
            _bisect_state(
                model, metas, part, part_changed, part_sink, stand_ins, reasons, lonely
            )
