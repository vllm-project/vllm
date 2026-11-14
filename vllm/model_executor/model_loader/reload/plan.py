# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load plan: which part of each checkpoint tensor this rank needs, and where
in live memory those bytes go.

`plan_loads` traces the model's `load_weights` once (`bake.py`) on
the real tensors, between `start_reload` and `finish_reload` so that it
sees the layout direct mode actually restored (the NCCL engine does it at its
first target update), and reduces each traced `copy_` to a box of the
checkpoint tensor and a strided region of the target's storage
(`geometry.py`). A name is
direct only if every byte it writes is a copy of one incoming byte; anything
else is a fallback, which the engine sends in full through `load_weights`.
The plan holds for every later update of the same model.

`load_planned` loads full checkpoint tensors through `get_dest`; the NCCL
engine uses it. `get_boxes` (with `get_dest`) is for engines that send or
read only the slices a rank needs (sharded RDT, NCCL M2N follow-ups).
"""

import heapq
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.model_executor.utils import storage_view

from .bake import Copy, Targets, Traced, trace_loads
from .direct import direct_reload_active, direct_reload_enabled
from .geometry import (
    Box,
    Fallback,
    Region,
    Write,
    box_and_region,
    byte_runs,
    merge_writes,
)
from .utils import stages_online_quant

if TYPE_CHECKING:
    from vllm.distributed.weight_transfer.base import ParamMeta

logger = init_logger(__name__)

__all__ = ["Box", "LoadPlan", "check_plannable", "load_planned", "plan_loads"]

Dest = tuple[tuple[Box, torch.Tensor], ...]


@dataclass(frozen=True)
class Entry:
    name: str
    shape: tuple[int, ...]  # from the metadata
    dtype: torch.dtype
    boxes: tuple[Box, ...] | None  # what this rank needs; None: nothing
    writes: tuple[Write, ...] | None  # where it goes; None: fallback
    reason: str | None = None  # why it is a fallback


@dataclass(frozen=True)
class LoadPlan:
    entries: dict[str, Entry]  # per declared name, in metadata order
    dests: dict[str, Dest]  # live tensors built at plan_loads; not data

    def get_boxes(self, name: str) -> tuple[Box, ...] | None:
        """Parts of the checkpoint tensor this rank needs; `None`: nothing.

        For engines that send or read only those slices (sharded RDT, NCCL
        M2N follow-ups); an engine that receives full tensors calls
        `load_planned` instead."""
        return self.entries[name].boxes

    def get_dest(self, name: str) -> list[tuple[Box, torch.Tensor]] | None:
        """(part of a box, live tensor) pairs: copying each part, reshaped to
        its tensor's shape, into that tensor is exactly what the loader would
        have written. `None`: fallback, send the full tensor through
        `load_weights`. Write only between `start_reload` and
        `finish_reload`; never read.

        `load_planned` copies full tensors through it; an engine that sends
        only `get_boxes` copies its slices the same way."""
        if self.entries[name].writes is None:
            return None
        return list(self.dests[name])


def plan_loads(model: torch.nn.Module, metadata: "Iterable[ParamMeta]") -> LoadPlan:
    """Trace `model.load_weights` over the declared checkpoint tensors once
    and return where each lands on this rank.

    Call between `start_reload` and `finish_reload`: the plan traces
    the real tensors in the layout `begin_direct_reload` applied. Incoming
    tensors are data-free traces, so nothing is loaded. Loaders' constant
    writes land, as on the `load_weights` path; so do writes into
    non-checkpoint GPU tensors fed by a read of a checkpoint tensor (a
    mirror), which are not restored: that name falls back and the update
    rewrites them. Module, parameter and buffer state a loader sets stays,
    as it would after `load_weights`.

    Raises:
        ValueError: a name is declared twice.
        RuntimeError: the model cannot be planned.

    """
    check_plannable(model)
    if not direct_reload_active(model):
        raise RuntimeError(
            "load plan: call plan_loads between start_reload and finish_reload; "
            "the plan takes its layout from what the direct reload restored"
        )
    metas: dict[str, ParamMeta] = {}
    for meta in metadata:
        if meta.name in metas:
            raise ValueError(f"load plan: {meta.name!r} is declared twice")
        metas[meta.name] = meta
    # TODO(#54477 F2): refuse when a name of the required set (cold-start
    # checkpoint names minus `is_unused_checkpoint_weight` and
    # VLLM_DIRECT_RELOAD_EXCLUDE_REGEX) is missing from `metas`.

    t0 = time.perf_counter()
    targets = Targets(model)
    traced = trace_loads(model, metas, targets)
    if traced.lonely:
        lonely = traced.lonely
        raise RuntimeError(
            f"load plan: {len(lonely)} fallback names write nothing themselves "
            f"(e.g. {lonely[:3]}); their loaders combine several checkpoint "
            "tensors. Move the combine into process_weights_after_loading."
        )
    for name, reason in _overlaps(traced, targets).items():
        traced.reasons.setdefault(name, reason)
    entries = {name: _entry(meta, traced, targets) for name, meta in metas.items()}
    dests = {
        name: tuple((w.src, _dest(targets, w, e.dtype)) for w in e.writes)
        for name, e in entries.items()
        if e.writes is not None
    }
    plan = LoadPlan(entries=entries, dests=dests)
    _report(plan, time.perf_counter() - t0)
    return plan


def load_planned(
    model: torch.nn.Module,
    plan: LoadPlan,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> None:
    """Load full checkpoint tensors as `plan` says: copy each direct name's
    boxes into its destinations, skip names this rank needs nothing of, and
    pass fallbacks to `model.load_weights`. Call between `start_reload`
    and `finish_reload`."""
    fallbacks = []
    for name, full in weights:
        dest = plan.get_dest(name)
        if dest is None:
            fallbacks.append((name, full))
        for box, tensor in dest or ():
            part = full[tuple(slice(start, stop) for start, stop in box)]
            tensor.copy_(part.reshape(tensor.shape))
    if fallbacks:
        model.load_weights(fallbacks)


def check_plannable(model: torch.nn.Module) -> None:
    """Refuse a configuration no load plan supports: a reload mode other than
    direct, and EPLB. Reads only configuration, so an engine can call it
    before `start_reload` changes the model. Direct reload's own refusals
    run in `begin_direct_reload`.

    Raises:
        RuntimeError: the configuration cannot be planned.

    """
    if not direct_reload_enabled():
        raise RuntimeError("load plan: requires VLLM_RELOAD_MODE=direct")
    for name, module in model.named_modules():
        moe_config = getattr(module, "moe_config", None)
        parallel = getattr(moe_config, "moe_parallel_config", None)
        if getattr(parallel, "enable_eplb", False):
            raise RuntimeError(f"load plan: EPLB is not supported ({name})")
        if stages_online_quant(module):
            # staged by `begin`: its loaders buffer the traced tensors
            raise RuntimeError(
                f"load plan: online quantization is not supported ({name})"
            )


def _entry(meta: "ParamMeta", traced: Traced, targets: Targets) -> Entry:
    name, shape = meta.name, tuple(meta.shape)
    full: Box = tuple((0, n) for n in shape)
    try:
        if name in traced.reasons:
            raise Fallback(traced.reasons[name])
        copies = traced.copies.get(name, ())
        found = [_write(c, shape, meta.dtype, targets) for c in copies]
    except Fallback as e:
        return Entry(name, shape, meta.dtype, (full,), None, str(e))
    writes = [w for w in found if w is not None]
    if not writes:  # another PP stage, a non-local expert, a tied alias
        return Entry(name, shape, meta.dtype, None, ())
    merged = merge_writes(writes)
    boxes = tuple(dict.fromkeys(w.src for w in merged))
    return Entry(name, shape, meta.dtype, boxes, tuple(merged))


def _write(
    copy: Copy, shape: tuple[int, ...], dtype: torch.dtype, targets: Targets
) -> Write | None:
    """`copy` as a box and region; `None` if it copies nothing.

    Raises:
        Fallback: it is not a plain copy of checkpoint bytes.

    """
    src = copy.src
    if src._geom is None:
        raise Fallback(src._lost or "not a view of the checkpoint tensor")
    if src.dtype != copy.dtype:
        raise Fallback(
            f"casts {src.dtype} to {copy.dtype} at copy_ "
            f"(send it as {copy.dtype} to make it direct)"
        )
    target = targets.by_storage.get(copy.storage)
    if target is None:
        raise Fallback("copies into a tensor that is not a checkpoint tensor")
    size = dtype.itemsize
    rel = copy.byte_offset - target.byte_offset
    if rel % size or target.byte_offset % size:
        raise Fallback("copies into a destination viewed with another element size")
    if 0 in copy.shape:
        return None
    geom = src._geom
    if tuple(geom.shape) != copy.shape:
        if geom.numel() != prod(copy.shape):
            raise Fallback("broadcasting copy_")
        try:
            geom = geom.view(copy.shape)
        except RuntimeError:
            raise Fallback("copy_ with a reshape that is not a view") from None
    dims = list(zip(copy.shape, geom.stride(), copy.stride))
    box, dst_shape, dst_stride = box_and_region(shape, geom.storage_offset(), dims)
    return Write(
        box, target.module, target.tensor, Region(rel // size, dst_shape, dst_stride)
    )


def _dest(targets: Targets, write: Write, dtype: torch.dtype) -> torch.Tensor:
    """The live tensor of `write`, in the checkpoint dtype."""
    target = targets.by_name[(write.module, write.tensor)]
    r = write.dst
    offset = target.byte_offset // dtype.itemsize + r.offset
    storage = target.real.untyped_storage()
    return storage_view(storage, offset, r.shape, r.stride, dtype)


def _overlaps(traced: Traced, targets: Targets) -> dict[str, str]:
    """Names whose writes share a byte with another write: direct writes land
    in any order, so these go through `load_weights` in arrival order."""
    runs: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    for name, copies in traced.copies.items():
        for c in copies:
            if c.storage in targets.by_storage and 0 not in c.shape:
                spans = byte_runs(c.byte_offset, c.shape, c.stride, c.dtype.itemsize)
                runs[c.storage] += [(a, b, name) for a, b in spans]
    reasons: dict[str, str] = {}
    for named in runs.values():
        named.sort()
        active: list[tuple[int, str]] = []  # heap of (stop, name)
        for start, stop, name in named:
            while active and active[0][0] <= start:
                heapq.heappop(active)
            for _, other in active:
                reasons.setdefault(name, f"writes overlap with {other!r}")
                reasons.setdefault(other, f"writes overlap with {name!r}")
            heapq.heappush(active, (stop, name))
    return reasons


def _nbytes(entry: Entry) -> int:
    """What the engine moves for `entry` on this rank."""
    if entry.writes is None:
        return prod(entry.shape) * entry.dtype.itemsize
    boxes = entry.boxes or ()
    return sum(prod(b - a for a, b in box) for box in boxes) * entry.dtype.itemsize


def _report(plan: LoadPlan, seconds: float) -> None:
    """Every name at debug; fallbacks grouped by reason and a summary at info."""
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # kind: [n, bytes]
    fallbacks: dict[str, list[str]] = defaultdict(list)
    for e in plan.entries.values():
        kind = "fallback" if e.writes is None else "direct" if e.boxes else "unneeded"
        nbytes = _nbytes(e)
        counts[kind][0] += 1
        counts[kind][1] += nbytes
        if e.reason is not None:
            fallbacks[e.reason].append(e.name)
        suffix = f": {e.reason}" if e.reason else ""
        logger.debug("load plan: %s %s (%d bytes)%s", kind, e.name, nbytes, suffix)
    for reason, names in fallbacks.items():
        logger.info(
            "load plan: %d fallback names, e.g. %s: %s", len(names), names[:3], reason
        )
    logger.info(
        "load plan: %d names: %d direct (%.1f MiB), %d fallback (%.1f MiB), "
        "%d not needed on this rank; traced in %.2f s",
        len(plan.entries),
        counts["direct"][0],
        counts["direct"][1] / (1 << 20),
        counts["fallback"][0],
        counts["fallback"][1] / (1 << 20),
        counts["unneeded"][0],
        seconds,
    )
