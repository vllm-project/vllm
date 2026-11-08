# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integer geometry of the load plan: one traced `copy_` as a box of the
checkpoint tensor plus a strided destination region, merging of writes, and
the byte runs a write covers. No torch state."""

import itertools
from dataclasses import dataclass
from math import prod

Box = tuple[tuple[int, int], ...]  # half-open (start, stop) per dim, HF coords


class Fallback(Exception):
    """A name cannot be written directly; the message is the reported reason."""


@dataclass(frozen=True)
class Region:
    """`as_strided` arguments relative to the target's storage offset, in
    checkpoint-dtype elements. `shape` is the shape of the write's box, or
    that shape with some dimensions split."""

    offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]


@dataclass(frozen=True)
class Write:
    src: Box  # part of one of the entry's boxes
    module: str  # target tensor, by name
    tensor: str
    dst: Region


def box_and_region(
    ckpt_shape: tuple[int, ...],
    src_offset: int,
    dims: list[tuple[int, int, int]],
) -> tuple[Box, tuple[int, ...], tuple[int, ...]]:
    """Reduce one `copy_` to a box of the (contiguous) checkpoint tensor and
    the destination shape and strides of the box's elements, in row-major box
    order.

    `dims` are the copied tensor's (size, source stride, destination
    stride); `src_offset` is where its first element sits in the checkpoint
    tensor. Each dim is assigned to the checkpoint dim its source stride steps
    through, split where it spans several (`flatten`); the parts assigned to
    one checkpoint dim must tile a contiguous range of it.

    Example: `P.copy_(W.t())` with `W` [4, 6] and `P` [6, 4] contiguous
    has dims `[(6, 1, 4), (4, 6, 1)]`: the box is `((0, 4), (0, 6))` and
    the destination `shape (4, 6), stride (1, 4)`. A box dim the loader
    splits and sends to unrelated places stays split: `6` becomes
    `(2, 3)` with two destination strides.

    Raises:
        Fallback: the copy is not a view of a box (step slice, expand, ...).

    """
    # elements per step of each checkpoint dim
    ckpt_strides = [prod(ckpt_shape[j + 1 :]) for j in range(len(ckpt_shape))]
    start = [(src_offset // c) % n for c, n in zip(ckpt_strides, ckpt_shape)]
    # per checkpoint dim: (step within it, size, destination stride)
    parts: list[list[tuple[int, int, int]]] = [[] for _ in ckpt_shape]
    for size, src_stride, dst_stride in dims:
        if size == 1:
            continue
        if src_stride <= 0:
            raise Fallback("repeats source elements (expand or broadcast)")
        if dst_stride <= 0:
            raise Fallback("writes one destination element twice")
        while True:
            dim = next(j for j, c in enumerate(ckpt_strides) if c <= src_stride)
            if src_stride % ckpt_strides[dim]:
                raise Fallback("source slice is not a box")
            step = src_stride // ckpt_strides[dim]
            if (size - 1) * step < ckpt_shape[dim]:
                parts[dim].append((step, size, dst_stride))
                break
            # spans checkpoint dims dim-1 and dim: split off the part in dim
            inner = ckpt_shape[dim] // step
            if dim == 0 or ckpt_shape[dim] % step or size % inner or start[dim]:
                raise Fallback("source slice is not a box")
            parts[dim].append((step, inner, dst_stride))
            size, src_stride, dst_stride = (
                size // inner,
                src_stride * inner,
                dst_stride * inner,
            )

    box, dst_shape, dst_strides = [], [], []
    for dim, dim_parts in enumerate(parts):
        dim_parts.sort(reverse=True)  # outer first
        extent = 1
        for step, size, _ in reversed(dim_parts):
            if step != extent:
                raise Fallback("source slice has a step or is not a box")
            extent *= size
        if start[dim] + extent > ckpt_shape[dim]:
            raise Fallback("source slice is not a box")
        box.append((start[dim], start[dim] + extent))
        merged: list[tuple[int, int]] = []  # (size, stride), merged where possible
        for _, size, stride in dim_parts or [(1, 1, 1)]:
            if merged and merged[-1][1] == stride * size:
                merged[-1] = (merged[-1][0] * size, stride)
            else:
                merged.append((size, stride))
        dst_shape += [n for n, _ in merged]
        dst_strides += [s for _, s in merged]
    return tuple(box), tuple(dst_shape), tuple(dst_strides)


def merge_writes(writes: list[Write]) -> list[Write]:
    """Merge writes of one name that continue one another (a loader's
    per-expert or per-row loop) into single writes."""
    out: list[Write] = []
    for w in writes:
        i = 0
        while i < len(out):
            merged = _merge(out[i], w) or _merge(w, out[i])
            if merged is None:
                i += 1
            else:
                del out[i]
                w, i = merged, 0  # the result may continue another write
        out.append(w)
    return sorted(out, key=lambda w: (w.module, w.tensor, w.dst.offset))


def _merge(a: Write, b: Write) -> Write | None:
    """`a` and `b` as one write if `b` continues `a` along exactly one
    box dim, into the same target, with an equal destination stride there."""
    rank = len(a.src)
    if (a.module, a.tensor) != (b.module, b.tensor) or not (
        len(a.dst.shape) == len(b.dst.shape) == rank
    ):
        return None  # other target, or a split dim
    diff = [k for k in range(rank) if a.src[k] != b.src[k]]
    if len(diff) != 1 or a.src[diff[0]][1] != b.src[diff[0]][0]:
        return None
    k = diff[0]
    for j in range(rank):
        if j != k and (
            a.dst.shape[j] != b.dst.shape[j]
            or (a.dst.shape[j] > 1 and a.dst.stride[j] != b.dst.stride[j])
        ):
            return None
    na, nb = a.dst.shape[k], b.dst.shape[k]
    delta = b.dst.offset - a.dst.offset
    stride_a = a.dst.stride[k] if na > 1 else delta
    stride_b = b.dst.stride[k] if nb > 1 else stride_a
    if stride_a != stride_b or delta != stride_a * na:
        return None
    src = tuple((a.src[j][0], b.src[j][1]) if j == k else a.src[j] for j in range(rank))
    shape = tuple(na + nb if j == k else n for j, n in enumerate(a.dst.shape))
    stride = tuple(stride_a if j == k else s for j, s in enumerate(a.dst.stride))
    return Write(src, a.module, a.tensor, Region(a.dst.offset, shape, stride))


_MAX_RUNS = 1 << 16


def byte_runs(
    byte_offset: int, shape: tuple[int, ...], stride: tuple[int, ...], itemsize: int
) -> list[tuple[int, int]]:
    """The bytes a strided write covers as [start, stop) runs; one covering
    span when there would be more than `_MAX_RUNS`."""
    dims = sorted(((n, s) for n, s in zip(shape, stride) if n > 1), key=lambda x: -x[1])
    run = 1  # contiguous elements at the innermost end
    while dims and dims[-1][1] == run:
        run *= dims.pop()[0]
    if prod(n for n, _ in dims) > _MAX_RUNS:
        span = sum((n - 1) * s for n, s in dims) + run
        return [(byte_offset, byte_offset + span * itemsize)]
    return [
        (byte_offset + sum(o) * itemsize, byte_offset + (sum(o) + run) * itemsize)
        for o in itertools.product(*(range(0, n * s, s) for n, s in dims))
    ]
