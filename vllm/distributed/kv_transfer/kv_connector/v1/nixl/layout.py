# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CacheLayout: meta-tensor-based cache layout for NIXL descriptor generation.

A CacheLayout wraps a PyTorch meta tensor whose strides encode the physical
memory layout of a KV cache region.  Two annotations — ``iter_axes`` and
``shard_axes`` — classify each dimension's role:

- **iter_axes**: dimensions that ``descriptors()`` iterates over to produce
  separate descriptor groups (e.g. K/V split, blocks).  Ordered outer→inner:
  the first entry is the outermost loop.
- **shard_axes**: dimensions sliced for TP sharding (e.g. heads).
- Remaining dimensions contribute to each descriptor's byte size.

Invariant: all non-iter dimensions (shard + remaining) must be C-contiguous
from the innermost end.  This ensures each NIXL descriptor covers a single
contiguous byte span.  Validated at construction and after narrow().

Usage:
    layout = build_attn_layout(num_blocks=100, num_kv_heads=8, ...)
    local  = layout.narrow(2, offset, num_heads)   # TP head slice
    descs  = local.descriptors(base_addr=0x7f00, device_id=0)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _dim_after_remove(dims: tuple[int, ...], removed: int) -> tuple[int, ...]:
    """Adjust dimension indices after one axis is removed via select()."""
    return tuple(d - 1 if d > removed else d for d in dims if d != removed)


@dataclass(frozen=True)
class CacheLayout:
    """Symbolic layout for one NIXL memory region.

    Fields:
        meta        — meta-device tensor encoding shape and strides
        iter_axes   — dim indices iterated to produce descriptors (ordered
                      outer→inner)
        shard_axes  — dim indices sliced for TP sharding
    """

    meta: torch.Tensor
    iter_axes: tuple[int, ...]
    shard_axes: tuple[int, ...]

    def __post_init__(self) -> None:
        ndim = self.meta.ndim
        for d in self.iter_axes:
            assert 0 <= d < ndim, f"iter_axes dim {d} out of range for {ndim}-D tensor"
        for d in self.shard_axes:
            assert 0 <= d < ndim, f"shard_axes dim {d} out of range for {ndim}-D tensor"
        overlap = set(self.iter_axes) & set(self.shard_axes)
        assert not overlap, f"iter_axes and shard_axes overlap on dims {overlap}"

    @property
    def descriptor_size_bytes(self) -> int:
        """Byte size of one descriptor: product of all non-iter dim sizes.

        Includes shard dims because after narrow() they represent the local
        TP slice that each descriptor must cover.
        """
        elem = self.meta.element_size()
        size = 1
        iter_set = set(self.iter_axes)
        for d in range(self.meta.ndim):
            if d not in iter_set:
                size *= self.meta.shape[d]
        return size * elem

    def narrow(self, axis: int, start: int, length: int) -> CacheLayout:
        """Slice a dimension (typically a shard axis for TP).

        narrow() preserves strides, so if non-iter dims were C-contiguous
        before, they remain so after.  Re-validated for safety.
        """
        layout = CacheLayout(
            meta=self.meta.narrow(axis, start, length),
            iter_axes=self.iter_axes,
            shard_axes=self.shard_axes,
        )
        layout._check_payload_contiguity()
        return layout

    def select(self, axis: int, index: int) -> CacheLayout:
        """Collapse one iteration dimension."""
        assert axis in self.iter_axes, (
            f"select() only works on iter_axes {self.iter_axes}, got axis={axis}"
        )
        return CacheLayout(
            meta=self.meta.select(axis, index),
            iter_axes=_dim_after_remove(self.iter_axes, axis),
            shard_axes=_dim_after_remove(self.shard_axes, axis),
        )

    def sub_block(self, ratio: int) -> CacheLayout:
        """View N blocks as N*ratio sub-blocks, each 1/ratio the block_size.

        The block_size dimension is the first non-iter, non-shard dim.
        Dim 0 (blocks, always an iter axis) gets multiplied by ratio.
        """
        if ratio == 1:
            return self
        tagged = set(self.iter_axes) | set(self.shard_axes)
        bsz_candidates = [d for d in range(self.meta.ndim) if d not in tagged]
        assert bsz_candidates, (
            "sub_block requires at least one payload dim for block_size"
        )
        bsz_dim = bsz_candidates[0]

        assert self.meta.shape[bsz_dim] % ratio == 0, (
            f"block_size dim {self.meta.shape[bsz_dim]} not divisible by ratio {ratio}"
        )
        blocks_dim = self.iter_axes[-1]
        assert self.meta.stride(blocks_dim) % ratio == 0, (
            f"page stride {self.meta.stride(blocks_dim)} not divisible by "
            f"ratio {ratio}; sub-blocks would drift"
        )

        new_shape = list(self.meta.shape)
        new_shape[blocks_dim] *= ratio
        new_shape[bsz_dim] //= ratio

        new_strides = list(self.meta.stride())
        new_strides[blocks_dim] = self.meta.stride(blocks_dim) // ratio

        return CacheLayout.from_physical(
            shape=tuple(new_shape),
            strides=tuple(new_strides),
            dtype=self.meta.dtype,
            iter_axes=self.iter_axes,
            shard_axes=self.shard_axes,
        )

    def descriptors(
        self,
        base_addr: int,
        device_id: int,
    ) -> list[tuple[int, int, int]]:
        """Generate (addr, size, device_id) tuples for all blocks.

        Iterates over iter_axes in order (first = outermost loop).
        For blocks-first attention with iter_axes=(1, 0) on shape
        (N, 2, H, B, D), this yields K descriptors for all blocks
        (idx=0), then V descriptors for all blocks (idx=1).
        """
        return self._iter_descriptors(0, base_addr, device_id)

    def _iter_descriptors(
        self,
        iter_idx: int,
        base_addr: int,
        device_id: int,
    ) -> list[tuple[int, int, int]]:
        """Recursively iterate over iter_axes in order."""
        if iter_idx >= len(self.iter_axes):
            return self._leaf_descriptors(base_addr, device_id)

        axis = self.iter_axes[iter_idx]
        result: list[tuple[int, int, int]] = []
        for idx in range(self.meta.shape[axis]):
            sub = self.select(axis, idx)
            result.extend(
                sub._iter_descriptors(
                    iter_idx,  # same position: select() removed this axis,
                    # so the next iter axis slides into place
                    base_addr,
                    device_id,
                )
            )
        return result

    def _leaf_descriptors(
        self,
        base_addr: int,
        device_id: int,
    ) -> list[tuple[int, int, int]]:
        """Emit a single descriptor when no iter_axes remain."""
        assert len(self.iter_axes) == 0
        elem = self.meta.element_size()
        size = self.descriptor_size_bytes
        offset = int(self.meta.storage_offset()) * elem
        return [(base_addr + offset, size, device_id)]

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        iter_axes: tuple[int, ...],
        shard_axes: tuple[int, ...],
    ) -> CacheLayout:
        """Build from an actual allocated KV cache tensor (local path).

        Captures the tensor's shape, strides, and storage_offset onto a
        meta-device tensor — no extra memory, just shape metadata.
        """
        meta = torch.empty(0, dtype=tensor.dtype, device="meta").as_strided(
            tensor.shape,
            tensor.stride(),
            tensor.storage_offset(),
        )
        layout = cls(meta=meta, iter_axes=iter_axes, shard_axes=shard_axes)
        layout._check_payload_contiguity()
        return layout

    @classmethod
    def from_physical(
        cls,
        shape: tuple[int, ...],
        strides: tuple[int, ...],
        dtype: torch.dtype,
        iter_axes: tuple[int, ...],
        shard_axes: tuple[int, ...],
        offset_bytes: int = 0,
    ) -> CacheLayout:
        """Build from raw shape/strides (remote path)."""
        elem = torch.empty(0, dtype=dtype).element_size()
        meta = torch.as_strided(
            torch.empty(1, dtype=dtype, device="meta"),
            size=shape,
            stride=strides,
            storage_offset=offset_bytes // elem,
        )
        layout = cls(meta=meta, iter_axes=iter_axes, shard_axes=shard_axes)
        layout._check_payload_contiguity()
        return layout

    def _check_payload_contiguity(self) -> None:
        """Assert non-iter dims are C-contiguous from the innermost end.

        NIXL copies [addr, addr+size) as a flat byte span, so the
        non-iteration portion of each descriptor must be contiguous.
        """
        non_iter = [d for d in range(self.meta.ndim) if d not in set(self.iter_axes)]
        if not non_iter:
            return
        expected_stride = 1
        for d in reversed(non_iter):
            if self.meta.stride(d) != expected_stride:
                raise ValueError(
                    f"Non-iter dims {non_iter} must be C-contiguous, "
                    f"but dim {d} has stride {self.meta.stride(d)} "
                    f"(expected {expected_stride}). "
                    f"shape={tuple(self.meta.shape)}, "
                    f"strides={tuple(self.meta.stride())}"
                )
            expected_stride *= self.meta.shape[d]


def _c_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """C-contiguous (row-major) strides for a shape, in elements."""
    s, out = 1, []
    for d in reversed(shape):
        out.append(s)
        s *= d
    return tuple(reversed(out))


def build_attn_layout(
    num_blocks: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    dtype: torch.dtype,
    split_kv: bool = False,
    page_stride_bytes: int | None = None,
) -> CacheLayout:
    """Build a CacheLayout for one attention layer region (remote path).

    For split_kv=True (FA CUDA with blocks-first, FlashInfer virtual split):
        shape ``(N, 2, H, B, D)``
        iter_axes=(1, 0)  — K/V outer, blocks inner
        shard_axes=(2,)

    For split_kv=False (ROCM, MLA, cross-layer blocks):
        shape ``(N, H, B, D)``
        iter_axes=(0,)    — blocks only
        shard_axes=(1,)
    """
    inner: tuple[int, ...]
    iter_axes: tuple[int, ...]
    shard_axes: tuple[int, ...]
    if split_kv:
        inner = (2, num_kv_heads, block_size, head_size)
        iter_axes = (1, 0)
        shard_axes = (2,)
    else:
        inner = (num_kv_heads, block_size, head_size)
        iter_axes = (0,)
        shard_axes = (1,)

    shape = (num_blocks, *inner)
    inner_strides = _c_strides(inner)

    if page_stride_bytes is not None:
        elem_size = torch.empty(0, dtype=dtype).element_size()
        strides = (page_stride_bytes // elem_size, *inner_strides)
    else:
        strides = _c_strides(shape)

    return CacheLayout.from_physical(
        shape=shape,
        strides=strides,
        dtype=dtype,
        iter_axes=iter_axes,
        shard_axes=shard_axes,
    )
