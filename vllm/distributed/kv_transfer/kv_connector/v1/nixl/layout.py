# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CacheLayout: meta-tensor-based cache layout for NIXL descriptor generation.

A CacheLayout wraps a PyTorch meta tensor whose strides encode the physical
memory layout of a KV cache region.  A single annotation — ``shard_axis`` —
separates the shape into iteration dimensions (before) and payload (from
shard_axis onward).

Blocks are always the outermost dimension (axis 0).  Any dimensions between
axis 0 and shard_axis are "split axes" that ``descriptors()`` iterates over
automatically (e.g. the K/V ``2`` in blocks-first attention).

Usage:
    layout = build_attn_layout(num_blocks=100, num_kv_heads=8, ...)
    local  = layout.narrow(2, offset, num_heads)   # TP head slice
    descs  = local.descriptors(base_addr=0x7f00, device_id=0)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CacheLayout:
    """Symbolic layout for one NIXL memory region.

    Fields:
        meta        — meta-device tensor encoding shape and strides
        shard_axis  — dimension sharded across TP ranks (heads, etc.)

    Axis 0 is always the block (page) dimension.  Dimensions in
    ``range(1, shard_axis)`` are iteration dimensions that ``descriptors()``
    recurses through.  Everything from ``shard_axis`` onward is the
    per-descriptor payload.
    """

    meta: torch.Tensor
    # Currently a single int for 1-D TP sharding.  Can be extended to a
    # tuple or dict keyed by mesh axis to support multi-axis sharding
    # (e.g. TP + PP), mirroring DistSpec's n-d mesh placements.
    shard_axis: int

    @property
    def descriptor_size_bytes(self) -> int:
        """Payload size of one descriptor (everything from shard_axis onward)."""
        elem = self.meta.element_size()
        payload = 1
        for d in range(self.shard_axis, self.meta.ndim):
            payload *= self.meta.shape[d]
        return payload * elem

    def narrow(self, axis: int, start: int, length: int) -> CacheLayout:
        return CacheLayout(
            meta=self.meta.narrow(axis, start, length),
            shard_axis=self.shard_axis,
        )

    def select(self, axis: int, index: int) -> CacheLayout:
        """Collapse one iteration dimension (between 0 and shard_axis)."""
        assert 0 < axis < self.shard_axis, (
            f"select() only works on iteration dims (1..{self.shard_axis - 1}), "
            f"got axis={axis}"
        )
        return CacheLayout(
            meta=self.meta.select(axis, index),
            shard_axis=self.shard_axis - 1,
        )

    def sub_block(self, ratio: int) -> CacheLayout:
        """View N blocks as N*ratio sub-blocks, each 1/ratio the block_size.

        The block_size dimension (at shard_axis + 1) is divided by ratio,
        and stride(0) is divided by ratio so sub-blocks tile the same memory.
        """
        bsz_dim = self.shard_axis + 1
        assert self.meta.shape[bsz_dim] % ratio == 0

        new_shape = list(self.meta.shape)
        new_shape[0] *= ratio
        new_shape[bsz_dim] //= ratio

        new_strides = list(self.meta.stride())
        new_strides[0] = self.meta.stride(0) // ratio

        return CacheLayout.from_physical(
            shape=tuple(new_shape),
            strides=tuple(new_strides),
            dtype=self.meta.dtype,
            shard_axis=self.shard_axis,
        )

    def descriptors(
        self,
        base_addr: int,
        device_id: int,
    ) -> list[tuple[int, int, int]]:
        """Generate (addr, size, device_id) tuples for all blocks.

        Recursively iterates all dimensions in ``range(1, shard_axis)``
        (the "split axes"), then emits one descriptor per block.
        For blocks-first attention this yields K descriptors for all blocks,
        then V descriptors for all blocks.
        """
        if self.shard_axis <= 1:
            return self._block_descriptors(base_addr, device_id)

        result: list[tuple[int, int, int]] = []
        for idx in range(self.meta.shape[1]):
            sub = self.select(1, idx)
            result.extend(sub.descriptors(base_addr, device_id))
        return result

    def _block_descriptors(
        self,
        base_addr: int,
        device_id: int,
    ) -> list[tuple[int, int, int]]:
        """One descriptor per block. shard_axis <= 1 guaranteed."""
        elem = self.meta.element_size()
        block_stride = self.meta.stride(0) * elem
        offset = self.meta.storage_offset() * elem
        size = self.descriptor_size_bytes
        return [
            (base_addr + offset + b * block_stride, size, device_id)
            for b in range(self.meta.shape[0])
        ]

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, shard_axis: int) -> CacheLayout:
        """Build from an actual allocated KV cache tensor (local path).

        Captures the tensor's shape, strides, and storage_offset onto a
        meta-device tensor — no extra memory, just shape metadata.
        """
        meta = torch.empty(0, dtype=tensor.dtype, device="meta").as_strided(
            tensor.shape,
            tensor.stride(),
            tensor.storage_offset(),
        )
        return cls(meta=meta, shard_axis=shard_axis)

    @classmethod
    def from_physical(
        cls,
        shape: tuple[int, ...],
        strides: tuple[int, ...],
        dtype: torch.dtype,
        shard_axis: int,
        offset_bytes: int = 0,
    ) -> CacheLayout:
        """Build from raw shape/strides (remote path)."""
        elem = torch.tensor([], dtype=dtype).element_size()
        meta = torch.as_strided(
            torch.empty(1, dtype=dtype, device="meta"),
            size=shape,
            stride=strides,
            storage_offset=offset_bytes // elem,
        )
        return cls(meta=meta, shard_axis=shard_axis)


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
    virtually_split_kv: bool,
    page_stride_bytes: int | None = None,
) -> CacheLayout:
    """Build a CacheLayout for one attention layer region (remote path).

    For virtually_split_kv=True (FlashInfer, FA CUDA with blocks-first):
        shape ``(N, 2, H, B, D)``  shard_axis=2
        The ``2`` between block and shard axes -> K/V split in descriptors().

    For virtually_split_kv=False (ROCM, MLA, cross-layer blocks):
        shape ``(N, H, B, D)``     shard_axis=1
        No split dims -> single descriptor group per block.
    """
    inner: tuple[int, ...]
    if virtually_split_kv:
        inner = (2, num_kv_heads, block_size, head_size)
        shard_axis = 2
    else:
        inner = (num_kv_heads, block_size, head_size)
        shard_axis = 1

    shape = (num_blocks, *inner)
    inner_strides = _c_strides(inner)

    if page_stride_bytes is not None:
        elem_size = torch.tensor([], dtype=dtype).element_size()
        strides = (page_stride_bytes // elem_size, *inner_strides)
    else:
        strides = _c_strides(shape)

    return CacheLayout.from_physical(
        shape=shape,
        strides=strides,
        dtype=dtype,
        shard_axis=shard_axis,
    )
