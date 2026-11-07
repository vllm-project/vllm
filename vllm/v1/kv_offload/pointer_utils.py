# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility functions for resolving device (gpu,tpu,etc) pointers."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np

from vllm.utils.math_utils import cdiv
from vllm.v1.kv_offload.base import (
    DevicePointerGroupInfo,
    DevicePointers,
)

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import (
        CanonicalKVCaches,
        GPULoadStoreSpec,
    )


def iter_pointer_groups(
    device_ptrs: DevicePointers,
    blocks_per_chunk: int,
) -> Iterator[DevicePointerGroupInfo]:
    """Iterate non-empty groups with pre-computed chunk/skip metadata."""
    dev_ptr_offset = 0
    for group_idx, (group_size, n_data_refs, block_idx) in enumerate(
        zip(
            device_ptrs.group_block_counts,
            device_ptrs.group_data_ref_counts,
            device_ptrs.block_indices,
        )
    ):
        if group_size == 0:
            continue
        skip = block_idx % blocks_per_chunk
        n_chunks = cdiv(group_size + skip, blocks_per_chunk)
        yield DevicePointerGroupInfo(
            group_idx=group_idx,
            group_size=group_size,
            n_data_refs=n_data_refs,
            skip=skip,
            n_chunks=n_chunks,
            dev_ptr_offset=dev_ptr_offset,
        )
        dev_ptr_offset += group_size * n_data_refs


def resolve_device_pointers(
    device_spec: GPULoadStoreSpec,
    kv_caches: CanonicalKVCaches,
) -> DevicePointers:
    """Convert logical block IDs to device memory pointers.
    GPU blocks_per_chunk is always 1, so resolution is a direct
    vectorized lookup: base_ptr + block_id * row_stride.
    """
    block_ids = device_spec.block_ids
    group_sizes = device_spec.group_sizes
    block_indices = device_spec.block_indices
    group_data_refs = kv_caches.group_data_refs

    assert len(group_sizes) == len(group_data_refs)
    assert len(block_indices) == len(group_sizes)

    num_copy_ops = sum(gs * len(refs) for gs, refs in zip(group_sizes, group_data_refs))

    ptrs = np.empty(num_copy_ops, dtype=np.uint64)
    sizes = np.empty(num_copy_ops, dtype=np.uint64)

    blk_offset = 0
    op_idx = 0

    for group_size, data_refs in zip(group_sizes, group_data_refs):
        if group_size == 0:
            continue
        group_block_ids = block_ids[blk_offset : blk_offset + group_size]
        group_block_ids_u64 = group_block_ids.astype(np.uint64)
        tensors = [kv_caches.tensors[ref.tensor_idx].tensor for ref in data_refs]
        base_ptrs = np.array([t.data_ptr() for t in tensors], dtype=np.uint64)
        row_strides = np.array([t.stride(0) for t in tensors], dtype=np.uint64)
        page_sizes = np.array(
            [ref.page_size_bytes for ref in data_refs], dtype=np.uint64
        )

        # Fill all refs at once through (num_refs, group_size) output views
        shape = (len(data_refs), group_size)
        end_idx = op_idx + len(data_refs) * group_size
        group_ptrs = ptrs[op_idx:end_idx].reshape(shape)
        np.multiply(group_block_ids_u64, row_strides[:, None], out=group_ptrs)
        group_ptrs += base_ptrs[:, None]
        sizes[op_idx:end_idx].reshape(shape)[:] = page_sizes[:, None]
        op_idx = end_idx
        blk_offset += group_size

    assert blk_offset == len(block_ids)
    assert op_idx == num_copy_ops

    return DevicePointers(
        ptrs=ptrs,
        sizes=sizes,
        group_block_counts=tuple(group_sizes),
        group_data_ref_counts=tuple(len(refs) for refs in group_data_refs),
        block_indices=tuple(block_indices),
    )
