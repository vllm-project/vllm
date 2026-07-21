# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure descriptor planning for compact CPU KV transfers.

Port of the canonical descriptor-planning algorithm from the product
``compact_transfer.py`` at ``af2904cbd``, adapted to accepted Jasl
``CompactGroupSliceConfig`` / ``CompactSliceConfig`` types and current APIs.

One direction-neutral :class:`CompactTransferPlan`; the worker later swaps
pointer roles for store vs. load.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from vllm.utils.math_utils import cdiv
from vllm.v1.kv_offload.config import CompactGroupSliceConfig
from vllm.v1.kv_offload.cpu.common import CompactCPUAddress


@dataclass(frozen=True)
class CompactTransferPlan:
    """Byte pointers and sizes for one compact transfer submission.

    All arrays are immutable NumPy arrays of ``np.uint64``.  The worker
    swaps the semantic role of GPU and CPU pointers for store vs. load.
    """

    gpu_ptrs: np.ndarray
    cpu_ptrs: np.ndarray
    sizes: np.ndarray
    num_cpu_addresses: int

    def __post_init__(self) -> None:
        if self.gpu_ptrs.dtype != np.uint64:
            raise TypeError("gpu_ptrs must be uint64")
        if self.cpu_ptrs.dtype != np.uint64:
            raise TypeError("cpu_ptrs must be uint64")
        if self.sizes.dtype != np.uint64:
            raise TypeError("sizes must be uint64")
        if self.gpu_ptrs.shape != self.cpu_ptrs.shape:
            raise ValueError("gpu_ptrs and cpu_ptrs must have the same shape")
        if self.sizes.shape != self.gpu_ptrs.shape:
            raise ValueError("sizes must match pointer arrays")
        if self.sizes.size == 0:
            raise ValueError("plan must have at least one descriptor")
        if self.num_cpu_addresses <= 0:
            raise ValueError("num_cpu_addresses must be positive")

    @property
    def num_bytes(self) -> int:
        return int(self.sizes.sum())

    @property
    def num_descriptors(self) -> int:
        return int(self.sizes.size)


def plan_compact_transfer(
    *,
    gpu_base_ptr: int,
    gpu_row_stride: int,
    cpu_base_ptr: int,
    cpu_region_size: int,
    gpu_block_ids: np.ndarray,
    group_sizes: Sequence[int],
    block_indices: Sequence[int],
    compact_addresses: Sequence[CompactCPUAddress],
    group_slice_configs: Sequence[CompactGroupSliceConfig],
    block_size_factor: int,
) -> CompactTransferPlan:
    """Plan compact packed-slice copies without touching runtime state.

    Compact addresses are grouped by scheduler group in the same order as GPU
    block IDs.  Each address owns one static payload whose slices are laid out
    consecutively; each slice contains ``block_size_factor`` native GPU
    sub-blocks.  Partial transfers select sub-blocks inside that static layout.

    Parameters
    ----------
    gpu_base_ptr:
        Base GPU pointer for the packed KV cache allocation.
    gpu_row_stride:
        Byte stride of one packed GPU block row (all layers collapsed).
    cpu_base_ptr:
        Base CPU pointer for the compact CPU allocation.
    cpu_region_size:
        Total allocated CPU region size in bytes.
    gpu_block_ids:
        Actual GPU block IDs whose packed rows must be transferred.  One entry
        per native GPU block in the batch, ordered by group.
    group_sizes:
        Number of GPU block IDs belonging to each group.
    block_indices:
        Per-group partial-chunk offset (in native blocks) into the first
        compact address.  When a group's block transfer starts mid-address,
        ``block_indices[i]`` selects the first sub-block within the first
        compact address for that group.
    compact_addresses:
        Compact CPU addresses owned by the target slots, ordered by group
        then address index.  Each address may have non-contiguous physical
        spans.
    group_slice_configs:
        Order-preserving slice accounting per group, matching the transported
        ``compact_slice_accounting``.
    block_size_factor:
        Number of native GPU sub-blocks packed into one compact CPU address.

    Returns
    -------
    CompactTransferPlan
        Flat descriptor arrays with one entry per contiguous copy segment.
    """
    # -- Precondition validation ------------------------------------------------
    if block_size_factor <= 0:
        raise ValueError("block_size_factor must be positive")
    if gpu_row_stride <= 0 or cpu_region_size <= 0:
        raise ValueError("GPU row stride and CPU region size must be positive")
    if len(group_sizes) != len(group_slice_configs):
        raise ValueError("group_sizes must match compact group slice configs")
    if len(block_indices) != len(group_slice_configs):
        raise ValueError("block_indices must match compact group slice configs")
    if sum(group_sizes) != len(gpu_block_ids):
        raise ValueError("group_sizes must cover every GPU block ID")

    gpu_ptr_values: list[int] = []
    cpu_ptr_values: list[int] = []
    size_values: list[int] = []

    # Group compact addresses by group index so we can consume them in order.
    addresses_by_group: list[list[CompactCPUAddress]] = [
        [] for _ in group_slice_configs
    ]
    for address in compact_addresses:
        if address.group_idx < 0 or address.group_idx >= len(addresses_by_group):
            raise ValueError(
                f"compact address has out-of-range group index {address.group_idx}"
            )
        addresses_by_group[address.group_idx].append(address)

    gpu_cursor = 0
    cpu_cursor = 0
    for expected_group_idx, (group_size, block_idx, group_slice_cfg) in enumerate(
        zip(group_sizes, block_indices, group_slice_configs)
    ):
        if group_slice_cfg.group_idx != expected_group_idx:
            raise ValueError(
                "compact group slice configs must be contiguous and ordered"
            )
        if group_size < 0 or block_idx < 0:
            raise ValueError("group sizes and block indices must be non-negative")

        first_sub_block = block_idx % block_size_factor
        num_cpu_blocks = cdiv(first_sub_block + group_size, block_size_factor)
        group_addresses = addresses_by_group[expected_group_idx]
        if len(group_addresses) != num_cpu_blocks:
            raise ValueError(
                f"compact addresses for group {expected_group_idx} do not cover "
                f"the GPU group: expected {num_cpu_blocks}, "
                f"got {len(group_addresses)}"
            )

        slice_base = 0
        for compact_slice in group_slice_cfg.slices:
            real_bytes = compact_slice.real_bytes_per_gpu_block
            if real_bytes <= 0:
                raise ValueError("compact slice payload must be positive")
            if (
                compact_slice.offset_bytes < 0
                or compact_slice.offset_bytes + real_bytes > gpu_row_stride
            ):
                raise ValueError("compact slice falls outside the packed GPU row")

            for logical_idx in range(group_size):
                compact_idx, sub_idx = divmod(
                    first_sub_block + logical_idx, block_size_factor
                )
                address = group_addresses[compact_idx]
                required_payload = slice_base + block_size_factor * real_bytes
                if any(
                    span.byte_offset + span.allocated_length > cpu_region_size
                    for span in address.physical_spans
                ):
                    raise ValueError("compact address falls outside the CPU region")
                if address.logical_length < required_payload:
                    raise ValueError(
                        "compact address is smaller than its static slice layout"
                    )

                gpu_block_id = int(gpu_block_ids[gpu_cursor + logical_idx])
                if gpu_block_id < 0:
                    raise ValueError("GPU block IDs must be non-negative")
                gpu_ptr = (
                    gpu_base_ptr
                    + gpu_block_id * gpu_row_stride
                    + compact_slice.offset_bytes
                )
                logical_offset = slice_base + sub_idx * real_bytes
                remaining = real_bytes
                span_logical_base = 0
                for span in address.physical_spans:
                    span_end = span_logical_base + span.logical_length
                    if logical_offset < span_end and remaining:
                        within_span = max(0, logical_offset - span_logical_base)
                        chunk = min(remaining, span.logical_length - within_span)
                        gpu_ptr_values.append(gpu_ptr + (real_bytes - remaining))
                        cpu_ptr_values.append(
                            cpu_base_ptr + span.byte_offset + within_span
                        )
                        size_values.append(chunk)
                        logical_offset += chunk
                        remaining -= chunk
                    span_logical_base = span_end
                if remaining:
                    raise ValueError(
                        "compact physical spans do not cover the requested range"
                    )
            slice_base += block_size_factor * real_bytes

        # Verify that the computed slice geometry matches the transported
        # group payload charge.  compact_real_bytes_per_rank is UNSCALED
        # (per-native-GPU-block), so multiply by block_size_factor to obtain
        # the per-address payload that the accumulated slice_base represents.
        expected = group_slice_cfg.compact_real_bytes_per_rank * block_size_factor
        if slice_base != expected:
            raise ValueError(
                "compact slice layout does not match the group payload charge: "
                f"computed {slice_base}, "
                f"expected {expected}"
            )
        for address in group_addresses:
            if address.logical_length != slice_base:
                raise ValueError(
                    "compact address payload does not match its scheduler group"
                )

        gpu_cursor += group_size
        cpu_cursor += num_cpu_blocks

    if gpu_cursor != len(gpu_block_ids):
        raise ValueError("GPU block IDs were not fully consumed")
    if cpu_cursor != len(compact_addresses):
        raise ValueError("compact addresses were not fully consumed")

    gpu_arr = np.asarray(gpu_ptr_values, dtype=np.uint64)
    cpu_arr = np.asarray(cpu_ptr_values, dtype=np.uint64)
    sz_arr = np.asarray(size_values, dtype=np.uint64)
    gpu_arr.flags.writeable = False
    cpu_arr.flags.writeable = False
    sz_arr.flags.writeable = False
    return CompactTransferPlan(
        gpu_ptrs=gpu_arr,
        cpu_ptrs=cpu_arr,
        sizes=sz_arr,
        num_cpu_addresses=cpu_cursor,
    )
