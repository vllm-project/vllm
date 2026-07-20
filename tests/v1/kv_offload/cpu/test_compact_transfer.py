# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused tests for the pure compact transfer descriptor planner.

Verifies the planning algorithm ported from the canonical product
``compact_transfer.py`` at ``af2904cbd``, adapted to accepted Jasl
``CompactGroupSliceConfig`` types and current APIs.

No mocks, global module patches, conftest, shared TP1, bounded-tail,
ExtentAllocator, compact_layout, scheduler, or lifecycle dependencies.
"""

import numpy as np
import pytest

from vllm.v1.kv_offload.config import (
    CompactGroupSliceConfig,
    CompactSliceConfig,
)
from vllm.v1.kv_offload.cpu.common import (
    CompactCPUAddress,
    CompactCPUAddressSpan,
)
from vllm.v1.kv_offload.cpu.compact_transfer import (
    CompactTransferPlan,
    plan_compact_transfer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _addr(
    byte_offset: int,
    logical_length: int,
    allocated_length: int,
    group_idx: int = 0,
    spans: tuple[CompactCPUAddressSpan, ...] = (),
) -> CompactCPUAddress:
    """Convenience: build a CompactCPUAddress."""
    return CompactCPUAddress(
        byte_offset=byte_offset,
        logical_length=logical_length,
        allocated_length=allocated_length,
        group_idx=group_idx,
        spans=spans,
    )


def _span(
    byte_offset: int, logical_length: int, allocated_length: int
) -> CompactCPUAddressSpan:
    return CompactCPUAddressSpan(
        byte_offset=byte_offset,
        logical_length=logical_length,
        allocated_length=allocated_length,
    )


def _single_slice_cfg(
    group_idx: int = 0,
    real_bytes: int = 80,
    padded: int = 100,
    layer_name: str = "k",
) -> tuple[CompactGroupSliceConfig, ...]:
    """One group with one packed slice.

    ``compact_real_bytes_per_rank`` is UNSCALED (per-native-GPU-block).
    Callers must multiply by ``block_size_factor`` when constructing
    ``CompactCPUAddress.logical_length``.
    """
    return (
        CompactGroupSliceConfig(
            group_idx=group_idx,
            slices=(
                CompactSliceConfig(
                    offset_bytes=0,
                    real_bytes_per_gpu_block=real_bytes,
                    padded_bytes_per_gpu_block=padded,
                    layer_name=layer_name,
                ),
            ),
            compact_real_bytes_per_rank=real_bytes,
            compact_padded_bytes_per_rank=padded,
        ),
    )


def _two_slice_cfg(
    group_idx: int = 0,
    real_k: int = 80,
    real_v: int = 40,
    padded_k: int = 100,
    padded_v: int = 60,
) -> tuple[CompactGroupSliceConfig, ...]:
    """One group with two packed slices (k and v).

    ``compact_real_bytes_per_rank`` is UNSCALED (per-native-GPU-block).
    """
    k = CompactSliceConfig(
        offset_bytes=0,
        real_bytes_per_gpu_block=real_k,
        padded_bytes_per_gpu_block=padded_k,
        layer_name="k",
    )
    v = CompactSliceConfig(
        offset_bytes=padded_k,
        real_bytes_per_gpu_block=real_v,
        padded_bytes_per_gpu_block=padded_v,
        layer_name="v",
    )
    total_real = real_k + real_v
    total_padded = padded_k + padded_v
    return (
        CompactGroupSliceConfig(
            group_idx=group_idx,
            slices=(k, v),
            compact_real_bytes_per_rank=total_real,
            compact_padded_bytes_per_rank=total_padded,
        ),
    )


# ---------------------------------------------------------------------------
# Plan construction validation
# ---------------------------------------------------------------------------


def test_plan_empty_raises() -> None:
    """Plan with zero descriptors must raise."""
    with pytest.raises(ValueError, match="at least one descriptor"):
        CompactTransferPlan(
            gpu_ptrs=np.array([], dtype=np.uint64),
            cpu_ptrs=np.array([], dtype=np.uint64),
            sizes=np.array([], dtype=np.uint64),
            num_cpu_addresses=0,
        )


def test_plan_zero_addresses_raises() -> None:
    """Zero num_cpu_addresses must raise."""
    with pytest.raises(ValueError, match="num_cpu_addresses"):
        CompactTransferPlan(
            gpu_ptrs=np.array([1], dtype=np.uint64),
            cpu_ptrs=np.array([2], dtype=np.uint64),
            sizes=np.array([3], dtype=np.uint64),
            num_cpu_addresses=0,
        )


def test_plan_mismatched_arrays_raises() -> None:
    """Mismatched pointer / size array lengths must raise."""
    with pytest.raises(ValueError, match="same shape"):
        CompactTransferPlan(
            gpu_ptrs=np.array([1], dtype=np.uint64),
            cpu_ptrs=np.array([2, 3], dtype=np.uint64),
            sizes=np.array([4], dtype=np.uint64),
            num_cpu_addresses=1,
        )


def test_plan_wrong_dtype_raises() -> None:
    """Non-uint64 arrays must raise."""
    with pytest.raises(TypeError, match="uint64"):
        CompactTransferPlan(
            gpu_ptrs=np.array([1], dtype=np.int64),
            cpu_ptrs=np.array([2], dtype=np.uint64),
            sizes=np.array([3], dtype=np.uint64),
            num_cpu_addresses=1,
        )


def test_plan_properties() -> None:
    """num_bytes and num_descriptors reflect arrays."""
    plan = CompactTransferPlan(
        gpu_ptrs=np.array([10, 20], dtype=np.uint64),
        cpu_ptrs=np.array([100, 200], dtype=np.uint64),
        sizes=np.array([80, 40], dtype=np.uint64),
        num_cpu_addresses=2,
    )
    assert plan.num_bytes == 120
    assert plan.num_descriptors == 2


# ---------------------------------------------------------------------------
# 1. Different GPU block IDs produce different GPU pointers
# ---------------------------------------------------------------------------


def test_different_block_ids_different_gpu_ptrs() -> None:
    """Different GPU block IDs must produce different GPU pointers."""
    plan = plan_compact_transfer(
        gpu_base_ptr=0x1000,
        gpu_row_stride=160,
        cpu_base_ptr=0xA000,
        cpu_region_size=2048,
        gpu_block_ids=np.array([0, 1], dtype=np.int64),
        group_sizes=[2],
        block_indices=[0],
        compact_addresses=[
            _addr(0, 80, 80, 0),
            _addr(100, 80, 80, 0),
        ],
        group_slice_configs=_single_slice_cfg(real_bytes=80),
        block_size_factor=1,
    )
    assert plan.num_descriptors == 2
    assert plan.gpu_ptrs[0] == 0x1000 + 0 * 160  # block 0
    assert plan.gpu_ptrs[1] == 0x1000 + 1 * 160  # block 1
    assert plan.gpu_ptrs[0] != plan.gpu_ptrs[1]
    assert plan.num_bytes == 160


# ---------------------------------------------------------------------------
# 2. Nonzero block_indices select correct sub-block
# ---------------------------------------------------------------------------


def test_nonzero_block_index_selects_sub_block() -> None:
    """block_idx=1 with block_size_factor=2 selects the second sub-block."""
    plan = plan_compact_transfer(
        gpu_base_ptr=0x1000,
        gpu_row_stride=160,
        cpu_base_ptr=0xA000,
        cpu_region_size=1024,
        gpu_block_ids=np.array([3], dtype=np.int64),
        group_sizes=[1],
        block_indices=[1],
        compact_addresses=[
            _addr(0, 160, 160, 0),
        ],
        group_slice_configs=_single_slice_cfg(real_bytes=80),
        block_size_factor=2,
    )
    assert plan.num_descriptors == 1
    # GPU: base + block_id * row_stride + slice offset
    assert plan.gpu_ptrs[0] == 0x1000 + 3 * 160
    # CPU: the second sub-block starts at offset 80 within the address
    assert plan.cpu_ptrs[0] == 0xA000 + 80
    assert plan.sizes[0] == 80
    assert plan.num_cpu_addresses == 1


# ---------------------------------------------------------------------------
# 3. blocks_per_chunk=2 (block_size_factor=2)
# ---------------------------------------------------------------------------


def test_blocks_per_chunk_two() -> None:
    """Two GPU blocks (block_size_factor=2) pack into one compact address."""
    plan = plan_compact_transfer(
        gpu_base_ptr=0x1000,
        gpu_row_stride=160,
        cpu_base_ptr=0xA000,
        cpu_region_size=1024,
        gpu_block_ids=np.array([0, 1], dtype=np.int64),
        group_sizes=[2],
        block_indices=[0],
        compact_addresses=[
            _addr(0, 160, 160, 0),
        ],
        group_slice_configs=_single_slice_cfg(real_bytes=80),
        block_size_factor=2,
    )
    assert plan.num_descriptors == 2
    assert plan.gpu_ptrs[0] == 0x1000 + 0 * 160
    assert plan.gpu_ptrs[1] == 0x1000 + 1 * 160
    assert plan.cpu_ptrs[0] == 0xA000 + 0
    assert plan.cpu_ptrs[1] == 0xA000 + 80
    assert plan.num_cpu_addresses == 1  # one address covers both blocks
    assert plan.num_bytes == 160


# ---------------------------------------------------------------------------
# 4. Two groups
# ---------------------------------------------------------------------------


def test_two_groups() -> None:
    """Two scheduler groups with separate compact addresses."""
    plan = plan_compact_transfer(
        gpu_base_ptr=0x1000,
        gpu_row_stride=160,
        cpu_base_ptr=0xA000,
        cpu_region_size=4096,
        gpu_block_ids=np.array([0, 3], dtype=np.int64),
        group_sizes=[1, 1],
        block_indices=[0, 0],
        compact_addresses=[
            _addr(0, 120, 120, 0),
            _addr(200, 120, 120, 1),
        ],
        group_slice_configs=(
            CompactGroupSliceConfig(
                group_idx=0,
                slices=(
                    CompactSliceConfig(
                        offset_bytes=0,
                        real_bytes_per_gpu_block=120,
                        padded_bytes_per_gpu_block=160,
                        layer_name="k",
                    ),
                ),
                compact_real_bytes_per_rank=120,
                compact_padded_bytes_per_rank=160,
            ),
            CompactGroupSliceConfig(
                group_idx=1,
                slices=(
                    CompactSliceConfig(
                        offset_bytes=0,
                        real_bytes_per_gpu_block=120,
                        padded_bytes_per_gpu_block=160,
                        layer_name="k",
                    ),
                ),
                compact_real_bytes_per_rank=120,
                compact_padded_bytes_per_rank=160,
            ),
        ),
        block_size_factor=1,
    )
    assert plan.num_descriptors == 2
    assert plan.gpu_ptrs[0] == 0x1000 + 0 * 160
    assert plan.cpu_ptrs[0] == 0xA000 + 0
    assert plan.gpu_ptrs[1] == 0x1000 + 3 * 160
    assert plan.cpu_ptrs[1] == 0xA000 + 200
    assert plan.num_cpu_addresses == 2
    assert plan.num_bytes == 240


# ---------------------------------------------------------------------------
# 5. Multiple packed slices per group
# ---------------------------------------------------------------------------


def test_two_slices_per_group() -> None:
    """Multiple packed slices (k and v) produce one descriptor per slice."""
    plan = plan_compact_transfer(
        gpu_base_ptr=0x1000,
        gpu_row_stride=160,
        cpu_base_ptr=0xA000,
        cpu_region_size=1024,
        gpu_block_ids=np.array([2], dtype=np.int64),
        group_sizes=[1],
        block_indices=[0],
        compact_addresses=[
            _addr(0, 120, 120, 0),
        ],
        group_slice_configs=_two_slice_cfg(),
        block_size_factor=1,
    )
    assert plan.num_descriptors == 2
    # Slice 0 (k): GPU offset 0
    assert plan.gpu_ptrs[0] == 0x1000 + 2 * 160 + 0
    assert plan.cpu_ptrs[0] == 0xA000 + 0
    assert plan.sizes[0] == 80
    # Slice 1 (v): GPU offset 100
    assert plan.gpu_ptrs[1] == 0x1000 + 2 * 160 + 100
    assert plan.cpu_ptrs[1] == 0xA000 + 80
    assert plan.sizes[1] == 40
    assert plan.gpu_ptrs[0] != plan.gpu_ptrs[1]
    assert plan.num_bytes == 120


def test_two_slices_with_block_size_factor_two() -> None:
    """Two slices with block_size_factor=2: each compact address holds
    2 sub-blocks worth of each slice, laid out consecutively.

    Descriptor order is per-slice outer, per-sub-block inner:
        [k for block 0, k for block 1, v for block 0, v for block 1]
    """
    plan = plan_compact_transfer(
        gpu_base_ptr=0x1000,
        gpu_row_stride=160,
        cpu_base_ptr=0xA000,
        cpu_region_size=2048,
        gpu_block_ids=np.array([0, 1], dtype=np.int64),
        group_sizes=[2],
        block_indices=[0],
        compact_addresses=[
            _addr(0, 240, 240, 0),  # 2 * (80 + 40) = 240
        ],
        group_slice_configs=_two_slice_cfg(),
        block_size_factor=2,
    )
    assert plan.num_descriptors == 4  # 2 blocks * 2 slices
    # Desc[0]: sub-block 0, slice k (offset 0)
    assert plan.sizes[0] == 80
    assert plan.cpu_ptrs[0] == 0xA000 + 0
    # Desc[1]: sub-block 1, slice k (offset 80 within address)
    assert plan.sizes[1] == 80
    assert plan.cpu_ptrs[1] == 0xA000 + 80
    # Desc[2]: sub-block 0, slice v (offset 2*80=160 within address)
    assert plan.sizes[2] == 40
    assert plan.cpu_ptrs[2] == 0xA000 + 160
    # Desc[3]: sub-block 1, slice v (offset 200 within address)
    assert plan.sizes[3] == 40
    assert plan.cpu_ptrs[3] == 0xA000 + 200
    assert plan.num_bytes == 240


# ---------------------------------------------------------------------------
# 6. Non-contiguous CPU spans split descriptors
# ---------------------------------------------------------------------------


def test_non_contiguous_spans_split_descriptor() -> None:
    """Non-contiguous physical spans split a single sub-block into multiple
    descriptors."""
    plan = plan_compact_transfer(
        gpu_base_ptr=0x1000,
        gpu_row_stride=160,
        cpu_base_ptr=0xA000,
        cpu_region_size=1024,
        gpu_block_ids=np.array([7], dtype=np.int64),
        group_sizes=[1],
        block_indices=[0],
        compact_addresses=[
            _addr(
                byte_offset=0,
                logical_length=80,
                allocated_length=180,
                group_idx=0,
                spans=(
                    _span(byte_offset=0, logical_length=32, allocated_length=32),
                    _span(
                        byte_offset=100,
                        logical_length=48,
                        allocated_length=148,
                    ),
                ),
            ),
        ],
        group_slice_configs=_single_slice_cfg(real_bytes=80),
        block_size_factor=1,
    )
    assert plan.num_descriptors == 2
    # First span: bytes 0-31 of the slice
    assert plan.gpu_ptrs[0] == 0x1000 + 7 * 160
    assert plan.cpu_ptrs[0] == 0xA000 + 0
    assert plan.sizes[0] == 32
    # Second span: bytes 32-79 of the slice
    assert plan.gpu_ptrs[1] == 0x1000 + 7 * 160 + 32
    assert plan.cpu_ptrs[1] == 0xA000 + 100
    assert plan.sizes[1] == 48
    assert plan.num_bytes == 80


# ---------------------------------------------------------------------------
# 7. Insufficient address size fails
# ---------------------------------------------------------------------------


def test_insufficient_address_size_fails() -> None:
    """Compact address too small for the required payload must raise."""
    # block_size_factor=2 -> required_payload = 2 * 80 = 160
    # address.logical_length = 80 < 160
    with pytest.raises(ValueError, match="smaller than"):
        plan_compact_transfer(
            gpu_base_ptr=0x1000,
            gpu_row_stride=160,
            cpu_base_ptr=0xA000,
            cpu_region_size=1024,
            gpu_block_ids=np.array([0, 1], dtype=np.int64),
            group_sizes=[2],
            block_indices=[0],
            compact_addresses=[
                _addr(0, 80, 80, 0),
            ],
            group_slice_configs=_single_slice_cfg(real_bytes=80),
            block_size_factor=2,
        )


def test_insufficient_single_block_address_fails() -> None:
    """Address with logical_length < real_bytes must raise."""
    with pytest.raises(ValueError, match="smaller than"):
        plan_compact_transfer(
            gpu_base_ptr=0x1000,
            gpu_row_stride=160,
            cpu_base_ptr=0xA000,
            cpu_region_size=1024,
            gpu_block_ids=np.array([0], dtype=np.int64),
            group_sizes=[1],
            block_indices=[0],
            compact_addresses=[
                _addr(0, 50, 50, 0),
            ],
            group_slice_configs=_single_slice_cfg(real_bytes=80),
            block_size_factor=1,
        )


# ---------------------------------------------------------------------------
# 8. Unknown group / order / count mismatches fail
# ---------------------------------------------------------------------------


def test_unknown_group_index_fails() -> None:
    """Compact address with out-of-range group index must raise."""
    with pytest.raises(ValueError, match="out-of-range group index"):
        plan_compact_transfer(
            gpu_base_ptr=0x1000,
            gpu_row_stride=160,
            cpu_base_ptr=0xA000,
            cpu_region_size=1024,
            gpu_block_ids=np.array([0], dtype=np.int64),
            group_sizes=[1],
            block_indices=[0],
            compact_addresses=[
                _addr(0, 80, 80, 99),
            ],
            group_slice_configs=_single_slice_cfg(),
            block_size_factor=1,
        )


def test_group_sizes_mismatch_fails() -> None:
    """Mismatched len(group_sizes) vs len(group_slice_configs) must raise."""
    with pytest.raises(ValueError, match="group_sizes must match"):
        plan_compact_transfer(
            gpu_base_ptr=0x1000,
            gpu_row_stride=160,
            cpu_base_ptr=0xA000,
            cpu_region_size=1024,
            gpu_block_ids=np.array([0, 1], dtype=np.int64),
            group_sizes=[2],
            block_indices=[0],
            compact_addresses=[
                _addr(0, 160, 160, 0),
            ],
            group_slice_configs=(
                CompactGroupSliceConfig(
                    group_idx=0,
                    slices=(
                        CompactSliceConfig(
                            offset_bytes=0,
                            real_bytes_per_gpu_block=160,
                            padded_bytes_per_gpu_block=160,
                            layer_name="k",
                        ),
                    ),
                    compact_real_bytes_per_rank=160,
                    compact_padded_bytes_per_rank=160,
                ),
                CompactGroupSliceConfig(
                    group_idx=1,
                    slices=(
                        CompactSliceConfig(
                            offset_bytes=0,
                            real_bytes_per_gpu_block=80,
                            padded_bytes_per_gpu_block=80,
                            layer_name="v",
                        ),
                    ),
                    compact_real_bytes_per_rank=80,
                    compact_padded_bytes_per_rank=80,
                ),
            ),
            block_size_factor=1,
        )


def test_group_order_mismatch_fails() -> None:
    """Non-contiguous group index must raise."""
    with pytest.raises(ValueError, match="contiguous and ordered"):
        plan_compact_transfer(
            gpu_base_ptr=0x1000,
            gpu_row_stride=160,
            cpu_base_ptr=0xA000,
            cpu_region_size=1024,
            gpu_block_ids=np.array([0], dtype=np.int64),
            group_sizes=[1],
            block_indices=[0],
            compact_addresses=[
                _addr(0, 80, 80, 0),
            ],
            group_slice_configs=(
                CompactGroupSliceConfig(
                    group_idx=1,
                    slices=(
                        CompactSliceConfig(
                            offset_bytes=0,
                            real_bytes_per_gpu_block=80,
                            padded_bytes_per_gpu_block=80,
                            layer_name="k",
                        ),
                    ),
                    compact_real_bytes_per_rank=80,
                    compact_padded_bytes_per_rank=80,
                ),
            ),
            block_size_factor=1,
        )


def test_address_count_mismatch_fails() -> None:
    """Address count not matching num_cpu_blocks must raise."""
    with pytest.raises(ValueError, match="do not cover"):
        plan_compact_transfer(
            gpu_base_ptr=0x1000,
            gpu_row_stride=160,
            cpu_base_ptr=0xA000,
            cpu_region_size=1024,
            gpu_block_ids=np.array([0, 1], dtype=np.int64),
            group_sizes=[2],
            block_indices=[0],
            compact_addresses=[
                _addr(0, 160, 160, 0),
            ],
            group_slice_configs=_single_slice_cfg(),
            # block_size_factor=1, group_size=2, num_cpu_blocks=2
            # but only 1 address provided
            block_size_factor=1,
        )


# ---------------------------------------------------------------------------
# 9. Region bounds fail
# ---------------------------------------------------------------------------


def test_gpu_slice_out_of_bounds_fails() -> None:
    """Slice offset + real_bytes exceeding gpu_row_stride must raise."""
    with pytest.raises(ValueError, match="outside the packed GPU row"):
        plan_compact_transfer(
            gpu_base_ptr=0x1000,
            gpu_row_stride=100,
            cpu_base_ptr=0xA000,
            cpu_region_size=1024,
            gpu_block_ids=np.array([0], dtype=np.int64),
            group_sizes=[1],
            block_indices=[0],
            compact_addresses=[
                _addr(0, 80, 80, 0),
            ],
            group_slice_configs=(
                CompactGroupSliceConfig(
                    group_idx=0,
                    slices=(
                        CompactSliceConfig(
                            offset_bytes=80,
                            real_bytes_per_gpu_block=80,
                            padded_bytes_per_gpu_block=80,
                            layer_name="k",
                        ),
                    ),
                    compact_real_bytes_per_rank=80,
                    compact_padded_bytes_per_rank=80,
                ),
            ),
            block_size_factor=1,
        )


def test_cpu_region_bounds_fails() -> None:
    """Compact address outside CPU region must raise."""
    with pytest.raises(ValueError, match="outside the CPU region"):
        plan_compact_transfer(
            gpu_base_ptr=0x1000,
            gpu_row_stride=160,
            cpu_base_ptr=0xA000,
            cpu_region_size=100,
            gpu_block_ids=np.array([0], dtype=np.int64),
            group_sizes=[1],
            block_indices=[0],
            compact_addresses=[
                _addr(200, 80, 80, 0),  # 200 + 80 = 280 > 100
            ],
            group_slice_configs=_single_slice_cfg(),
            block_size_factor=1,
        )


def test_cpu_region_bounds_span_fails() -> None:
    """Multi-span address where a span exceeds CPU region must raise."""
    with pytest.raises(ValueError, match="outside the CPU region"):
        plan_compact_transfer(
            gpu_base_ptr=0x1000,
            gpu_row_stride=160,
            cpu_base_ptr=0xA000,
            cpu_region_size=150,
            gpu_block_ids=np.array([0], dtype=np.int64),
            group_sizes=[1],
            block_indices=[0],
            compact_addresses=[
                _addr(
                    byte_offset=0,
                    logical_length=80,
                    allocated_length=110,
                    group_idx=0,
                    spans=(
                        _span(0, 30, 30),
                        # second span ends at 100 + 80 = 180 > 150 (cpu_region_size)
                        _span(100, 50, 80),
                    ),
                ),
            ],
            group_slice_configs=_single_slice_cfg(real_bytes=80),
            block_size_factor=1,
        )


# ---------------------------------------------------------------------------
# 10. All IDs / addresses consumed
# ---------------------------------------------------------------------------


def test_extra_gpu_ids_not_consumed_fails() -> None:
    """More GPU block IDs than group_sizes claim must raise.

    The planner validates at the start that sum(group_sizes) == len(gpu_block_ids);
    this checks the early-bound ``must cover every GPU block ID`` error.
    """
    with pytest.raises(ValueError, match="must cover every GPU block ID"):
        plan_compact_transfer(
            gpu_base_ptr=0x1000,
            gpu_row_stride=160,
            cpu_base_ptr=0xA000,
            cpu_region_size=1024,
            gpu_block_ids=np.array([0, 1], dtype=np.int64),
            group_sizes=[1],
            block_indices=[0],
            compact_addresses=[
                _addr(0, 80, 80, 0),
            ],
            group_slice_configs=_single_slice_cfg(),
            block_size_factor=1,
        )


def test_extra_addresses_not_consumed_fails() -> None:
    """More compact addresses than needed must raise.

    The planner validates per group that the address count matches
    ``num_cpu_blocks``, which fires before the final consumption check.
    """
    with pytest.raises(ValueError, match="do not cover"):
        plan_compact_transfer(
            gpu_base_ptr=0x1000,
            gpu_row_stride=160,
            cpu_base_ptr=0xA000,
            cpu_region_size=1024,
            gpu_block_ids=np.array([0], dtype=np.int64),
            group_sizes=[1],
            block_indices=[0],
            compact_addresses=[
                _addr(0, 80, 80, 0),
                _addr(100, 80, 80, 0),  # extra
            ],
            group_slice_configs=_single_slice_cfg(),
            block_size_factor=1,
        )


# ---------------------------------------------------------------------------
# Precondition guards
# ---------------------------------------------------------------------------


def test_negative_block_size_factor_fails() -> None:
    """Non-positive block_size_factor must raise."""
    with pytest.raises(ValueError, match="block_size_factor must be positive"):
        plan_compact_transfer(
            gpu_base_ptr=0x1000,
            gpu_row_stride=160,
            cpu_base_ptr=0xA000,
            cpu_region_size=1024,
            gpu_block_ids=np.array([0], dtype=np.int64),
            group_sizes=[1],
            block_indices=[0],
            compact_addresses=[
                _addr(0, 80, 80, 0),
            ],
            group_slice_configs=_single_slice_cfg(),
            block_size_factor=0,
        )


def test_negative_gpu_block_id_fails() -> None:
    """Negative GPU block ID must raise."""
    with pytest.raises(ValueError, match="GPU block IDs must be non-negative"):
        plan_compact_transfer(
            gpu_base_ptr=0x1000,
            gpu_row_stride=160,
            cpu_base_ptr=0xA000,
            cpu_region_size=1024,
            gpu_block_ids=np.array([-1], dtype=np.int64),
            group_sizes=[1],
            block_indices=[0],
            compact_addresses=[
                _addr(0, 80, 80, 0),
            ],
            group_slice_configs=_single_slice_cfg(),
            block_size_factor=1,
        )


def test_sum_group_sizes_equals_gpu_block_ids_len() -> None:
    """Sum of group_sizes must equal len(gpu_block_ids) -- valid case."""
    plan = plan_compact_transfer(
        gpu_base_ptr=0x1000,
        gpu_row_stride=160,
        cpu_base_ptr=0xA000,
        cpu_region_size=1024,
        gpu_block_ids=np.array([0, 1], dtype=np.int64),
        group_sizes=[1, 1],  # sum = 2, OK
        block_indices=[0, 0],
        compact_addresses=[
            _addr(0, 120, 120, 0),
            _addr(200, 120, 120, 1),
        ],
        group_slice_configs=(
            CompactGroupSliceConfig(
                group_idx=0,
                slices=(
                    CompactSliceConfig(
                        offset_bytes=0,
                        real_bytes_per_gpu_block=120,
                        padded_bytes_per_gpu_block=120,
                        layer_name="k",
                    ),
                ),
                compact_real_bytes_per_rank=120,
                compact_padded_bytes_per_rank=120,
            ),
            CompactGroupSliceConfig(
                group_idx=1,
                slices=(
                    CompactSliceConfig(
                        offset_bytes=0,
                        real_bytes_per_gpu_block=120,
                        padded_bytes_per_gpu_block=120,
                        layer_name="k",
                    ),
                ),
                compact_real_bytes_per_rank=120,
                compact_padded_bytes_per_rank=120,
            ),
        ),
        block_size_factor=1,
    )
    assert plan.num_descriptors == 2
    assert plan.num_bytes == 240
    assert plan.num_cpu_addresses == 2


def test_sum_group_sizes_mismatch_fails() -> None:
    """Sum of group_sizes not matching gpu_block_ids length must raise."""
    with pytest.raises(ValueError, match="must cover every GPU block ID"):
        plan_compact_transfer(
            gpu_base_ptr=0x1000,
            gpu_row_stride=160,
            cpu_base_ptr=0xA000,
            cpu_region_size=1024,
            gpu_block_ids=np.array([0], dtype=np.int64),
            group_sizes=[2],  # sum=2 but only 1 ID provided
            block_indices=[0],
            compact_addresses=[
                _addr(0, 160, 160, 0),
                _addr(200, 160, 160, 0),
            ],
            group_slice_configs=_single_slice_cfg(real_bytes=80),
            block_size_factor=2,
        )


# ---------------------------------------------------------------------------
# Immutable output
# ---------------------------------------------------------------------------


def test_plan_arrays_immutable() -> None:
    """Plan arrays should be read-only."""
    plan = plan_compact_transfer(
        gpu_base_ptr=0x1000,
        gpu_row_stride=160,
        cpu_base_ptr=0xA000,
        cpu_region_size=1024,
        gpu_block_ids=np.array([0], dtype=np.int64),
        group_sizes=[1],
        block_indices=[0],
        compact_addresses=[
            _addr(0, 80, 80, 0),
        ],
        group_slice_configs=_single_slice_cfg(),
        block_size_factor=1,
    )
    assert not plan.gpu_ptrs.flags.writeable
    assert not plan.cpu_ptrs.flags.writeable
    assert not plan.sizes.flags.writeable
    # Verify mutating raises
    with pytest.raises(ValueError):
        plan.gpu_ptrs[0] = 99  # type: ignore[index]


# ---------------------------------------------------------------------------
# Basic sanity (smoke test for config consistency)
# ---------------------------------------------------------------------------


def test_plan_slice_layout_charge_match() -> None:
    """slice_base accum after processing a group must equal the transported
    compact_real_bytes_per_rank."""
    plan = plan_compact_transfer(
        gpu_base_ptr=0x1000,
        gpu_row_stride=160,
        cpu_base_ptr=0xA000,
        cpu_region_size=1024,
        gpu_block_ids=np.array([0], dtype=np.int64),
        group_sizes=[1],
        block_indices=[0],
        compact_addresses=[
            _addr(0, 120, 120, 0),
        ],
        group_slice_configs=_two_slice_cfg(),
        block_size_factor=1,
    )
    assert plan.num_descriptors == 2
    assert plan.num_bytes == 120  # 80 + 40
    assert plan.num_cpu_addresses == 1


def test_slice_charge_mismatch_fails() -> None:
    """Mismatched slice layout vs transported charge must raise."""
    # compact_real_bytes_per_rank is deliberately wrong (should be 120 not 999)
    with pytest.raises(ValueError, match="does not match"):
        plan_compact_transfer(
            gpu_base_ptr=0x1000,
            gpu_row_stride=160,
            cpu_base_ptr=0xA000,
            cpu_region_size=1024,
            gpu_block_ids=np.array([0], dtype=np.int64),
            group_sizes=[1],
            block_indices=[0],
            compact_addresses=[
                _addr(0, 160, 160, 0),
            ],
            group_slice_configs=(
                CompactGroupSliceConfig(
                    group_idx=0,
                    slices=(
                        CompactSliceConfig(
                            offset_bytes=0,
                            real_bytes_per_gpu_block=80,
                            padded_bytes_per_gpu_block=100,
                            layer_name="k",
                        ),
                    ),
                    compact_real_bytes_per_rank=999,  # wrong
                    compact_padded_bytes_per_rank=999,
                ),
            ),
            block_size_factor=1,
        )


def test_address_logical_length_mismatch_fails() -> None:
    """Address logical_length not matching slice_base must raise.

    ``CompactCPUAddress.logical_length`` must equal
    ``compact_real_bytes_per_rank * block_size_factor``.  This test
    provides an address whose logical_length is too small for the slice
    geometry, which should fail the per-address invariant check.
    """
    # real_bytes=80, block_size_factor=2 -> slice_base = 160
    # But address.logical_length = 80 < 160
    with pytest.raises(
        ValueError,
        match="compact address is smaller than its static slice layout",
    ):
        plan_compact_transfer(
            gpu_base_ptr=0x1000,
            gpu_row_stride=160,
            cpu_base_ptr=0xA000,
            cpu_region_size=1024,
            gpu_block_ids=np.array([0, 1], dtype=np.int64),
            group_sizes=[2],
            block_indices=[0],
            compact_addresses=[
                _addr(0, 80, 80, 0),  # should be 160 for block_size_factor=2
            ],
            group_slice_configs=_single_slice_cfg(real_bytes=80),
            block_size_factor=2,
        )


def test_address_logical_length_not_equal_slice_base() -> None:
    """Address logical_length not equal to slice_base (but >= required payload)
    must raise the exact-equality check.

    The early invariant check (``logical_length < required_payload``) passes,
    but the per-address exact-equality check at the end of the group loop
    fires because ``logical_length != slice_base``.
    """
    # real_bytes=80, block_size_factor=2 -> slice_base = 160
    # address.logical_length = 200 >= 160 (passes early check)
    # But 200 != 160 (fails exact-equality check)
    with pytest.raises(ValueError, match="does not match"):
        plan_compact_transfer(
            gpu_base_ptr=0x1000,
            gpu_row_stride=160,
            cpu_base_ptr=0xA000,
            cpu_region_size=1024,
            gpu_block_ids=np.array([0, 1], dtype=np.int64),
            group_sizes=[2],
            block_indices=[0],
            compact_addresses=[
                _addr(0, 200, 200, 0),  # >= 160 but != 160
            ],
            group_slice_configs=_single_slice_cfg(real_bytes=80),
            block_size_factor=2,
        )


def test_address_scaling_mismatch_fails() -> None:
    """Aggregate charge check: slice_base must equal
    compact_real_bytes_per_rank * block_size_factor.

    Provide a group config whose compact_real_bytes_per_rank is wrong
    (should be 80 but instead 40), so the aggregate check fires.
    """
    # real_bytes=80, block_size_factor=2 -> slice_base = 160
    # compact_real_bytes_per_rank=40 * 2 = 80 != 160
    with pytest.raises(ValueError, match="does not match"):
        plan_compact_transfer(
            gpu_base_ptr=0x1000,
            gpu_row_stride=160,
            cpu_base_ptr=0xA000,
            cpu_region_size=1024,
            gpu_block_ids=np.array([0, 1], dtype=np.int64),
            group_sizes=[2],
            block_indices=[0],
            compact_addresses=[
                _addr(0, 160, 160, 0),
            ],
            group_slice_configs=(
                CompactGroupSliceConfig(
                    group_idx=0,
                    slices=(
                        CompactSliceConfig(
                            offset_bytes=0,
                            real_bytes_per_gpu_block=80,
                            padded_bytes_per_gpu_block=100,
                            layer_name="k",
                        ),
                    ),
                    compact_real_bytes_per_rank=40,  # wrong: should be 80
                    compact_padded_bytes_per_rank=60,
                ),
            ),
            block_size_factor=2,
        )
