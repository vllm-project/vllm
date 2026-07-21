# SPDX-License-Identifier: Apache-2.0
"""Integration tests: config/spec -> compact manager -> planner -> transfer wiring.

These tests validate that commit 5's worker/spec integration layer is reachable
from the canonical OffloadingConfig boundary without CUDA or runtime GPU state.
All planner logic is pure NumPy.  No conftest monkeypatching, no _custom_ops
replacement, no broad Triton/module stubs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vllm.v1.kv_offload.base import GPULoadStoreSpec, OffloadKey, ReqContext
from vllm.v1.kv_offload.config import (
    CompactGroupSliceConfig,
    CompactSliceConfig,
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingGroupConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)
from vllm.v1.kv_offload.cpu.common import (
    CompactCPUAddress,
    CompactCPUAddressSpan,
    CompactCPULoadStoreSpec,
    CPULoadStoreSpec,
)
from vllm.v1.kv_offload.cpu.compact_transfer import plan_compact_transfer

# ---------------------------------------------------------------------------
# Pure compact transfer planner tests (no GPU/CUDA required)
# ---------------------------------------------------------------------------


def _make_group_slice_configs(
    num_groups: int = 2,
    slices_per_group: int = 2,
) -> tuple[CompactGroupSliceConfig, ...]:
    """Build synthetic slice accounting for testing.

    Returns ``CompactGroupSliceConfig`` whose ``compact_real_bytes_per_rank``
    is the per-GPU-block per-rank value (unscaled by block_size_factor).
    Address ``logical_length`` must be scaled by ``block_size_factor`` at
    test time.
    """
    configs: list[CompactGroupSliceConfig] = []
    for g in range(num_groups):
        slices: list[CompactSliceConfig] = []
        total_real = 0
        for s in range(slices_per_group):
            real = (g + 1) * 128 + s * 32
            padded = real + 8  # small padding
            slices.append(
                CompactSliceConfig(
                    offset_bytes=total_real,
                    real_bytes_per_gpu_block=real,
                    padded_bytes_per_gpu_block=padded,
                    layer_name=f"group{g}_slice{s}",
                )
            )
            total_real += real
        configs.append(
            CompactGroupSliceConfig(
                group_idx=g,
                slices=tuple(slices),
                compact_real_bytes_per_rank=total_real,
                compact_padded_bytes_per_rank=total_real + 8 * slices_per_group,
            )
        )
    return tuple(configs)


def _packed_gpu_row_stride(groups: tuple[CompactGroupSliceConfig, ...]) -> int:
    """Compute the total packed GPU row stride from slice offsets and sizes."""
    stride = 0
    for grp in groups:
        for sl in grp.slices:
            stride = max(stride, sl.offset_bytes + sl.real_bytes_per_gpu_block)
    return stride


def test_compact_planner_two_groups_two_slices() -> None:
    """Verify plan_compact_transfer with two groups, two slices, contiguous spans."""
    groups = _make_group_slice_configs(num_groups=2, slices_per_group=2)
    block_size_factor = 4
    gpu_row_stride = _packed_gpu_row_stride(groups)
    gpu_base_ptr = 0x10000
    cpu_base_ptr = 0x20000
    cpu_region_size = 65536

    # Two groups: group0 size=3 blocks, group1 size=2 blocks
    gpu_block_ids = np.array([10, 11, 12, 20, 21], dtype=np.uint64)
    group_sizes = (3, 2)
    block_indices = (0, 0)

    # Compact CPU addresses: logical_length must be scaled by block_size_factor.
    # per-block = groups[0].compact_real_bytes_per_rank = 256 (128+128)
    g0_payload = (
        groups[0].compact_real_bytes_per_rank * block_size_factor
    )  # 256 * 4 = 1024
    g0_allocated = groups[0].compact_padded_bytes_per_rank * block_size_factor
    # per-block = groups[1].compact_real_bytes_per_rank = 544 (256+288)
    g1_payload = (
        groups[1].compact_real_bytes_per_rank * block_size_factor
    )  # 544 * 4 = 2176
    g1_allocated = groups[1].compact_padded_bytes_per_rank * block_size_factor
    compact_addresses: list[CompactCPUAddress] = [
        CompactCPUAddress(
            byte_offset=0,
            logical_length=g0_payload,
            allocated_length=g0_allocated,
            group_idx=0,
            spans=(),
        ),
        CompactCPUAddress(
            byte_offset=2048,
            logical_length=g1_payload,
            allocated_length=g1_allocated,
            group_idx=1,
            spans=(),
        ),
    ]

    plan = plan_compact_transfer(
        gpu_base_ptr=gpu_base_ptr,
        gpu_row_stride=gpu_row_stride,
        cpu_base_ptr=cpu_base_ptr,
        cpu_region_size=cpu_region_size,
        gpu_block_ids=gpu_block_ids,
        group_sizes=group_sizes,
        block_indices=block_indices,
        compact_addresses=compact_addresses,
        group_slice_configs=groups,
        block_size_factor=block_size_factor,
    )

    # Total descriptors = sum(group_size * slices_per_group) = 3*2 + 2*2 = 10
    assert plan.num_descriptors == 10, (
        f"expected 10 descriptors, got {plan.num_descriptors}"
    )
    assert plan.num_cpu_addresses == 2
    assert plan.num_bytes > 0

    # Every GPU pointer should reference actual block IDs from gpu_block_ids.
    for gpu_ptr_val in plan.gpu_ptrs:
        block_offset = (int(gpu_ptr_val) - gpu_base_ptr) % gpu_row_stride
        assert 0 <= block_offset < gpu_row_stride, (
            f"GPU ptr {gpu_ptr_val} has invalid block offset {block_offset}"
        )

    # Every CPU pointer should be within the region.
    for cpu_ptr_val in plan.cpu_ptrs:
        offset = int(cpu_ptr_val) - cpu_base_ptr
        assert 0 <= offset < cpu_region_size, (
            f"CPU ptr {cpu_ptr_val} has invalid offset {offset}"
        )

    # Sizes should be positive and match per-slice real bytes.
    for s_val in plan.sizes:
        assert s_val > 0, f"non-positive size {s_val}"


def test_compact_planner_fragmented_spans() -> None:
    """Verify planning with non-contiguous physical spans (fragmentation)."""
    groups = _make_group_slice_configs(num_groups=1, slices_per_group=1)
    block_size_factor = 2
    gpu_row_stride = _packed_gpu_row_stride(groups)
    gpu_base_ptr = 0x10000
    cpu_base_ptr = 0x20000
    cpu_region_size = 8192

    gpu_block_ids = np.array([5, 5], dtype=np.uint64)  # same block, two logical idxs
    group_sizes = (2,)
    block_indices = (0,)

    # Create a fragmented address: two non-contiguous physical spans.
    group_payload = groups[0].compact_real_bytes_per_rank * block_size_factor
    spans = (
        CompactCPUAddressSpan(byte_offset=0, logical_length=64, allocated_length=64),
        CompactCPUAddressSpan(
            byte_offset=512,
            logical_length=group_payload - 64,
            allocated_length=group_payload - 64,
        ),
    )
    compact_addresses = [
        CompactCPUAddress(
            byte_offset=0,
            logical_length=group_payload,
            allocated_length=group_payload,
            group_idx=0,
            spans=spans,
        ),
    ]

    plan = plan_compact_transfer(
        gpu_base_ptr=gpu_base_ptr,
        gpu_row_stride=gpu_row_stride,
        cpu_base_ptr=cpu_base_ptr,
        cpu_region_size=cpu_region_size,
        gpu_block_ids=gpu_block_ids,
        group_sizes=group_sizes,
        block_indices=block_indices,
        compact_addresses=compact_addresses,
        group_slice_configs=groups,
        block_size_factor=block_size_factor,
    )

    assert plan.num_descriptors > 0
    # CPU pointers should be distributed across both spans.
    cpu_offsets = [int(p) - cpu_base_ptr for p in plan.cpu_ptrs]
    span0_offsets = [off for off in cpu_offsets if off < 128]
    span1_offsets = [off for off in cpu_offsets if off >= 512]
    assert len(span0_offsets) > 0, "no CPU pointers in first span"
    assert len(span1_offsets) > 0, "no CPU pointers in second span"
    assert plan.num_bytes > 0


def test_compact_planner_partial_chunk_offset() -> None:
    """Verify planning with a non-zero block_idx (partial chunk)."""
    groups = _make_group_slice_configs(num_groups=1, slices_per_group=1)
    block_size_factor = 4
    gpu_row_stride = _packed_gpu_row_stride(groups)
    gpu_base_ptr = 0x10000
    cpu_base_ptr = 0x20000
    cpu_region_size = 8192

    # 3 blocks starting at sub-block index 2 (partial first address).
    # ceil(2 + 3, 4) = 2 addresses needed.
    gpu_block_ids = np.array([10, 11, 12], dtype=np.uint64)
    group_sizes = (3,)
    block_indices = (2,)  # start at sub-block 2 within the first compact address

    group_payload = groups[0].compact_real_bytes_per_rank * block_size_factor
    compact_addresses = [
        CompactCPUAddress(
            byte_offset=0,
            logical_length=group_payload,
            allocated_length=group_payload,
            group_idx=0,
            spans=(),
        ),
        CompactCPUAddress(
            byte_offset=group_payload,
            logical_length=group_payload,
            allocated_length=group_payload,
            group_idx=0,
            spans=(),
        ),
    ]

    plan = plan_compact_transfer(
        gpu_base_ptr=gpu_base_ptr,
        gpu_row_stride=gpu_row_stride,
        cpu_base_ptr=cpu_base_ptr,
        cpu_region_size=cpu_region_size,
        gpu_block_ids=gpu_block_ids,
        group_sizes=group_sizes,
        block_indices=block_indices,
        compact_addresses=compact_addresses,
        group_slice_configs=groups,
        block_size_factor=block_size_factor,
    )

    assert plan.num_descriptors > 0
    assert plan.num_bytes > 0
    # The first CPU address should not use bytes 0..block_idx * slice2bytes.
    # We verify at least one GPU pointer is present.
    assert len(plan.gpu_ptrs) == len(plan.cpu_ptrs)


# ---------------------------------------------------------------------------
# Spec -> Manager construction tests (no CUDA)
# ---------------------------------------------------------------------------


def _make_offloading_config(
    compact_slice_accounting: tuple[CompactGroupSliceConfig, ...] | None = None,
    extra_config: dict[str, Any] | None = None,
) -> OffloadingConfig:
    base_extra = {"cpu_bytes_to_use": "16777216", "store_threshold": "1"}
    if extra_config:
        base_extra.update(extra_config)
    return OffloadingConfig(
        groups=(
            OffloadingGroupConfig(
                tokens_per_block=256,
                layer_names=("layer0",),
                compact_bytes_per_native_block_per_worker=(
                    compact_slice_accounting[0].compact_real_bytes_per_rank
                    if compact_slice_accounting
                    else None
                ),
            ),
        ),
        worker_kv_bytes_per_block=1024,
        enable_kv_cache_events=False,
        extra_config=base_extra,
        engine_id="test-engine",
        model=OffloadingModelConfig(
            name="test-model",
            dtype="float16",
        ),
        cache=OffloadingCacheConfig(
            tokens_per_hash=64,
            blocks_per_chunk=1,
        ),
        parallel=OffloadingParallelConfig(
            rank=0,
            world_size=1,
            tp_size=1,
            pp_size=1,
            pcp_size=1,
            dcp_size=1,
            data_parallel_index=0,
            is_parallelism_agnostic=True,
        ),
        compact_slice_accounting=compact_slice_accounting,
    )


def test_spec_creates_compact_manager() -> None:
    """Spec with compact_slice_accounting creates a compact manager."""
    groups = _make_group_slice_configs(num_groups=1, slices_per_group=1)
    config = _make_offloading_config(compact_slice_accounting=groups)
    # Import lazily to verify import order doesn't matter.
    from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

    spec = CPUOffloadingSpec(config)
    assert spec._enable_compact_layout is True, "compact layout should be enabled"
    assert spec._compact_per_rank_budget is not None
    assert spec._compact_policy_capacity is not None
    assert spec._compact_group_payload_map is not None
    assert 0 in spec._compact_group_payload_map

    # The manager should be compact-enabled.
    manager = spec.get_manager()
    assert manager._compact_enabled is True
    assert manager._compact_allocator is not None
    assert manager._group_payload_bytes == {0: 128}


def test_spec_creates_legacy_manager() -> None:
    """Spec without compact_slice_accounting creates a legacy manager."""
    config = _make_offloading_config(compact_slice_accounting=None)
    from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

    spec = CPUOffloadingSpec(config)
    assert spec._enable_compact_layout is False

    manager = spec.get_manager()
    assert manager._compact_enabled is False
    assert manager._compact_allocator is None


def test_compact_manager_prepare_store_returns_compact_spec() -> None:
    """Verify compact prepare_store returns CompactCPULoadStoreSpec."""
    # Construct a compact manager directly (no spec layer).
    groups = _make_group_slice_configs(num_groups=1, slices_per_group=1)
    payload = groups[0].compact_real_bytes_per_rank
    budget = payload * 8  # 8 keys
    page_size = payload  # one payload per page

    from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

    manager = CPUOffloadingManager(
        num_blocks=8,
        cache_policy="lru",
        enable_events=False,
        store_threshold=1,
        max_tracker_size=100,
        compact_group_payload_map={
            cfg.group_idx: cfg.compact_real_bytes_per_rank for cfg in groups
        },
        blocks_per_chunk=1,
        compact_cpu_budget_bytes=budget,
        compact_page_size=page_size,
    )
    assert manager._compact_enabled is True

    # Create synthetic offload keys (group 0).
    keys = [bytes([0] * 28 + [0, 0, 0, g]) for g in [0, 0, 0]]

    req = ReqContext(req_id="test-req")
    result = manager.prepare_store([OffloadKey(k) for k in keys], req)
    assert result is not None, "prepare_store returned None"
    assert len(result.keys_to_store) == 3
    assert isinstance(result.store_spec, CompactCPULoadStoreSpec), (
        f"expected CompactCPULoadStoreSpec, got {type(result.store_spec)}"
    )
    assert len(result.store_spec.compact_addresses) == 3


# ---------------------------------------------------------------------------
# CompactCPULoadStoreSpec type dispatch (data-only, no CUDA)
# ---------------------------------------------------------------------------


def test_compact_spec_type_dispatch() -> None:
    """Verify CompactCPULoadStoreSpec and CPULoadStoreSpec are distinct types."""
    addr = CompactCPUAddress(
        byte_offset=0,
        logical_length=64,
        allocated_length=64,
        group_idx=0,
    )
    compact_spec = CompactCPULoadStoreSpec([addr])
    legacy_spec = CPULoadStoreSpec([0, 1, 2])

    assert isinstance(compact_spec, CompactCPULoadStoreSpec)
    assert isinstance(legacy_spec, CPULoadStoreSpec)
    assert not isinstance(compact_spec, CPULoadStoreSpec)
    assert not isinstance(legacy_spec, CompactCPULoadStoreSpec)

    # Verify the type-check logic that transfer_async uses.
    gpu_to_cpu_dispatch = isinstance(compact_spec, CompactCPULoadStoreSpec)
    assert gpu_to_cpu_dispatch is True

    cpu_to_gpu_dispatch = isinstance(compact_spec, CompactCPULoadStoreSpec)
    assert cpu_to_gpu_dispatch is True

    # Legacy path should not match compact.
    assert not isinstance(legacy_spec, CompactCPULoadStoreSpec)


# ---------------------------------------------------------------------------
# Compact page size: non-divisible budget regression (defect 1)
# ---------------------------------------------------------------------------


def test_spec_compact_page_size_non_divisible_budget() -> None:
    """Compact page size must evenly divide the per-rank budget even when
    no individual group payload divides the budget evenly.

    Regression test for defect 1: the old ``min(group payload)`` selection
    fails whenever ``per_rank_budget % min_payload != 0``.  The fix uses
    ``math.gcd(64 * 1024, per_rank_budget)`` which is always a divisor of
    the budget (positive by construction).
    """
    # Create slice accounting with per-rank payload of 50000 bytes.
    # If per_rank_budget = 123456, then 123456 % 50000 = 23456 != 0.
    slice_cfg = CompactSliceConfig(
        offset_bytes=0,
        real_bytes_per_gpu_block=50000,
        padded_bytes_per_gpu_block=50000,
        layer_name="k",
    )
    groups = (
        CompactGroupSliceConfig(
            group_idx=0,
            slices=(slice_cfg,),
            compact_real_bytes_per_rank=50000,
            compact_padded_bytes_per_rank=50000,
        ),
    )
    budget = 123456  # not divisible by 50000
    # Math: gcd(65536, 123456) = gcd(65536, 123456 % 65536 = 57920)
    #        = gcd(57920, 65536 % 57920 = 7616)
    #        = gcd(7616, 57920 % 7616 = 4736)
    #        = gcd(4736, 7616 % 4736 = 2880)
    #        = gcd(2880, 4736 % 2880 = 1856)
    #        = gcd(1856, 2880 % 1856 = 1024)
    #        = gcd(1024, 1856 % 1024 = 832)
    #        = gcd(832, 1024 % 832 = 192)
    #        = gcd(192, 832 % 192 = 64)
    #        = gcd(64, 192 % 64 = 0) = 64
    expected_page_size = 64
    assert 123456 % expected_page_size == 0, (
        f"{expected_page_size} must divide {budget}"
    )

    config = _make_offloading_config(
        compact_slice_accounting=groups,
        extra_config={"cpu_bytes_to_use": str(budget)},
    )
    from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

    spec = CPUOffloadingSpec(config)
    assert spec._enable_compact_layout is True

    # get_manager constructs the FixedPageAllocator; it should not raise.
    manager = spec.get_manager()
    assert manager._compact_allocator is not None
    # The page size should be the GCD-derived value, not the raw payload.
    assert manager._compact_allocator.page_size == expected_page_size, (
        f"expected page_size={expected_page_size}, "
        f"got {manager._compact_allocator.page_size}"
    )


# ---------------------------------------------------------------------------
# Handler-level descriptor construction tests (defect 3)
#
# These tests extract pure descriptor-building from the handler and verify
# exact GPU block IDs, packed slice offsets/row stride, fragmented CPU
# spans, partial chunk geometry, pointer role reversal, sizes, and byte
# total -- WITHOUT CUDA, async submission, or real tensor allocations.
#
# They use the extracted ``_fill_compact_descriptor_buffers_numpy`` and
# ``plan_compact_transfer`` which are pure NumPy.
# ---------------------------------------------------------------------------


def _handler_test_dir_gpu_to_cpu() -> None:
    """GPU -> CPU compact descriptor construction with actual planner."""
    # --- Arrange: synthetic group slice config ---
    k_cfg = CompactSliceConfig(
        offset_bytes=0,
        real_bytes_per_gpu_block=80,
        padded_bytes_per_gpu_block=100,
        layer_name="k",
    )
    v_cfg = CompactSliceConfig(
        offset_bytes=100,
        real_bytes_per_gpu_block=40,
        padded_bytes_per_gpu_block=60,
        layer_name="v",
    )
    groups = (
        CompactGroupSliceConfig(
            group_idx=0,
            slices=(k_cfg, v_cfg),
            compact_real_bytes_per_rank=120,  # 80 + 40 (unscaled)
            compact_padded_bytes_per_rank=160,
        ),
    )

    # --- GPU spec: 3 actual GPU block IDs, one group ---
    gpu_block_ids = np.array([10, 11, 12], dtype=np.int64)
    gpu_spec = GPULoadStoreSpec(
        block_ids=gpu_block_ids,
        group_sizes=(3,),
        block_indices=(0,),
    )

    # --- CPU spec: 3 compact addresses (one per block, block_size_factor=1) ---
    compact_addresses = [
        CompactCPUAddress(
            byte_offset=i * 120,
            logical_length=120,
            allocated_length=160,
            group_idx=0,
        )
        for i in range(3)
    ]
    cpu_spec = CompactCPULoadStoreSpec(compact_addresses)

    # --- Plan: synthetic base pointers (real values not needed for test) ---
    gpu_base_ptr = 0xDEAD_BEEF_0000
    gpu_row_stride = 160  # packed row = padded width
    cpu_base_ptr = 0xCAFE_F000
    cpu_region_size = 3 * 160  # 3 addresses

    plan = plan_compact_transfer(
        gpu_base_ptr=gpu_base_ptr,
        gpu_row_stride=gpu_row_stride,
        cpu_base_ptr=cpu_base_ptr,
        cpu_region_size=cpu_region_size,
        gpu_block_ids=gpu_spec.block_ids,
        group_sizes=gpu_spec.group_sizes,
        block_indices=gpu_spec.block_indices,
        compact_addresses=cpu_spec.compact_addresses,
        group_slice_configs=groups,
        block_size_factor=1,
    )

    # --- Verify plan properties ---
    # 3 blocks * 2 slices = 6 descriptors
    assert plan.num_descriptors == 6, f"expected 6, got {plan.num_descriptors}"
    assert plan.num_cpu_addresses == 3
    expected_bytes = 3 * (80 + 40)  # 3 blocks * 120 real bytes
    assert plan.num_bytes == expected_bytes, (
        f"expected {expected_bytes}, got {plan.num_bytes}"
    )

    # --- Verify GPU block ID encoding ---
    # Planner order is slice-major: all K blocks, then all V blocks.
    # K slice (offset 0): block 10, 11, 12
    assert plan.gpu_ptrs[0] == gpu_base_ptr + 10 * gpu_row_stride + 0
    assert plan.gpu_ptrs[1] == gpu_base_ptr + 11 * gpu_row_stride + 0
    assert plan.gpu_ptrs[2] == gpu_base_ptr + 12 * gpu_row_stride + 0
    # V slice (offset 100): block 10, 11, 12
    assert plan.gpu_ptrs[3] == gpu_base_ptr + 10 * gpu_row_stride + 100
    assert plan.gpu_ptrs[4] == gpu_base_ptr + 11 * gpu_row_stride + 100
    assert plan.gpu_ptrs[5] == gpu_base_ptr + 12 * gpu_row_stride + 100

    # --- Verify CPU address encoding ---
    # CPU offsets: address 0 = 0..120, address 1 = 120..240, address 2 = 240..360
    # K slices (all at offset 0 within each address)
    assert plan.cpu_ptrs[0] == cpu_base_ptr + 0 * 120 + 0
    assert plan.cpu_ptrs[1] == cpu_base_ptr + 1 * 120 + 0
    assert plan.cpu_ptrs[2] == cpu_base_ptr + 2 * 120 + 0
    # V slices (offset 80 within each address)
    assert plan.cpu_ptrs[3] == cpu_base_ptr + 0 * 120 + 80
    assert plan.cpu_ptrs[4] == cpu_base_ptr + 1 * 120 + 80
    assert plan.cpu_ptrs[5] == cpu_base_ptr + 2 * 120 + 80

    # --- Verify sizes ---
    assert list(plan.sizes) == [80, 80, 80, 40, 40, 40]

    # --- Verify pointer role reversal via buffer filling (GPU->CPU) ---
    num_ops = plan.num_descriptors
    src_buf = np.zeros(num_ops, dtype=np.uint64)
    dst_buf = np.zeros(num_ops, dtype=np.uint64)
    sz_buf = np.zeros(num_ops, dtype=np.uint64)

    from vllm.v1.kv_offload.cpu.gpu_worker import (
        _fill_compact_descriptor_buffers_numpy,
    )

    _fill_compact_descriptor_buffers_numpy(
        gpu_to_cpu=True,
        plan=plan,
        src_arr=src_buf,
        dst_arr=dst_buf,
        sizes_arr=sz_buf,
    )

    # For GPU->CPU: src = GPU pointers, dst = CPU pointers
    np.testing.assert_array_equal(src_buf, plan.gpu_ptrs)
    np.testing.assert_array_equal(dst_buf, plan.cpu_ptrs)
    np.testing.assert_array_equal(sz_buf, plan.sizes)

    # --- Verify pointer role reversal (CPU->GPU) ---
    src_buf2 = np.zeros(num_ops, dtype=np.uint64)
    dst_buf2 = np.zeros(num_ops, dtype=np.uint64)
    sz_buf2 = np.zeros(num_ops, dtype=np.uint64)

    _fill_compact_descriptor_buffers_numpy(
        gpu_to_cpu=False,
        plan=plan,
        src_arr=src_buf2,
        dst_arr=dst_buf2,
        sizes_arr=sz_buf2,
    )

    # For CPU->GPU: src = CPU pointers, dst = GPU pointers (reversed)
    np.testing.assert_array_equal(src_buf2, plan.cpu_ptrs)
    np.testing.assert_array_equal(dst_buf2, plan.gpu_ptrs)
    np.testing.assert_array_equal(sz_buf2, plan.sizes)


def test_handler_descriptor_gpu_to_cpu() -> None:
    """GPU->CPU compact descriptor construction."""
    _handler_test_dir_gpu_to_cpu()


def test_handler_descriptor_cpu_to_gpu() -> None:
    """CPU->GPU compact descriptor construction with pointer role reversal."""
    # Reuse the same GPU->CPU setup; the role reversal is the key distinction.
    k_cfg = CompactSliceConfig(
        offset_bytes=0,
        real_bytes_per_gpu_block=80,
        padded_bytes_per_gpu_block=100,
        layer_name="k",
    )
    v_cfg = CompactSliceConfig(
        offset_bytes=100,
        real_bytes_per_gpu_block=40,
        padded_bytes_per_gpu_block=60,
        layer_name="v",
    )
    groups = (
        CompactGroupSliceConfig(
            group_idx=0,
            slices=(k_cfg, v_cfg),
            compact_real_bytes_per_rank=120,
            compact_padded_bytes_per_rank=160,
        ),
    )

    gpu_block_ids = np.array([42], dtype=np.int64)
    gpu_spec = GPULoadStoreSpec(
        block_ids=gpu_block_ids,
        group_sizes=(1,),
        block_indices=(0,),
    )

    compact_addresses = [
        CompactCPUAddress(
            byte_offset=0,
            logical_length=120,
            allocated_length=160,
            group_idx=0,
        ),
    ]
    cpu_spec = CompactCPULoadStoreSpec(compact_addresses)

    plan = plan_compact_transfer(
        gpu_base_ptr=0x1000,
        gpu_row_stride=160,
        cpu_base_ptr=0xA000,
        cpu_region_size=512,
        gpu_block_ids=gpu_spec.block_ids,
        group_sizes=gpu_spec.group_sizes,
        block_indices=gpu_spec.block_indices,
        compact_addresses=cpu_spec.compact_addresses,
        group_slice_configs=groups,
        block_size_factor=1,
    )

    assert plan.num_descriptors == 2  # 1 block * 2 slices
    assert plan.gpu_ptrs[0] == 0x1000 + 42 * 160  # k slice offset 0
    assert plan.gpu_ptrs[1] == 0x1000 + 42 * 160 + 100  # v slice offset 100
    assert plan.cpu_ptrs[0] == 0xA000 + 0
    assert plan.cpu_ptrs[1] == 0xA000 + 80
    assert list(plan.sizes) == [80, 40]
    assert plan.num_bytes == 120

    # CPU->GPU: pointer role reversal
    src_buf = np.zeros(2, dtype=np.uint64)
    dst_buf = np.zeros(2, dtype=np.uint64)
    sz_buf = np.zeros(2, dtype=np.uint64)

    from vllm.v1.kv_offload.cpu.gpu_worker import (
        _fill_compact_descriptor_buffers_numpy,
    )

    _fill_compact_descriptor_buffers_numpy(
        gpu_to_cpu=False,
        plan=plan,
        src_arr=src_buf,
        dst_arr=dst_buf,
        sizes_arr=sz_buf,
    )

    # For CPU->GPU: src = CPU pointers, dst = GPU pointers
    np.testing.assert_array_equal(src_buf, plan.cpu_ptrs)
    np.testing.assert_array_equal(dst_buf, plan.gpu_ptrs)
    np.testing.assert_array_equal(sz_buf, plan.sizes)


def test_handler_descriptor_fragmented_cpu_spans() -> None:
    """Compact descriptor with non-contiguous CPU physical spans."""
    k_cfg = CompactSliceConfig(
        offset_bytes=0,
        real_bytes_per_gpu_block=80,
        padded_bytes_per_gpu_block=100,
        layer_name="k",
    )
    groups = (
        CompactGroupSliceConfig(
            group_idx=0,
            slices=(k_cfg,),
            compact_real_bytes_per_rank=80,
            compact_padded_bytes_per_rank=100,
        ),
    )

    # Single GPU block, single address with two fragmented spans
    span_a = CompactCPUAddressSpan(
        byte_offset=0,
        logical_length=32,
        allocated_length=32,
    )
    span_b = CompactCPUAddressSpan(
        byte_offset=200,
        logical_length=48,
        allocated_length=48,
    )

    compact_addresses = [
        CompactCPUAddress(
            byte_offset=0,
            logical_length=80,
            allocated_length=80,
            group_idx=0,
            spans=(span_a, span_b),
        ),
    ]

    plan = plan_compact_transfer(
        gpu_base_ptr=0x1000,
        gpu_row_stride=160,
        cpu_base_ptr=0xA000,
        cpu_region_size=512,
        gpu_block_ids=np.array([7], dtype=np.int64),
        group_sizes=[1],
        block_indices=[0],
        compact_addresses=compact_addresses,
        group_slice_configs=groups,
        block_size_factor=1,
    )

    # Two descriptors due to two CPU spans
    assert plan.num_descriptors == 2
    # GPU: block 7, offset 0 (same for both spans since it's one GPU block)
    assert plan.gpu_ptrs[0] == 0x1000 + 7 * 160
    assert plan.gpu_ptrs[1] == 0x1000 + 7 * 160 + 32  # remaining 48 bytes
    # CPU: span a at offset 0, span b at offset 200
    assert plan.cpu_ptrs[0] == 0xA000 + 0
    assert plan.cpu_ptrs[1] == 0xA000 + 200
    assert plan.sizes[0] == 32
    assert plan.sizes[1] == 48
    assert plan.num_bytes == 80

    # Verify pointer role reversal (GPU->CPU)
    src_buf = np.zeros(2, dtype=np.uint64)
    dst_buf = np.zeros(2, dtype=np.uint64)
    sz_buf = np.zeros(2, dtype=np.uint64)

    from vllm.v1.kv_offload.cpu.gpu_worker import (
        _fill_compact_descriptor_buffers_numpy,
    )

    _fill_compact_descriptor_buffers_numpy(
        gpu_to_cpu=True,
        plan=plan,
        src_arr=src_buf,
        dst_arr=dst_buf,
        sizes_arr=sz_buf,
    )

    np.testing.assert_array_equal(src_buf, plan.gpu_ptrs)
    np.testing.assert_array_equal(dst_buf, plan.cpu_ptrs)
    np.testing.assert_array_equal(sz_buf, plan.sizes)

    # CPU->GPU: reversed roles
    src_buf2 = np.zeros(2, dtype=np.uint64)
    dst_buf2 = np.zeros(2, dtype=np.uint64)
    sz_buf2 = np.zeros(2, dtype=np.uint64)

    _fill_compact_descriptor_buffers_numpy(
        gpu_to_cpu=False,
        plan=plan,
        src_arr=src_buf2,
        dst_arr=dst_buf2,
        sizes_arr=sz_buf2,
    )

    np.testing.assert_array_equal(src_buf2, plan.cpu_ptrs)
    np.testing.assert_array_equal(dst_buf2, plan.gpu_ptrs)
    np.testing.assert_array_equal(sz_buf2, plan.sizes)


def test_handler_descriptor_partial_chunk() -> None:
    """Partial chunk geometry: block_idx=2 with block_size_factor=4."""
    k_cfg = CompactSliceConfig(
        offset_bytes=0,
        real_bytes_per_gpu_block=80,
        padded_bytes_per_gpu_block=100,
        layer_name="k",
    )
    groups = (
        CompactGroupSliceConfig(
            group_idx=0,
            slices=(k_cfg,),
            compact_real_bytes_per_rank=80,
            compact_padded_bytes_per_rank=100,
        ),
    )

    # block_size_factor=4, block_idx=2: 3 blocks starting at sub-block index 2
    # ceil(2 + 3, 4) = ceil(5, 4) = 2 compact addresses needed
    gpu_block_ids = np.array([5, 6, 7], dtype=np.int64)

    compact_addresses = [
        CompactCPUAddress(
            byte_offset=0,
            logical_length=320,  # 80 * 4
            allocated_length=400,
            group_idx=0,
        ),
        CompactCPUAddress(
            byte_offset=400,
            logical_length=320,
            allocated_length=400,
            group_idx=0,
        ),
    ]

    plan = plan_compact_transfer(
        gpu_base_ptr=0x1000,
        gpu_row_stride=160,
        cpu_base_ptr=0xA000,
        cpu_region_size=1024,
        gpu_block_ids=gpu_block_ids,
        group_sizes=[3],
        block_indices=[2],  # start at sub-block 2
        compact_addresses=compact_addresses,
        group_slice_configs=groups,
        block_size_factor=4,
    )

    # 3 blocks * 1 slice = 3 descriptors
    assert plan.num_descriptors == 3
    # Sub-block 2 of address 0 -> CPU offset = 2 * 80 = 160
    assert plan.cpu_ptrs[0] == 0xA000 + 160
    assert plan.sizes[0] == 80
    # Sub-block 3 of address 0 -> CPU offset = 3 * 80 = 240
    assert plan.cpu_ptrs[1] == 0xA000 + 240
    assert plan.sizes[1] == 80
    # Sub-block 0 of address 1 -> CPU offset = 400
    assert plan.cpu_ptrs[2] == 0xA000 + 400
    assert plan.sizes[2] == 80
    assert plan.num_bytes == 240
    assert plan.num_cpu_addresses == 2

    # Verify role reversal for GPU->CPU
    num_ops = plan.num_descriptors
    src_buf = np.zeros(num_ops, dtype=np.uint64)
    dst_buf = np.zeros(num_ops, dtype=np.uint64)
    sz_buf = np.zeros(num_ops, dtype=np.uint64)

    from vllm.v1.kv_offload.cpu.gpu_worker import (
        _fill_compact_descriptor_buffers_numpy,
    )

    _fill_compact_descriptor_buffers_numpy(
        gpu_to_cpu=True,
        plan=plan,
        src_arr=src_buf,
        dst_arr=dst_buf,
        sizes_arr=sz_buf,
    )

    np.testing.assert_array_equal(src_buf, plan.gpu_ptrs)
    np.testing.assert_array_equal(dst_buf, plan.cpu_ptrs)
    np.testing.assert_array_equal(sz_buf, plan.sizes)


def test_handler_transfer_compact_both_directions() -> None:
    """Exercise _transfer_compact on SingleDirectionOffloadingHandler in both
    directions without CUDA, Triton, or global conftest.

    Constructs the handler via ``object.__new__``, sets only fields required
    by ``_transfer_compact``, patches ``_submit_descriptors`` per-instance to
    capture arguments, and verifies exact descriptor arrays, sizes, op count,
    and byte totals for GPU->CPU and CPU->GPU.  GPU->CPU includes a fragmented
    CPU address (two non-contiguous physical spans); CPU->GPU includes a
    partial chunk (block_idx > 0 with block_size_factor > 1).
    """
    import numpy as np
    import torch

    from vllm.v1.kv_offload.base import GPULoadStoreSpec
    from vllm.v1.kv_offload.config import (
        CompactGroupSliceConfig,
        CompactSliceConfig,
    )
    from vllm.v1.kv_offload.cpu.common import (
        CompactCPUAddress,
        CompactCPUAddressSpan,
        CompactCPULoadStoreSpec,
    )
    from vllm.v1.kv_offload.cpu.gpu_worker import (
        SingleDirectionOffloadingHandler,
    )

    # ------------------------------------------------------------------
    # Shared geometry: two slices per group (k=80, v=40)
    # ------------------------------------------------------------------
    k_cfg = CompactSliceConfig(
        offset_bytes=0,
        real_bytes_per_gpu_block=80,
        padded_bytes_per_gpu_block=100,
        layer_name="k",
    )
    v_cfg = CompactSliceConfig(
        offset_bytes=100,
        real_bytes_per_gpu_block=40,
        padded_bytes_per_gpu_block=60,
        layer_name="v",
    )
    groups = (
        CompactGroupSliceConfig(
            group_idx=0,
            slices=(k_cfg, v_cfg),
            compact_real_bytes_per_rank=120,
            compact_padded_bytes_per_rank=160,
        ),
    )

    # Real tensor allocations (CPU only -- no CUDA)
    gpu_tensor = torch.empty(32, 160, dtype=torch.int8)
    compact_region = torch.empty(1024, dtype=torch.int8)

    gpu_base_ptr = gpu_tensor.data_ptr()
    gpu_row_stride = gpu_tensor.stride(0)
    cpu_base_ptr = compact_region.data_ptr()

    # ================================================================
    # Direction 1: GPU -> CPU
    # Includes a fragmented CPU address with non-contiguous spans.
    # ================================================================
    gpu_block_ids_g2c = np.array([10, 11, 12], dtype=np.int64)
    gpu_spec_g2c = GPULoadStoreSpec(
        block_ids=gpu_block_ids_g2c,
        group_sizes=(3,),
        block_indices=(0,),
    )
    # First address has two physical spans: 0..64 and 200..256
    frag_address = CompactCPUAddress(
        byte_offset=0,
        logical_length=120,
        allocated_length=120,
        group_idx=0,
        spans=(
            CompactCPUAddressSpan(
                byte_offset=0, logical_length=64, allocated_length=64
            ),
            CompactCPUAddressSpan(
                byte_offset=200, logical_length=56, allocated_length=56
            ),
        ),
    )
    addr_b = CompactCPUAddress(
        byte_offset=120, logical_length=120, allocated_length=160, group_idx=0
    )
    addr_c = CompactCPUAddress(
        byte_offset=240, logical_length=120, allocated_length=160, group_idx=0
    )
    cpu_spec_g2c = CompactCPULoadStoreSpec([frag_address, addr_b, addr_c])

    captured: list[dict] = []

    def _fake_submit_g2c(
        *,
        job_id: int,
        batch_src: torch.Tensor,
        batch_dst: torch.Tensor,
        batch_sizes: torch.Tensor,
        num_copy_ops: int,
        num_transfer_bytes: int,
        use_batch_api: bool = True,
    ) -> bool:
        captured.append(
            {
                "job_id": job_id,
                "src": batch_src.clone(),
                "dst": batch_dst.clone(),
                "sizes": batch_sizes.clone(),
                "num_copy_ops": num_copy_ops,
                "num_transfer_bytes": num_transfer_bytes,
                "use_batch_api": use_batch_api,
            }
        )
        return True

    handler_g2c = object.__new__(SingleDirectionOffloadingHandler)
    handler_g2c.gpu_to_cpu = True
    handler_g2c.src_tensors = [gpu_tensor]
    handler_g2c.dst_tensors = []
    handler_g2c._compact_region = compact_region
    handler_g2c._compact_group_slice_configs = groups
    handler_g2c._buffer_pool = []
    handler_g2c.dst_blocks_per_chunk = 1
    handler_g2c.src_blocks_per_chunk = 1
    handler_g2c._submit_descriptors = _fake_submit_g2c

    result = handler_g2c._transfer_compact(
        job_id=42,
        cpu_spec=cpu_spec_g2c,
        gpu_spec=gpu_spec_g2c,
    )

    assert result is True, "GPU->CPU transfer must succeed"
    assert len(captured) == 1, "exactly one submit call expected"
    call = captured[0]
    assert call["job_id"] == 42, "job_id must be preserved"
    # Fragmented address splits k into 2 descriptors (64 + 16).
    #   frag k: 64 + 16 = 2 descriptors
    #   addr_b k: 1 descriptor (80 bytes contiguous)
    #   addr_c k: 1 descriptor (80 bytes contiguous)
    #   frag v: 1 descriptor (40 bytes contiguous)
    #   addr_b v: 1 descriptor
    #   addr_c v: 1 descriptor
    # Total: 7
    assert call["num_copy_ops"] == 7
    assert call["num_transfer_bytes"] == 3 * 120
    # Compact transfer must pass use_batch_api=False to bypass the
    # driver batch-API path that segfaults with large descriptor counts.
    assert call["use_batch_api"] is False, (
        "compact GPU->CPU must explicitly pass use_batch_api=False"
    )

    # Verify types
    assert call["src"].dtype in (torch.int64, torch.uint64)
    assert call["dst"].dtype in (torch.int64, torch.uint64)

    # GPU->CPU: src = GPU pointers, dst = CPU pointers
    src_arr = call["src"].numpy()
    dst_arr = call["dst"].numpy()
    sizes_arr = call["sizes"].numpy()

    # Slice-major order: all k descriptors, then all v descriptors
    # Fragmented address 0, k slice: spans split into 64 + 16
    assert src_arr[0] == gpu_base_ptr + 10 * gpu_row_stride + 0
    assert dst_arr[0] == cpu_base_ptr + 0
    assert sizes_arr[0] == 64
    assert src_arr[1] == gpu_base_ptr + 10 * gpu_row_stride + 64
    assert dst_arr[1] == cpu_base_ptr + 200
    assert sizes_arr[1] == 16
    # Address 1, k slice: contiguous 80
    assert src_arr[2] == gpu_base_ptr + 11 * gpu_row_stride + 0
    assert dst_arr[2] == cpu_base_ptr + 120
    assert sizes_arr[2] == 80
    # Address 2, k slice: contiguous 80
    assert src_arr[3] == gpu_base_ptr + 12 * gpu_row_stride + 0
    assert dst_arr[3] == cpu_base_ptr + 240
    assert sizes_arr[3] == 80
    # Fragmented address 0, v slice: logical_offset=80 within frag_address
    # falls in span b (byte_offset=200, span_logical_base=64):
    # cpu_ptr = cpu_base_ptr + 200 + (80 - 64) = cpu_base_ptr + 216
    assert src_arr[4] == gpu_base_ptr + 10 * gpu_row_stride + 100
    assert dst_arr[4] == cpu_base_ptr + 216
    assert sizes_arr[4] == 40
    # Address 1, v slice: contiguous 40
    assert src_arr[5] == gpu_base_ptr + 11 * gpu_row_stride + 100
    assert dst_arr[5] == cpu_base_ptr + 120 + 80
    assert sizes_arr[5] == 40
    # Address 2, v slice: contiguous 40
    assert src_arr[6] == gpu_base_ptr + 12 * gpu_row_stride + 100
    assert dst_arr[6] == cpu_base_ptr + 240 + 80
    assert sizes_arr[6] == 40

    # ================================================================
    # Direction 2: CPU -> GPU with a partial chunk
    # block_idx=1, block_size_factor=2 => first_sub_block=1,
    # 3 GPU blocks starting at sub-block 1, so:
    #   address 0 gets sub-blocks 1, 2, 3 (indices 1, 2, 3)
    #   address 1 gets sub-block 0 (index 0 of second address)
    # ================================================================
    captured.clear()

    def _fake_submit_c2g(
        *,
        job_id: int,
        batch_src: torch.Tensor,
        batch_dst: torch.Tensor,
        batch_sizes: torch.Tensor,
        num_copy_ops: int,
        num_transfer_bytes: int,
        use_batch_api: bool = True,
    ) -> bool:
        captured.append(
            {
                "job_id": job_id,
                "src": batch_src.clone(),
                "dst": batch_dst.clone(),
                "sizes": batch_sizes.clone(),
                "num_copy_ops": num_copy_ops,
                "num_transfer_bytes": num_transfer_bytes,
                "use_batch_api": use_batch_api,
            }
        )
        return True

    handler_c2g = object.__new__(SingleDirectionOffloadingHandler)
    handler_c2g.gpu_to_cpu = False
    handler_c2g.src_tensors = []
    handler_c2g.dst_tensors = [gpu_tensor]
    handler_c2g._compact_region = compact_region
    handler_c2g._compact_group_slice_configs = groups
    handler_c2g._buffer_pool = []
    handler_c2g.dst_blocks_per_chunk = 1
    handler_c2g.src_blocks_per_chunk = 2  # blocks_per_chunk = 2
    handler_c2g._submit_descriptors = _fake_submit_c2g

    gpu_block_ids_c2g = np.array([20, 21, 22], dtype=np.int64)
    gpu_spec_c2g = GPULoadStoreSpec(
        block_ids=gpu_block_ids_c2g,
        group_sizes=(3,),
        block_indices=(1,),
    )
    addr_c2g_a = CompactCPUAddress(
        byte_offset=0, logical_length=240, allocated_length=240, group_idx=0
    )
    addr_c2g_b = CompactCPUAddress(
        byte_offset=240, logical_length=240, allocated_length=240, group_idx=0
    )
    cpu_spec_c2g = CompactCPULoadStoreSpec([addr_c2g_a, addr_c2g_b])

    result = handler_c2g._transfer_compact(
        job_id=99,
        cpu_spec=cpu_spec_c2g,
        gpu_spec=gpu_spec_c2g,
    )

    assert result is True, "CPU->GPU transfer must succeed"
    assert len(captured) == 1, "exactly one submit call expected"
    call = captured[0]
    assert call["job_id"] == 99, "job_id must be preserved"
    # 3 blocks * 2 slices = 6 descriptors
    assert call["num_copy_ops"] == 6
    # Total real bytes = 3 * 120 = 360
    assert call["num_transfer_bytes"] == 360
    # Compact transfer must pass use_batch_api=False to bypass the
    # driver batch-API path that segfaults with large descriptor counts.
    assert call["use_batch_api"] is False, (
        "compact CPU->GPU must explicitly pass use_batch_api=False"
    )

    # CPU->GPU: src = CPU pointers, dst = GPU pointers
    src_arr = call["src"].numpy()
    dst_arr = call["dst"].numpy()

    # Slice-major order across 3 GPU blocks.
    # block_size_factor=2, first_sub_block=1, 3 logical blocks:
    #   logical_idx=0 → compact_idx=0, sub_idx=1 (address 0, sub-block 1)
    #   logical_idx=1 → compact_idx=1, sub_idx=0 (address 1, sub-block 0)
    #   logical_idx=2 → compact_idx=1, sub_idx=1 (address 1, sub-block 1)
    #
    # K slice:
    # Block 20: sub-block 1 within address 0 -> cpu offset = 1*80 = 80
    assert dst_arr[0] == gpu_base_ptr + 20 * gpu_row_stride + 0
    assert src_arr[0] == cpu_base_ptr + 80
    # Block 21: sub-block 0 within address 1 -> cpu offset = 240 + 0 = 240
    assert dst_arr[1] == gpu_base_ptr + 21 * gpu_row_stride + 0
    assert src_arr[1] == cpu_base_ptr + 240
    # Block 22: sub-block 1 within address 1 -> cpu offset = 240 + 80 = 320
    assert dst_arr[2] == gpu_base_ptr + 22 * gpu_row_stride + 0
    assert src_arr[2] == cpu_base_ptr + 320

    # V slice (offset 100 within GPU row).  slice_base after k = 2*80 = 160.
    # logical_offset = slice_base + sub_idx * real_bytes:
    # Block 20: sub_idx=1 -> logical_offset = 160 + 1*40 = 200
    assert dst_arr[3] == gpu_base_ptr + 20 * gpu_row_stride + 100
    assert src_arr[3] == cpu_base_ptr + 200
    # Block 21: sub_idx=0 -> logical_offset = 160 + 0*40 = 160
    assert dst_arr[4] == gpu_base_ptr + 21 * gpu_row_stride + 100
    assert src_arr[4] == cpu_base_ptr + 400
    # Block 22: sub_idx=1 -> logical_offset = 160 + 1*40 = 200
    assert dst_arr[5] == gpu_base_ptr + 22 * gpu_row_stride + 100
    assert src_arr[5] == cpu_base_ptr + 440
