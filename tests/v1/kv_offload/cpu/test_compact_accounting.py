"""Focused unit tests for compact CPU KV accounting (lean commit 1).

Excludes dead fields, DeepSeek-specific prefer_early_eviction, and
bounded-tail policy.
"""

import pytest
import torch

from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_offload.cpu.common import (
    CompactCPUAddress,
    CompactCPUAddressSpan,
    CompactCPULoadStoreSpec,
)
from vllm.v1.kv_offload.cpu.compact_accounting import (
    CompactGroupCharge,
    build_compact_group_charges,
    build_compact_layout_accounting,
)


def _packed_fixture() -> KVCacheConfig:
    a0 = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=10,
        dtype=torch.uint8,
        page_size_padded=100,
    )
    a1 = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=5,
        dtype=torch.uint8,
        page_size_padded=60,
    )
    b0 = SlidingWindowSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=15,
        dtype=torch.uint8,
        page_size_padded=100,
        sliding_window=4,
    )
    b1 = SlidingWindowSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=5,
        dtype=torch.uint8,
        page_size_padded=60,
        sliding_window=4,
    )
    return KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[
            KVCacheTensor(
                size=8 * 160, shared_by=["a0", "b0"], offset=0, block_stride=160
            ),
            KVCacheTensor(
                size=8 * 160, shared_by=["a1", "b1"], offset=100, block_stride=160
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["a0", "a1"],
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=4, kv_cache_specs={"a0": a0, "a1": a1}
                ),
            ),
            KVCacheGroupSpec(
                layer_names=["b0", "b1"],
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=2, kv_cache_specs={"b0": b0, "b1": b1}
                ),
            ),
        ],
    )


# -----------------------------------------------------------------------
# CompactCPUAddress / CompactCPUAddressSpan / CompactCPULoadStoreSpec
# -----------------------------------------------------------------------


def test_compact_address_valid() -> None:
    addr = CompactCPUAddress(
        byte_offset=100, logical_length=200, allocated_length=256, group_idx=1
    )
    assert addr.byte_offset == 100
    assert addr.physical_spans == (CompactCPUAddressSpan(100, 200, 256),)


def test_compact_address_validation() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        CompactCPUAddress(-1, 1, 1)
    with pytest.raises(ValueError, match="positive"):
        CompactCPUAddress(0, 0, 1)
    with pytest.raises(ValueError, match="must be >="):
        CompactCPUAddress(0, 10, 5)
    with pytest.raises(ValueError, match="non-negative"):
        CompactCPUAddress(0, 1, 1, group_idx=-1)


def test_compact_address_with_spans() -> None:
    spans = (CompactCPUAddressSpan(100, 150, 256), CompactCPUAddressSpan(356, 50, 100))
    addr = CompactCPUAddress(
        byte_offset=100,
        logical_length=200,
        allocated_length=356,
        group_idx=0,
        spans=spans,
    )
    assert addr.physical_spans == spans


def test_compact_address_span_validation() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        CompactCPUAddressSpan(-1, 1, 1)
    with pytest.raises(ValueError, match="positive"):
        CompactCPUAddressSpan(0, 0, 1)
    with pytest.raises(ValueError, match="cover"):
        CompactCPUAddressSpan(0, 100, 50)


def test_compact_address_overlapping_spans_raises() -> None:
    spans = (CompactCPUAddressSpan(100, 100, 150), CompactCPUAddressSpan(200, 100, 100))
    with pytest.raises(ValueError, match="must not overlap"):
        CompactCPUAddress(
            byte_offset=100, logical_length=200, allocated_length=250, spans=spans
        )


def test_compact_cpu_load_store_spec() -> None:
    spec = CompactCPULoadStoreSpec([CompactCPUAddress(0, 100, 256)])
    assert len(spec.compact_addresses) == 1


# -----------------------------------------------------------------------
# CompactGroupCharge
# -----------------------------------------------------------------------


def test_compact_group_charge_no_prefer_early_eviction() -> None:
    charge = CompactGroupCharge(
        group_idx=0,
        native_block_tokens=4,
        compact_real_bytes_per_rank=120,
        compact_real_bytes_server=240,
    )
    assert not hasattr(charge, "prefer_early_eviction")


# -----------------------------------------------------------------------
# build_compact_group_charges
# -----------------------------------------------------------------------


def test_build_group_charges_basic() -> None:
    charges = build_compact_group_charges(
        _packed_fixture(), world_size=2, block_size_factor=1
    )
    assert len(charges) == 2
    assert charges[0].compact_real_bytes_per_rank == 120
    assert charges[1].compact_real_bytes_per_rank == 80


def test_build_group_charges_block_size_factor() -> None:
    charges = build_compact_group_charges(
        _packed_fixture(), world_size=2, block_size_factor=3
    )
    assert charges[0].compact_real_bytes_per_rank == 360


def test_build_group_charges_validation() -> None:
    empty = KVCacheConfig(num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[])
    with pytest.raises(ValueError, match="requires at least one KV group"):
        build_compact_group_charges(empty, world_size=1, block_size_factor=1)
    with pytest.raises(ValueError, match="world_size"):
        build_compact_group_charges(
            _packed_fixture(), world_size=0, block_size_factor=1
        )
    with pytest.raises(ValueError, match="block_size_factor"):
        build_compact_group_charges(
            _packed_fixture(), world_size=1, block_size_factor=0
        )


# -----------------------------------------------------------------------
# build_compact_layout_accounting
# -----------------------------------------------------------------------


def test_dead_fields_excluded() -> None:
    accounting = build_compact_layout_accounting(
        _packed_fixture(), world_size=2, block_size_factor=1, cpu_budget_bytes=5_600
    )
    assert not hasattr(accounting, "packed_row_stride_bytes_per_rank")
    assert not hasattr(accounting, "current_slot_bytes_server")
    assert not hasattr(accounting, "current_num_slots")
    assert not hasattr(accounting, "compact_common_prefix_capacity_tokens")
    for group in accounting.groups:
        assert not hasattr(group, "prefer_early_eviction")


def test_accounting_live_fields() -> None:
    accounting = build_compact_layout_accounting(
        _packed_fixture(), world_size=2, block_size_factor=1, cpu_budget_bytes=5_600
    )
    assert accounting.world_size == 2
    assert accounting.block_size_factor == 1
    assert accounting.cpu_budget_bytes == 5_600
    assert len(accounting.groups) == 2


def test_accounting_group_slices() -> None:
    accounting = build_compact_layout_accounting(
        _packed_fixture(), world_size=2, block_size_factor=1, cpu_budget_bytes=5_600
    )
    full, sliding = accounting.groups

    assert full.native_block_tokens == 4
    assert full.slot_offsets == (0, 100)
    assert full.compact_real_bytes_per_rank == 120
    assert full.compact_padded_bytes_per_rank == 160
    assert [s.layer_name for s in full.slices] == ["a0", "a1"]
    assert full.slices[0].real_bytes_per_gpu_block == 80
    assert full.slices[1].real_bytes_per_gpu_block == 40
    assert full.slices[0].shared_by == ("a0", "b0")
    assert full.slices[1].shared_by == ("a1", "b1")

    assert sliding.native_block_tokens == 2
    assert sliding.compact_real_bytes_per_rank == 80
    assert sliding.compact_padded_bytes_per_rank == 160


def test_accounting_block_size_factor() -> None:
    accounting = build_compact_layout_accounting(
        _packed_fixture(), world_size=2, block_size_factor=3, cpu_budget_bytes=50_000
    )
    for group in accounting.groups:
        slice_real = sum(s.real_bytes_per_gpu_block for s in group.slices)
        slice_padded = sum(s.padded_bytes_per_gpu_block for s in group.slices)
        assert slice_real * 3 == group.compact_real_bytes_per_rank
        assert slice_padded * 3 == group.compact_padded_bytes_per_rank


def test_accounting_fails_closed() -> None:
    unpacked = _packed_fixture()
    for t in unpacked.kv_cache_tensors:
        t.block_stride = 0
    with pytest.raises(ValueError, match="packed KV layout"):
        build_compact_layout_accounting(
            unpacked, world_size=2, block_size_factor=1, cpu_budget_bytes=5_600
        )

    shifted = _packed_fixture()
    shifted.kv_cache_tensors[0].offset = 1
    with pytest.raises(ValueError, match="do not cover the packed row"):
        build_compact_layout_accounting(
            shifted, world_size=2, block_size_factor=1, cpu_budget_bytes=5_600
        )

    ambiguous = _packed_fixture()
    ambiguous.kv_cache_tensors[0].shared_by.append("a1")
    with pytest.raises(ValueError, match="appears in multiple packed slots"):
        build_compact_layout_accounting(
            ambiguous, world_size=2, block_size_factor=1, cpu_budget_bytes=5_600
        )

    undersized = _packed_fixture()
    undersized.kv_cache_tensors[1].offset = 50
    with pytest.raises(ValueError, match="real payload .* exceeds packed slot"):
        build_compact_layout_accounting(
            undersized, world_size=2, block_size_factor=1, cpu_budget_bytes=5_600
        )

    with pytest.raises(ValueError, match="world_size"):
        build_compact_layout_accounting(
            _packed_fixture(), world_size=0, block_size_factor=1, cpu_budget_bytes=5_600
        )
    with pytest.raises(ValueError, match="block_size_factor"):
        build_compact_layout_accounting(
            _packed_fixture(), world_size=1, block_size_factor=0, cpu_budget_bytes=5_600
        )
    with pytest.raises(ValueError, match="cpu_budget_bytes"):
        build_compact_layout_accounting(
            _packed_fixture(), world_size=1, block_size_factor=1, cpu_budget_bytes=0
        )


def test_accounting_slice_order_matches_layer_names() -> None:
    a0 = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=10,
        dtype=torch.uint8,
        page_size_padded=100,
    )
    a1 = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=5,
        dtype=torch.uint8,
        page_size_padded=60,
    )
    config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[
            KVCacheTensor(
                size=4 * 160, shared_by=["x0", "y0"], offset=0, block_stride=160
            ),
            KVCacheTensor(
                size=4 * 160, shared_by=["x1", "y1"], offset=100, block_stride=160
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["x1", "x0"],
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=4, kv_cache_specs={"x0": a0, "x1": a1}
                ),
            ),
            KVCacheGroupSpec(
                layer_names=["y0", "y1"],
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=4, kv_cache_specs={"y0": a0, "y1": a1}
                ),
            ),
        ],
    )
    accounting = build_compact_layout_accounting(
        config, world_size=1, block_size_factor=1, cpu_budget_bytes=10_000
    )
    assert [s.layer_name for s in accounting.groups[0].slices] == ["x1", "x0"]
    assert [s.layer_name for s in accounting.groups[1].slices] == ["y0", "y1"]


def test_accounting_no_packed_slot_raises() -> None:
    a0 = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=10,
        dtype=torch.uint8,
        page_size_padded=100,
    )
    config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[
            KVCacheTensor(size=200, shared_by=["real"], offset=0, block_stride=100)
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["real", "ghost"],
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=4, kv_cache_specs={"real": a0, "ghost": a0}
                ),
            )
        ],
    )
    with pytest.raises(ValueError, match="no packed slot"):
        build_compact_layout_accounting(
            config, world_size=1, block_size_factor=1, cpu_budget_bytes=5_000
        )


def test_accounting_duplicate_layer_raises() -> None:
    a0 = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=10,
        dtype=torch.uint8,
        page_size_padded=100,
    )
    config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[
            KVCacheTensor(size=200, shared_by=["dup"], offset=0, block_stride=100)
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["dup", "dup"],
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=4, kv_cache_specs={"dup": a0}
                ),
            )
        ],
    )
    with pytest.raises(ValueError, match="appears multiple times"):
        build_compact_layout_accounting(
            config, world_size=1, block_size_factor=1, cpu_budget_bytes=5_000
        )


# -----------------------------------------------------------------------
# MambaSpec accounting
# -----------------------------------------------------------------------


def test_mamba_compact_group_charge() -> None:
    """MambaSpec real-payload accounting via _real_page_size_bytes.

    MambaSpec without page_size_padded returns the sum of its tensor
    products; with padding the real payload is the unpadded sum.
    """
    # Unpadded Mamba
    mamba = MambaSpec(
        block_size=1,
        shapes=((64, 128), (128,)),
        dtypes=(torch.bfloat16, torch.float32),
    )
    config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["m0"], kv_cache_spec=mamba),
        ],
    )
    charges = build_compact_group_charges(config, world_size=1, block_size_factor=1)
    assert len(charges) == 1
    # (64*128*2) + (128*4) = 16384 + 512 = 16896
    assert charges[0].compact_real_bytes_per_rank == 16896
    assert charges[0].compact_real_bytes_server == 16896
    assert charges[0].native_block_tokens == 1


def test_mamba_padded_compact_group_charge() -> None:
    """MambaSpec with page_size_padded includes the padding in page_size_bytes
    but _real_page_size_bytes strips it."""

    mamba = MambaSpec(
        block_size=1,
        shapes=((64, 128),),
        dtypes=(torch.bfloat16,),
        page_size_padded=32768,
    )
    config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["m0"], kv_cache_spec=mamba),
        ],
    )
    charges = build_compact_group_charges(config, world_size=1, block_size_factor=1)
    assert len(charges) == 1
    # page_size_bytes is 32768 (padded), but real payload is 64*128*2 = 16384
    assert mamba.page_size_bytes == 32768
    assert charges[0].compact_real_bytes_per_rank == 16384
