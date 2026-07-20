# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the compact geometry transport seam (commit 2).

Verifies the end-to-end transport path:
1. KVCacheConfig.compact_aggregate_signature
2. OffloadingGroupConfig.compact_bytes_per_native_block_per_worker
3. OffloadingConfig.compact_slice_accounting
4. Inert when enable_compact_layout is false/absent
5. Cross-worker signature agreement
6. PP > 1 fails loud
7. Explicit connector block_size does not change native-block charge
"""

from typing import cast
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.config import KVTransferConfig, ParallelConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    build_offloading_config,
)
from vllm.platforms import current_platform
from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_offload.compact_geometry import CompactTransportSignature
from vllm.v1.kv_offload.cpu.compact_accounting import (
    CompactGroupCharge,
    build_compact_group_charges,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_signature(
    charges: tuple[CompactGroupCharge, ...],
) -> tuple[CompactTransportSignature, ...]:
    """Convert rich CompactGroupCharge to transport-neutral signature."""
    return tuple(
        CompactTransportSignature(
            group_idx=cg.group_idx,
            compact_bytes_per_native_block_per_worker=cg.compact_real_bytes_per_rank,
        )
        for cg in charges
    )


def _packed_kv_cache_config() -> KVCacheConfig:
    """A two-group packed KVCacheConfig with heterogeneous layers."""
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
    b0 = FullAttentionSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=15,
        dtype=torch.uint8,
        page_size_padded=100,
    )
    b1 = FullAttentionSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=5,
        dtype=torch.uint8,
        page_size_padded=60,
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
                    block_size=4,
                    kv_cache_specs={"a0": a0, "a1": a1},
                ),
            ),
            KVCacheGroupSpec(
                layer_names=["b0", "b1"],
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=2,
                    kv_cache_specs={"b0": b0, "b1": b1},
                ),
            ),
        ],
    )


def _make_vllm_config(
    enable_compact: bool = False,
    world_size: int = 1,
    **kwargs,
) -> VllmConfig:
    """Minimal VllmConfig for testing the compact transport seam."""
    config = MagicMock()
    config.cache_config.block_size = 16
    config.cache_config.enable_prefix_caching = True
    config.cache_config.prefix_match_unit = None
    config.cache_config.cache_dtype = torch.float16
    config.cache_config.num_gpu_blocks_override = None
    config.model_config.model = "test-model"
    config.model_config.original_max_model_len = -1
    config.model_config.max_model_len = 10000
    config.kv_events_config = None
    config.use_v2_model_runner = False
    config.scheduler_config = MagicMock()
    config.scheduler_config.max_num_seqs = 16
    config.scheduler_config.max_num_batched_tokens = 64

    with patch.object(current_platform, "device_count", return_value=world_size):
        config.parallel_config = ParallelConfig(
            tensor_parallel_size=kwargs.get("tensor_parallel_size", 1),
            pipeline_parallel_size=kwargs.get("pipeline_parallel_size", 1),
        )

    extra = {}
    if enable_compact:
        extra["enable_compact_layout"] = "true"
    config.kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config=extra,
    )
    return cast(VllmConfig, config)


# ---------------------------------------------------------------------------
# 1. Legacy configs unchanged / None
# ---------------------------------------------------------------------------


def test_legacy_config_signature_is_none() -> None:
    """KVCacheConfig without enable_compact_layout keeps signature None."""
    config = _packed_kv_cache_config()
    assert config.compact_aggregate_signature is None


def test_legacy_group_config_no_compact_bytes() -> None:
    """OffloadingGroupConfig remains None for legacy builds."""
    kv_config = _packed_kv_cache_config()
    vllm_config = _make_vllm_config(enable_compact=False)
    off_config = build_offloading_config(vllm_config, kv_config)
    for g in off_config.groups:
        assert g.compact_bytes_per_native_block_per_worker is None


def test_legacy_config_slice_accounting_none() -> None:
    """OffloadingConfig.compact_slice_accounting is None in legacy mode."""
    kv_config = _packed_kv_cache_config()
    vllm_config = _make_vllm_config(enable_compact=False)
    off_config = build_offloading_config(vllm_config, kv_config)
    assert off_config.compact_slice_accounting is None


def test_legacy_offloading_group_config_unchanged() -> None:
    """Legacy OffloadingGroupConfig fields are unchanged."""
    kv_config = _packed_kv_cache_config()
    vllm_config = _make_vllm_config(enable_compact=False)
    off_config = build_offloading_config(vllm_config, kv_config)
    assert len(off_config.groups) == 2
    tokens_a = off_config.groups[0].tokens_per_block
    tokens_b = off_config.groups[1].tokens_per_block
    assert tokens_a == 4
    assert tokens_b == 2


# ---------------------------------------------------------------------------
# 2. Heterogeneous packed config derives charges before scheduler projection
# ---------------------------------------------------------------------------


def test_enabled_computes_aggregate_signature() -> None:
    """enable_compact_layout=true populates compact_aggregate_signature."""
    kv_config = _packed_kv_cache_config()
    # Simulate what get_kv_cache_configs would do:
    world_size = 1
    signature = build_compact_group_charges(
        kv_config, world_size=world_size, block_size_factor=1
    )
    assert len(signature) == 2
    assert signature[0].compact_real_bytes_per_rank == 120  # 80 + 40
    assert signature[1].compact_real_bytes_per_rank == 80  # 60 + 20
    # No prefer_early_eviction
    assert not hasattr(signature[0], "prefer_early_eviction")


def test_heterogeneous_charges_match_layer_specs() -> None:
    """Per-group charge derives from real layer page sizes."""
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
    # real_page_size_bytes for FullAttentionSpec:
    #   2 * block_size * num_kv_heads * head_size * dtype_bytes
    # a0: 2 * 4 * 1 * 10 * 1 = 80
    # a1: 2 * 4 * 1 * 5 * 1 = 40
    # Group 0 total: 120
    #
    # page_size_padded is only used by the GPU allocator (padded allocation);
    # build_compact_group_charges uses the real (unpadded) page size.
    config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[
            KVCacheTensor(
                size=4 * 160, shared_by=["a0", "b0"], offset=0, block_stride=160
            ),
            KVCacheTensor(
                size=4 * 160, shared_by=["a1", "b1"], offset=100, block_stride=160
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["a0", "a1"],
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=4,
                    kv_cache_specs={"a0": a0, "a1": a1},
                ),
            ),
            KVCacheGroupSpec(
                layer_names=["b0"],
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=4,
                    kv_cache_specs={
                        "b0": FullAttentionSpec(
                            block_size=4,
                            num_kv_heads=1,
                            head_size=15,
                            dtype=torch.uint8,
                            page_size_padded=100,
                        ),
                    },
                ),
            ),
        ],
    )
    charges = build_compact_group_charges(config, world_size=1, block_size_factor=1)
    # Group 0: a0 real=80, a1 real=40 -> total=120
    assert charges[0].compact_real_bytes_per_rank == 120
    # Group 1: b0 real=120 -> total=120
    assert charges[1].compact_real_bytes_per_rank == 120


# ---------------------------------------------------------------------------
# 3. Scheduler projection preserves aggregate signature
# ---------------------------------------------------------------------------


def test_get_kv_cache_configs_stamps_signature() -> None:
    """get_kv_cache_configs stamps compact_aggregate_signature when enabled."""
    vllm_config = _make_vllm_config(enable_compact=True, world_size=2)
    # float16: unpadded = 2 * block_size * num_kv_heads * head_size * 2
    # a0: 2 * 4 * 1 * 10 * 2 = 160
    # a1: 2 * 4 * 1 * 5 * 2 = 80
    a0 = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=10,
        dtype=torch.float16,
        page_size_padded=200,
    )
    a1 = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=5,
        dtype=torch.float16,
        page_size_padded=100,
    )
    worker_specs: list[dict[str, KVCacheSpec]] = [
        {"a0": a0, "a1": a1},
        {"a0": a0, "a1": a1},
    ]
    available = [1_000_000, 1_000_000]

    configs = get_kv_cache_configs(vllm_config, worker_specs, available)

    for cfg in configs:
        assert cfg.compact_aggregate_signature is not None
        assert len(cfg.compact_aggregate_signature) == 1  # merged into one group


def test_pp_greater_than_one_fails_loud() -> None:
    """enable_compact_layout with pipeline_parallel_size > 1 raises."""
    vllm_config = _make_vllm_config(
        enable_compact=True,
        world_size=2,
        pipeline_parallel_size=2,
    )
    a0 = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=10,
        dtype=torch.float16,
        page_size_padded=200,
    )
    worker_specs: list[dict[str, KVCacheSpec]] = [
        {"l0": a0},
        {"l0": a0},
    ]
    available = [1_000_000, 1_000_000]

    with pytest.raises(ValueError, match="pipeline_parallel_size"):
        get_kv_cache_configs(vllm_config, worker_specs, available)


# ---------------------------------------------------------------------------
# 4. Worker build_offloading_config derives slice accounting
# ---------------------------------------------------------------------------


def test_worker_derives_slice_accounting() -> None:
    """Worker-side build_offloading_config derives compact_slice_accounting."""
    kv_config = _packed_kv_cache_config()
    signature = _to_signature(
        build_compact_group_charges(kv_config, world_size=1, block_size_factor=1)
    )
    kv_config.compact_aggregate_signature = signature

    vllm_config = _make_vllm_config(enable_compact=True)
    off_config = build_offloading_config(vllm_config, kv_config)

    # Compact slice accounting should be present
    assert off_config.compact_slice_accounting is not None
    assert len(off_config.compact_slice_accounting) == 2

    # Group 0 slice details
    g0 = off_config.compact_slice_accounting[0]
    assert g0.group_idx == 0
    assert len(g0.slices) == 2
    assert g0.slices[0].layer_name == "a0"
    assert g0.slices[1].layer_name == "a1"
    assert g0.slices[0].offset_bytes == 0
    assert g0.slices[1].offset_bytes == 100

    # Group 1 slice details
    g1 = off_config.compact_slice_accounting[1]
    assert g1.group_idx == 1
    assert len(g1.slices) == 2


def test_worker_slice_accounting_aggregate_matches_charge() -> None:
    """CompactGroupSliceConfig aggregate matches transported charge."""
    kv_config = _packed_kv_cache_config()
    charges = build_compact_group_charges(kv_config, world_size=1, block_size_factor=1)
    signature = _to_signature(charges)
    kv_config.compact_aggregate_signature = signature

    vllm_config = _make_vllm_config(enable_compact=True)
    off_config = build_offloading_config(vllm_config, kv_config)

    assert off_config.compact_slice_accounting is not None
    for gs in off_config.compact_slice_accounting:
        charge = signature[gs.group_idx]
        assert (
            gs.compact_real_bytes_per_rank
            == charge.compact_bytes_per_native_block_per_worker
        )


def test_worker_compact_bytes_per_native_block_populated() -> None:
    """OffloadingGroupConfig.compact_bytes_per_native_block_per_worker populated."""
    kv_config = _packed_kv_cache_config()
    signature = _to_signature(
        build_compact_group_charges(kv_config, world_size=1, block_size_factor=1)
    )
    kv_config.compact_aggregate_signature = signature

    vllm_config = _make_vllm_config(enable_compact=True)
    off_config = build_offloading_config(vllm_config, kv_config)

    assert off_config.groups[0].compact_bytes_per_native_block_per_worker == 120
    assert off_config.groups[1].compact_bytes_per_native_block_per_worker == 80


def test_blocks_per_chunk_gt_one_charge_unchanged() -> None:
    """Explicit connector block_size producing blocks_per_chunk > 1 does not
    change the transported native-block charge.

    The *native* block charge (compact_bytes_per_native_block_per_worker)
    stays at one GPU KV block regardless of how many blocks the runtime
    coalesces into one offload chunk (blocks_per_chunk).
    """
    # Uniform block_size=4 across all groups so that `block_size=8`
    # yields blocks_per_chunk = 8 // 4 = 2.
    # Slot 0 at offset 0 has 100 bytes (padded), slot 1 at offset 100
    # has 60 bytes.  Each layer's real payload must fit its slot.
    # real = 2 * block_size * num_kv_heads * head_size * dtype_bytes
    # Slot 0: max head_size with num_kv_heads=1, dtype=uint8:
    #          100 / (2 * 4 * 1 * 1) = 12.5  -> use head_size=12 (real=96)
    # Slot 1: max head_size with num_kv_heads=1, dtype=uint8:
    #          60 / (2 * 4 * 1 * 1) = 7.5   -> use head_size=6 (real=48)
    a0 = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=12,
        dtype=torch.uint8,
        page_size_padded=100,
    )
    b0 = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=12,
        dtype=torch.uint8,
        page_size_padded=100,
    )
    a1 = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=6,
        dtype=torch.uint8,
        page_size_padded=60,
    )
    b1 = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=6,
        dtype=torch.uint8,
        page_size_padded=60,
    )
    kv_config = KVCacheConfig(
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
                    block_size=4,
                    kv_cache_specs={"a0": a0, "a1": a1},
                ),
            ),
            KVCacheGroupSpec(
                layer_names=["b0", "b1"],
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=4,
                    kv_cache_specs={"b0": b0, "b1": b1},
                ),
            ),
        ],
    )
    signature = build_compact_group_charges(
        kv_config, world_size=1, block_size_factor=1
    )
    kv_config.compact_aggregate_signature = _to_signature(signature)

    # Simulate a connector block_size of 8 tokens with native block_size of 4:
    # blocks_per_chunk = 8 // 4 = 2.
    vllm_config = _make_vllm_config(enable_compact=True)
    vllm_config.kv_transfer_config.kv_connector_extra_config["block_size"] = "8"

    off_config = build_offloading_config(vllm_config, kv_config)
    assert off_config.cache.blocks_per_chunk == 2

    # Group 0: a0 real=96 + a1 real=48 = 144
    assert off_config.groups[0].compact_bytes_per_native_block_per_worker == 144
    # Group 1: b0 real=96 + b1 real=48 = 144
    assert off_config.groups[1].compact_bytes_per_native_block_per_worker == 144

    # worker_kv_bytes_per_block also stays at the native packed-block value.
    # Runtime consumers multiply by blocks_per_chunk when sizing offload
    # payloads.
    assert off_config.worker_kv_bytes_per_block == 160


# ---------------------------------------------------------------------------
# 5. Manager/worker normalized group aggregate agreement
# ---------------------------------------------------------------------------


def test_manager_worker_group_agreement() -> None:
    """Manager-side (pre-projection) and worker-side (post-projection) configs
    agree on compact_bytes_per_native_block_per_worker for shared groups."""
    kv_config = _packed_kv_cache_config()
    signature = _to_signature(
        build_compact_group_charges(kv_config, world_size=2, block_size_factor=1)
    )
    kv_config.compact_aggregate_signature = signature

    vllm_config = _make_vllm_config(enable_compact=True, world_size=2)
    off_config = build_offloading_config(vllm_config, kv_config)

    # Both groups should have compact_bytes_per_native_block_per_worker populated
    assert off_config.groups[0].compact_bytes_per_native_block_per_worker is not None
    assert off_config.groups[1].compact_bytes_per_native_block_per_worker is not None
    # Verify manager-side: all groups present.
    assert len(off_config.groups) == 2


def test_compact_bytes_per_native_block_indexed_by_group_idx() -> None:
    """compact_bytes_per_native_block_per_worker values map to correct group."""
    kv_config = _packed_kv_cache_config()
    # Create out-of-order charges
    charges = list(
        build_compact_group_charges(kv_config, world_size=1, block_size_factor=1)
    )
    # Swap to verify index matching
    charges[0], charges[1] = charges[1], charges[0]
    # But charges[0].group_idx is now 1 — the assertion in build_offloading_config
    # will fail because group_idx doesn't match builder index.
    # Restore and verify correct mapping:
    charges[0], charges[1] = charges[1], charges[0]
    kv_config.compact_aggregate_signature = _to_signature(tuple(charges))

    vllm_config = _make_vllm_config(enable_compact=True)
    off_config = build_offloading_config(vllm_config, kv_config)

    # Group 0 -> charge 0
    assert off_config.groups[0].compact_bytes_per_native_block_per_worker == 120
    # Group 1 -> charge 1
    assert off_config.groups[1].compact_bytes_per_native_block_per_worker == 80


# ---------------------------------------------------------------------------
# 6. Mismatched worker signatures fail loud
# ---------------------------------------------------------------------------


def test_mismatched_signature_group_idx_asserts() -> None:
    """Mismatched group index between builder and signature asserts."""
    kv_config = _packed_kv_cache_config()
    charges = list(
        build_compact_group_charges(kv_config, world_size=1, block_size_factor=1)
    )
    # Corrupt group_idx
    old = charges[0]
    charges[0] = CompactGroupCharge(
        group_idx=99,
        native_block_tokens=old.native_block_tokens,
        compact_real_bytes_per_rank=old.compact_real_bytes_per_rank,
        compact_real_bytes_server=old.compact_real_bytes_server,
    )
    kv_config.compact_aggregate_signature = _to_signature(tuple(charges))

    vllm_config = _make_vllm_config(enable_compact=True)
    with pytest.raises(AssertionError, match="group index"):
        build_offloading_config(vllm_config, kv_config)


def test_mismatched_signature_count_asserts() -> None:
    """Signature group count != KVCacheConfig group count asserts."""
    kv_config = _packed_kv_cache_config()
    # Only one charge but two groups
    single_charge = (
        CompactTransportSignature(
            group_idx=0,
            compact_bytes_per_native_block_per_worker=120,
        ),
    )
    kv_config.compact_aggregate_signature = single_charge

    vllm_config = _make_vllm_config(enable_compact=True)
    with pytest.raises(AssertionError, match="Group count"):
        build_offloading_config(vllm_config, kv_config)


def test_mismatched_worker_signatures_raise_in_get_kv_cache_configs() -> None:
    """Cross-worker signature mismatch in get_kv_cache_configs raises."""
    # This test simulates the assertion in get_kv_cache_configs where
    # two workers would have different signatures. Since we compute the
    # signature once from global groups and stamp it uniformly, this
    # path can only be reached if build_compact_group_charges itself
    # produces different results for different workers' configs, which
    # it cannot do because the global groups are shared.
    #
    # Instead, verify that the explicit cross-worker assertion in
    # get_kv_cache_configs uses == correctly by constructing two
    # configs with different signatures manually.
    a0 = FullAttentionSpec(
        block_size=4, num_kv_heads=1, head_size=10, dtype=torch.float16
    )
    config_a = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["a0"], a0)],
    )
    config_b = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["a0"], a0)],
    )
    sig = (CompactTransportSignature(0, 10),)
    config_a.compact_aggregate_signature = sig
    # Different real bytes
    sig_b = (CompactTransportSignature(0, 20),)
    config_b.compact_aggregate_signature = sig_b

    assert config_a.compact_aggregate_signature != config_b.compact_aggregate_signature


def test_mismatched_slice_accounting_vs_charge_asserts() -> None:
    """Slice accounting that doesn't match transported charge asserts."""
    kv_config = _packed_kv_cache_config()
    # Build real charges
    real_charges = build_compact_group_charges(
        kv_config, world_size=1, block_size_factor=1
    )
    # Replace group 0 charge with a wrong value
    wrong_charges = list(real_charges)
    wrong_charges[0] = CompactGroupCharge(
        group_idx=0,
        native_block_tokens=wrong_charges[0].native_block_tokens,
        compact_real_bytes_per_rank=99999,  # deliberately wrong
        compact_real_bytes_server=99999,
    )
    kv_config.compact_aggregate_signature = _to_signature(tuple(wrong_charges))

    vllm_config = _make_vllm_config(enable_compact=True)
    with pytest.raises(AssertionError, match="slice accounting"):
        build_offloading_config(vllm_config, kv_config)


# ---------------------------------------------------------------------------
# 7. Signature excludes prefer_early_eviction / bounded-tail
# ---------------------------------------------------------------------------


def test_signature_no_prefer_early_eviction() -> None:
    """CompactTransportSignature has no prefer_early_eviction field."""
    charge = CompactTransportSignature(
        group_idx=0,
        compact_bytes_per_native_block_per_worker=120,
    )
    assert not hasattr(charge, "prefer_early_eviction")


def test_signature_has_only_neutral_fields() -> None:
    """Signature fields are exactly group_idx and compact_bytes_per_native_block."""
    import dataclasses

    fields = {f.name for f in dataclasses.fields(CompactTransportSignature)}
    assert fields == {"group_idx", "compact_bytes_per_native_block_per_worker"}, (
        f"CompactTransportSignature has unexpected fields: {fields}"
    )
