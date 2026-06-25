# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for contiguous KV cache packing."""

from unittest.mock import MagicMock

import pytest
import torch

from vllm.v1.core.kv_cache_utils import (
    _get_kv_cache_config_packed,
    get_kv_cache_config_from_groups,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)


def _make_mla_spec(page_size: int, block_size: int = 256) -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        page_size_padded=page_size,
        cache_dtype_str="fp8_ds_mla",
        model_version="deepseek_v4",
        alignment=576,
    )


def _make_full_spec() -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.float16,
    )


def _make_sw_spec() -> SlidingWindowSpec:
    return SlidingWindowSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.float16,
        sliding_window=128,
    )


def _make_groups(n_c4, n_c128, n_swa):
    PS_C4_MLA = 37440
    PS_C4_IDX = 8640
    PS_C128 = 1728
    PS_SWA = 37440

    mla_specs = {}
    for i in range(n_c4):
        mla_specs[f"c4_mla.{i}"] = _make_mla_spec(PS_C4_MLA)
        mla_specs[f"c4_idx.{i}"] = _make_mla_spec(PS_C4_IDX)
    for i in range(n_c128):
        mla_specs[f"c128_mla.{i}"] = _make_mla_spec(PS_C128)

    mla_group = KVCacheGroupSpec(
        layer_names=list(mla_specs.keys()),
        kv_cache_spec=UniformTypeKVCacheSpecs(block_size=256, kv_cache_specs=mla_specs),
    )

    swa_specs = {}
    for i in range(n_swa):
        swa_specs[f"swa.{i}"] = _make_mla_spec(PS_SWA)

    swa_group = KVCacheGroupSpec(
        layer_names=list(swa_specs.keys()),
        kv_cache_spec=UniformTypeKVCacheSpecs(block_size=256, kv_cache_specs=swa_specs),
    )

    return [mla_group, swa_group]


def _mock_vllm_config(kv_connector_extra_config: dict[str, str] | None = None):
    config = MagicMock()
    config.cache_config.num_gpu_blocks_override = None
    config.kv_transfer_config = None
    if kv_connector_extra_config is not None:
        config.kv_transfer_config = MagicMock()
        config.kv_transfer_config.kv_connector_extra_config = kv_connector_extra_config
    return config


def _run(n_c4=3, n_c128=2, n_swa=5, mem=100 * 1024 * 1024):
    groups = _make_groups(n_c4, n_c128, n_swa)
    return _get_kv_cache_config_packed(_mock_vllm_config(), groups, mem)


def _page_sizes_by_layer(
    groups: list[KVCacheGroupSpec],
) -> dict[str, int]:
    page_sizes = {}
    for group in groups:
        specs = group.kv_cache_spec.kv_cache_specs
        for layer_name in group.layer_names:
            page_sizes[layer_name] = specs[layer_name].page_size_bytes
    return page_sizes


class TestInterleavedPacking:
    def test_all_tensors_have_block_stride(self):
        _, tensors = _run()
        for t in tensors:
            assert t.block_stride > 0

    def test_all_tensors_share_same_size(self):
        _, tensors = _run()
        sizes = set(t.size for t in tensors)
        assert len(sizes) == 1
        assert sizes.pop() > 0

    def test_offsets_within_one_block(self):
        _, tensors = _run()
        for t in tensors:
            assert t.offset < t.block_stride

    def test_all_layers_accounted_for(self):
        n_c4, n_c128, n_swa = 5, 4, 7
        _, tensors = _run(n_c4=n_c4, n_c128=n_c128, n_swa=n_swa)
        all_names = set()
        for t in tensors:
            all_names.update(t.shared_by)
        expected = n_c4 * 2 + n_c128 + n_swa
        assert len(all_names) == expected

    def test_strided_views_are_independent(self):
        groups = _make_groups(n_c4=3, n_c128=2, n_swa=5)
        page_sizes = _page_sizes_by_layer(groups)
        num_blocks, tensors = _get_kv_cache_config_packed(
            _mock_vllm_config(), groups, 100 * 1024 * 1024
        )
        backing = torch.zeros(tensors[0].size, dtype=torch.uint8)
        views = []
        for t in tensors:
            page_size = page_sizes[t.shared_by[0]]
            v = torch.as_strided(
                backing,
                size=(num_blocks, page_size),
                stride=(t.block_stride, 1),
                storage_offset=t.offset,
            )
            views.append(v)

        for i, v in enumerate(views):
            v.fill_(i + 1)

        for i, v in enumerate(views):
            assert (v == i + 1).all(), f"View {i} was corrupted"

    def test_hma_attention_groups_keep_default_backing(self):
        full = _make_full_spec()
        sw = _make_sw_spec()
        page_size = full.page_size_bytes
        groups = [
            KVCacheGroupSpec(["full.0", "full.1"], full),
            KVCacheGroupSpec(["sw.0", "sw.2"], sw),
            KVCacheGroupSpec(["sw.1", "sw.3"], sw),
        ]

        config = get_kv_cache_config_from_groups(
            _mock_vllm_config(), groups, available_memory=page_size * 2 * 32
        )

        assert config.num_blocks == 32
        assert sum(t.size for t in config.kv_cache_tensors) == page_size * 2 * 32
        assert config.kv_cache_tensors == [
            KVCacheTensor(size=page_size * 32, shared_by=["full.0", "sw.0", "sw.1"]),
            KVCacheTensor(size=page_size * 32, shared_by=["full.1", "sw.2", "sw.3"]),
        ]

    def test_hma_attention_groups_use_packed_backing_with_enable_cross_layers(self):
        full = _make_full_spec()
        sw = _make_sw_spec()
        page_size = full.page_size_bytes
        groups = [
            KVCacheGroupSpec(["full.0", "full.1"], full),
            KVCacheGroupSpec(["sw.0", "sw.2"], sw),
            KVCacheGroupSpec(["sw.1", "sw.3"], sw),
        ]

        config = get_kv_cache_config_from_groups(
            _mock_vllm_config({"enable_cross_layers_blocks": "True"}),
            groups,
            available_memory=page_size * 2 * 32,
        )

        assert config.num_blocks == 32
        assert {t.size for t in config.kv_cache_tensors} == {page_size * 2 * 32}
        assert config.kv_cache_tensors == [
            KVCacheTensor(
                size=page_size * 2 * 32,
                shared_by=["full.0", "sw.0", "sw.1"],
                offset=0,
                block_stride=page_size * 2,
            ),
            KVCacheTensor(
                size=page_size * 2 * 32,
                shared_by=["full.1", "sw.2", "sw.3"],
                offset=page_size,
                block_stride=page_size * 2,
            ),
        ]

    def test_single_group_attention_keeps_unpacked_layout(self):
        spec = _make_full_spec()
        groups = [KVCacheGroupSpec(["full.0", "full.1"], spec)]

        config = get_kv_cache_config_from_groups(
            _mock_vllm_config(), groups, available_memory=spec.page_size_bytes * 2 * 32
        )

        assert sum(t.size for t in config.kv_cache_tensors) == (
            spec.page_size_bytes * 2 * 32
        )
        assert [t.block_stride for t in config.kv_cache_tensors] == [0, 0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
