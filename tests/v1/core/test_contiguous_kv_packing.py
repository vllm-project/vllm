# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for contiguous KV cache packing in _get_kv_cache_config_deepseek_v4."""

from unittest.mock import MagicMock

import pytest
import torch

from vllm.v1.core.kv_cache_utils import _get_kv_cache_config_deepseek_v4
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    MLAAttentionSpec,
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


def _mock_vllm_config():
    config = MagicMock()
    config.scheduler_config.num_gpu_blocks_override = None
    return config


def _run(n_c4=3, n_c128=2, n_swa=5, mem=100 * 1024 * 1024):
    groups = _make_groups(n_c4, n_c128, n_swa)
    return _get_kv_cache_config_deepseek_v4(_mock_vllm_config(), groups, mem)


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

    def test_mla_and_swa_in_separate_tensors(self):
        _, tensors = _run(n_c4=3, n_swa=3)
        for t in tensors:
            names = t.shared_by
            has_mla = any("c4_mla" in n or "c4_idx" in n or "c128" in n for n in names)
            has_swa = any("swa" in n for n in names)
            assert not (has_mla and has_swa), (
                "MLA and SWA layers must be in separate tensors"
            )

    def test_strided_views_are_independent(self):
        num_blocks, tensors = _run()
        backing = torch.zeros(tensors[0].size, dtype=torch.uint8)
        views = []
        for t in tensors:
            page_size = t.block_stride // 1
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
