"""Tests for contiguous KV block packing in _get_kv_cache_config_deepseek_v4."""

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


def _split_tensors(tensors):
    mla = [t for t in tensors if any("mla" in n or "idx" in n for n in t.shared_by)]
    swa = [t for t in tensors if any("swa" in n for n in t.shared_by)]
    return mla, swa


class TestMlaContiguousPacking:
    def test_all_mla_share_one_size(self):
        _, tensors = _run()
        mla, _ = _split_tensors(tensors)
        assert len(set(t.size for t in mla)) == 1

    def test_all_mla_share_one_block_stride(self):
        _, tensors = _run()
        mla, _ = _split_tensors(tensors)
        strides = set(t.block_stride for t in mla)
        assert len(strides) == 1
        assert strides.pop() > 0

    def test_mla_offsets_are_unique(self):
        _, tensors = _run(n_c4=5, n_c128=4)
        mla, _ = _split_tensors(tensors)
        offsets = [t.offset for t in mla]
        assert len(offsets) == len(set(offsets))

    def test_mla_offsets_fit_within_block_stride(self):
        _, tensors = _run(n_c4=5, n_c128=4)
        mla, _ = _split_tensors(tensors)
        stride = mla[0].block_stride
        for t in mla:
            assert 0 <= t.offset < stride, f"offset {t.offset} >= block_stride {stride}"

    def test_mla_all_layers_accounted_for(self):
        n_c4, n_c128 = 5, 4
        _, tensors = _run(n_c4=n_c4, n_c128=n_c128)
        mla, _ = _split_tensors(tensors)
        all_names = set()
        for t in mla:
            all_names.update(t.shared_by)
        expected_count = n_c4 * 2 + n_c128  # c4_mla + c4_idx + c128_mla
        assert len(all_names) == expected_count

    def test_size_equals_stride_times_num_blocks(self):
        num_blocks, tensors = _run()
        mla, _ = _split_tensors(tensors)
        for t in mla:
            assert t.size == t.block_stride * num_blocks


class TestSwaContiguousPacking:
    def test_swa_packed_separately_from_mla(self):
        _, tensors = _run()
        mla, swa = _split_tensors(tensors)
        assert len(swa) > 0
        assert len(mla) > 0
        assert set(t.size for t in swa) != set(t.size for t in mla)

    def test_swa_share_one_size(self):
        _, tensors = _run()
        _, swa = _split_tensors(tensors)
        assert len(set(t.size for t in swa)) == 1

    def test_swa_share_one_block_stride(self):
        _, tensors = _run()
        _, swa = _split_tensors(tensors)
        strides = set(t.block_stride for t in swa)
        assert len(strides) == 1
        assert strides.pop() > 0

    def test_swa_offsets_are_unique(self):
        _, tensors = _run(n_swa=10)
        _, swa = _split_tensors(tensors)
        offsets = [t.offset for t in swa]
        assert len(offsets) == len(set(offsets))

    def test_swa_all_layers_accounted_for(self):
        n_swa = 7
        _, tensors = _run(n_swa=n_swa)
        _, swa = _split_tensors(tensors)
        all_names = set()
        for t in swa:
            all_names.update(t.shared_by)
        assert len(all_names) == n_swa

    def test_swa_size_equals_stride_times_num_blocks(self):
        num_blocks, tensors = _run()
        _, swa = _split_tensors(tensors)
        for t in swa:
            assert t.size == t.block_stride * num_blocks


class TestStridedViewCorrectness:
    def test_views_are_independent(self):
        page_sizes = [1728, 8640, 37440]
        layer_tuple_bytes = sum(page_sizes)
        num_tuples = 3
        block_stride = layer_tuple_bytes * num_tuples
        num_blocks = 4

        backing = torch.zeros(block_stride * num_blocks, dtype=torch.uint8)

        views = []
        for t in range(num_tuples):
            for ps_idx, ps in enumerate(page_sizes):
                offset = t * layer_tuple_bytes + sum(page_sizes[:ps_idx])
                view = torch.as_strided(
                    backing,
                    size=(num_blocks, ps),
                    stride=(block_stride, 1),
                    storage_offset=offset,
                )
                views.append((f"tuple{t}_ps{ps}", view))

        for i, (name, view) in enumerate(views):
            view.fill_(i + 1)
            for j, (_, other) in enumerate(views):
                if j <= i:
                    continue
                assert other.sum() == 0, f"Writing to {name} corrupted view {j}"

        for i, (name, view) in enumerate(views):
            assert view.sum() == (i + 1) * view.numel()

    def test_block_isolation(self):
        ps = 37440
        n_layers = 5
        block_stride = ps * n_layers
        num_blocks = 3

        backing = torch.zeros(block_stride * num_blocks, dtype=torch.uint8)
        views = [
            torch.as_strided(backing, (num_blocks, ps), (block_stride, 1), layer * ps)
            for layer in range(n_layers)
        ]

        views[2][1].fill_(42)
        assert views[2][0].sum() == 0
        assert views[2][2].sum() == 0
        for i in range(n_layers):
            if i != 2:
                assert views[i].sum() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
