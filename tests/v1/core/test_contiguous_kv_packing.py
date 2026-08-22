# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for contiguous KV cache packing.

Every cache group packs its layers densely into one block; groups overlay each other
(a block ID is owned by one group at a time), so the packed block stride is the
largest group's packing. The layout decides whether the layer dim sits outside the
block dim (a contiguous region per layer) or inside it (all layers' pages within each
block); the allocation is the same either way.
"""

from unittest.mock import MagicMock

import pytest
import torch

from vllm.config import CacheConfig
from vllm.v1.core.kv_cache_utils import (
    _get_kv_cache_bytes_per_block,
    _pool_bytes_per_block,
    get_kv_cache_config_from_groups,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    KVCacheLayout,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.utils import allocate_kv_cache

MEMORY = 8 * 1024 * 1024


def _mla(head_size: int) -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=64, num_kv_heads=1, head_size=head_size, dtype=torch.uint8
    )


def _full() -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=64, dtype=torch.float16
    )


def _uniform_group(specs: dict) -> KVCacheGroupSpec:
    return KVCacheGroupSpec(
        list(specs),
        UniformTypeKVCacheSpecs(block_size=64, kv_cache_specs=specs),
    )


def _mixed_page_groups(n_mla=3, n_idx=3, n_swa=5):
    """A DeepSeek-V4-style hybrid: two groups with different page mixes."""
    g1 = {f"mla.{i}": _mla(512) for i in range(n_mla)}
    g1.update({f"idx.{i}": _mla(128) for i in range(n_idx)})
    g2 = {f"swa.{i}": _mla(512) for i in range(n_swa)}
    return [_uniform_group(g1), _uniform_group(g2)], g1, g2


def _mock_vllm_config(layout: str | None):
    config = MagicMock()
    config.cache_config = CacheConfig()
    config.cache_config.num_gpu_blocks_override = None
    config.cache_config.kv_cache_layout = layout
    return config


def _pages(groups) -> dict[str, int]:
    return {
        name: group.kv_cache_spec.kv_cache_specs[name].page_size_bytes
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        else group.kv_cache_spec.page_size_bytes
        for group in groups
        for name in group.layer_names
    }


def _expected_bytes_per_block(groups) -> int:
    pages = _pages(groups)
    return max(sum(pages[n] for n in g.layer_names) for g in groups)


def _bind(config, layout: str):
    return allocate_kv_cache(config, torch.device("cpu"), KVCacheLayout[layout], None)


class TestDensePacking:
    def test_bytes_per_block_is_largest_group(self):
        groups, g1, g2 = _mixed_page_groups()
        assert _get_kv_cache_bytes_per_block(groups) == _expected_bytes_per_block(
            groups
        )

        config = get_kv_cache_config_from_groups(
            _mock_vllm_config("BLHNC"), groups, MEMORY
        )
        # Groups overlay: both start at offset 0.
        assert [tensor.offset for tensor in config.kv_cache_tensors].count(0) == 2
        assert [tensor.layers for tensor in config.kv_cache_tensors] == [
            list(g1)[:3],
            list(g1)[3:],
            list(g2),
        ]

    def test_layers_within_a_group_are_dense(self):
        groups, _, _ = _mixed_page_groups()
        pages = _pages(groups)
        config = get_kv_cache_config_from_groups(
            _mock_vllm_config("BLHNC"), groups, MEMORY
        )
        offsets = {
            name: tensor.offset + i * tensor.layer_stride
            for tensor in config.kv_cache_tensors
            for i, name in enumerate(tensor.layers)
        }
        for group in groups:
            expected = 0
            for name in group.layer_names:
                assert offsets[name] == expected
                expected += pages[name]

    @pytest.mark.parametrize("layout", ["LBNHC", "BLHNC"])
    def test_allocation_is_layout_invariant(self, layout):
        specs = {f"l.{i}": _full() for i in range(4)}
        groups = [KVCacheGroupSpec(list(specs), _full())]
        config = get_kv_cache_config_from_groups(
            _mock_vllm_config(layout), groups, MEMORY
        )
        (tensor,) = config.kv_cache_tensors
        page = _full().page_size_bytes
        assert tensor.layers == list(specs)
        assert config.num_blocks == MEMORY // (4 * page)
        assert tensor.size == 4 * page * config.num_blocks
        if layout == "LBNHC":
            assert (tensor.layer_stride, tensor.block_stride) == (
                page * config.num_blocks,
                page,
            )
        else:
            assert (tensor.layer_stride, tensor.block_stride) == (page, 4 * page)

    @pytest.mark.parametrize("layout", ["LBNHC", "BLHNC"])
    def test_single_group_mixed_pages_follows_layout(self, layout):
        specs = {"mla.0": _mla(512), "mla.1": _mla(512), "idx.0": _mla(128)}
        groups = [_uniform_group(specs)]
        config = get_kv_cache_config_from_groups(
            _mock_vllm_config(layout), groups, MEMORY
        )
        block_stride = _expected_bytes_per_block(groups)
        mla_tensor, idx_tensor = config.kv_cache_tensors
        assert mla_tensor.layers == ["mla.0", "mla.1"]
        assert idx_tensor.layers == ["idx.0"]
        assert {t.size for t in config.kv_cache_tensors} == {
            block_stride * config.num_blocks
        }
        if layout == "LBNHC":
            assert (
                idx_tensor.offset == 2 * _mla(512).page_size_bytes * config.num_blocks
            )
            assert mla_tensor.block_stride == _mla(512).page_size_bytes
        else:
            assert idx_tensor.offset == 2 * _mla(512).page_size_bytes
            assert mla_tensor.block_stride == block_stride

    def test_overlaid_groups_alias_and_stay_isolated(self):
        groups, g1, g2 = _mixed_page_groups()
        # Overlay models resolve to a block-outer layout at backend selection (the
        # model's backend declares it); mirror that here.
        config = get_kv_cache_config_from_groups(
            _mock_vllm_config("BLNHC"), groups, MEMORY
        )
        assert config.num_blocks == MEMORY // _expected_bytes_per_block(groups)
        assert _pool_bytes_per_block(groups) == _expected_bytes_per_block(groups)

        views = _bind(config, "BLNHC")
        assert set(views) == set(g1) | set(g2)
        assert views["swa.0"].data_ptr() == views["mla.0"].data_ptr()

        # A block is owned by one group at a time: writes to group-owned
        # blocks never disturb the other group's blocks.
        for i, name in enumerate(g1):
            views[name][0].fill_(i + 1)
            views[name][2].fill_(i + 1)
        for i, name in enumerate(g2):
            views[name][1].fill_(100 + i)
            views[name][3].fill_(100 + i)
        for i, name in enumerate(g1):
            assert (views[name][0].to(torch.int32) == i + 1).all()
            assert (views[name][2].to(torch.int32) == i + 1).all()
        for i, name in enumerate(g2):
            assert (views[name][1].to(torch.int32) == 100 + i).all()
            assert (views[name][3].to(torch.int32) == 100 + i).all()
        # Layers within a group are disjoint.
        views["mla.0"][0].fill_(77)
        for i, name in enumerate(list(g1)[1:], start=1):
            assert (views[name][0].to(torch.int32) == i + 1).all()

    def test_layer_compact_layout_rejected_for_overlaid_groups(self):
        # The layout has a single writer (backend-selection resolution); a layer-
        # compact layout reaching an overlay model's allocation is an error, not a
        # silent flip.
        groups, _, _ = _mixed_page_groups()
        with pytest.raises(
            ValueError, match="cannot express this model's mixed page sizes"
        ):
            get_kv_cache_config_from_groups(_mock_vllm_config("LBNHC"), groups, MEMORY)

    def test_unresolved_layout_rejected(self):
        groups, _, _ = _mixed_page_groups()
        with pytest.raises(ValueError, match="has not been resolved"):
            get_kv_cache_config_from_groups(_mock_vllm_config(None), groups, MEMORY)

    def test_head_outer_layout_rejected_for_mixed_pages(self):
        groups, _, _ = _mixed_page_groups()
        with pytest.raises(
            ValueError, match="cannot express this model's mixed page sizes"
        ):
            get_kv_cache_config_from_groups(_mock_vllm_config("LHBNC"), groups, MEMORY)

    @pytest.mark.parametrize("layout", ["LBNHC", "BLHNC"])
    def test_bound_views_round_trip(self, layout):
        specs = {"mla.0": _mla(512), "mla.1": _mla(512), "idx.0": _mla(128)}
        groups = [_uniform_group(specs)]
        config = get_kv_cache_config_from_groups(
            _mock_vllm_config(layout), groups, MEMORY
        )
        views = _bind(config, layout)
        for i, name in enumerate(specs):
            views[name].fill_(i + 1)
        for i, name in enumerate(specs):
            assert (views[name].to(torch.int32) == i + 1).all(), name
            assert views[name].shape[0] == config.num_blocks


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
