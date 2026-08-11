# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for KV cache packing.

Every cache group packs its layers densely into one block; groups overlay
each other (a block ID is owned by one group at a time), so the packed block
stride is the largest group's packing. The layout decides whether the layer
dim sits outside the block dim (a contiguous region per layer) or inside it
(all layers' pages within each block); the allocation is the same either way.
"""

from unittest.mock import MagicMock

import pytest
import torch

from vllm.v1.attention.backends.utils import (
    get_kv_cache_layout,
    set_kv_cache_layout,
)
from vllm.v1.core.kv_cache_utils import (
    _get_packed_kv_cache_layout,
    _pool_bytes_per_block,
    get_kv_cache_config_from_groups,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.utils import allocate_and_reshape_kv_cache

MEMORY = 8 * 1024 * 1024


@pytest.fixture(autouse=True)
def reset_layout():
    yield
    set_kv_cache_layout(None)


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


def _mock_vllm_config():
    config = MagicMock()
    config.cache_config.num_gpu_blocks_override = None
    config.cache_config.kv_cache_layout = None
    return config


def _pages(groups) -> dict[str, int]:
    return {
        name: group.kv_cache_spec.kv_cache_specs[name].page_size_bytes
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        else group.kv_cache_spec.page_size_bytes
        for group in groups
        for name in group.layer_names
    }


def _packed_block_stride(groups) -> int:
    pages = _pages(groups)
    return max(sum(pages[n] for n in g.layer_names) for g in groups)


def _bind(config):
    return allocate_and_reshape_kv_cache(
        config, torch.device("cpu"), get_kv_cache_layout(), None
    )


class TestDensePacking:
    def test_packed_block_stride_is_largest_group_packing(self):
        groups, g1, g2 = _mixed_page_groups()
        block_stride, runs = _get_packed_kv_cache_layout(groups)
        assert block_stride == _packed_block_stride(groups)
        # Groups overlay: both start at offset 0.
        assert [offset for offset, _, _ in runs].count(0) == 2
        # Same-spec layers merge into one tensor, in layer order.
        assert [layers for _, layers, _ in runs] == [
            list(g1)[:3],
            list(g1)[3:],
            list(g2),
        ]

    def test_layers_within_a_group_are_dense(self):
        groups, _, _ = _mixed_page_groups()
        pages = _pages(groups)
        _, runs = _get_packed_kv_cache_layout(groups)
        offsets = {
            name: run_offset + i * pages[name]
            for run_offset, layers, _ in runs
            for i, name in enumerate(layers)
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
        set_kv_cache_layout(layout)
        config = get_kv_cache_config_from_groups(_mock_vllm_config(), groups, MEMORY)
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
        set_kv_cache_layout(layout)
        config = get_kv_cache_config_from_groups(_mock_vllm_config(), groups, MEMORY)
        block_stride = _packed_block_stride(groups)
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
        config = get_kv_cache_config_from_groups(_mock_vllm_config(), groups, MEMORY)
        assert config.num_blocks == MEMORY // _packed_block_stride(groups)
        assert _pool_bytes_per_block(groups) == _packed_block_stride(groups)

        views = _bind(config)
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

    def test_overlaid_groups_select_a_block_outer_layout(self):
        groups, _, _ = _mixed_page_groups()
        vllm_config = _mock_vllm_config()
        get_kv_cache_config_from_groups(vllm_config, groups, MEMORY)
        assert not get_kv_cache_layout().is_layer_compact
        assert vllm_config.cache_config.kv_cache_layout == "BLHNC"

    def test_explicit_layer_outer_layout_rejected_for_overlays(self):
        groups, _, _ = _mixed_page_groups()
        set_kv_cache_layout("LBNHC")
        with pytest.raises(ValueError, match="places layers outside blocks"):
            get_kv_cache_config_from_groups(_mock_vllm_config(), groups, MEMORY)

    def test_head_outer_layout_rejected_for_mixed_pages(self):
        groups, _, _ = _mixed_page_groups()
        set_kv_cache_layout("LHBNC")
        with pytest.raises(ValueError, match="hoists heads outside the block"):
            get_kv_cache_config_from_groups(_mock_vllm_config(), groups, MEMORY)

    @pytest.mark.parametrize("layout", ["LBNHC", "BLHNC"])
    def test_bound_views_round_trip(self, layout):
        specs = {"mla.0": _mla(512), "mla.1": _mla(512), "idx.0": _mla(128)}
        groups = [_uniform_group(specs)]
        set_kv_cache_layout(layout)
        config = get_kv_cache_config_from_groups(_mock_vllm_config(), groups, MEMORY)
        views = _bind(config)
        for i, name in enumerate(specs):
            views[name].fill_(i + 1)
        for i, name in enumerate(specs):
            assert (views[name].to(torch.int32) == i + 1).all(), name
            assert views[name].shape[0] == config.num_blocks
