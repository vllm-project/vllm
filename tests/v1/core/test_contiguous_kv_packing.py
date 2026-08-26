# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for contiguous KV cache packing.

Every cache group packs its layers densely into one block; groups overlay each other
(a block ID is owned by one group at a time), so the packed block stride is the
largest group's packing. The layout decides whether the layer dim sits outside the
block dim (a contiguous region per layer) or inside it (all layers' pages within each
block); the allocation is the same either way.
"""

from dataclasses import replace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.config import CacheConfig
from vllm.v1.core.kv_cache_utils import (
    _get_kv_cache_bytes_per_block,
    _max_memory_usage_bytes_from_groups,
    _pool_bytes_per_block,
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheGroupSpec,
    KVCacheLayout,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
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


MAIN_KV_PAGE_BYTES = 2_048
COMPRESSED_PAGE_BYTES = 128
NUM_CACHE_TUPLES = 3
BYTES_PER_BLOCK = NUM_CACHE_TUPLES * (MAIN_KV_PAGE_BYTES + COMPRESSED_PAGE_BYTES)


def _main_kv_name(layer_index: int) -> str:
    return f"model.layers.{layer_index}.self_attn"


def _compressed_name(layer_index: int) -> str:
    return f"model.layers.{layer_index}.self_attn.indexer.compressed_key_cache"


def _compressor_state_name(layer_index: int) -> str:
    return f"model.layers.{layer_index}.self_attn.indexer.raw_key_cache"


def _make_csa_linear_specs(
    num_mamba: int = 7,
    num_tuples: int = NUM_CACHE_TUPLES,
    *,
    main_kv_indices: list[int] | None = None,
    mamba_indices: list[int] | None = None,
    include_replicated: bool = True,
) -> dict[str, KVCacheSpec]:
    main_kv_indices = main_kv_indices or list(range(num_tuples))
    mamba_indices = mamba_indices or list(range(num_mamba))
    assert len(main_kv_indices) == num_tuples
    assert len(mamba_indices) == num_mamba

    specs: dict[str, KVCacheSpec] = {}
    for layer_index in mamba_indices:
        specs[f"model.layers.{layer_index}.linear_attn"] = MambaSpec(
            block_size=16,
            shapes=((32,),),
            dtypes=(torch.bfloat16,),
        )
    if include_replicated:
        specs["model.layers.0.ple"] = MambaSpec(
            block_size=16,
            shapes=((24,),),
            dtypes=(torch.bfloat16,),
            tp_replicated=True,
        )
    for layer_index in main_kv_indices:
        specs[_main_kv_name(layer_index)] = FullAttentionSpec(
            block_size=16,
            num_kv_heads=2,
            head_size=16,
            head_size_v=16,
            dtype=torch.bfloat16,
        )
        specs[_compressed_name(layer_index)] = MLAAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=16,
            dtype=torch.bfloat16,
            tokens_per_state=4,
        )
        specs[_compressor_state_name(layer_index)] = CircularBufferSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=8,
            head_size_v=0,
            dtype=torch.bfloat16,
        )
    return specs


def _shared_layout_config():
    config = _mock_vllm_config("BLNHC")
    config.scheduler_config.disable_hybrid_kv_cache_manager = False
    config.speculative_config = None
    config.model_config.max_model_len = 16
    config.model_config.get_total_num_hidden_layers.return_value = 64
    config.model_config.get_total_num_kv_heads.return_value = 2
    config.model_config.get_num_kv_heads.return_value = 2
    config.parallel_config.pipeline_parallel_size = 1
    config.parallel_config.decode_context_parallel_size = 1
    config.cache_config.block_size = 16
    config.cache_config.enable_prefix_caching = False
    config.cache_config.prefix_match_unit = None
    config.cache_config.mamba_cache_mode = "none"
    return config


def _placements_by_layer(kv_cache_config) -> dict[str, tuple[int, int]]:
    return {
        layer_name: (
            tensor.offset + index * tensor.layer_stride,
            tensor.block_stride,
        )
        for tensor in kv_cache_config.kv_cache_tensors
        for index, layer_name in enumerate(tensor.layers)
    }


class TestCSALinearPacking:
    def test_layer_types_form_compressed_sparse_compressor_state_and_mamba_groups(
        self,
    ):
        groups = get_kv_cache_groups(_shared_layout_config(), _make_csa_linear_specs())

        assert len(groups) == 6
        compressed_sparse, compressor_state, *mamba = groups
        assert compressed_sparse.layer_names == [
            name
            for i in range(NUM_CACHE_TUPLES)
            for name in (_main_kv_name(i), _compressed_name(i))
        ]
        assert compressor_state.layer_names == [
            _compressor_state_name(i) for i in range(NUM_CACHE_TUPLES)
        ]
        assert [len(group.layer_names) for group in mamba] == [3, 2, 2, 1]
        assert all(not group.kv_cache_spec.tp_replicated for group in mamba[:-1])
        assert mamba[-1].kv_cache_spec.tp_replicated
        assert compressed_sparse.kv_cache_spec.prefix_cacheable
        assert not compressor_state.kv_cache_spec.prefix_cacheable
        assert {
            compressor_state.kv_cache_spec.kv_cache_specs[name].page_size_bytes
            for name in compressor_state.layer_names
        } == {COMPRESSED_PAGE_BYTES}
        assert all(
            group.kv_cache_spec.page_size_bytes == MAIN_KV_PAGE_BYTES for group in mamba
        )

    def test_shared_tensors_match_expected_memory_and_ownership(self):
        config = _shared_layout_config()
        groups = get_kv_cache_groups(config, _make_csa_linear_specs())

        assert _get_kv_cache_bytes_per_block(groups) == BYTES_PER_BLOCK
        assert _max_memory_usage_bytes_from_groups(config, groups) == (
            BYTES_PER_BLOCK * 6
        )
        kv_cache_config = get_kv_cache_config_from_groups(
            config,
            groups,
            available_memory=BYTES_PER_BLOCK * 32,
        )

        assert kv_cache_config.num_blocks == 32
        assert {tensor.size for tensor in kv_cache_config.kv_cache_tensors} == {
            BYTES_PER_BLOCK * 32
        }
        assert all(
            tensor.block_stride == BYTES_PER_BLOCK
            for tensor in kv_cache_config.kv_cache_tensors
        )

        placements = _placements_by_layer(kv_cache_config)
        mamba_groups = groups[2:]
        for index in range(NUM_CACHE_TUPLES):
            main_placement = placements[_main_kv_name(index)]
            for group in mamba_groups:
                if index < len(group.layer_names):
                    assert placements[group.layer_names[index]] == main_placement
            assert (
                placements[_compressed_name(index)]
                == placements[_compressor_state_name(index)]
            )

        views = _bind(kv_cache_config, "BLNHC")
        for index in range(NUM_CACHE_TUPLES):
            main_view = views[_main_kv_name(index)]
            for group in mamba_groups:
                if index < len(group.layer_names):
                    assert views[group.layer_names[index]].data_ptr() == (
                        main_view.data_ptr()
                    )
            assert views[_compressed_name(index)].data_ptr() == (
                views[_compressor_state_name(index)].data_ptr()
            )

    def test_missing_and_duplicate_layer_roles_are_rejected(self):
        incomplete = _make_csa_linear_specs(num_tuples=2)
        del incomplete[_compressor_state_name(0)]
        with pytest.raises(ValueError, match="matching transformer-layer indices"):
            get_kv_cache_groups(_shared_layout_config(), incomplete)

        duplicate = _make_csa_linear_specs(num_tuples=1)
        duplicate["model.layers.0.other_raw_cache"] = duplicate[
            _compressor_state_name(0)
        ]
        with pytest.raises(ValueError, match="duplicate compressor-state"):
            get_kv_cache_groups(_shared_layout_config(), duplicate)

    def test_strict_spec_types_and_head_geometry_are_enforced(self):
        specs = _make_csa_linear_specs(num_tuples=2)
        specs["model.layers.63.local_attn"] = SlidingWindowSpec(
            block_size=16,
            num_kv_heads=2,
            head_size=16,
            dtype=torch.bfloat16,
            sliding_window=16,
        )
        with pytest.raises(ValueError, match="unsupported cache owners"):
            get_kv_cache_groups(_shared_layout_config(), specs)

        specs = _make_csa_linear_specs(num_tuples=1)
        main_kv = specs[_main_kv_name(0)]
        assert isinstance(main_kv, FullAttentionSpec)
        specs[_main_kv_name(0)] = replace(main_kv, num_kv_heads=1)
        with pytest.raises(ValueError, match="TP-local KV-head geometry"):
            get_kv_cache_groups(_shared_layout_config(), specs)

    def test_equal_page_sizes_do_not_change_layer_pairing(self):
        specs = _make_csa_linear_specs(num_tuples=1)
        compressed = specs[_compressed_name(0)]
        assert isinstance(compressed, MLAAttentionSpec)
        specs[_compressed_name(0)] = replace(compressed, head_size=256)

        config = _shared_layout_config()
        groups = get_kv_cache_groups(config, specs)
        kv_cache_config = get_kv_cache_config_from_groups(
            config,
            groups,
            available_memory=2 * MAIN_KV_PAGE_BYTES * 8,
        )
        placements = _placements_by_layer(kv_cache_config)

        assert placements[_main_kv_name(0)] == placements["model.layers.0.linear_attn"]
        assert placements[_compressed_name(0)] == placements[_compressor_state_name(0)]

    def test_compressor_state_and_mamba_pages_must_fit_their_owners(self):
        compressor_state_too_large = _make_csa_linear_specs(num_tuples=1)
        compressor_state = compressor_state_too_large[_compressor_state_name(0)]
        assert isinstance(compressor_state, CircularBufferSpec)
        compressor_state_too_large[_compressor_state_name(0)] = replace(
            compressor_state, head_size=100
        )
        with pytest.raises(ValueError, match="violate CSA geometry"):
            get_kv_cache_groups(_shared_layout_config(), compressor_state_too_large)

        state_too_large = _make_csa_linear_specs(num_mamba=1, num_tuples=1)
        state_name = "model.layers.0.linear_attn"
        state = state_too_large[state_name]
        assert isinstance(state, MambaSpec)
        state_too_large[state_name] = replace(
            state,
            shapes=((MAIN_KV_PAGE_BYTES + 1,),),
            dtypes=(torch.uint8,),
        )
        with pytest.raises(ValueError, match="main_kv tensor page"):
            get_kv_cache_groups(_shared_layout_config(), state_too_large)

    def test_pipeline_partitions_balance_mamba_owners(self):
        specs = _make_csa_linear_specs(
            num_mamba=6,
            num_tuples=4,
            main_kv_indices=[1, 5, 9, 13],
            mamba_indices=[0, 2, 3, 4, 6, 7],
        )
        config = _shared_layout_config()
        config.parallel_config.pipeline_parallel_size = 2
        config.model_config.get_total_num_hidden_layers.return_value = 16

        groups = get_kv_cache_groups(config, specs)

        assert len(groups) == 6
        assert [len(group.layer_names) for group in groups[2:-1]] == [2, 2, 2]
        assert groups[-1].layer_names == ["model.layers.0.ple"]

    def test_prefix_hits_are_aligned_and_ignore_private_compressor_state(self):
        config = _shared_layout_config()
        config.cache_config.enable_prefix_caching = True
        config.cache_config.prefix_match_unit = 16
        groups = get_kv_cache_groups(
            config,
            _make_csa_linear_specs(num_tuples=1),
        )
        kv_cache_config = get_kv_cache_config_from_groups(
            config,
            groups,
            available_memory=2 * MAIN_KV_PAGE_BYTES,
        )

        assert resolve_kv_cache_block_sizes(kv_cache_config, config) == (16, 16)

        config.cache_config.prefix_match_unit = 2
        with pytest.raises(ValueError, match="compression ratio"):
            get_kv_cache_groups(
                config,
                _make_csa_linear_specs(num_tuples=1),
            )

    def test_packed_mamba_views_use_owner_offsets_and_block_stride(self):
        config = _shared_layout_config()
        groups = get_kv_cache_groups(config, _make_csa_linear_specs())
        kv_cache_config = get_kv_cache_config_from_groups(
            config,
            groups,
            available_memory=BYTES_PER_BLOCK * 3,
        )
        views = _bind(kv_cache_config, "BLNHC")

        for index in range(NUM_CACHE_TUPLES):
            main_view = views[_main_kv_name(index)]
            assert main_view.stride(0) * main_view.element_size() == BYTES_PER_BLOCK
            for group in groups[2:]:
                if index < len(group.layer_names):
                    mamba_view = views[group.layer_names[index]]
                    assert mamba_view.data_ptr() == main_view.data_ptr()
                    assert (
                        mamba_view.stride(0) * mamba_view.element_size()
                        == BYTES_PER_BLOCK
                    )


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
