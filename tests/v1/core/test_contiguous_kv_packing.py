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
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    _get_kv_cache_bytes_per_block,
    _get_packed_kv_cache_groups,
    _pool_bytes_per_block,
    generate_scheduler_kv_cache_config,
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
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
    iter_layer_specs,
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
) -> dict[str, KVCacheSpec]:
    specs: dict[str, KVCacheSpec] = {}
    for layer_index in range(num_mamba):
        specs[f"model.layers.{layer_index}.linear_attn"] = MambaSpec(
            block_size=16,
            shapes=((32,),),
            dtypes=(torch.bfloat16,),
        )
    specs["model.layers.0.ple"] = MambaSpec(
        block_size=16,
        shapes=((24,),),
        dtypes=(torch.bfloat16,),
        tp_replicated=True,
    )
    for layer_index in range(num_tuples):
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


class TestCSALinearGrouping:
    """A CSA + linear-attention model (sparse attention with a compressor ring,
    plus sharded GDN and one TP-replicated PLE state) goes through the generic
    packed-group path; no model-specific branch is involved."""

    @staticmethod
    def _mamba_groups(groups):
        """(group, per-layer mamba spec) for every group holding mamba state."""
        out = []
        for group in groups:
            spec = next(iter(iter_layer_specs(group.kv_cache_spec)))
            if isinstance(spec, MambaSpec):
                out.append((group, spec))
        return out

    def test_replicated_state_gets_its_own_group(self):
        """The PLE state is TP-replicated and a different size, so it must not
        share a manager group with the sharded GDN states (the NIXL worker
        requires exactly one single-layer replicated group)."""
        groups = get_kv_cache_groups(_shared_layout_config(), _make_csa_linear_specs())

        mamba_groups = self._mamba_groups(groups)
        replicated = [g for g, spec in mamba_groups if spec.tp_replicated]
        assert len(replicated) == 1
        assert replicated[0].layer_names == ["model.layers.0.ple"]
        sharded = [g for g, spec in mamba_groups if not spec.tp_replicated]
        assert sharded, "GDN states must keep their own groups"
        assert sorted(n for g in sharded for n in g.layer_names) == sorted(
            f"model.layers.{i}.linear_attn" for i in range(7)
        )

    def test_roles_land_in_separate_groups(self):
        groups = get_kv_cache_groups(_shared_layout_config(), _make_csa_linear_specs())

        owner = next(g for g in groups if _main_kv_name(0) in g.layer_names)
        assert sorted(owner.layer_names) == sorted(
            [
                *(_main_kv_name(i) for i in range(NUM_CACHE_TUPLES)),
                *(_compressed_name(i) for i in range(NUM_CACHE_TUPLES)),
            ]
        )
        assert owner.kv_cache_spec.prefix_cacheable

        scratch = next(g for g in groups if _compressor_state_name(0) in g.layer_names)
        assert scratch.layer_names == [
            _compressor_state_name(i) for i in range(NUM_CACHE_TUPLES)
        ]
        assert not scratch.kv_cache_spec.prefix_cacheable

    def test_every_group_fits_one_packed_block(self):
        config = _shared_layout_config()
        groups = get_kv_cache_groups(config, _make_csa_linear_specs())
        bytes_per_block = _get_kv_cache_bytes_per_block(groups)

        pages = _pages(groups)
        for group in groups:
            assert sum(pages[n] for n in group.layer_names) <= bytes_per_block

        kv_cache_config = get_kv_cache_config_from_groups(
            config, groups, available_memory=bytes_per_block * 32
        )
        assert kv_cache_config.num_blocks == 32
        # Groups overlay from byte 0 of each block, so a layer never addresses
        # past the block it belongs to.
        for tensor in kv_cache_config.kv_cache_tensors:
            assert tensor.block_stride == bytes_per_block
            assert tensor.offset < bytes_per_block

    def test_scratch_group_survives_computed_block_truncation(self):
        """The scratch group contributes no computed blocks, so truncating a
        lookup result must skip it: its block size is the ring capacity, which
        neither divides the hit length nor bounds the (empty) block list."""
        config = _shared_layout_config()
        config.cache_config.enable_prefix_caching = True
        groups = get_kv_cache_groups(config, _make_csa_linear_specs())
        kv_cache_config = get_kv_cache_config_from_groups(
            config, groups, available_memory=BYTES_PER_BLOCK * 64
        )
        scheduler_config = generate_scheduler_kv_cache_config([kv_cache_config])
        manager = KVCacheManager(
            scheduler_config,
            max_model_len=8192,
            enable_caching=True,
            hash_block_size=16,
            scheduler_block_size=16,
        )
        blocks = manager.create_kv_cache_blocks(
            tuple(
                manager.block_pool.get_new_blocks(3)
                if group.kv_cache_spec.prefix_cacheable
                else []
                for group in scheduler_config.kv_cache_groups
            )
        )
        truncated = manager.truncate_computed_blocks(blocks, 48)
        scratch_index = next(
            i
            for i, group in enumerate(scheduler_config.kv_cache_groups)
            if not group.kv_cache_spec.prefix_cacheable
        )
        assert truncated.blocks[scratch_index] == []

    def test_prefix_hits_respect_compression_alignment(self):
        config = _shared_layout_config()
        config.cache_config.enable_prefix_caching = True
        config.cache_config.prefix_match_unit = 16
        config.cache_config.mamba_cache_mode = "align"
        specs = {
            name: replace(spec, mamba_cache_mode="align")
            if isinstance(spec, MambaSpec)
            else spec
            for name, spec in _make_csa_linear_specs(num_tuples=1).items()
        }
        groups = get_kv_cache_groups(config, specs)
        kv_cache_config = get_kv_cache_config_from_groups(
            config, groups, available_memory=8 * MAIN_KV_PAGE_BYTES
        )
        assert resolve_kv_cache_block_sizes(kv_cache_config, config) == (16, 16)

        # 2 tokens is not a multiple of the compression ratio 4: a prefix hit
        # could land inside a partially filled compressed state.
        config.cache_config.prefix_match_unit = 2
        with pytest.raises(ValueError, match="per-state compression"):
            resolve_kv_cache_block_sizes(kv_cache_config, config)

    def test_scratch_ring_does_not_drag_hash_granularity(self):
        config = _shared_layout_config()
        config.cache_config.enable_prefix_caching = True
        config.cache_config.mamba_cache_mode = "align"
        specs = {
            name: replace(spec, mamba_cache_mode="align")
            if isinstance(spec, MambaSpec)
            else spec
            for name, spec in _make_csa_linear_specs(num_tuples=1).items()
        }
        groups = get_kv_cache_groups(config, specs)
        kv_cache_config = get_kv_cache_config_from_groups(
            config, groups, available_memory=8 * MAIN_KV_PAGE_BYTES
        )
        # The hash granularity is the GCD over prefix-cacheable groups only;
        # the 4-token scratch ring is excluded (it would drag it to 4).
        assert resolve_kv_cache_block_sizes(kv_cache_config, config) == (16, 16)

    def test_compressed_attention_hashes_can_be_finer_than_cache_hits(self):
        config = _shared_layout_config()
        config.cache_config.enable_prefix_caching = True
        specs = {
            "compressed.4": MLAAttentionSpec(
                block_size=256,
                num_kv_heads=1,
                head_size=16,
                dtype=torch.bfloat16,
                tokens_per_state=4,
            ),
            "compressed.128": MLAAttentionSpec(
                block_size=256,
                num_kv_heads=1,
                head_size=16,
                dtype=torch.bfloat16,
                tokens_per_state=128,
            ),
            "compressor_state.4": SlidingWindowMLASpec(
                block_size=4,
                num_kv_heads=1,
                head_size=8,
                head_size_v=0,
                dtype=torch.bfloat16,
                sliding_window=8,
            ),
        }
        groups = get_kv_cache_groups(config, specs)
        kv_cache_config = get_kv_cache_config_from_groups(
            config, groups, available_memory=8 * MAIN_KV_PAGE_BYTES
        )

        # Hashes are computed every 4 tokens, but without an align-mode Mamba
        # group cache hits remain on the 256-token scheduler boundary.
        assert resolve_kv_cache_block_sizes(kv_cache_config, config) == (256, 4)

    @pytest.mark.parametrize("wide", ["unbalanced_attention", "unsplittable_state"])
    def test_mamba_split_measures_the_block_the_other_groups_already_force(self, wide):
        """Whatever fixes the block stride -- a bucket with unequal layer counts
        per page size, which is emitted whole, or a single-layer state that
        cannot be split at all -- the mamba layers must be sized against it.
        Sizing them against a narrower bucket splits them past what the block
        already fits, spending a pool block per extra group for no saving."""
        config = _mock_vllm_config("BLNHC")
        config.speculative_config = None
        specs = {}
        if wide == "unbalanced_attention":
            for i in range(3):
                specs[f"wide.{i}"] = _mla(1024)
            specs["wide.odd"] = _mla(800)
        else:
            specs["wide.state"] = MambaSpec(
                block_size=16,
                shapes=((51_200,),),
                dtypes=(torch.bfloat16,),
                tp_replicated=True,
            )
        # Mixed and balanced, but far narrower than the bucket above.
        for i in range(10):
            specs[f"narrow.a.{i}"] = FullAttentionSpec(
                block_size=16, num_kv_heads=1, head_size=8, dtype=torch.uint8
            )
            specs[f"narrow.b.{i}"] = FullAttentionSpec(
                block_size=16, num_kv_heads=1, head_size=16, dtype=torch.uint8
            )
        for i in range(100):
            specs[f"gdn.{i}"] = MambaSpec(
                block_size=16, shapes=((512,),), dtypes=(torch.bfloat16,)
            )

        groups = _get_packed_kv_cache_groups(config, specs)
        gdn = [g for g in groups if g.layer_names[0].startswith("gdn.")]

        assert _get_kv_cache_bytes_per_block(groups) == sum(
            specs[name].page_size_bytes for name in specs if name.startswith("wide.")
        )
        # That block holds every GDN state at once, so the repeat pattern alone
        # decides the split; the cap must not add groups on top of it.
        assert len(gdn) == 10


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
