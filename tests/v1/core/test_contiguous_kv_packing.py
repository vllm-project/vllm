# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for contiguous KV cache packing."""

from dataclasses import replace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.v1.core.kv_cache_utils import (
    _get_kv_cache_config_packed,
    _get_kv_cache_groups_uniform_groups,
    _max_memory_usage_bytes_from_groups,
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.gpu.attn_utils import _reshape_mamba_kv_cache


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


def _packing_by_layer(
    tensors: list[KVCacheTensor],
) -> dict[str, tuple[int, int]]:
    return {
        layer_name: (tensor.offset, tensor.block_stride)
        for tensor in tensors
        for layer_name in tensor.shared_by
    }


def _make_views(
    groups: list[KVCacheGroupSpec],
    num_blocks: int,
    tensors: list[KVCacheTensor],
) -> dict[str, torch.Tensor]:
    page_sizes = _page_sizes_by_layer(groups)
    packing = _packing_by_layer(tensors)
    backing = torch.zeros(tensors[0].size, dtype=torch.uint8)
    return {
        layer_name: torch.as_strided(
            backing,
            size=(num_blocks, page_size),
            stride=(packing[layer_name][1], 1),
            storage_offset=packing[layer_name][0],
        )
        for layer_name, page_size in page_sizes.items()
    }


def _make_page_group(prefix: str, page_sizes: list[int]) -> KVCacheGroupSpec:
    specs = {
        f"{prefix}.{i}": MagicMock(page_size_bytes=page_size)
        for i, page_size in enumerate(page_sizes)
    }
    return KVCacheGroupSpec(
        layer_names=list(specs),
        kv_cache_spec=UniformTypeKVCacheSpecs(block_size=256, kv_cache_specs=specs),
    )


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
            indexes_kv_by_block_stride=True,
        )
        specs[_compressed_name(layer_index)] = MLAAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=16,
            dtype=torch.bfloat16,
            compress_ratio=4,
            indexes_kv_by_block_stride=True,
        )
        specs[_compressor_state_name(layer_index)] = CircularBufferSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=8,
            head_size_v=0,
            dtype=torch.bfloat16,
            indexes_kv_by_block_stride=True,
        )
    return specs


def _shared_layout_config():
    config = _mock_vllm_config()
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


class TestInterleavedPacking:
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
            group.kv_cache_spec.page_size_bytes == MAIN_KV_PAGE_BYTES
            for group in mamba
            for name in group.layer_names
        )

    def test_shared_tensors_match_expected_memory_and_ownership(self):
        config = _shared_layout_config()
        groups = get_kv_cache_groups(config, _make_csa_linear_specs())

        assert _max_memory_usage_bytes_from_groups(config, groups) == (
            BYTES_PER_BLOCK * 6
        )
        kv_cache_config = get_kv_cache_config_from_groups(
            config,
            groups,
            available_memory=BYTES_PER_BLOCK * 32,
        )

        assert kv_cache_config.num_blocks == 32
        assert len(kv_cache_config.kv_cache_tensors) == 2 * NUM_CACHE_TUPLES
        assert sum(t.size for t in kv_cache_config.kv_cache_tensors) == (
            BYTES_PER_BLOCK * 32
        )
        assert all(
            tensor.block_stride == 0 and tensor.offset == 0
            for tensor in kv_cache_config.kv_cache_tensors
        )

        tensors_by_owner = {
            tensor.shared_by[0]: tensor for tensor in kv_cache_config.kv_cache_tensors
        }
        mamba_owners = {
            name
            for i in range(NUM_CACHE_TUPLES)
            for name in tensors_by_owner[_main_kv_name(i)].shared_by[1:]
        }
        assert mamba_owners == {
            *(f"model.layers.{i}.linear_attn" for i in range(7)),
            "model.layers.0.ple",
        }
        for i in range(NUM_CACHE_TUPLES):
            assert tensors_by_owner[_compressed_name(i)].shared_by == [
                _compressed_name(i),
                _compressor_state_name(i),
            ]

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

        groups = get_kv_cache_groups(_shared_layout_config(), specs)
        kv_cache_config = get_kv_cache_config_from_groups(
            _shared_layout_config(),
            groups,
            available_memory=2 * MAIN_KV_PAGE_BYTES * 8,
        )

        assert [tensor.shared_by for tensor in kv_cache_config.kv_cache_tensors] == [
            [
                _main_kv_name(0),
                *(f"model.layers.{i}.linear_attn" for i in range(7)),
                "model.layers.0.ple",
            ],
            [_compressed_name(0), _compressor_state_name(0)],
        ]

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
            state, shapes=((MAIN_KV_PAGE_BYTES + 1,),), dtypes=(torch.uint8,)
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

    def test_prefix_hits_are_aligned_and_ignore_the_private_compressor_state(self):
        config = _shared_layout_config()
        config.cache_config.enable_prefix_caching = True
        config.cache_config.prefix_match_unit = 16
        groups = get_kv_cache_groups(config, _make_csa_linear_specs(num_tuples=1))
        kv_cache_config = get_kv_cache_config_from_groups(
            config, groups, available_memory=2 * MAIN_KV_PAGE_BYTES
        )

        assert resolve_kv_cache_block_sizes(kv_cache_config, config) == (16, 16)

        config.cache_config.prefix_match_unit = 2
        with pytest.raises(ValueError, match="compression ratio"):
            get_kv_cache_groups(config, _make_csa_linear_specs(num_tuples=1))

    def test_packed_mamba_views_use_layer_offsets_and_block_stride(self):
        backing = torch.zeros(3 * 64, dtype=torch.uint8)
        left = _reshape_mamba_kv_cache(backing, 16, 3, (0, 64))
        right = _reshape_mamba_kv_cache(backing, 16, 3, (16, 64))

        for block_id in range(3):
            left[block_id].fill_(block_id + 1)
            right[block_id].fill_(block_id + 11)

        assert left.stride(0) == right.stride(0) == 64
        assert left.storage_offset() == 0
        assert right.storage_offset() == 16
        for block_id in range(3):
            assert (left[block_id] == block_id + 1).all()
            assert (right[block_id] == block_id + 11).all()

        contiguous = _reshape_mamba_kv_cache(backing, 16, 3, None)
        assert contiguous.shape == (3, 1, 1, 16)
        assert contiguous.stride(0) == 16

    def test_compact_cache_overlays_fp32_state_group(self):
        full_specs = {}
        state_specs = {}
        for i in range(2):
            full_specs[f"mla.{i}"] = MLAAttentionSpec(
                block_size=256,
                num_kv_heads=1,
                head_size=512,
                dtype=torch.uint8,
                page_size_padded=32768,
                indexes_kv_by_block_stride=True,
                compress_ratio=4,
            )
            full_specs[f"indexer.{i}"] = MLAAttentionSpec(
                block_size=256,
                num_kv_heads=1,
                head_size=68,
                dtype=torch.uint8,
                page_size_padded=4608,
                compress_ratio=4,
            )
            state_specs[f"mla_state.{i}"] = SlidingWindowMLASpec(
                block_size=4,
                num_kv_heads=1,
                head_size=2048,
                dtype=torch.float32,
                sliding_window=8,
                indexes_kv_by_block_stride=True,
            )
            state_specs[f"indexer_state.{i}"] = SlidingWindowMLASpec(
                block_size=4,
                num_kv_heads=1,
                head_size=512,
                dtype=torch.float32,
                sliding_window=8,
                indexes_kv_by_block_stride=True,
            )

        grouped_specs = [
            UniformTypeKVCacheSpecs(block_size=256, kv_cache_specs=full_specs),
            UniformTypeKVCacheSpecs(block_size=4, kv_cache_specs=state_specs),
        ]
        groups = _get_kv_cache_groups_uniform_groups(grouped_specs)

        assert len(groups) == 2
        assert {full_specs[f"indexer.{i}"].page_size_bytes for i in range(2)} == {4608}
        assert {full_specs[f"indexer.{i}"].real_page_size_bytes for i in range(2)} == {
            4352
        }
        assert {
            state_specs[f"indexer_state.{i}"].page_size_bytes for i in range(2)
        } == {8192}

        full_group_bytes = 2 * (32768 + 4608)
        state_group_bytes = 2 * (32768 + 8192)
        bytes_per_block = max(full_group_bytes, state_group_bytes)
        num_blocks, tensors = _get_kv_cache_config_packed(
            _mock_vllm_config(), groups, bytes_per_block * 32
        )
        assert num_blocks == 32
        assert {tensor.block_stride for tensor in tensors} == {bytes_per_block}

        packing = _packing_by_layer(tensors)
        assert packing["mla.0"][0] == packing["mla_state.0"][0] == 0
        assert packing["indexer.0"][0] == 32768
        assert packing["indexer_state.0"][0] == 32768

    def test_deepseek_v4_pro_stride(self):
        groups = [
            _make_page_group("full", [32768, 4608] * 30 + [1024] * 31),
            _make_page_group("c4_state", [32768, 8192] * 30),
            _make_page_group("c128_state", [32768] * 31),
            _make_page_group("swa.0", [32768] * 31),
            _make_page_group("swa.1", [32768] * 30),
        ]
        expected_stride = 1_228_800

        num_blocks, tensors = _get_kv_cache_config_packed(
            _mock_vllm_config(), groups, expected_stride * 32
        )

        assert num_blocks == 32
        assert {tensor.block_stride for tensor in tensors} == {expected_stride}
        assert {tensor.size for tensor in tensors} == {expected_stride * 32}

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
        groups = _make_groups(n_c4=3, n_c128=2, n_swa=5)
        _, tensors = _get_kv_cache_config_packed(
            _mock_vllm_config(), groups, 100 * 1024 * 1024
        )
        page_sizes = _page_sizes_by_layer(groups)
        packing = _packing_by_layer(tensors)
        for layer_name, page_size in page_sizes.items():
            offset, block_stride = packing[layer_name]
            assert offset + page_size <= block_stride

    def test_layouts_are_disjoint_within_each_group(self):
        groups = _make_groups(n_c4=3, n_c128=2, n_swa=5)
        _, tensors = _get_kv_cache_config_packed(
            _mock_vllm_config(), groups, 100 * 1024 * 1024
        )
        page_sizes = _page_sizes_by_layer(groups)
        packing = _packing_by_layer(tensors)

        for group in groups:
            ranges = sorted(
                (packing[name][0], packing[name][0] + page_sizes[name])
                for name in group.layer_names
            )
            assert all(left[1] <= right[0] for left, right in zip(ranges, ranges[1:]))

    def test_all_layers_accounted_for(self):
        n_c4, n_c128, n_swa = 5, 4, 7
        _, tensors = _run(n_c4=n_c4, n_c128=n_c128, n_swa=n_swa)
        all_names = set()
        for t in tensors:
            all_names.update(t.shared_by)
        expected = n_c4 * 2 + n_c128 + n_swa
        assert len(all_names) == expected

    def test_group_owned_blocks_do_not_alias(self):
        groups = _make_groups(n_c4=3, n_c128=2, n_swa=5)
        num_blocks, tensors = _get_kv_cache_config_packed(
            _mock_vllm_config(), groups, 100 * 1024 * 1024
        )
        views = _make_views(groups, num_blocks, tensors)

        expected = {}
        value = 1
        for block_id, group in enumerate(groups):
            for layer_name in group.layer_names:
                views[layer_name][block_id].fill_(value)
                expected[layer_name] = (block_id, value)
                value += 1

        for layer_name, (block_id, value) in expected.items():
            assert (views[layer_name][block_id] == value).all()

        # Once the first group releases its block, another group may reuse it.
        for layer_name in groups[1].layer_names:
            views[layer_name][0].fill_(255)
        for layer_name in groups[1].layer_names:
            assert (views[layer_name][0] == 255).all()

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
