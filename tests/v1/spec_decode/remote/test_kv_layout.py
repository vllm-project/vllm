# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise standalone KV descriptors through the real CPU storage allocator."""

from dataclasses import replace

import pytest
import torch

from vllm.v1.core.single_type_kv_cache_manager import (
    ChunkedLocalAttentionManager,
    CrossAttentionManager,
    FullAttentionManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    HiddenStateCacheSpec,
    HiSparseHotPageSpec,
    KVQuantMode,
    MLAAttentionSpec,
    SparseCacheRole,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.spec_decode.remote.protocol import SequenceKey
from vllm.v1.spec_decode.remote.runtime.kv import DraftBlockPool
from vllm.v1.spec_decode.remote.runtime.kv_layout import (
    make_standalone_kv_config,
    select_standalone_kv_layer_config,
)
from vllm.v1.spec_decode.remote.runtime.step import DraftStepPlanner, DraftStepRow
from vllm.v1.worker.utils import allocate_kv_cache


def _spec(kind):
    kwargs = dict(
        block_size=4,
        num_kv_heads=2,
        head_size=3,
        dtype=torch.float32 if kind == "full" else torch.bfloat16,
    )
    if kind == "mla":
        return MLAAttentionSpec(**(kwargs | {"num_kv_heads": 1, "head_size": 5}))
    if kind == "swa":
        return AttentionSpec(**kwargs, head_size_v=4)
    return AttentionSpec(**kwargs)


def _layer_config(kind):
    spec = _spec(kind)
    if kind == "swa":
        return SlidingWindowManager.layer_config(spec, sliding_window=5)
    return FullAttentionManager.layer_config(spec)


@pytest.mark.parametrize("kind", ["full", "mla", "swa"])
@pytest.mark.parametrize("layout_name", ["LBHNC", "LBNHC", "LHBNC"])
def test_dense_own_kv_layers_and_blocks_never_alias(kind, layout_name):
    layer_config = _layer_config(kind)
    spec = layer_config.spec
    names = ["draft.0", "draft.1", "draft.2"]
    assert (
        select_standalone_kv_layer_config(dict.fromkeys(names, layer_config))
        is layer_config
    )
    page_group_bytes = len(names) * spec.page_size_bytes
    budget = page_group_bytes * 6 - 1
    layout = KVCacheLayout[layout_name]
    config = make_standalone_kv_config(layer_config, names, budget, layout)
    caches = allocate_kv_cache(config, torch.device("cpu"), layout)
    assert config.num_blocks == 5
    group = config.kv_cache_groups[0]
    assert all(value is spec for value in group.kv_cache_specs.values())
    assert group.manager_config == layer_config.manager_config
    assert group.is_eagle_group
    for layer, cache in enumerate(caches.values()):
        assert cache.shape == (
            5,
            spec.num_kv_heads,
            4,
            spec.head_size + spec.head_size_v,
        )
        assert cache.untyped_storage().nbytes() == page_group_bytes * 5
        for block in range(1, 5):
            cache[block].fill_(10 * layer + block)
    for layer, cache in enumerate(caches.values()):
        assert not cache[0].count_nonzero().item()
        for block in range(1, 5):
            assert torch.all(cache[block] == 10 * layer + block)
    assert len({cache.untyped_storage().data_ptr() for cache in caches.values()}) == 1
    assert page_group_bytes * 5 <= budget


@pytest.mark.parametrize("kind", ["full", "mla", "swa"])
def test_each_layer_budget_reserves_a_null_block(kind):
    layer_config = _layer_config(kind)
    spec = layer_config.spec
    names = ["a"] if kind == "full" else ["a", "b"]
    page_group_bytes = len(names) * spec.page_size_bytes
    with pytest.raises(ValueError, match="null block plus one block"):
        make_standalone_kv_config(
            layer_config, names, 2 * page_group_bytes - 1, KVCacheLayout.LBNHC
        )
    config = make_standalone_kv_config(
        layer_config, names, 2 * page_group_bytes, KVCacheLayout.LBNHC
    )
    pool = DraftBlockPool(config.num_blocks, spec.block_size)
    assert pool.ensure(0, spec.block_size) == [1]
    assert pool.num_free_blocks == 0


@pytest.mark.parametrize("layout_name", ["LBHNC", "LBNHC", "LHBNC"])
def test_mla_kernel_block_splitting_preserves_latent_vectors(layout_name):
    layer_config = _layer_config("mla")
    spec = layer_config.spec
    layout = KVCacheLayout[layout_name]
    config = make_standalone_kv_config(
        layer_config, ["a", "b"], 6 * spec.page_size_bytes, layout
    )
    caches = allocate_kv_cache(config, torch.device("cpu"), layout, [2])
    for layer, cache in enumerate(caches.values()):
        assert cache.shape == (6, 1, 2, 5)
        for kernel_block in range(2, 6):
            cache[kernel_block].fill_(10 * layer + kernel_block)
    for layer, cache in enumerate(caches.values()):
        assert not cache[:2].count_nonzero().item()
        for kernel_block in range(2, 6):
            assert torch.all(cache[kernel_block] == 10 * layer + kernel_block)


def test_sliding_window_planner_keeps_full_context_for_rejection():
    layer_config = _layer_config("swa")
    spec = layer_config.spec
    config = make_standalone_kv_config(
        layer_config, ["draft"], 8 * spec.page_size_bytes, KVCacheLayout.LBNHC
    )
    pool = DraftBlockPool(config.num_blocks, spec.block_size)
    planner = DraftStepPlanner(
        pool, max_num_reqs=1, max_model_len=24, num_speculative_tokens=3
    )
    key = SequenceKey("verifier", 1, 1)
    assert planner.open(key)
    plan, failures = planner.plan([DraftStepRow(key, 12, 1)])
    assert not failures
    planner.commit(plan)
    before = pool.blocks(planner.req_index(key))
    assert len(before) == 4  # Query plus two lookahead positions, beyond the window.
    plan, failures = planner.plan([DraftStepRow(key, 3, 1, num_rejected=2)])
    assert not failures
    planner.commit(plan)
    assert planner.num_computed(key) == 13
    assert pool.blocks(planner.req_index(key)) == before
    planner.close(key)
    assert pool.num_free_blocks == config.num_blocks - 1


@pytest.mark.parametrize("kind", ["full", "mla", "swa"])
def test_mixed_or_differently_sized_layers_are_rejected(kind):
    layer_config = _layer_config(kind)
    with pytest.raises(ValueError, match="one uniform KV group"):
        select_standalone_kv_layer_config(
            {
                "a": layer_config,
                "b": replace(
                    layer_config, spec=replace(layer_config.spec, head_size=7)
                ),
            }
        )
    with pytest.raises(ValueError, match="one uniform KV group"):
        select_standalone_kv_layer_config(
            {"a": layer_config, "b": _layer_config("swa" if kind != "swa" else "mla")}
        )
    with pytest.raises(ValueError, match="one uniform KV group"):
        select_standalone_kv_layer_config(
            {
                "a": layer_config,
                "b": replace(
                    layer_config,
                    manager_config=SlidingWindowManager.config(sliding_window=8),
                ),
            }
        )


@pytest.mark.parametrize(
    "spec",
    [
        replace(_spec("mla"), tokens_per_state=2),
        replace(_spec("mla"), state_content_bytes=32),
        replace(_spec("swa"), page_size_padded=256),
        replace(_spec("full"), num_head_slots=3),
        replace(_spec("full"), dtype=torch.uint8),
        replace(_spec("full"), kv_quant_mode=KVQuantMode.FP8_PER_TENSOR),
    ],
)
def test_compressed_packed_padded_and_quantized_pages_are_rejected(spec):
    with pytest.raises(ValueError, match="dense unquantized KV pages"):
        select_standalone_kv_layer_config(
            {"draft": FullAttentionManager.layer_config(spec)}
        )


@pytest.mark.parametrize(
    "changes,manager_config",
    [
        ({"model_version": "deepseek_v4"}, None),
        ({"is_index_group_leader": True}, None),
        ({"cache_role": SparseCacheRole.INDEXER}, None),
        ({}, CrossAttentionManager.config()),
        ({}, SlidingWindowManager.config(sliding_window=8)),
        ({"head_size_v": 3}, None),
        ({"num_kv_heads": 2}, None),
        ({"topk_buffer_width": 8}, None),
        ({"alignment": 1}, None),
    ],
)
def test_sparse_and_nonstandard_mla_require_a_separate_contract(
    changes, manager_config
):
    layer_config = _layer_config("mla")
    layer_config = replace(
        layer_config,
        spec=replace(layer_config.spec, **changes),
        manager_config=manager_config or layer_config.manager_config,
    )
    with pytest.raises(ValueError, match="dense MLA without an indexer"):
        select_standalone_kv_layer_config({"draft": layer_config})


@pytest.mark.parametrize(
    "spec",
    [
        HiddenStateCacheSpec(
            block_size=4, num_kv_heads=1, head_size=5, dtype=torch.bfloat16
        ),
        HiSparseHotPageSpec(block_size=4, page_size=64),
    ],
)
def test_other_cache_spec_subclasses_are_not_implicitly_admitted(spec):
    with pytest.raises(ValueError, match="unsupported"):
        select_standalone_kv_layer_config(
            {"draft": FullAttentionManager.layer_config(spec)}
        )


@pytest.mark.parametrize(
    "manager_config",
    [
        CrossAttentionManager.config(),
        ChunkedLocalAttentionManager.config(attention_chunk_size=8),
        SlidingWindowManager.config(sliding_window=8, bounded_replay=True),
    ],
)
def test_noncausal_and_chunked_attention_are_rejected(manager_config):
    with pytest.raises(ValueError, match="causal token-addressed KV"):
        select_standalone_kv_layer_config(
            {"draft": replace(_layer_config("full"), manager_config=manager_config)}
        )


def test_nonempty_unique_layer_names_and_hardware_alignment_are_required():
    layer_config = _layer_config("full")
    for names in ([], ["a", "a"]):
        with pytest.raises(ValueError, match="nonempty and unique"):
            make_standalone_kv_config(layer_config, names, 4096, KVCacheLayout.LBNHC)
    with pytest.raises(ValueError, match="block alignment"):
        make_standalone_kv_config(
            replace(
                layer_config,
                spec=replace(layer_config.spec, block_stride_alignment=1024),
            ),
            ["a"],
            4096,
            KVCacheLayout.LBNHC,
        )


@pytest.mark.parametrize("kind", ["full", "mla", "swa"])
@pytest.mark.parametrize("layout_name", ["BLHNC", "BLNHC", "BHLNC"])
def test_all_cache_types_require_layer_compact_layout(kind, layout_name):
    with pytest.raises(ValueError, match="layer-compact"):
        make_standalone_kv_config(
            _layer_config(kind), ["draft"], 4096, KVCacheLayout[layout_name]
        )
