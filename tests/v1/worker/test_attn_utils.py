# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache view creation and sleep-mode wake-up regressions.

Guards that a page_size_padded spec strides the block dimension by the padded page
while keeping per-block content compact, so padding bytes at the end of each page are
never addressed by the logical view. Also covers packed Mamba views and selective
state initialization after wake-up.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention.utils import dense_kv_cache_views
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheLayout,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    compute_layout_strides,
    create_kv_cache_views,
)
from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
from vllm.v1.worker.gpu_model_runner import GPUModelRunner as GPUModelRunnerV1
from vllm.v1.worker.utils import (
    AttentionGroup,
    allocate_kv_cache,
    copy_kv_cache_blocks_inplace,
)


class FakeAttentionBackend:
    pass


def test_create_packed_mamba_kv_cache_views_preserves_block_layout():
    spec = MambaSpec(
        block_size=1,
        shapes=((2,), (4,)),
        dtypes=(torch.float32, torch.float16),
    )
    num_blocks = 3
    page_size = spec.page_size_bytes
    offset = 8
    block_stride = 32
    raw = torch.full((num_blocks * block_stride,), -1, dtype=torch.int8)
    kv_cache_tensor = KVCacheTensor(
        size=raw.numel(),
        layers=["mamba"],
        layer_stride=page_size,
        block_stride=block_stride,
        offset=offset,
    )

    (cache,) = create_kv_cache_views(
        raw,
        spec,
        num_blocks,
        KVCacheLayout.BLHNC,
        kv_cache_tensor,
    )

    assert cache.shape == (num_blocks, 1, 1, page_size)
    assert cache.stride() == (block_stride, page_size, page_size, 1)
    assert cache.data_ptr() == raw.data_ptr() + offset

    cache.fill_(0)
    raw_blocks = raw.view(num_blocks, block_stride)
    assert torch.count_nonzero(raw_blocks[:, offset : offset + page_size]) == 0
    assert torch.all(raw_blocks[:, :offset] == -1)
    assert torch.all(raw_blocks[:, offset + page_size :] == -1)

    layer = SimpleNamespace(
        get_state_shape=lambda: spec.shapes,
        get_state_dtype=lambda: spec.dtypes,
    )
    MambaBase.bind_kv_cache(layer, cache)
    for state, shape in zip(layer.kv_cache, spec.shapes):
        assert state.shape == (num_blocks, *shape)
        assert state.stride(0) * state.element_size() == block_stride


def _make_hybrid_attn_groups(mamba_layer_names):
    mamba_spec = MambaSpec(
        block_size=1,
        shapes=((2,), (3,)),
        dtypes=(torch.float32, torch.float32),
    )
    attention_spec = FullAttentionSpec(
        block_size=1,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    return [
        [
            AttentionGroup(
                backend=FakeAttentionBackend,
                layer_names=mamba_layer_names,
                kv_cache_spec=mamba_spec,
                kv_cache_group_id=0,
            ),
            AttentionGroup(
                backend=FakeAttentionBackend,
                layer_names=["attention"],
                kv_cache_spec=attention_spec,
                kv_cache_group_id=0,
            ),
        ]
    ]


def _run_post_kv_cache_wake_up(runner_cls, groups, forward_context):
    runner = SimpleNamespace(
        attn_groups=groups,
        compilation_config=SimpleNamespace(static_forward_context=forward_context),
    )
    if runner_cls is GPUModelRunnerV1:
        runner.init_fp8_kv_scales = lambda: None
    else:
        runner.block_tables = SimpleNamespace(
            init_block_table_layout_tensors=lambda: None
        )
    runner_cls.post_kv_cache_wake_up(runner)


@pytest.mark.parametrize(
    "runner_cls", [GPUModelRunnerV1, GPUModelRunnerV2], ids=["mrv1", "mrv2"]
)
def test_post_kv_cache_wake_up_zeros_only_mamba_state(runner_cls):
    groups = _make_hybrid_attn_groups(["mamba"])
    mamba_backing = torch.ones(4, 6)
    mamba_states = (mamba_backing[:, ::2], mamba_backing[:, 1::2])
    attention_cache = torch.ones(4)
    forward_context = {
        "mamba": SimpleNamespace(kv_cache=mamba_states),
        "attention": SimpleNamespace(kv_cache=attention_cache),
    }

    _run_post_kv_cache_wake_up(runner_cls, groups, forward_context)

    assert all(torch.count_nonzero(state) == 0 for state in mamba_states)
    assert torch.count_nonzero(mamba_backing) == 0
    assert torch.all(attention_cache == 1)


def test_post_kv_cache_wake_up_deduplicates_shared_views():
    groups = _make_hybrid_attn_groups(["mamba", "mamba_alias"])
    state = torch.ones(4, 2)
    forward_context = {
        "mamba": SimpleNamespace(kv_cache=(state,)),
        "mamba_alias": SimpleNamespace(kv_cache=(state,)),
        "attention": SimpleNamespace(kv_cache=torch.ones(4)),
    }
    version = state._version

    _run_post_kv_cache_wake_up(GPUModelRunnerV1, groups, forward_context)

    assert state._version == version + 1
    assert torch.count_nonzero(state) == 0


def test_reshape_padded_kv_cache_strides_by_padded_page():
    num_blocks = 3
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
        page_size_padded=384,
    )
    assert spec.real_page_size_bytes == 256

    raw = torch.zeros(spec.page_size_bytes * num_blocks, dtype=torch.int8)
    (kv_cache,) = dense_kv_cache_views(raw, spec, num_blocks, 1, KVCacheLayout.LBHNC)

    elem_size = 4  # float32
    # Content dim packs K and V: 2 * head_size.
    assert kv_cache.shape == (num_blocks, 1, 16, 2 * spec.head_size)
    assert kv_cache.dtype == spec.dtype
    assert kv_cache.stride(0) == spec.page_size_padded // elem_size
    assert kv_cache[1].storage_offset() == spec.page_size_padded // elem_size
    # Within one block the (unpadded) content stays compact.
    assert kv_cache[0].is_contiguous()


@pytest.mark.parametrize(
    ("kernel_block_sizes", "expected_num_blocks", "expected_num_states"),
    [
        (None, 4, 64),
        ([256], 4, 64),
        ([64], 16, 16),
    ],
)
def test_allocate_compressed_mla_cache(
    kernel_block_sizes: list[int] | None,
    expected_num_blocks: int,
    expected_num_states: int,
):
    spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        tokens_per_state=4,
    )
    num_pages = 4
    config = KVCacheConfig(
        num_blocks=num_pages,
        kv_cache_tensors=[
            KVCacheTensor(
                size=num_pages * spec.page_size_bytes,
                layers=["layer.0"],
                layer_stride=num_pages * spec.page_size_bytes,
                block_stride=spec.page_size_bytes,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(["layer.0"], spec)],
    )

    caches = allocate_kv_cache(
        config, torch.device("cpu"), KVCacheLayout.LBHNC, kernel_block_sizes
    )

    assert caches["layer.0"].shape == (expected_num_blocks, 1, expected_num_states, 128)


@pytest.mark.parametrize("layout", list(KVCacheLayout))
def test_copy_kv_cache_blocks_shared_storage(layout: KVCacheLayout):
    num_blocks = 4
    num_layers = 2
    spec = FullAttentionSpec(
        block_size=2,
        num_kv_heads=2,
        head_size=2,
        dtype=torch.float32,
    )
    raw = torch.zeros(num_blocks * num_layers * spec.page_size_bytes, dtype=torch.int8)
    caches = dense_kv_cache_views(raw, spec, num_blocks, num_layers, layout)

    for layer_idx, cache in enumerate(caches):
        for block_idx in range(num_blocks):
            cache[block_idx].fill_(10 * layer_idx + block_idx)

    expected = [[cache[i].clone() for i in range(num_blocks)] for cache in caches]
    copies = [KVCacheBlockCopy(src_block_id=0, dst_block_id=2)]

    copy_kv_cache_blocks_inplace(caches, num_blocks, copies)

    for layer_idx, cache in enumerate(caches):
        torch.testing.assert_close(cache[2], expected[layer_idx][0])
        torch.testing.assert_close(cache[1], expected[layer_idx][1])


def test_fixed_block_stride_propagates_outward_in_lhbnc():
    num_blocks = 3
    num_layers = 2
    spec = FullAttentionSpec(
        block_size=2,
        num_kv_heads=2,
        head_size=2,
        dtype=torch.float32,
    )
    natural = compute_layout_strides(spec, num_blocks, num_layers, KVCacheLayout.LHBNC)
    block_stride = natural[1] + 8

    strides = compute_layout_strides(
        spec,
        num_blocks,
        num_layers,
        KVCacheLayout.LHBNC,
        fixed_strides=(None, block_stride, None, None, None),
    )

    assert strides[1] == block_stride
    assert strides[2] == block_stride * num_blocks
    assert strides[0] == strides[2] * spec.num_heads


def test_copy_kv_cache_blocks_separate_head_groups():
    # LHBNC stores each head group separately, so a block's bytes are scattered
    # across L*H regions.
    layout = KVCacheLayout.LHBNC
    num_blocks = 4
    num_layers = 2
    spec = FullAttentionSpec(
        block_size=2,
        num_kv_heads=2,
        head_size=2,
        dtype=torch.float32,
        num_head_slots=2,
        state_content_bytes=2 * 2 * 4,
    )
    raw = torch.zeros(num_blocks * num_layers * spec.page_size_bytes, dtype=torch.int8)
    caches = dense_kv_cache_views(raw, spec, num_blocks, num_layers, layout)

    for layer_idx, cache in enumerate(caches):
        for block_idx in range(num_blocks):
            for head_idx in range(cache.shape[1]):
                cache[block_idx, head_idx].fill_(
                    100 * layer_idx + 10 * head_idx + block_idx
                )

    expected = [[cache[i].clone() for i in range(num_blocks)] for cache in caches]
    copy_kv_cache_blocks_inplace(
        caches,
        num_blocks,
        [KVCacheBlockCopy(src_block_id=0, dst_block_id=2)],
    )

    for layer_idx, cache in enumerate(caches):
        torch.testing.assert_close(cache[2], expected[layer_idx][0])
        torch.testing.assert_close(cache[1], expected[layer_idx][1])


@pytest.mark.parametrize(
    "layout,num_layers",
    [
        (KVCacheLayout.LBHNC, 2),
        # Splitting needs a manager block to be one dense page, which a
        # block-outermost layout only gives when the block holds one layer.
        (KVCacheLayout.BLHNC, 1),
    ],
)
def test_copy_kv_cache_blocks_with_virtual_block_splitting(
    layout: KVCacheLayout, num_layers: int
):
    num_blocks = 4
    physical_per_logical = 2
    spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
    )
    raw = torch.zeros(num_blocks * num_layers * spec.page_size_bytes, dtype=torch.int8)
    caches = dense_kv_cache_views(
        raw,
        spec,
        num_blocks,
        num_layers,
        layout,
        kernel_block_size=spec.block_size // physical_per_logical,
    )

    for layer_idx, cache in enumerate(caches):
        for block_idx in range(cache.shape[0]):
            cache[block_idx].fill_(100 * layer_idx + block_idx)
    expected = [[cache[i].clone() for i in range(cache.shape[0])] for cache in caches]

    copy_kv_cache_blocks_inplace(
        caches,
        num_blocks,
        [KVCacheBlockCopy(src_block_id=0, dst_block_id=2)],
    )

    dst_start = 2 * physical_per_logical
    for layer_idx, cache in enumerate(caches):
        for physical_idx in range(physical_per_logical):
            torch.testing.assert_close(
                cache[dst_start + physical_idx], expected[layer_idx][physical_idx]
            )
