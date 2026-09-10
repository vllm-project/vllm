# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regression for the packed-DSV4 KV zeroer geometry (upstream #50276).

Four DSV4 fp8_ds_mla layers share one cross-layer allocation: each block row packs
the four alignment-padded pages side by side (layout BLHNC). The zeroer must step by
the full packed row per block while zeroing only each layer's meaningful page — never
the alignment padding, never an adjacent layer's page.

CPU only: the zeroer's ``__init__`` precomputes the segment tables; no kernel is
launched.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention.utils import dense_kv_cache_views
from vllm.v1.kv_cache_interface import KVCacheLayout, MLAAttentionSpec
from vllm.v1.worker.utils import (
    AttentionGroup,
    KVBlockZeroer,
    allocate_kv_cache,
)

pytestmark = pytest.mark.cpu_test

NUM_BLOCKS = 100
NUM_LAYERS = 4
ALIGNMENT = 576


def test_packed_dsv4_zeroer_zeroes_only_each_layers_page():
    spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        cache_dtype_str="fp8_ds_mla",
        model_version="deepseek_v4",
        tokens_per_state=4,
        alignment=ALIGNMENT,
        # DeepseekV4 fp8_ds_mla: 584B per token, published by the layer.
        state_content_bytes=584,
    )
    # DSV4: 448B NoPE + 128B RoPE + 8B fp8 scale = 584B per stored state.
    assert spec.state_content_size_bytes == 584
    assert spec.num_states == 64
    unpadded_page = spec.unpadded_page_size_bytes
    padded_page = spec.page_size_bytes
    assert unpadded_page == 64 * 584
    assert padded_page > unpadded_page, "alignment must pad the page"

    raw = torch.zeros(NUM_BLOCKS * NUM_LAYERS * padded_page, dtype=torch.int8)
    views = dense_kv_cache_views(raw, spec, NUM_BLOCKS, NUM_LAYERS, KVCacheLayout.BLHNC)
    base = raw.data_ptr()
    block_row = NUM_LAYERS * padded_page
    for i, view in enumerate(views):
        assert view.data_ptr() - base == i * padded_page
        assert view.stride(0) * view.element_size() == block_row

    zeroer = KVBlockZeroer(
        torch.device("cpu"),
        attn_groups_iter=iter(
            [
                AttentionGroup(
                    backend=None,
                    layer_names=[f"layer.{i}" for i in range(NUM_LAYERS)],
                    kv_cache_spec=spec,
                    kv_cache_group_id=0,
                )
            ]
        ),
        kernel_block_sizes=[spec.block_size],
        static_forward_context={
            f"layer.{i}": SimpleNamespace(kv_cache=views[i]) for i in range(NUM_LAYERS)
        },
        num_blocks=NUM_BLOCKS,
    )
    seg_addrs, seg_block_strides, seg_page_sizes, _, _, n_segs = zeroer._meta

    assert n_segs == NUM_LAYERS
    # Segments step by the full packed row per block...
    assert (seg_block_strides * 4 == block_row).all()
    # ...but zero only the layer's meaningful page: no alignment padding, no
    # adjacent layer's page.
    assert (seg_page_sizes * 4 == unpadded_page).all()
    assert sorted(a - base for a in seg_addrs.tolist()) == [
        i * padded_page for i in range(NUM_LAYERS)
    ]
    # Block 99 of the highest-offset layer stays within the packed backing.
    last_end = (
        max(seg_addrs.tolist())
        + (NUM_BLOCKS - 1) * int(seg_block_strides[0]) * 4
        + int(seg_page_sizes[0]) * 4
    )
    assert last_end <= base + raw.numel()


def test_overlaid_zeroer_dedups_segments_with_max_span():
    """Two groups overlay one allocation; the zeroer must emit one segment per distinct
    byte offset, spanning the widest overlaid page, so a newly allocated block is fully
    zeroed no matter which group owns it."""
    from unittest.mock import MagicMock

    from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups
    from vllm.v1.kv_cache_interface import (
        KVCacheGroupSpec,
        KVCacheLayout,
        UniformTypeKVCacheSpecs,
    )

    def make_spec(head_size):
        return MLAAttentionSpec(
            block_size=64, num_kv_heads=1, head_size=head_size, dtype=torch.uint8
        )

    g1_specs = {"g1.big": make_spec(512), "g1.small": make_spec(128)}
    g2_specs = {"g2.huge": make_spec(1024)}
    groups = [
        KVCacheGroupSpec(
            list(g1_specs),
            UniformTypeKVCacheSpecs(block_size=64, kv_cache_specs=g1_specs),
        ),
        KVCacheGroupSpec(
            list(g2_specs),
            UniformTypeKVCacheSpecs(block_size=64, kv_cache_specs=g2_specs),
        ),
    ]
    from vllm.config import CacheConfig

    vllm_config = MagicMock()
    vllm_config.cache_config = CacheConfig()
    vllm_config.cache_config.num_gpu_blocks_override = None
    vllm_config.cache_config.kv_cache_layout = "BLHNC"
    config = get_kv_cache_config_from_groups(vllm_config, groups, 8 * 1024 * 1024)
    views = allocate_kv_cache(config, torch.device("cpu"), KVCacheLayout.BLHNC, None)
    buf_ptr = views["g1.big"].data_ptr()

    attn_groups = [
        AttentionGroup(
            backend=None,
            layer_names=list(specs),
            kv_cache_spec=next(iter(specs.values())),
            kv_cache_group_id=gid,
        )
        for gid, specs in enumerate((g1_specs, g2_specs))
    ]
    zeroer = KVBlockZeroer(
        torch.device("cpu"),
        attn_groups_iter=iter(attn_groups),
        kernel_block_sizes=[64, 64],
        static_forward_context={
            name: SimpleNamespace(kv_cache=views[name]) for name in views
        },
        num_blocks=config.num_blocks,
    )
    seg_addrs, seg_block_strides, seg_page_sizes, _, _, n_segs = zeroer._meta

    # g1.big and g2.huge overlay at offset 0 -> one segment with g2's wider
    # span; g1.small keeps its own segment.
    assert n_segs == 2
    pages = {n: s.page_size_bytes for n, s in (g1_specs | g2_specs).items()}
    by_offset = {
        a - buf_ptr: p * 4 for a, p in zip(seg_addrs.tolist(), seg_page_sizes.tolist())
    }
    assert by_offset[0] == max(pages["g1.big"], pages["g2.huge"])
    assert by_offset[pages["g1.big"]] == pages["g1.small"]
    packed_block_stride = max(sum(pages[n] for n in g) for g in (g1_specs, g2_specs))
    assert (seg_block_strides * 4 == packed_block_stride).all()
