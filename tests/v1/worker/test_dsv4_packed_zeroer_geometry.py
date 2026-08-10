# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regression for the packed-DSV4 KV zeroer geometry (upstream #50276).

Four DSV4 fp8_ds_mla layers share one cross-layer allocation: each block row
packs the four alignment-padded pages side by side (layout BLHNC). The zeroer
must step by the full packed row per block while zeroing only each layer's
meaningful page — never the alignment padding, never an adjacent layer's page.

CPU only: the zeroer's ``__init__`` precomputes the segment tables; no kernel
is launched.
"""

from types import SimpleNamespace

import torch

from vllm.v1.kv_cache_interface import (
    KVCacheLayout,
    MLAAttentionSpec,
    reshape_kv_cache,
)
from vllm.v1.worker.utils import AttentionGroup, KVBlockZeroer

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
    assert spec.storage_block_size == 64
    unpadded_page = spec.unpadded_page_size_bytes
    padded_page = spec.page_size_bytes
    assert unpadded_page == 64 * 584
    assert padded_page > unpadded_page, "alignment must pad the page"

    raw = torch.zeros(NUM_BLOCKS * NUM_LAYERS * padded_page, dtype=torch.int8)
    views = reshape_kv_cache(raw, spec, NUM_BLOCKS, NUM_LAYERS, KVCacheLayout.BLHNC)
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
