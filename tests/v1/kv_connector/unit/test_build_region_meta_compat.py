#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify new build_region_meta produces identical descriptors to the
ORIGINAL upstream manual byte-loop code."""

from math import prod

import torch

from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    _DIM4_B,
    _DIM4_H,
    AttentionSpec,
    FullAttentionSpec,
    KVCacheLayout,
    KVQuantMode,
    MLAAttentionSpec,
    num_states_for,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
    build_region_meta,
)


# ── ORIGINAL upstream _build_fa_local (verbatim manual byte loop) ──

def original_build_fa_local(
    base_addresses, block_len_per_layer, block_stride_per_layer,
    num_blocks, device_id, block_size_ratio, virtually_split,
):
    result = []
    for i, base_addr in enumerate(base_addresses):
        if virtually_split:
            kv_block_len = block_len_per_layer[i] // 2
        else:
            kv_block_len = block_len_per_layer[i]
        kv_block_len = kv_block_len // block_size_ratio
        page_stride = block_stride_per_layer[i] // block_size_ratio

        for block_id in range(num_blocks):
            addr = base_addr + block_id * page_stride
            result.append((addr, kv_block_len, device_id))

        if virtually_split:
            second_split = block_len_per_layer[i] // 2 // block_size_ratio
            for block_id in range(num_blocks):
                addr = base_addr + block_id * page_stride
                v_addr = addr + kv_block_len
                result.append((v_addr, second_split, device_id))
    return result


# ── ORIGINAL upstream _build_fa_remote (verbatim manual byte loop) ──

def original_build_fa_remote(
    block_len_per_layer, block_stride_per_layer,
    kv_caches_base_addr, remote_block_lens,
    num_blocks, device_id,
    rank_offset_factor, num_attn_reads,
    block_size_ratio, virtually_split,
):
    result = []
    for i, base_addr in enumerate(kv_caches_base_addr):
        local_block_len = block_len_per_layer[i]
        if virtually_split:
            local_block_len = local_block_len // 2

        remote_kv_block_len = local_block_len // block_size_ratio
        if block_size_ratio > 1:
            local_block_len = remote_kv_block_len

        local_block_len = local_block_len // num_attn_reads
        rank_offset = rank_offset_factor * remote_kv_block_len

        page_stride = block_stride_per_layer[i]
        for block_id in range(num_blocks):
            block_offset = block_id * page_stride
            addr = base_addr + block_offset + rank_offset
            result.append((addr, local_block_len, device_id))

        if virtually_split:
            second_split = block_len_per_layer[i] // 2
            second_split = second_split // num_attn_reads
            for block_id in range(num_blocks):
                block_offset = block_id * page_stride
                addr = base_addr + block_offset + rank_offset
                v_addr = addr + remote_block_lens[i] // 2
                result.append((v_addr, second_split, device_id))
    return result


# ── Helpers ──

def view_to_descriptors(view, base_addr, device_id):
    elem = view.element_size()
    block_stride = view.stride(_DIM4_B) * elem
    payload = prod(view.shape[1:]) * elem
    offset = view.storage_offset() * elem
    return [
        (base_addr + offset + b * block_stride, payload, device_id)
        for b in range(view.shape[_DIM4_B])
    ]


def new_build_fa_local_from_build_region_meta(
    base_addresses, block_len_per_layer, block_stride_per_layer,
    num_blocks, device_id, block_size_ratio, virtually_split,
    spec, block_size, layout,
):
    """New code path: build_region_meta + view_to_descriptors."""
    result = []
    for i, base_addr in enumerate(base_addresses):
        block_len = block_len_per_layer[i]
        metas = build_region_meta(
            spec=spec,
            num_blocks=num_blocks,
            block_size=block_size,
            layout=layout,
            block_stride_bytes=block_stride_per_layer[i] // block_size_ratio,
            region_content_bytes=block_len // block_size_ratio,
            virtually_split=virtually_split,
        )
        for meta in metas:
            result.extend(view_to_descriptors(meta, base_addr, device_id))
    return result


# ── Test configs ──

CONFIGS = []

# Standard attention: 8 heads, 64 head_size, fp16, block_size=16
for num_blocks in [4, 16]:
    for bsr in [1]:
        for vs in [False, True]:
            for layout_str in ["HND", "NHD"]:
                CONFIGS.append(dict(
                    label=f"attn_h8_bs16_nb{num_blocks}_bsr{bsr}_vs{vs}_{layout_str}",
                    num_kv_heads=8, head_size=64, dtype=torch.float16,
                    block_size=16, num_blocks=num_blocks * bsr,
                    block_size_ratio=bsr, virtually_split=vs,
                    layout_str=layout_str,
                    is_mla=False,
                ))

# MLA: 1 head, 512 head_size, fp16, block_size=64
for num_blocks in [4]:
    for layout_str in ["HND", "NHD"]:
        CONFIGS.append(dict(
            label=f"mla_h1_bs64_nb{num_blocks}_{layout_str}",
            num_kv_heads=1, head_size=512, dtype=torch.float16,
            block_size=64, num_blocks=num_blocks,
            block_size_ratio=1, virtually_split=False,
            layout_str=layout_str,
            is_mla=True,
        ))

# Small heads (GQA): 2 heads, 128 head_size
for vs in [False, True]:
    CONFIGS.append(dict(
        label=f"gqa_h2_hs128_vs{vs}",
        num_kv_heads=2, head_size=128, dtype=torch.float16,
        block_size=16, num_blocks=8,
        block_size_ratio=1, virtually_split=vs,
        layout_str="HND",
        is_mla=False,
    ))


def test_build_fa_local_compat(cfg):
    """Compare original upstream _build_fa_local vs new build_region_meta."""
    num_kv_heads = cfg["num_kv_heads"]
    head_size = cfg["head_size"]
    dtype = cfg["dtype"]
    block_size = cfg["block_size"]
    num_blocks = cfg["num_blocks"]
    bsr = cfg["block_size_ratio"]
    vs = cfg["virtually_split"]
    layout = KVCacheLayout.from_layout_string(cfg["layout_str"])
    is_mla = cfg["is_mla"]

    elem = get_dtype_size(dtype)

    if is_mla:
        content_per_head = head_size * elem
        block_len = num_kv_heads * block_size * content_per_head
        spec = MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=dtype,
        )
    else:
        content_per_head = 2 * head_size * elem
        if vs:
            block_len = num_kv_heads * block_size * content_per_head
        else:
            block_len = num_kv_heads * block_size * head_size * elem
        spec = FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            head_size_v=head_size,
            dtype=dtype,
        )

    page_stride = block_len + 256  # some padding
    base_addr = 0x1000_0000
    device_id = 0

    base_addresses = [base_addr]
    block_len_per_layer = [block_len]
    block_stride_per_layer = [page_stride]

    old = original_build_fa_local(
        base_addresses, block_len_per_layer, block_stride_per_layer,
        num_blocks, device_id, bsr, vs,
    )
    new = new_build_fa_local_from_build_region_meta(
        base_addresses, block_len_per_layer, block_stride_per_layer,
        num_blocks, device_id, bsr, vs,
        spec, block_size, layout,
    )

    assert old == new, (
        f"Mismatch for {cfg['label']}:\n"
        f"  old[0:5] = {old[:5]}\n"
        f"  new[0:5] = {new[:5]}\n"
        f"  old_len={len(old)}, new_len={len(new)}"
    )


if __name__ == "__main__":
    for cfg in CONFIGS:
        print(f"Testing {cfg['label']}...", end=" ")
        try:
            test_build_fa_local_compat(cfg)
            print("PASS")
        except AssertionError as e:
            print(f"FAIL\n{e}")
        except Exception as e:
            print(f"ERROR: {e}")
