# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify new _build_fa_remote and _build_fa_local produce identical
descriptors to the old code paths."""

from dataclasses import dataclass
from math import prod

import pytest
import torch

from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    _DIM4_B,
    _DIM4_H,
    KVCacheLayout,
    num_states_for,
)

# ── Old _build_fa_local (verbatim from upstream/main before rewrite) ──


def old_build_fa_local(
    base_addresses: list[int],
    block_len_per_layer: list[int],
    block_stride_per_layer: list[int],
    num_blocks: int,
    device_id: int,
    block_size_ratio: int,
    virtually_split: bool,
) -> list[tuple[int, int, int]]:
    """Old _build_fa_local: manual byte loop."""
    result: list[tuple[int, int, int]] = []
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


def new_build_fa_local(
    base_addresses: list[int],
    block_len_per_layer: list[int],
    block_stride_per_layer: list[int],
    num_blocks: int,
    device_id: int,
    block_size_ratio: int,
    virtually_split: bool,
    num_heads: int,
    block_size: int,
    tokens_per_state: int,
    dtype: torch.dtype,
    layout: KVCacheLayout,
) -> list[tuple[int, int, int]]:
    """New _build_fa_local: uses build_region_meta_4d + view_to_descriptors."""
    result: list[tuple[int, int, int]] = []
    for i, base_addr in enumerate(base_addresses):
        block_len = block_len_per_layer[i]
        region_len = block_len // 2 if virtually_split else block_len

        meta_4d = build_region_meta_4d(
            num_blocks=num_blocks,
            num_heads=num_heads,
            block_size=block_size,
            tokens_per_state=tokens_per_state,
            block_len_bytes=region_len // block_size_ratio,
            block_stride_bytes=block_stride_per_layer[i] // block_size_ratio,
            dtype=dtype,
            layout=layout,
        )

        result.extend(view_to_descriptors(meta_4d, base_addr, device_id))

        if virtually_split:
            v_base = base_addr + block_len // 2
            result.extend(view_to_descriptors(meta_4d, v_base, device_id))
    return result


# ── Old _build_fa_remote (verbatim from upstream/main before rewrite) ──


def old_build_fa_remote(
    block_len_per_layer: list[int],
    block_stride_per_layer: list[int],
    kv_caches_base_addr: list[int],
    remote_block_lens: list[int],
    num_blocks: int,
    device_id: int,
    rank_offset_factor: int,
    num_attn_reads: int,
    block_size_ratio: int,
    virtually_split: bool,
) -> list[tuple[int, int, int]]:
    """Old _build_fa_remote logic (pre-rewrite).

    remote_block_lens: nixl_agent_meta.block_lens (remote per-region sizes).
    block_len_per_layer: local per-region sizes.
    """
    result: list[tuple[int, int, int]] = []
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


# ── New code (extracted from worker.py) ─────────────────────────────────


def build_region_meta_4d(
    num_blocks,
    num_heads,
    block_size,
    tokens_per_state,
    block_len_bytes,
    block_stride_bytes,
    dtype,
    layout,
):
    elem = get_dtype_size(dtype)
    n_states = num_states_for(block_size, tokens_per_state)
    c_elems = block_len_bytes // (num_heads * n_states * elem)

    logical_4d = (num_blocks, num_heads, n_states, c_elems)
    layer_order = layout.layer_stride_order
    phys_shape = tuple(logical_4d[p] for p in layer_order)
    phys_strides = list(torch.empty(phys_shape, device="meta").stride())
    inv = [list(layer_order).index(i) for i in range(4)]
    phys_strides[inv[_DIM4_B]] = block_stride_bytes // elem

    meta = torch.as_strided(
        torch.empty(1, dtype=dtype, device="meta"),
        size=phys_shape,
        stride=tuple(phys_strides),
    )
    return meta.permute(*inv)


def view_to_descriptors(view, base_addr, device_id):
    elem = view.element_size()
    block_stride = view.stride(_DIM4_B) * elem
    payload = prod(view.shape[1:]) * elem
    offset = view.storage_offset() * elem
    return [
        (base_addr + offset + b * block_stride, payload, device_id)
        for b in range(view.shape[_DIM4_B])
    ]


def new_build_fa_remote(
    block_len_per_layer: list[int],
    block_stride_per_layer: list[int],
    kv_caches_base_addr: list[int],
    num_blocks: int,
    device_id: int,
    block_size_ratio: int,
    virtually_split: bool,
    # New params
    remote_num_heads: int,
    block_size: int,
    tokens_per_state: int,
    dtype: torch.dtype,
    layout: KVCacheLayout,
    remote_tp_size: int,
    remote_tp_rank: int,
    local_tp: int,
    local_rank: int,
    total_kv_heads: int,
    slice_fn,
) -> list[tuple[int, int, int]]:
    """New _build_fa_remote logic (post-rewrite)."""
    result: list[tuple[int, int, int]] = []
    for i, base_addr in enumerate(kv_caches_base_addr):
        block_len = block_len_per_layer[i]
        region_len = block_len // 2 if virtually_split else block_len

        meta_4d = build_region_meta_4d(
            num_blocks=num_blocks,
            num_heads=remote_num_heads,
            block_size=block_size,
            tokens_per_state=tokens_per_state,
            block_len_bytes=region_len,
            block_stride_bytes=block_stride_per_layer[i],
            dtype=dtype,
            layout=layout,
        )

        slices = slice_fn(
            meta_4d,
            my_tp=remote_tp_size,
            my_rank=remote_tp_rank,
            other_tp=local_tp,
            other_rank=local_rank,
            total_num_kv_heads=total_kv_heads,
        )
        if not slices:
            continue

        view = slices[0]

        result.extend(view_to_descriptors(view, base_addr, device_id))

        if virtually_split:
            v_base = base_addr + block_len // 2
            result.extend(view_to_descriptors(view, v_base, device_id))
    return result


# ── Old rank_offset_factor computation (from tp_mapping.py) ─────────────


def compute_rank_offset_factor(
    tp_rank,
    tp_size,
    remote_tp_size,
    total_num_kv_heads,
    is_mla,
    attn_ranks,
):
    if is_mla or tp_size <= remote_tp_size:
        return 0
    elif tp_size > total_num_kv_heads:
        local_head = tp_rank * total_num_kv_heads // tp_size
        p_start = attn_ranks[0] * total_num_kv_heads // remote_tp_size
        return local_head - p_start
    else:
        return tp_rank % (tp_size // remote_tp_size)


def compute_attn_ranks(tp_rank, tp_size, remote_tp_size, total_num_kv_heads, is_mla):
    import numpy as np

    if is_mla or tp_size >= remote_tp_size:
        return [tp_rank * remote_tp_size // tp_size]
    else:
        abs_tp = remote_tp_size // tp_size
        start = tp_rank * abs_tp
        heads = np.arange(start, start + abs_tp) * total_num_kv_heads // remote_tp_size
        _, unique_idx = np.unique(heads, return_index=True)
        return (start + np.sort(unique_idx)).tolist()


# ── Tests ───────────────────────────────────────────────────────────────


@dataclass
class FakeSpec:
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    block_size: int
    tokens_per_state: int = 1


def _make_slice_fn(total_kv_heads):
    """Build a slice_for_tp_transfer that mimics AttentionSpec."""

    def slice_fn(tensor, my_tp, my_rank, other_tp, other_rank, total_num_kv_heads):
        if total_num_kv_heads >= my_tp:
            my_start = my_rank * total_num_kv_heads // my_tp
            my_end = (my_rank + 1) * total_num_kv_heads // my_tp
        else:
            my_start, my_end = 0, total_num_kv_heads

        if total_num_kv_heads >= other_tp:
            other_start = other_rank * total_num_kv_heads // other_tp
            other_end = (other_rank + 1) * total_num_kv_heads // other_tp
        else:
            other_start, other_end = 0, total_num_kv_heads

        overlap_start = max(my_start, other_start)
        overlap_end = min(my_end, other_end)
        if overlap_start >= overlap_end:
            return []

        local_h_start = overlap_start - my_start
        local_h_len = overlap_end - overlap_start
        return [tensor.narrow(_DIM4_H, local_h_start, local_h_len)]

    return slice_fn


def _run_comparison(
    total_kv_heads: int,
    head_size: int,
    block_size: int,
    local_tp: int,
    local_rank: int,
    remote_tp: int,
    remote_rank: int,
    num_blocks: int = 4,
    num_regions: int = 2,
    dtype: torch.dtype = torch.float16,
    is_mla: bool = False,
    virtually_split: bool = False,
    block_size_ratio: int = 1,
    layout_str: str = "HND",
):
    """Run old vs new and compare descriptor lists."""
    elem = get_dtype_size(dtype)

    if is_mla:
        remote_num_heads = 1  # MLA: single latent
    elif total_kv_heads >= remote_tp:
        remote_num_heads = total_kv_heads // remote_tp
    else:
        remote_num_heads = total_kv_heads

    region_c = 2 * head_size * elem if virtually_split else head_size * elem

    block_len = remote_num_heads * block_size * region_c
    block_stride = block_len  # no padding for simplicity

    block_lens = [block_len] * num_regions
    block_strides = [block_stride] * num_regions
    base_addrs = [i * block_stride * num_blocks for i in range(num_regions)]
    device_id = 0

    attn_ranks = compute_attn_ranks(
        local_rank, local_tp, remote_tp, total_kv_heads, is_mla
    )
    num_attn_reads = len(attn_ranks)
    rank_offset_factor = compute_rank_offset_factor(
        local_rank, local_tp, remote_tp, total_kv_heads, is_mla, attn_ranks
    )

    # local block_len_per_layer: what the local side sees.
    # GQA replicated: when total_kv_heads < local_tp, every rank holds ALL heads.
    if is_mla:
        local_num_heads = 1
    elif total_kv_heads >= local_tp:
        local_num_heads = total_kv_heads // local_tp
    else:
        local_num_heads = total_kv_heads
    local_block_len = local_num_heads * block_size * region_c
    local_block_lens = [local_block_len] * num_regions

    old = old_build_fa_remote(
        block_len_per_layer=local_block_lens,
        block_stride_per_layer=block_strides,
        kv_caches_base_addr=base_addrs,
        remote_block_lens=block_lens,
        num_blocks=num_blocks,
        device_id=device_id,
        rank_offset_factor=rank_offset_factor,
        num_attn_reads=num_attn_reads,
        block_size_ratio=block_size_ratio,
        virtually_split=virtually_split,
    )

    layout = KVCacheLayout.from_layout_string(layout_str)
    new = new_build_fa_remote(
        block_len_per_layer=block_lens,
        block_stride_per_layer=block_strides,
        kv_caches_base_addr=base_addrs,
        num_blocks=num_blocks,
        device_id=device_id,
        block_size_ratio=block_size_ratio,
        virtually_split=virtually_split,
        remote_num_heads=remote_num_heads,
        block_size=block_size,
        tokens_per_state=1,
        dtype=dtype,
        layout=layout,
        remote_tp_size=remote_tp,
        remote_tp_rank=remote_rank,
        local_tp=local_tp,
        local_rank=local_rank,
        total_kv_heads=total_kv_heads,
        slice_fn=_make_slice_fn(total_kv_heads),
    )

    return old, new


# ─────────────── Actual test cases ────────────────────────────────────


CONFIGS = [
    # (name, total_kv_heads, head_size, block_size, local_tp, local_rank,
    #  remote_tp, remote_rank, layout_str)
    ("1p1d_HND", 8, 128, 16, 1, 0, 1, 0, "HND"),
    ("1p1d_NHD", 8, 128, 16, 1, 0, 1, 0, "NHD"),
    ("D2_P1_r0_HND", 8, 128, 16, 2, 0, 1, 0, "HND"),
    ("D2_P1_r1_HND", 8, 128, 16, 2, 1, 1, 0, "HND"),
    ("D4_P1_r0_HND", 8, 128, 16, 4, 0, 1, 0, "HND"),
    ("D4_P1_r3_HND", 8, 128, 16, 4, 3, 1, 0, "HND"),
    ("D1_P2_r0_HND", 8, 128, 16, 1, 0, 2, 0, "HND"),
    ("D1_P2_r0_P1_HND", 8, 128, 16, 1, 0, 2, 1, "HND"),
    ("D1_P4_r0_HND", 8, 128, 16, 1, 0, 4, 0, "HND"),
    # GQA: fewer KV heads than TP
    ("GQA_D4_P1_r0", 2, 128, 16, 4, 0, 1, 0, "HND"),
    # GQA_D4_P1_r2 excluded: old code had a bug where rank_offset_factor=1
    # produces an out-of-bounds offset (8192 into an 8192-byte block).
    # slice_for_tp_transfer correctly returns offset 0 for this case because
    # total_kv_heads < local_tp means all heads are replicated to every rank.
]

# Virtually-split configs (FlashInfer-like: K+V in one block)
VS_CONFIGS = [
    # (name, total_kv_heads, head_size, block_size, local_tp, local_rank,
    #  remote_tp, remote_rank, layout_str)
    ("vs_1p1d_HND", 8, 128, 16, 1, 0, 1, 0, "HND"),
    ("vs_1p1d_NHD", 8, 128, 16, 1, 0, 1, 0, "NHD"),
    ("vs_D2_P1_r0", 8, 128, 16, 2, 0, 1, 0, "HND"),
    ("vs_D2_P1_r1", 8, 128, 16, 2, 1, 1, 0, "HND"),
    ("vs_D1_P2_r0", 8, 128, 16, 1, 0, 2, 0, "HND"),
]


@pytest.mark.parametrize(
    "name,total_kv,hs,bs,ltp,lr,rtp,rr,lay",
    CONFIGS,
    ids=[c[0] for c in CONFIGS],
)
def test_old_vs_new(name, total_kv, hs, bs, ltp, lr, rtp, rr, lay):
    old, new = _run_comparison(
        total_kv_heads=total_kv,
        head_size=hs,
        block_size=bs,
        local_tp=ltp,
        local_rank=lr,
        remote_tp=rtp,
        remote_rank=rr,
        layout_str=lay,
    )
    assert len(old) == len(new), f"Length mismatch: old={len(old)} new={len(new)}"
    for j, (o, n) in enumerate(zip(old, new)):
        assert o == n, (
            f"Descriptor {j} mismatch:\n  old={o}\n  new={n}\n  config={name}"
        )


@pytest.mark.parametrize(
    "name,total_kv,hs,bs,ltp,lr,rtp,rr,lay",
    VS_CONFIGS,
    ids=[c[0] for c in VS_CONFIGS],
)
def test_old_vs_new_virtually_split(name, total_kv, hs, bs, ltp, lr, rtp, rr, lay):
    old, new = _run_comparison(
        total_kv_heads=total_kv,
        head_size=hs,
        block_size=bs,
        local_tp=ltp,
        local_rank=lr,
        remote_tp=rtp,
        remote_rank=rr,
        layout_str=lay,
        virtually_split=True,
    )
    assert len(old) == len(new), f"Length mismatch: old={len(old)} new={len(new)}"
    for j, (o, n) in enumerate(zip(old, new)):
        assert o == n, (
            f"Descriptor {j} mismatch:\n  old={o}\n  new={n}\n  config={name}"
        )


# ── _build_fa_local comparison helper ────────────────────────────────────


def _run_local_comparison(
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int = 4,
    num_regions: int = 2,
    dtype: torch.dtype = torch.float16,
    virtually_split: bool = False,
    block_size_ratio: int = 1,
    layout_str: str = "HND",
):
    """Run old vs new _build_fa_local and compare."""
    elem = get_dtype_size(dtype)

    region_c = 2 * head_size * elem if virtually_split else head_size * elem

    block_len = num_heads * block_size * region_c
    block_stride = block_len

    block_lens = [block_len] * num_regions
    block_strides = [block_stride] * num_regions
    base_addrs = [i * block_stride * num_blocks for i in range(num_regions)]
    device_id = 0

    old = old_build_fa_local(
        base_addresses=base_addrs,
        block_len_per_layer=block_lens,
        block_stride_per_layer=block_strides,
        num_blocks=num_blocks * block_size_ratio,
        device_id=device_id,
        block_size_ratio=block_size_ratio,
        virtually_split=virtually_split,
    )

    layout = KVCacheLayout.from_layout_string(layout_str)
    new = new_build_fa_local(
        base_addresses=base_addrs,
        block_len_per_layer=block_lens,
        block_stride_per_layer=block_strides,
        num_blocks=num_blocks * block_size_ratio,
        device_id=device_id,
        block_size_ratio=block_size_ratio,
        virtually_split=virtually_split,
        num_heads=num_heads,
        block_size=block_size,
        tokens_per_state=1,
        dtype=dtype,
        layout=layout,
    )

    return old, new


LOCAL_CONFIGS = [
    # (name, num_heads, head_size, block_size, layout_str)
    ("local_HND_h8", 8, 128, 16, "HND"),
    ("local_NHD_h8", 8, 128, 16, "NHD"),
    ("local_HND_h1", 1, 512, 16, "HND"),
    ("local_HND_h32", 32, 64, 16, "HND"),
]

LOCAL_VS_CONFIGS = [
    ("local_vs_HND_h8", 8, 128, 16, "HND"),
    ("local_vs_NHD_h8", 8, 128, 16, "NHD"),
]


@pytest.mark.parametrize(
    "name,nh,hs,bs,lay",
    LOCAL_CONFIGS,
    ids=[c[0] for c in LOCAL_CONFIGS],
)
def test_local_old_vs_new(name, nh, hs, bs, lay):
    old, new = _run_local_comparison(
        num_heads=nh,
        head_size=hs,
        block_size=bs,
        layout_str=lay,
    )
    assert len(old) == len(new), f"Length mismatch: old={len(old)} new={len(new)}"
    for j, (o, n) in enumerate(zip(old, new)):
        assert o == n, (
            f"Local descriptor {j} mismatch:\n  old={o}\n  new={n}\n  config={name}"
        )


@pytest.mark.parametrize(
    "name,nh,hs,bs,lay",
    LOCAL_VS_CONFIGS,
    ids=[c[0] for c in LOCAL_VS_CONFIGS],
)
def test_local_old_vs_new_virtually_split(name, nh, hs, bs, lay):
    old, new = _run_local_comparison(
        num_heads=nh,
        head_size=hs,
        block_size=bs,
        layout_str=lay,
        virtually_split=True,
    )
    assert len(old) == len(new), f"Length mismatch: old={len(old)} new={len(new)}"
    for j, (o, n) in enumerate(zip(old, new)):
        assert o == n, (
            f"Local descriptor {j} mismatch:\n  old={o}\n  new={n}\n  config={name}"
        )
