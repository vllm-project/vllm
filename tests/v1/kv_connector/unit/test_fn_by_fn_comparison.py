#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Function-by-function comparison: old _build_fa_remote vs new code.

Isolates each sub-function so we can see EXACTLY where any divergence occurs.

Old code decomposes into:
  A) get_backend_aware_kv_block_len  → local_block_len
  B) rank_offset_factor * remote_kv_block_len → rank_offset
  C) for block_id: addr = base + block_id * page_size + rank_offset → descriptors

New code decomposes into:
  1) _make_remote_meta_4d → 4D meta tensor
  2) slice_for_tp_transfer → narrowed view (head slicing)
  3) _view_to_descriptors  → descriptors

We verify:
  - (1) produces correct shape [B, H_remote, N, C]
  - (2) produces storage_offset * elem == rank_offset from (B)
  -      narrowed H == local_block_len / (N * C * elem) from (A)
  - (3) produces identical (addr, len, dev) tuples to (C)
"""

import sys
from dataclasses import dataclass
from math import prod

import numpy as np
import torch

from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    _DIM4_B,
    _DIM4_H,
    KVCacheLayout,
    num_states_for,
)

# ═══════════════════════════════════════════════════════════════════════
# OLD sub-functions (verbatim from upstream/main)
# ═══════════════════════════════════════════════════════════════════════


def old_get_backend_aware_kv_block_len(
    block_len_per_layer_i: int,
    virtually_split: bool,
    first_split: bool = True,
) -> int:
    if virtually_split:
        return block_len_per_layer_i // 2
    return block_len_per_layer_i


def old_compute_rank_offset_factor(
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


def old_compute_attn_ranks(
    tp_rank, tp_size, remote_tp_size, total_num_kv_heads, is_mla
):
    if is_mla or tp_size >= remote_tp_size:
        return [tp_rank * remote_tp_size // tp_size]
    else:
        abs_tp = remote_tp_size // tp_size
        start = tp_rank * abs_tp
        heads = np.arange(start, start + abs_tp) * total_num_kv_heads // remote_tp_size
        _, unique_idx = np.unique(heads, return_index=True)
        return (start + np.sort(unique_idx)).tolist()


# ═══════════════════════════════════════════════════════════════════════
# NEW sub-functions (extracted from worker.py)
# ═══════════════════════════════════════════════════════════════════════


def new_make_remote_meta_4d(
    num_blocks,
    remote_num_heads,
    block_size,
    tokens_per_state,
    block_len_bytes,
    block_stride_bytes,
    dtype,
    layout,
):
    elem = get_dtype_size(dtype)
    n_states = num_states_for(block_size, tokens_per_state)
    c_elems = block_len_bytes // (remote_num_heads * n_states * elem)

    logical_4d = (num_blocks, remote_num_heads, n_states, c_elems)
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


def new_slice_for_tp_transfer(
    tensor, my_tp, my_rank, other_tp, other_rank, total_num_kv_heads
):
    """Mimics AttentionSpec.slice_for_tp_transfer."""
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


def new_view_to_descriptors(view, base_addr, device_id):
    elem = view.element_size()
    block_stride = view.stride(_DIM4_B) * elem
    payload = prod(view.shape[1:]) * elem
    offset = view.storage_offset() * elem
    return [
        (base_addr + offset + b * block_stride, payload, device_id)
        for b in range(view.shape[_DIM4_B])
    ]


# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class Config:
    name: str
    total_kv_heads: int
    head_size: int
    block_size: int
    local_tp: int
    local_rank: int
    remote_tp: int
    remote_rank: int
    layout_str: str
    virtually_split: bool = False
    block_size_ratio: int = 1
    is_mla: bool = False
    dtype: torch.dtype = torch.float16


CONFIGS = [
    Config("1p1d_HND", 8, 128, 16, 1, 0, 1, 0, "HND"),
    Config("1p1d_NHD", 8, 128, 16, 1, 0, 1, 0, "NHD"),
    Config("D2_P1_r0_HND", 8, 128, 16, 2, 0, 1, 0, "HND"),
    Config("D2_P1_r1_HND", 8, 128, 16, 2, 1, 1, 0, "HND"),
    Config("D4_P1_r0_HND", 8, 128, 16, 4, 0, 1, 0, "HND"),
    Config("D4_P1_r3_HND", 8, 128, 16, 4, 3, 1, 0, "HND"),
    Config("D1_P2_r0_HND", 8, 128, 16, 1, 0, 2, 0, "HND"),
    Config("D1_P2_r0_from_P1", 8, 128, 16, 1, 0, 2, 1, "HND"),
    Config("D1_P4_r0_HND", 8, 128, 16, 1, 0, 4, 0, "HND"),
    Config("GQA_D4_P1_r0", 2, 128, 16, 4, 0, 1, 0, "HND"),
    # vs = virtually split
    Config("vs_1p1d_HND", 8, 128, 16, 1, 0, 1, 0, "HND", virtually_split=True),
    Config("vs_1p1d_NHD", 8, 128, 16, 1, 0, 1, 0, "NHD", virtually_split=True),
    Config("vs_D2_P1_r0", 8, 128, 16, 2, 0, 1, 0, "HND", virtually_split=True),
    Config("vs_D2_P1_r1", 8, 128, 16, 2, 1, 1, 0, "HND", virtually_split=True),
    Config("vs_D1_P2_r0", 8, 128, 16, 1, 0, 2, 0, "HND", virtually_split=True),
    # Qwen3-0.6B-like (1p1d, head_size=64, kv_heads=8, block_size=128)
    Config("qwen3_1p1d", 8, 64, 128, 1, 0, 1, 0, "HND", virtually_split=True),
]


# ═══════════════════════════════════════════════════════════════════════
# Comparison engine
# ═══════════════════════════════════════════════════════════════════════


def run_one(cfg: Config, verbose: bool = True) -> bool:
    """Compare old vs new for one config. Returns True if all match."""
    elem = get_dtype_size(cfg.dtype)
    num_blocks = 4
    num_regions = 2

    # Remote num_heads
    if cfg.is_mla:
        remote_num_heads = 1
    elif cfg.total_kv_heads >= cfg.remote_tp:
        remote_num_heads = cfg.total_kv_heads // cfg.remote_tp
    else:
        remote_num_heads = cfg.total_kv_heads

    # Local num_heads
    if cfg.is_mla:
        local_num_heads = 1
    elif cfg.total_kv_heads >= cfg.local_tp:
        local_num_heads = cfg.total_kv_heads // cfg.local_tp
    else:
        local_num_heads = cfg.total_kv_heads

    if cfg.virtually_split:
        region_c_bytes = 2 * cfg.head_size * elem  # K+V in one region
    else:
        region_c_bytes = cfg.head_size * elem

    # Remote block dimensions
    remote_block_len = remote_num_heads * cfg.block_size * region_c_bytes
    remote_block_stride = remote_block_len  # no padding

    # Local block dimensions
    local_block_len = local_num_heads * cfg.block_size * region_c_bytes

    base_addrs = [i * remote_block_stride * num_blocks for i in range(num_regions)]

    # Old: rank_offset computation
    attn_ranks = old_compute_attn_ranks(
        cfg.local_rank,
        cfg.local_tp,
        cfg.remote_tp,
        cfg.total_kv_heads,
        cfg.is_mla,
    )
    num_attn_reads = len(attn_ranks)
    rank_offset_factor = old_compute_rank_offset_factor(
        cfg.local_rank,
        cfg.local_tp,
        cfg.remote_tp,
        cfg.total_kv_heads,
        cfg.is_mla,
        attn_ranks,
    )

    layout = KVCacheLayout.from_layout_string(cfg.layout_str)

    all_ok = True

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Config: {cfg.name}")
        print(
            f"  total_kv_heads={cfg.total_kv_heads}, head_size={cfg.head_size}, "
            f"block_size={cfg.block_size}"
        )
        print(
            f"  local_tp={cfg.local_tp}, local_rank={cfg.local_rank}, "
            f"remote_tp={cfg.remote_tp}, remote_rank={cfg.remote_rank}"
        )
        print(f"  virtually_split={cfg.virtually_split}, layout={cfg.layout_str}")
        print(
            f"  remote_num_heads={remote_num_heads}, local_num_heads={local_num_heads}"
        )
        print(
            f"  remote_block_len={remote_block_len}, local_block_len={local_block_len}"
        )
        print(
            f"  rank_offset_factor={rank_offset_factor}, "
            f"num_attn_reads={num_attn_reads}"
        )

    for region_idx in range(num_regions):
        base_addr = base_addrs[region_idx]

        # ─── Function 1: _make_remote_meta_4d ───
        region_len = remote_block_len // 2 if cfg.virtually_split else remote_block_len
        meta_4d = new_make_remote_meta_4d(
            num_blocks=num_blocks,
            remote_num_heads=remote_num_heads,
            block_size=cfg.block_size,
            tokens_per_state=1,
            block_len_bytes=region_len,
            block_stride_bytes=remote_block_stride,
            dtype=cfg.dtype,
            layout=layout,
        )

        expected_shape = (
            num_blocks,
            remote_num_heads,
            cfg.block_size,
            region_len // (remote_num_heads * cfg.block_size * elem),
        )
        shape_ok = tuple(meta_4d.shape) == expected_shape
        stride_B_bytes = meta_4d.stride(_DIM4_B) * elem
        stride_B_ok = stride_B_bytes == remote_block_stride

        if verbose:
            print(f"\n  [Region {region_idx}] Function 1: _make_remote_meta_4d")
            print(
                f"    shape:    {tuple(meta_4d.shape)}  expected: {expected_shape}  "
                f"{'OK' if shape_ok else 'MISMATCH!'}"
            )
            ok_str = "OK" if stride_B_ok else "MISMATCH!"
            print(
                f"    stride(B): {stride_B_bytes} bytes"
                f"  expected: {remote_block_stride}"
                f"  {ok_str}"
            )
            print(f"    strides (elems): {meta_4d.stride()}")

        if not shape_ok or not stride_B_ok:
            all_ok = False

        # ─── Function 2: slice_for_tp_transfer ───
        slices = new_slice_for_tp_transfer(
            meta_4d,
            my_tp=cfg.remote_tp,
            my_rank=cfg.remote_rank,
            other_tp=cfg.local_tp,
            other_rank=cfg.local_rank,
            total_num_kv_heads=cfg.total_kv_heads,
        )

        # Old: rank_offset = rank_offset_factor * remote_kv_block_len
        old_local_bl = old_get_backend_aware_kv_block_len(
            local_block_len, cfg.virtually_split
        )
        old_remote_kv_bl = old_local_bl // cfg.block_size_ratio
        old_rank_offset = rank_offset_factor * old_remote_kv_bl

        if not slices:
            if verbose:
                print("    slice_for_tp_transfer: [] (no overlap)")
                if old_rank_offset == 0 and num_attn_reads == 0:
                    print("    Old code would also skip → OK")
            continue

        view = slices[0]
        new_offset_bytes = view.storage_offset() * elem
        new_payload = prod(view.shape[1:]) * elem

        # Old payload
        old_payload_k = old_local_bl // cfg.block_size_ratio
        if cfg.block_size_ratio > 1:
            old_payload_k = old_payload_k  # already divided
        old_payload_k = old_payload_k // num_attn_reads

        offset_ok = new_offset_bytes == old_rank_offset
        payload_ok = new_payload == old_payload_k

        if verbose:
            print(f"\n  [Region {region_idx}] Function 2: slice_for_tp_transfer")
            print(f"    view shape: {tuple(view.shape)}")
            print(
                f"    new offset (bytes): {new_offset_bytes}  "
                f"old rank_offset: {old_rank_offset}  "
                f"{'OK' if offset_ok else 'MISMATCH!'}"
            )
            print(
                f"    new K payload: {new_payload}  old K payload: {old_payload_k}  "
                f"{'OK' if payload_ok else 'MISMATCH!'}"
            )

        if not offset_ok or not payload_ok:
            all_ok = False

        # ─── Function 3: _view_to_descriptors (K) ───
        new_k_descs = new_view_to_descriptors(view, base_addr, 0)

        old_k_descs = []
        for block_id in range(num_blocks):
            block_offset = block_id * remote_block_stride
            addr = base_addr + block_offset + old_rank_offset
            old_k_descs.append((addr, old_payload_k, 0))

        k_match = new_k_descs == old_k_descs
        if verbose:
            print(f"\n  [Region {region_idx}] Function 3: _view_to_descriptors (K)")
            print(f"    K descriptors match: {'OK' if k_match else 'MISMATCH!'}")
            if not k_match:
                for j in range(min(len(old_k_descs), len(new_k_descs))):
                    if old_k_descs[j] != new_k_descs[j]:
                        print(
                            f"      desc[{j}] old={old_k_descs[j]} new={new_k_descs[j]}"
                        )

        if not k_match:
            all_ok = False

        # ─── Function 3b: _view_to_descriptors (V, if virtually_split) ───
        if cfg.virtually_split:
            new_v_base = base_addr + remote_block_len // 2
            new_v_descs = new_view_to_descriptors(view, new_v_base, 0)

            old_v_payload = (
                old_get_backend_aware_kv_block_len(
                    local_block_len, cfg.virtually_split, first_split=False
                )
                // num_attn_reads
            )

            old_v_descs = []
            for block_id in range(num_blocks):
                block_offset = block_id * remote_block_stride
                addr = base_addr + block_offset + old_rank_offset
                v_addr = addr + remote_block_len // 2
                old_v_descs.append((v_addr, old_v_payload, 0))

            v_match = new_v_descs == old_v_descs
            if verbose:
                print(
                    f"\n  [Region {region_idx}] Function 3b: _view_to_descriptors (V)"
                )
                print(f"    V descriptors match: {'OK' if v_match else 'MISMATCH!'}")
                if not v_match:
                    for j in range(min(len(old_v_descs), len(new_v_descs))):
                        if old_v_descs[j] != new_v_descs[j]:
                            print(
                                f"      desc[{j}]"
                                f" old={old_v_descs[j]}"
                                f" new={new_v_descs[j]}"
                            )

            if not v_match:
                all_ok = False

    return all_ok


def main():
    passed = 0
    failed = 0
    for cfg in CONFIGS:
        ok = run_one(cfg, verbose=True)
        if ok:
            passed += 1
            print(f"\n  >>> {cfg.name}: ALL FUNCTIONS MATCH <<<")
        else:
            failed += 1
            print(f"\n  >>> {cfg.name}: MISMATCH DETECTED <<<")

    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {passed} passed, {failed} failed out of {len(CONFIGS)}")
    print(f"{'=' * 70}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
