# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch wrapper of the FlyDSL NVFP4 sparse prefill attention."""

import functools

import numpy as np
import torch

from vllm.models.minimax_m3.amd.ops.flydsl_utils import (
    LOG2E,
    cu_count,
    run_compiled,
    unit_scale,
)

from . import kernel as _kernel
from .kernel import BLK, HG, TOPK, D


@functools.cache
def _get_kernel(rounds: int):
    return _kernel.compile_nvfp4_prefill_fp16(rounds)


def supports_nvfp4_sparse_prefill(
    num_heads: int, num_kv_heads: int, head_dim: int, topk_blocks: int, block_size: int
) -> bool:
    from vllm.platforms.rocm import on_gfx950

    return (
        on_gfx950()
        and head_dim == D
        and num_heads == num_kv_heads * HG
        and topk_blocks == TOPK
        and block_size == BLK
    )


def minimax_m3_nvfp4_sparse_prefill(
    q: torch.Tensor,  # [T, H, 128] bf16
    kv_cache: torch.Tensor,  # [pages, 2 * Hkv, 128, 72] uint8
    topk: torch.Tensor,  # [Hkv, T, 16] int32 logical blocks, -1 = none
    block_table: torch.Tensor,  # [reqs, max_blocks] int32
    query_req_id: torch.Tensor,  # [T] int32
    query_abs_pos: torch.Tensor,  # [T] int32
    num_kv_heads: int,
    sm_scale: float,
    out: torch.Tensor,  # [T, H, 128] bf16
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> None:
    """Token t attends to the causal tokens of its ``topk`` blocks."""
    n_tok, n_heads, head_dim = q.shape
    if n_tok == 0:
        return
    assert head_dim == D and n_heads == num_kv_heads * HG
    assert q.dtype == torch.bfloat16 and out.dtype == torch.bfloat16
    assert out.shape == q.shape
    assert q.is_contiguous() and out.is_contiguous()
    assert kv_cache.dtype == torch.uint8 and tuple(kv_cache.shape[1:]) == (
        2 * num_kv_heads,
        BLK,
        72,
    )
    assert kv_cache.stride(1) == BLK * 72 and kv_cache.stride(3) == 1
    assert topk.dtype == torch.int32 and tuple(topk.shape) == (
        num_kv_heads,
        n_tok,
        TOPK,
    )
    assert topk.stride(2) == 1 and topk.stride(1) == TOPK
    for t in (block_table, query_req_id, query_abs_pos):
        assert t.dtype == torch.int32 and t.stride(-1) == 1
    assert kv_cache.stride(0) < 2**31
    unit = unit_scale(q.device)
    ks = unit if k_scale is None else k_scale
    vs = unit if v_scale is None else v_scale
    c = float(np.float32(sm_scale) * np.float32(LOG2E))
    # Up to one workgroup per CU the one-round combine is ~1 us faster.
    n_wg = (n_tok + 3) // 4 * num_kv_heads
    rounds = 2 if n_wg > cu_count(q.device) else 1
    run_compiled(
        _get_kernel(rounds),
        q.data_ptr(),
        kv_cache.data_ptr(),
        topk.data_ptr(),
        block_table.data_ptr(),
        query_req_id.data_ptr(),
        query_abs_pos.data_ptr(),
        out.data_ptr(),
        ks.data_ptr(),
        vs.data_ptr(),
        n_tok,
        n_heads,
        num_kv_heads,
        topk.stride(0),
        block_table.stride(0),
        kv_cache.stride(0),
        c,
        torch.cuda.current_stream(q.device),
    )
