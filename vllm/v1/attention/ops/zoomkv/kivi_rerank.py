# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KIVI dense/sparse token rerank (PyTorch + optional CUDA)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from vllm.v1.attention.ops.zoomkv.quant_pack import (
    _unpack_along_last_dim,
    chunk_minmax_to_scale_zero,
)


def dequant_kcache_4bit(
    packed: torch.Tensor,
    chunk_min: torch.Tensor,
    chunk_max: torch.Tensor,
    chunk_ids: torch.Tensor,
    group_size: int,
    bits: int = 4,
) -> torch.Tensor:
    assert packed.dim() == 5, (
        f"packed_K must be chunk-major [bs, kv, n_chunks, n_pack, g]; "
        f"got shape={tuple(packed.shape)}"
    )
    bs, kv, n_chunks_pkd, feat_per_int, pkd_group = packed.shape
    assert pkd_group == group_size
    D = chunk_min.shape[-1]
    nk = chunk_ids.shape[-1]
    scale, mn = chunk_minmax_to_scale_zero(chunk_min, chunk_max, bits=bits)

    chunk_ids_c = chunk_ids.clamp(min=0, max=max(n_chunks_pkd - 1, 0))
    chunk_gather_idx = (
        chunk_ids_c.unsqueeze(-1)
        .unsqueeze(-1)
        .expand(bs, kv, nk, feat_per_int, group_size)
    )
    packed_chunks = torch.gather(packed, 2, chunk_gather_idx)
    packed_gather = packed_chunks.transpose(-1, -2).contiguous()
    codes = _unpack_along_last_dim(packed_gather, bits=bits)

    chunk_gather_min = chunk_ids_c.unsqueeze(-1).expand(bs, kv, nk, D)
    scale_c = torch.gather(scale, 2, chunk_gather_min).unsqueeze(3)
    mn_c = torch.gather(mn, 2, chunk_gather_min).unsqueeze(3)
    return codes.to(scale_c.dtype) * scale_c + mn_c


def partial_chunk_kivi_qk_ref(
    chunk_ids: torch.Tensor,
    dense_mask: torch.Tensor,
    packed_K: torch.Tensor,
    chunk_min: torch.Tensor,
    chunk_max: torch.Tensor,
    raw_q: torch.Tensor,
    *,
    group_size: int,
    dense_topk: int,
    sparse_topk: int,
    bits: int = 4,
    out_scores: torch.Tensor | None = None,
    out_indices: torch.Tensor | None = None,
    neg_inf: float = -3.0e38,
) -> tuple[torch.Tensor, torch.Tensor]:
    bs, kv, nk = chunk_ids.shape
    D = chunk_min.shape[-1]
    g = group_size

    K_chunks = dequant_kcache_4bit(
        packed_K,
        chunk_min,
        chunk_max,
        chunk_ids,
        group_size=g,
        bits=bits,
    )
    tok_scores = (K_chunks * raw_q.view(bs, kv, 1, 1, D)).sum(dim=-1)

    sorted_scores, sorted_pos = tok_scores.topk(g, dim=-1, largest=True)
    offs = torch.arange(g, device=chunk_ids.device, dtype=torch.int64)
    token_ids = chunk_ids.unsqueeze(-1) * g + offs
    sorted_tok = torch.gather(token_ids, -1, sorted_pos)

    k_eff = torch.where(
        dense_mask,
        torch.full((), int(dense_topk), device=chunk_ids.device, dtype=torch.int64),
        torch.full((), int(sparse_topk), device=chunk_ids.device, dtype=torch.int64),
    )
    rank = torch.arange(g, device=chunk_ids.device, dtype=torch.int64).view(1, 1, 1, g)
    keep = rank < k_eff.unsqueeze(-1)
    masked_scores = sorted_scores.masked_fill(~keep, neg_inf).to(
        out_scores.dtype if out_scores is not None else torch.float32,
    )
    masked_idx = sorted_tok.masked_fill(~keep, 0)

    flat_scores = masked_scores.reshape(bs, kv, nk * g)
    flat_idx = masked_idx.reshape(bs, kv, nk * g)

    if out_scores is None:
        out_scores = flat_scores.contiguous()
    else:
        out_scores.copy_(flat_scores)
    if out_indices is None:
        out_indices = flat_idx.contiguous()
    else:
        out_indices.copy_(flat_idx)
    return out_scores, out_indices


@lru_cache
def _try_load_kivi_cuda() -> Any | None:
    """Best-effort JIT load of the KIVI CUDA extension (D in {128,256})."""
    try:
        import vllm._zoomkv_C as kernel

        if hasattr(kernel, "partial_chunk_kivi_qk_dense_sparse"):
            return kernel
    except ImportError:
        pass
    try:
        from torch.utils.cpp_extension import load
    except ImportError:
        return None
    cu_path = Path(__file__).with_name("kivi_qk_dot.cu")
    if not cu_path.exists():
        return None
    try:
        return load(
            name="vllm_zoomkv_kivi_qk_dot",
            sources=[str(cu_path)],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    except Exception:
        return None


def partial_chunk_kivi_qk(
    chunk_ids: torch.Tensor,
    dense_mask: torch.Tensor,
    packed_K: torch.Tensor,
    chunk_min: torch.Tensor,
    chunk_max: torch.Tensor,
    raw_q: torch.Tensor,
    *,
    group_size: int,
    dense_topk: int,
    sparse_topk: int,
    bits: int = 4,
    out_scores: torch.Tensor | None = None,
    out_indices: torch.Tensor | None = None,
    prefer_cuda: bool = True,
    strict: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch to CUDA when available; otherwise PyTorch reference."""
    strict = (
        os.environ.get("VLLM_ZOOMKV_STRICT_KERNELS", "0") == "1"
        if strict is None
        else strict
    )
    head_dim = int(chunk_min.shape[-1])
    use_cuda = (
        prefer_cuda
        and chunk_ids.is_cuda
        and head_dim in (128, 256)
        and group_size in (8, 16)
        and bits == 4
        and raw_q.dtype == torch.bfloat16
    )
    if use_cuda:
        kernel = _try_load_kivi_cuda()
        if kernel is not None:
            bs, kv, nk = chunk_ids.shape
            if out_scores is None:
                out_scores = torch.empty(
                    bs,
                    kv,
                    nk * group_size,
                    dtype=torch.float32,
                    device=chunk_ids.device,
                )
            if out_indices is None:
                out_indices = torch.empty(
                    bs, kv, nk * group_size, dtype=torch.int64, device=chunk_ids.device
                )
            kernel.partial_chunk_kivi_qk_dense_sparse(
                chunk_ids.to(torch.int64),
                dense_mask.bool(),
                packed_K,
                chunk_min.to(torch.bfloat16),
                chunk_max.to(torch.bfloat16),
                raw_q.to(torch.bfloat16),
                int(dense_topk),
                int(sparse_topk),
                int(group_size),
                out_scores,
                out_indices,
            )
            return out_scores, out_indices
        if strict:
            raise RuntimeError("ZoomKV strict mode: KIVI CUDA kernel unavailable")

    return partial_chunk_kivi_qk_ref(
        chunk_ids,
        dense_mask,
        packed_K,
        chunk_min,
        chunk_max,
        raw_q,
        group_size=group_size,
        dense_topk=dense_topk,
        sparse_topk=sparse_topk,
        bits=bits,
        out_scores=out_scores,
        out_indices=out_indices,
    )
