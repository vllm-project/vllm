# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused DSA (DeepSeek Sparse Attention) indexer top-k for Blackwell (SM100).

Streams KV in tiles and fuses fp8 MQA scoring (tcgen05 UMMA) + an online bucketed
gate + compact top-k, so the ``[num_q, seq_len]`` logit matrix is never
materialized. This module orchestrates the three C++ primitives
(``dsa_litetopk_seed_prep`` / ``_scan`` / ``_select``) and allocates scratch,
mirroring the reference implementation contributed to FlashInfer.

Opt-in via ``VLLM_SPARSE_INDEXER_FUSED=1``; the caller falls back to the dense
``fp8_fp4_mqa_logits`` + ``top_k_per_row_prefill`` path when unsupported.
"""

import functools

import torch

from vllm import _custom_ops as ops
from vllm.utils.deep_gemm import fp8_fp4_mqa_logits

# The kernel is specialized to the GLM/DeepSeek DSA indexer head shape.
_NUM_HEADS = 32
_HEAD_DIM = 128
_NUM_BUCKETS = 256
_SAMPLE_LEN = 8192
_REFRESH_EVERY = 64


@functools.cache
def dsa_litetopk_available() -> bool:
    """True iff the fused kernel was compiled (SM100 build) and registered."""
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability()[0] != 10:  # Blackwell SM100 only
        return False
    return hasattr(torch.ops._C, "dsa_litetopk_seed_prep")


def dsa_litetopk_indexer(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_scales: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk: int,
    out_indices: torch.Tensor,
    *,
    num_buckets: int = _NUM_BUCKETS,
    sample_len: int = _SAMPLE_LEN,
    cand_cap: int | None = None,
    refresh_every: int = _REFRESH_EVERY,
) -> None:
    """Fused indexer top-k. Writes ``topk`` KV indices per row into ``out_indices``.

    Args:
        q: ``[num_q, 32, 128]`` fp8_e4m3 indexer queries.
        kv: ``[seq_len, 128]`` fp8_e4m3 indexer keys (gathered, contiguous).
        kv_scales: ``[seq_len]`` fp32 per-position dequant scales.
        weights: ``[num_q, 32]`` fp32 per-head gate weights (q-scale folded in).
        cu_seqlen_ks: ``[num_q]`` int32 per-row causal start into ``kv``.
        cu_seqlen_ke: ``[num_q]`` int32 per-row causal end into ``kv``.
        topk: number of KV indices to select per row.
        out_indices: ``[num_q, topk]`` int32 output buffer (written in place).
    """
    num_q = q.shape[0]
    seq_len = kv.shape[0]
    dev = q.device
    cap = cand_cap if cand_cap is not None else max(4 * topk, 16384)

    q = q.contiguous()
    kv = kv.contiguous()
    kv_scales = kv_scales.contiguous()
    weights = weights.contiguous().float()

    # Gate calibration sample: score q against a bounded [num_q, sample_len]
    # prefix (<< seq_len). This only sets the gate threshold; recall is
    # guaranteed by the full scan below, so an approximate sample is fine.
    sl = min(sample_len, seq_len)
    ks0 = torch.zeros(num_q, dtype=torch.int32, device=dev)
    ke_s = torch.full((num_q,), sl, dtype=torch.int32, device=dev)
    sample_logits = fp8_fp4_mqa_logits(
        (q, None), (kv[:sl], kv_scales[:sl]), weights, ks0, ke_s, clean_logits=False
    ).contiguous()

    origin = torch.empty(num_q, dtype=torch.float32, device=dev)
    inv_delta = torch.empty(num_q, dtype=torch.float32, device=dev)
    th_bucket = torch.empty(num_q, dtype=torch.int32, device=dev)
    bcount = torch.zeros(num_q, num_buckets, dtype=torch.int32, device=dev)
    cand_val = torch.empty(num_q, cap, dtype=torch.float32, device=dev)
    cand_idx = torch.empty(num_q, cap, dtype=torch.int32, device=dev)
    cand_cnt = torch.empty(num_q, dtype=torch.int32, device=dev)
    out_val = torch.empty(num_q, topk, dtype=torch.float32, device=dev)

    # emit_limit=0: sample calibrates the gate only, emits no seeds; the full
    # scan (cu_start=cu_seqlen_ks) produces candidates with correct indices.
    ops.dsa_litetopk_seed_prep(
        sample_logits,
        num_buckets,
        topk,
        cap,
        0,
        0.0,
        0,
        1,
        origin,
        inv_delta,
        th_bucket,
        bcount,
        cand_val,
        cand_idx,
        cand_cnt,
    )
    ops.dsa_litetopk_scan(
        q,
        kv,
        kv_scales,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        origin,
        inv_delta,
        th_bucket,
        cand_val,
        cand_idx,
        cand_cnt,
        bcount,
        num_buckets,
        topk,
        refresh_every,
        -1,
        0,
        0,
    )
    # GATE4: cand_val is stored in bucket space, so select rebases with the
    # identity affine (origin=0, inv=1); the true origin/inv were folded in at
    # scan time.
    zero = torch.zeros(num_q, dtype=torch.float32, device=dev)
    one = torch.ones(num_q, dtype=torch.float32, device=dev)
    ops.dsa_litetopk_select(
        cand_val,
        cand_idx,
        cand_cnt,
        zero,
        one,
        th_bucket,
        num_buckets,
        topk,
        out_val,
        out_indices,
    )
