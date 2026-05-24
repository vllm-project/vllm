# SPDX-License-Identifier: Apache-2.0
"""PN50 — Fused split/reshape/cat/.contiguous() Triton kernel for Qwen3.5/3.6
GDN (Gated Linear Attention) projection prelude.

Backport of SGLang PR #21019 (merged 2026-03-23, commit 5bdc07d). Author of
the original kernel: Yuan Luo (@yuan-luo). Apache-2.0, full attribution
preserved below.

What this replaces
------------------
In `vllm/model_executor/layers/mamba/gdn_linear_attn.py:562-566`, the
Qwen3.5/3.6 contiguous-projection branch performs:

    mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
    z = z.reshape(z.size(0), -1, self.head_v_dim)
    b, a = ba.chunk(2, dim=-1)
    b = b.contiguous()
    a = a.contiguous()

This is ~5-6 separate kernel launches plus 2 explicit copies. PN50 fuses
this into a single Triton kernel with grid `(batch, num_heads_qk)`.

Claimed gain (SGLang H200 Qwen3.5-35B-A3B tp=2): +7.4% TPS, -10.8% TTFT,
-31.2% ITL P95. On Genesis 27B Lorbus INT4 + TQ k8v4 + MTP K=3 expect
modest gain (memory-bound layer; A5000 PCIe slower than H200).

Numerical safety
----------------
Pure data-copy kernel — no math, no reductions. Cannot introduce any
numerical drift vs the unfused PyTorch chain. Same dtype in/out. Layout
of output is identical to the unfused path (same downstream consumer in
the GatedDeltaNet attention).

Shape constraints
-----------------
- num_heads_v % num_heads_qk == 0 (V_PER_GROUP integer)
- head_qk and head_v must be powers of 2 ≥ 16 (Triton tl.arange limit)
- mixed_qkvz/mixed_ba must be contiguous (kernel uses scalar pointer arith)

For Genesis 27B Lorbus (head_k=head_v=128, num_heads_qk=2/tp, num_heads_v=16/tp):
  V_PER_GROUP = 8, V_PER_GROUP * HEAD_V = 1024 — fits Triton block limits.

Genesis-side guards
-------------------
The wrapper `fused_qkvzba_split_reshape_cat_contiguous` adds:
- input contiguity check (fall-through to PyTorch if non-contiguous)
- shape sanity check (V_PER_GROUP integer, head_qk/v power-of-2)
- explicit per-call try/except → fall through to original PyTorch path
  on any kernel failure (no production hang)

Author: Sandermage (Sander) Barzov Aleksandr backport. Original kernel
credit: Yuan Luo / SGLang.
"""
from __future__ import annotations

import logging

import torch

log = logging.getLogger("genesis.kernels.pn50")

_TRITON_OK = False
try:
    import triton
    import triton.language as tl
    _TRITON_OK = True
except Exception as _e:
    log.warning("PN50: Triton not importable (%s); kernel will be skipped", _e)


if _TRITON_OK:
    @triton.jit
    def _pn50_fused_qkvzba_split_reshape_cat_contiguous_kernel(
        mixed_qkv,
        z,
        b,
        a,
        mixed_qkvz,
        mixed_ba,
        NUM_HEADS_QK: tl.constexpr,
        NUM_HEADS_V: tl.constexpr,
        HEAD_QK: tl.constexpr,
        HEAD_V: tl.constexpr,
    ):
        i_bs, i_qk = tl.program_id(0), tl.program_id(1)

        V_PER_GROUP: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK

        # Input dimensions (contiguous layout)
        TOTAL_Q: tl.constexpr = NUM_HEADS_QK * HEAD_QK
        TOTAL_K: tl.constexpr = NUM_HEADS_QK * HEAD_QK
        TOTAL_V: tl.constexpr = NUM_HEADS_V * HEAD_V
        TOTAL_QKVZ: tl.constexpr = TOTAL_Q + TOTAL_K + TOTAL_V + TOTAL_V
        TOTAL_BA: tl.constexpr = NUM_HEADS_V * 2

        # Output dimensions
        QKV_DIM_T: tl.constexpr = TOTAL_Q + TOTAL_K + TOTAL_V

        blk_q_ptr = (
            mixed_qkvz + i_bs * TOTAL_QKVZ + i_qk * HEAD_QK + tl.arange(0, HEAD_QK)
        )
        blk_k_ptr = (
            mixed_qkvz + i_bs * TOTAL_QKVZ + TOTAL_Q
            + i_qk * HEAD_QK + tl.arange(0, HEAD_QK)
        )
        blk_v_ptr = (
            mixed_qkvz + i_bs * TOTAL_QKVZ + TOTAL_Q + TOTAL_K
            + i_qk * V_PER_GROUP * HEAD_V + tl.arange(0, V_PER_GROUP * HEAD_V)
        )
        blk_z_ptr = (
            mixed_qkvz + i_bs * TOTAL_QKVZ + TOTAL_Q + TOTAL_K + TOTAL_V
            + i_qk * V_PER_GROUP * HEAD_V + tl.arange(0, V_PER_GROUP * HEAD_V)
        )

        blk_q_st_ptr = (
            mixed_qkv + i_bs * QKV_DIM_T + i_qk * HEAD_QK + tl.arange(0, HEAD_QK)
        )
        blk_k_st_ptr = (
            mixed_qkv + i_bs * QKV_DIM_T + NUM_HEADS_QK * HEAD_QK
            + i_qk * HEAD_QK + tl.arange(0, HEAD_QK)
        )
        blk_v_st_ptr = (
            mixed_qkv + i_bs * QKV_DIM_T + NUM_HEADS_QK * HEAD_QK * 2
            + i_qk * V_PER_GROUP * HEAD_V + tl.arange(0, V_PER_GROUP * HEAD_V)
        )
        blk_z_st_ptr = (
            z + i_bs * NUM_HEADS_V * HEAD_V
            + i_qk * V_PER_GROUP * HEAD_V + tl.arange(0, V_PER_GROUP * HEAD_V)
        )

        tl.store(blk_q_st_ptr, tl.load(blk_q_ptr))
        tl.store(blk_k_st_ptr, tl.load(blk_k_ptr))
        tl.store(blk_v_st_ptr, tl.load(blk_v_ptr))
        tl.store(blk_z_st_ptr, tl.load(blk_z_ptr))

        for i in tl.static_range(V_PER_GROUP):
            blk_b_ptr = mixed_ba + i_bs * TOTAL_BA + i_qk * V_PER_GROUP + i
            blk_b_st_ptr = b + i_bs * NUM_HEADS_V + i_qk * V_PER_GROUP + i
            tl.store(blk_b_st_ptr, tl.load(blk_b_ptr))

        for i in tl.static_range(V_PER_GROUP):
            blk_a_ptr = mixed_ba + i_bs * TOTAL_BA + NUM_HEADS_V + i_qk * V_PER_GROUP + i
            blk_a_st_ptr = a + i_bs * NUM_HEADS_V + i_qk * V_PER_GROUP + i
            tl.store(blk_a_st_ptr, tl.load(blk_a_ptr))


def _fallback_pytorch(
    mixed_qkvz: torch.Tensor,
    mixed_ba: torch.Tensor,
    num_heads_qk: int,
    num_heads_v: int,
    head_qk: int,
    head_v: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference PyTorch implementation. Used as fallback AND in TDD."""
    qkv_size = num_heads_qk * head_qk * 2 + num_heads_v * head_v
    z_size = num_heads_v * head_v
    mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
    z = z.reshape(z.size(0), num_heads_v, head_v)
    b, a = mixed_ba.chunk(2, dim=-1)
    return mixed_qkv.contiguous(), z.contiguous(), b.contiguous(), a.contiguous()


def fused_qkvzba_split_reshape_cat_contiguous(
    mixed_qkvz: torch.Tensor,
    mixed_ba: torch.Tensor,
    num_heads_qk: int,
    num_heads_v: int,
    head_qk: int,
    head_v: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused split/reshape/cat for CONTIGUOUS input format (Qwen3.5/3.6).

    Input:
        mixed_qkvz: [batch, all_q | all_k | all_v | all_z] contiguous
        mixed_ba:   [batch, all_b | all_a] contiguous

    Output (matches the unfused PyTorch chain bit-for-bit):
        mixed_qkv: [batch, all_q | all_k | all_v]  (z stripped)
        z:         [batch, num_heads_v, head_v]
        b:         [batch, num_heads_v]
        a:         [batch, num_heads_v]

    Falls through to PyTorch reference on:
      * Triton not available
      * Non-contiguous input (kernel uses scalar pointer arith)
      * Non-CUDA device
      * V_PER_GROUP not integer
      * Any kernel launch failure (caught + re-routed)
    """
    # Defensive guards — fall through to PyTorch on any constraint violation.
    if not _TRITON_OK:
        return _fallback_pytorch(mixed_qkvz, mixed_ba, num_heads_qk,
                                 num_heads_v, head_qk, head_v)
    if not (mixed_qkvz.is_cuda and mixed_ba.is_cuda):
        return _fallback_pytorch(mixed_qkvz, mixed_ba, num_heads_qk,
                                 num_heads_v, head_qk, head_v)
    if not (mixed_qkvz.is_contiguous() and mixed_ba.is_contiguous()):
        return _fallback_pytorch(mixed_qkvz, mixed_ba, num_heads_qk,
                                 num_heads_v, head_qk, head_v)
    if num_heads_v % num_heads_qk != 0:
        return _fallback_pytorch(mixed_qkvz, mixed_ba, num_heads_qk,
                                 num_heads_v, head_qk, head_v)
    # head_qk / head_v must be power-of-2 for Triton tl.arange
    if (head_qk & (head_qk - 1)) != 0 or (head_v & (head_v - 1)) != 0:
        return _fallback_pytorch(mixed_qkvz, mixed_ba, num_heads_qk,
                                 num_heads_v, head_qk, head_v)

    batch, seq_len = mixed_qkvz.shape[0], 1
    qkv_dim_t = num_heads_qk * head_qk * 2 + num_heads_v * head_v

    try:
        mixed_qkv = torch.empty(
            [batch * seq_len, qkv_dim_t],
            dtype=mixed_qkvz.dtype, device=mixed_qkvz.device,
        )
        z = torch.empty(
            [batch * seq_len, num_heads_v, head_v],
            dtype=mixed_qkvz.dtype, device=mixed_qkvz.device,
        )
        b = torch.empty(
            [batch * seq_len, num_heads_v],
            dtype=mixed_ba.dtype, device=mixed_ba.device,
        )
        a = torch.empty_like(b)
        grid = (batch * seq_len, num_heads_qk)
        _pn50_fused_qkvzba_split_reshape_cat_contiguous_kernel[grid](
            mixed_qkv, z, b, a,
            mixed_qkvz, mixed_ba,
            num_heads_qk, num_heads_v, head_qk, head_v,
            num_warps=1, num_stages=3,
        )
        return mixed_qkv, z, b, a
    except Exception as e:
        log.warning(
            "PN50 kernel raised (%s); falling through to PyTorch reference. "
            "Shape: qkvz=%s ba=%s qk=%d v=%d",
            e, tuple(mixed_qkvz.shape), tuple(mixed_ba.shape), num_heads_qk, num_heads_v,
        )
        return _fallback_pytorch(mixed_qkvz, mixed_ba, num_heads_qk,
                                 num_heads_v, head_qk, head_v)
