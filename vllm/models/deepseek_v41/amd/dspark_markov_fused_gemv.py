# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Markov bias for DSpark sequential draft sampling.

The stock sequential stage runs, per draft step (k=5 steps per decode step,
num_reqs rows each):

    markov_embed = markov_w1[prev]        # nn.Embedding gather   [B, R]
    bias         = markov_w2 @ markov_embed  # replicated bf16 GEMV [B, V]
    logits_i     = base_logits[:, i] + bias  # elementwise add      [B, V]

-- three kernels and a 66 MB weight read (markov_w2 is [129280, 256] bf16,
replicated on every rank) per step. This module fuses all three ops into ONE
Triton kernel per step:

    out[b, v] = bf16( bf16( sum_r w2[v, r] * w1[prev[b], r] ) + base[b, v] )

with the dot accumulated in fp32 at a fixed sequential order (deterministic;
no atomics). The rounding sequence mirrors the stock chain: the GEMV result
is rounded to bf16 exactly like the bf16 GEMM output, then added to the base
logits in fp32 and rounded once (torch bf16 elementwise-add semantics). The
only allowed deviation is the fp32 accumulation ORDER inside the dot, so the
result is NOT guaranteed bit-exact with rocBLAS. The gumbel sampler is
untouched -- the RNG stream is identical.

The draft step is fully captured by the DSpark FULL CUDA graph, and a Triton
kernel launch is capturable like any other, so the fused kernel replays
in-graph (the choice is baked at capture).

Kill switch: VLLM_DISABLE_MARKOV_FUSED_GEMV=1 (read per call, so eager
in-process A/B works). Ineligible shapes (non-bf16/fp16, non-contiguous
weights, TP-sharded w2, mismatched vocab, confidence-head mode) fall back to
the stock chain.

Draft-batch gate: the fusion wins by launch elimination at tiny draft
batches and loses to rocBLAS at large ones -- the kernel's grid is
(v-blocks, batch), so every batch row re-reads the full w2 weight and
its memory traffic scales O(batch * weight bytes), while the stock GEMM
tiles over batch and re-reads w2 O(batch / tile) times. The measured
crossover on MI355X is between 2 and 4 concurrent requests (the ship
shape is 1), so the fusion engages at num draft rows <=
VLLM_DRAFT_SAMPLING_FUSION_MAX_TOKENS (default 2, shared with the
sample-epilogue fusion) and falls back to the stock chain above. Read
per call; under CUDA graphs the choice is baked per captured size.
"""

import os

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

_BLOCK_V = 128
_BLOCK_R = 128


def markov_fused_gemv_disabled() -> bool:
    return os.environ.get("VLLM_DISABLE_MARKOV_FUSED_GEMV", "0") == "1"


def draft_sampling_fusion_max_tokens() -> int:
    """Draft-batch gate shared by the fused draft-sampling levers (this
    markov GEMV and the sample epilogue): engage at num draft rows <=
    this value, fall back to the stock chain above."""
    return int(os.environ.get("VLLM_DRAFT_SAMPLING_FUSION_MAX_TOKENS", "2"))


@triton.jit
def _markov_bias_add_kernel(
    prev_ptr,  # [B] int32/int64 previously sampled token ids
    w1_ptr,  # [V, R] embedding-style table (markov_w1.weight)
    w2_ptr,  # [V, R] replicated lm-head-style table (markov_w2.weight)
    base_ptr,  # [B, V] base-logits slice (row stride base_stride)
    out_ptr,  # [B, V] contiguous output
    vocab_size,
    rank,
    base_stride,
    BLOCK_V: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_v = tl.program_id(0)
    b = tl.program_id(1)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_v = offs_v < vocab_size
    tok = tl.load(prev_ptr + b).to(tl.int64)

    acc = tl.zeros((BLOCK_V,), dtype=tl.float32)
    for r0 in range(0, rank, BLOCK_R):
        offs_r = r0 + tl.arange(0, BLOCK_R)
        mask_r = offs_r < rank
        x = tl.load(w1_ptr + tok * rank + offs_r, mask=mask_r, other=0.0).to(tl.float32)
        w = tl.load(
            w2_ptr + offs_v[:, None].to(tl.int64) * rank + offs_r[None, :],
            mask=mask_v[:, None] & mask_r[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)

    # Round the GEMV to the storage dtype exactly like the stock bf16 GEMM
    # output, then add in fp32 and round once (torch bf16 add semantics).
    out_dtype = w2_ptr.dtype.element_ty
    y = acc.to(out_dtype)
    base = tl.load(
        base_ptr + b.to(tl.int64) * base_stride + offs_v,
        mask=mask_v,
        other=0.0,
    )
    out = (y.to(tl.float32) + base.to(tl.float32)).to(out_dtype)
    tl.store(out_ptr + b.to(tl.int64) * vocab_size + offs_v, out, mask=mask_v)


def fused_markov_bias_add(
    prev_ids: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    base_logits_slice: torch.Tensor,
) -> torch.Tensor | None:
    """One fused kernel for ``markov_embed -> markov_bias -> + base``.

    Returns the biased logits ``[B, V]`` (contiguous, same layout as the
    stock ``base_logits[:, i] + bias`` result), or ``None`` when the fused
    kernel does not apply -- the caller must then run the stock chain.
    """
    if markov_fused_gemv_disabled():
        logger.warning_once(
            "markov fused GEMV is DISABLED (env override "
            "VLLM_DISABLE_MARKOV_FUSED_GEMV=1); using the stock "
            "embed + GEMV + add chain"
        )
        return None
    max_tokens = draft_sampling_fusion_max_tokens()
    if prev_ids.shape[0] > max_tokens:
        logger.warning_once(
            "markov fused GEMV gated off (draft rows %d > %d, "
            "VLLM_DRAFT_SAMPLING_FUSION_MAX_TOKENS; the fused kernel "
            "re-reads w2 per row, so large draft batches keep the stock "
            "GEMM); using the stock embed + GEMV + add chain",
            prev_ids.shape[0],
            max_tokens,
        )
        return None
    if (
        prev_ids.ndim != 1
        or prev_ids.dtype not in (torch.int32, torch.int64)
        or w1.ndim != 2
        or w2.ndim != 2
        or base_logits_slice.ndim != 2
        or w1.shape != w2.shape
        or w1.dtype != w2.dtype
        or w1.dtype != base_logits_slice.dtype
        or w1.dtype not in (torch.bfloat16, torch.float16)
        or not w1.is_contiguous()
        or not w2.is_contiguous()
        or base_logits_slice.stride(-1) != 1
        or base_logits_slice.shape[-1] != w2.shape[0]
        or base_logits_slice.shape[0] != prev_ids.shape[0]
        or not prev_ids.is_cuda
        or w1.device != prev_ids.device
    ):
        return None

    vocab_size, rank = w2.shape
    batch = prev_ids.shape[0]
    out = torch.empty(
        (batch, vocab_size), dtype=base_logits_slice.dtype, device=w2.device
    )
    grid = (triton.cdiv(vocab_size, _BLOCK_V), batch)
    _markov_bias_add_kernel[grid](
        prev_ids,
        w1,
        w2,
        base_logits_slice,
        out,
        vocab_size,
        rank,
        base_logits_slice.stride(0),
        BLOCK_V=_BLOCK_V,
        BLOCK_R=_BLOCK_R,
        num_warps=4,
    )
    logger.warning_once(
        "markov fused GEMV engaged (fused embed-gather + w2 GEMV + "
        "base-logits add in one Triton kernel, draft sequential sampling; "
        "disable var VLLM_DISABLE_MARKOV_FUSED_GEMV)"
    )
    return out
