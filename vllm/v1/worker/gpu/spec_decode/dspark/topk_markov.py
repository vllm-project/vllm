# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy candidate-pruned Markov sampling, fused across draft positions."""

from typing import TYPE_CHECKING

import torch

from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead


def can_fuse_markov_walk(
    head: "DSparkMarkovHead", logits: torch.Tensor, topk: int
) -> bool:
    """Whether the greedy kernel supports this Markov head and logits."""
    if not current_platform.is_cuda() or not logits.is_cuda:
        return False
    w1 = head.markov_w1.weight
    w2 = getattr(head.markov_w2, "weight", None)
    return (
        isinstance(
            head.markov_w2.quant_method,
            (UnquantizedEmbeddingMethod, UnquantizedLinearMethod),
        )
        and w2 is not None
        and w1.is_cuda
        and w1.device == w2.device == logits.device
        and w1.dtype == w2.dtype == logits.dtype
        and w1.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and w1.ndim == w2.ndim == 2
        and w1.stride(1) == w2.stride(1) == 1
        and w1.shape[1] == w2.shape[1] <= 1024
        and head.markov_w2.tp_size == 1
        and topk <= 128
    )


@triton.jit
def _markov_walk_kernel(
    logits_ptr,
    base_values_ptr,
    base_ids_ptr,
    markov_ids_ptr,
    markov_biases_ptr,
    w1_ptr,
    w2_ptr,
    input_ids_ptr,
    anchors_ptr,
    output_ptr,
    d2t_ptr,
    logits_req_stride: tl.constexpr,
    logits_step_stride: tl.constexpr,
    w1_stride: tl.constexpr,
    w2_stride: tl.constexpr,
    output_stride: tl.constexpr,
    SCALE: tl.constexpr,
    VOCAB: tl.constexpr,
    STEPS: tl.constexpr,
    K: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
    HAS_D2T: tl.constexpr,
):
    req = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_K)
    valid = cols < K
    prev = tl.load(input_ids_ptr + tl.load(anchors_ptr + req)).to(tl.int64)

    for step in range(STEPS):
        base_offset = (req * STEPS + step) * K + cols
        base_ids = tl.load(base_ids_ptr + base_offset, valid, other=0).to(tl.int64)
        bias_ids = tl.load(markov_ids_ptr + prev * K + cols, valid, other=0).to(
            tl.int64
        )
        # The cached score overwrites the online score for overlapping IDs.
        overlap = (
            tl.sum(
                ((base_ids[:, None] == bias_ids[None, :]) & valid[None, :]).to(
                    tl.int32
                ),
                axis=1,
            )
            > 0
        )
        base_valid = valid & ~overlap
        correction = tl.full((BLOCK_K,), 0, tl.float32)
        for start in range(0, RANK, BLOCK_R):
            ranks = start + tl.arange(0, BLOCK_R)
            embed = tl.load(w1_ptr + prev * w1_stride + ranks, ranks < RANK, 0).to(
                tl.float32
            )
            weight = tl.load(
                w2_ptr + base_ids[:, None] * w2_stride + ranks[None, :],
                base_valid[:, None] & (ranks[None, :] < RANK),
                other=0,
            ).to(tl.float32)
            correction += tl.sum(weight * embed[None, :], axis=1)
        base = tl.load(base_values_ptr + base_offset, valid, other=0).to(tl.float32)
        base_scores = (base + correction * SCALE).to(logits_ptr.dtype.element_ty)
        bias_base = tl.load(
            logits_ptr + req * logits_req_stride + step * logits_step_stride + bias_ids,
            valid,
            other=0,
        ).to(tl.float32)
        bias = tl.load(markov_biases_ptr + prev * K + cols, valid, other=0).to(
            tl.float32
        )
        bias_scores = (bias_base + bias).to(logits_ptr.dtype.element_ty)
        scores = tl.cat(
            tl.where(base_valid, base_scores.to(tl.float32), -float("inf")),
            tl.where(valid, bias_scores.to(tl.float32), -float("inf")),
        )
        ids = tl.cat(base_ids, bias_ids)
        # Dense argmax selects the first NaN, otherwise the lowest ID at a tie.
        nan_ids = tl.min(tl.where(scores != scores, ids, VOCAB), axis=0)
        maximum = tl.max(tl.where(scores != scores, -float("inf"), scores), axis=0)
        token = tl.min(tl.where(scores == maximum, ids, VOCAB), axis=0)
        token = tl.where(maximum == -float("inf"), 0, token)
        token = tl.where(nan_ids < VOCAB, nan_ids, token)
        if HAS_D2T:
            token += tl.load(d2t_ptr + token)
        tl.store(output_ptr + req * output_stride + step, token)
        prev = token


def markov_walk_topk(
    *,
    base_logits: torch.Tensor,
    base_values: torch.Tensor,
    base_ids: torch.Tensor,
    markov_ids: torch.Tensor,
    markov_biases: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    input_ids: torch.Tensor,
    anchor_indices: torch.Tensor,
    draft_tokens: torch.Tensor,
    d2t: torch.Tensor | None,
    scale: float,
) -> None:
    """Write a greedy chain with one program per request.

    Base logits are [batch, steps, vocab] with contiguous vocabulary rows;
    top-k values/IDs are contiguous [batch, steps, k]. Cached Markov values/IDs
    are contiguous [target_vocab, k]. Weights are unquantized, row-contiguous
    [vocab, rank]; d2t holds optional draft-to-target ID offsets. FP32 reduction
    can round differently from the unfused matrix multiplication.
    """
    batch, steps, k = base_ids.shape
    rank = w1.shape[1]
    _markov_walk_kernel[(batch,)](
        base_logits,
        base_values,
        base_ids,
        markov_ids,
        markov_biases,
        w1,
        w2,
        input_ids,
        anchor_indices,
        draft_tokens,
        d2t,
        base_logits.stride(0),
        base_logits.stride(1),
        w1.stride(0),
        w2.stride(0),
        draft_tokens.stride(0),
        scale,
        base_logits.shape[-1],
        steps,
        k,
        rank,
        triton.next_power_of_2(k),
        min(triton.next_power_of_2(rank), 64),
        d2t is not None,
        num_warps=4,
        enable_fp_fusion=False,
    )
