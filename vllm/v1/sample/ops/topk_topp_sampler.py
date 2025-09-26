# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from datetime import timedelta
from tkinter import NO
from typing import Optional

import torch
import torch.nn as nn
import triton
import triton.language as tl
import time
from packaging import version

from vllm import envs
from vllm.config import LogprobsMode
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

try:
    import flashinfer.sampling
    is_flashinfer_available = True
except ImportError:
    is_flashinfer_available = False
    
def g_str(s):
    return "\033[32m" + s + "\033[0m"
def r_str(s):
    return "\033[31m" + s + "\033[0m"
def b_str(s):
    return "\033[34m" + s + "\033[0m"
def y_str(s):
    return "\033[33m" + s + "\033[0m"
def c_str(s):
    return "\033[36m" + s + "\033[0m"
def m_str(s):
    return "\033[35m" + s + "\033[0m"


class TopKTopPSampler(nn.Module):
    """
    Module that performs optional top-k and top-p filtering followed by
    weighted random sampling of logits.

    Implementations may update the logits tensor in-place.
    """

    def __init__(self, logprobs_mode: LogprobsMode = "raw_logprobs") -> None:
        super().__init__()
        self.logprobs_mode = logprobs_mode
        # flashinfer optimization does not apply if intermediate
        # logprobs/logits after top_k/top_p need to be returned
        if logprobs_mode not in ("processed_logits", "processed_logprobs"
                                 ) and current_platform.is_cuda():
            if is_flashinfer_available:
                flashinfer_version = flashinfer.__version__
                if version.parse(flashinfer_version) < version.parse("0.2.3"):
                    logger.warning_once(
                        "FlashInfer version >= 0.2.3 required. "
                        "Falling back to default sampling implementation.")
                    self.forward = self.forward_native
                elif envs.VLLM_USE_FLASHINFER_SAMPLER is not False:
                    # NOTE(woosuk): The V0 sampler doesn't use FlashInfer for
                    # sampling unless VLLM_USE_FLASHINFER_SAMPLER=1 (i.e., by
                    # default it is unused). For backward compatibility, we set
                    # `VLLM_USE_FLASHINFER_SAMPLER` as None by default and
                    # interpret it differently in V0 and V1 samplers: In V0,
                    # None means False, while in V1, None means True. This is
                    # why we use the condition
                    # `envs.VLLM_USE_FLASHINFER_SAMPLER is not False` here.
                    logger.info_once(
                        "Using FlashInfer for top-p & top-k sampling.")
                    self.forward = self.forward_cuda
                else:
                    logger.warning_once(
                        "FlashInfer is available, but it is not enabled. "
                        "Falling back to the PyTorch-native implementation of "
                        "top-p & top-k sampling. For the best performance, "
                        "please set VLLM_USE_FLASHINFER_SAMPLER=1.")
                    self.forward = self.forward_native
            else:
                logger.warning_once(
                    "FlashInfer is not available. Falling back to the PyTorch-"
                    "native implementation of top-p & top-k sampling. For the "
                    "best performance, please install FlashInfer.")
                self.forward = self.forward_native
        else:
            self.forward = self.forward_native

        self.apply_top_k_top_p = apply_top_k_top_p

    def forward_native(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: Optional[torch.Tensor],
        p: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        PyTorch-native implementation of top-k and top-p sampling.

        The logits tensor may be updated in-place.
        """
        logits = self.apply_top_k_top_p(logits, k, p)
        logits_to_return = None
        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        return random_sample(probs, generators), logits_to_return

    def forward_cuda(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: Optional[torch.Tensor],
        p: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """More optimized implementation for top-k and top-p sampling."""
        # We prefer `random_sample` over `flashinfer_sample` when sorting is
        # not needed. This is because `random_sample` does not require
        # CPU-GPU synchronization while `flashinfer_sample` does.
        if (k is None and p is None) or generators:
            if generators:
                logger.warning_once("FlashInfer 0.2.3+ does not support "
                                    "per-request generators. Falling back to "
                                    "PyTorch-native implementation.")
            return self.forward_native(logits, generators, k, p)
        assert self.logprobs_mode not in (
            "processed_logits", "processed_logprobs"
        ), "FlashInfer does not support returning logits/logprobs"
        # flashinfer sampling functions expect contiguous logits.
        # In flex_attn/triton_attn fp32 inference, logits can be non-contiguous
        # because of slicing operation in logits_processor.
        return flashinfer_sample(logits.contiguous(), k, p, generators), None


def original_apply_top_k_top_p(
    logits: torch.Tensor,
    k: Optional[torch.Tensor],
    p: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply top-k and top-p masks to the logits.
    """
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)
    
    if p is None:
        if k is None:
            return logits

        # Avoid sorting vocab for top-k only case.
        logits = apply_top_k_only(logits, k)
    else:
        if k is not None:
            # Apply top-k.
            top_k_mask = logits_sort.size(1) - k.to(torch.long)  # shape: B
            # Get all the top_k values.
            top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
            top_k_mask = logits_sort < top_k_mask
            logits_sort.masked_fill_(top_k_mask, -float("inf"))


        if p is not None:
            # Apply top-p.
            probs_sort = logits_sort.softmax(dim=-1)
            probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
            top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
            # at least one
            top_p_mask[:, -1] = False
            logits_sort.masked_fill_(top_p_mask, -float("inf"))

        # Re-sort the probabilities.
        logits = logits_sort.scatter(dim=-1, index=logits_idx, src=logits_sort)
    return logits

@triton.jit
def _topk_kernel(LOGITS, PROBS, K, B, 
                      N: tl.constexpr,
                      BLOCK_SIZE: tl.constexpr,
                      NUM_TILES: tl.constexpr):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for row_id in tl.range(pid, B, num_programs):
        k_pivot = -float('inf')
        p_pivot = -float('inf')

        LOGITS_ROW = LOGITS + row_id * N
        PROBS_ROW = PROBS + pid * N

        search_addr = LOGITS_ROW
        search_range = N
        search_iters = NUM_TILES

        max_logit = -float('inf')

        k = tl.load(K + row_id)
        if not (k == N): # All tokens are valid
            min_logit = float('inf')

            # Zeroth pass: Compute avg and std from a sample block
            # May produce incorrect results if N < BLOCK_SIZE
            offs = tl.arange(0, BLOCK_SIZE)
            mask_n = offs < N
            logits_blk = tl.load(LOGITS_ROW + offs, mask=mask_n, other=0.0)
            avg_logit = tl.sum(logits_blk) / N
            sq_avg_logit = tl.sum(logits_blk * logits_blk) / N
            std_logit = tl.sqrt(sq_avg_logit - avg_logit * avg_logit)

            outlier_pivot = avg_logit + 2.8 * std_logit
            num_outliers = tl.zeros((), dtype=tl.uint32)
            # First pass: compute max and min logits and gather outliers
            for i in range(0,search_iters):
                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask_n = offs_n < search_range
                logits_blk = tl.load(search_addr + offs_n, mask=mask_n, other=avg_logit)

                max_logit = tl.maximum(max_logit, tl.max(logits_blk))
                min_logit = tl.minimum(min_logit, tl.min(logits_blk))
                outlier_mask = (logits_blk > outlier_pivot) & mask_n
                num_blk_outliers = tl.sum(outlier_mask)
                cumulative_pos = tl.cast(tl.cumsum(outlier_mask) - 1 + num_outliers, tl.int32) 
                num_outliers += num_blk_outliers
                write_pos = tl.where(outlier_mask, cumulative_pos, -1)
                tl.store(PROBS_ROW + write_pos, logits_blk, mask=outlier_mask)
                
            max_range = max_logit
            min_range = min_logit
            if num_outliers > k:
                max_range = max_logit
                min_range = outlier_pivot 
                search_addr = PROBS_ROW
                search_range = tl.cast(num_outliers, tl.int32)
                search_iters = tl.cast((num_outliers + BLOCK_SIZE - 1) // BLOCK_SIZE, tl.int32)

            # Second passes: Quaternary search for pivots (nlog_4(n))
            num_iters = 0
            while k_pivot == -float('inf') and num_iters < 18:
                k_pivot_0 = (max_range - min_range) * 1.0 / 4.0 + min_range
                k_pivot_1 = (max_range - min_range) * 2.0 / 4.0 + min_range
                k_pivot_2 = (max_range - min_range) * 3.0 / 4.0 + min_range
                k_pivots_num_0 = tl.zeros((), dtype=tl.uint32)
                k_pivots_num_1 = tl.zeros((), dtype=tl.uint32)
                k_pivots_num_2 = tl.zeros((), dtype=tl.uint32)

                for i in range(0, search_iters):
                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask_n = offs_n < search_range
                    logits_blk = tl.load(search_addr + offs_n, mask=mask_n, other=-float('inf'))

                    k_pivots_num_0 += tl.sum(logits_blk > k_pivot_0)
                    k_pivots_num_1 += tl.sum(logits_blk > k_pivot_1)
                    k_pivots_num_2 += tl.sum(logits_blk > k_pivot_2)

                # Check if any of the pivots are equal to k
                if k_pivots_num_0 == k:
                    k_pivot = k_pivot_0 
                elif k_pivots_num_1 == k:
                    k_pivot = k_pivot_1
                elif k_pivots_num_2 == k:
                    k_pivot = k_pivot_2
                # If none of the pivots are equal to k, we updatae the range
                elif k_pivots_num_2 > k:
                    min_range = k_pivot_2
                elif k_pivots_num_1 > k:
                    min_range = k_pivot_1
                elif k_pivots_num_0 > k:
                    min_range = k_pivot_0
                if k_pivots_num_0 < k:
                    max_range = k_pivot_0
                elif k_pivots_num_1 < k:
                    max_range = k_pivot_1
                elif k_pivots_num_2 < k:
                    max_range = k_pivot_2
                
                num_iters += 1
                if num_iters >= 18:
                    k_pivot = k_pivot_0

        # Third pass: Apply top-k mask
        if k_pivot != -float('inf') or p_pivot != -float('inf'):
            for i in range(0, NUM_TILES):
                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask_n = offs_n < N
                logits_blk = tl.load(LOGITS_ROW + offs_n, mask=mask_n)
                mask = (logits_blk > k_pivot) & (logits_blk > p_pivot)
                logits_blk = tl.where(mask, logits_blk, -float('inf'))
                tl.store(LOGITS_ROW + offs_n, logits_blk, mask=mask_n)


@triton.jit
def _topk_topp_kernel(LOGITS, PROBS, K, P, B, 
                      N: tl.constexpr,
                      BLOCK_SIZE: tl.constexpr,
                      NUM_TILES: tl.constexpr,
                      DEBUG_PTR):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for row_id in tl.range(pid, B, num_programs):
        k_pivot = -float('inf')
        p_pivot = -float('inf')

        LOGITS_ROW = LOGITS + row_id * N
        PROBS_ROW = PROBS + pid * N

        search_addr = LOGITS_ROW
        search_range = N
        search_iters = NUM_TILES

        max_logit = -float('inf')
        avg_logit = -float('inf')

        k = tl.load(K + row_id)
        if not (k == N): # All tokens are valid
            min_logit = float('inf')

            # Zeroth pass: Compute avg and std from a sample block
            # May produce incorrect results if N < BLOCK_SIZE
            offs = tl.arange(0, BLOCK_SIZE)
            mask_n = offs < N
            logits_blk = tl.load(LOGITS_ROW + offs, mask=mask_n, other=0.0)
            avg_logit = tl.sum(logits_blk) / N
            sq_avg_logit = tl.sum(logits_blk * logits_blk) / N
            std_logit = tl.sqrt(sq_avg_logit - avg_logit * avg_logit)

            outlier_pivot = avg_logit + 2.8 * std_logit
            num_outliers = tl.zeros((), dtype=tl.uint32)
            # First pass: compute max and min logits and gather outliers
            for i in range(0,search_iters):
                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask_n = offs_n < search_range
                logits_blk = tl.load(search_addr + offs_n, mask=mask_n, other=avg_logit)

                max_logit = tl.maximum(max_logit, tl.max(logits_blk))
                min_logit = tl.minimum(min_logit, tl.min(logits_blk))
                outlier_mask = (logits_blk > outlier_pivot) & mask_n
                num_blk_outliers = tl.sum(outlier_mask)
                cumulative_pos = tl.cast(tl.cumsum(outlier_mask) - 1 + num_outliers, tl.int32) 
                num_outliers += num_blk_outliers
                write_pos = tl.where(outlier_mask, cumulative_pos, -1)
                tl.store(PROBS_ROW + write_pos, logits_blk, mask=outlier_mask)
                
            max_range = max_logit
            min_range = min_logit
            if num_outliers > k:
                max_range = max_logit
                min_range = outlier_pivot 
                search_addr = PROBS_ROW
                search_range = tl.cast(num_outliers, tl.int32)
                search_iters = tl.cast((num_outliers + BLOCK_SIZE - 1) // BLOCK_SIZE, tl.int32)

            # Second passes: Quaternary search for pivots (nlog_4(n))
            num_iters = 0
            while k_pivot == -float('inf') and num_iters < 18:
                k_pivot_0 = (max_range - min_range) * 1.0 / 4.0 + min_range
                k_pivot_1 = (max_range - min_range) * 2.0 / 4.0 + min_range
                k_pivot_2 = (max_range - min_range) * 3.0 / 4.0 + min_range
                k_pivots_num_0 = tl.zeros((), dtype=tl.uint32)
                k_pivots_num_1 = tl.zeros((), dtype=tl.uint32)
                k_pivots_num_2 = tl.zeros((), dtype=tl.uint32)

                for i in range(0, search_iters):
                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask_n = offs_n < search_range
                    logits_blk = tl.load(search_addr + offs_n, mask=mask_n, other=-float('inf'))

                    k_pivots_num_0 += tl.sum(logits_blk > k_pivot_0)
                    k_pivots_num_1 += tl.sum(logits_blk > k_pivot_1)
                    k_pivots_num_2 += tl.sum(logits_blk > k_pivot_2)

                # Check if any of the pivots are equal to k
                if k_pivots_num_0 == k:
                    k_pivot = k_pivot_0 
                elif k_pivots_num_1 == k:
                    k_pivot = k_pivot_1
                elif k_pivots_num_2 == k:
                    k_pivot = k_pivot_2
                # If none of the pivots are equal to k, we updatae the range
                elif k_pivots_num_2 > k:
                    min_range = k_pivot_2
                elif k_pivots_num_1 > k:
                    min_range = k_pivot_1
                elif k_pivots_num_0 > k:
                    min_range = k_pivot_0
                if k_pivots_num_0 < k:
                    max_range = k_pivot_0
                elif k_pivots_num_1 < k:
                    max_range = k_pivot_1
                elif k_pivots_num_2 < k:
                    max_range = k_pivot_2
                
                num_iters += 1
                if num_iters >= 18:
                    k_pivot = k_pivot_0

        p = tl.load(P + row_id)
        if p != 1.0:
            max_probs = 0.0
            min_probs = 1.0
            sum_exp_logits = 0.0

            # Third pass: Compute exp logits and sum
            for i in range(0, search_iters):
                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask_n = offs_n < search_range
                probs_blk = tl.load(search_addr + offs_n, mask=mask_n, other=-float('inf'))
                probs_blk = tl.where(probs_blk > k_pivot, probs_blk, -float('inf'))
                probs_blk = probs_blk - max_logit
                probs_blk = tl.exp(probs_blk)
                sum_exp_logits += tl.sum(probs_blk)
                tl.store(PROBS_ROW + offs_n, probs_blk, mask=mask_n)

            # Fourth pass: Compute probs (softmax)
            exp_avg = tl.exp(avg_logit)
            for i in range(0, search_iters):
                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask_n = offs_n < search_range
                probs_blk = tl.load(PROBS_ROW + offs_n, mask=mask_n, other=exp_avg)
                probs_blk = probs_blk / sum_exp_logits
                min_probs = tl.minimum(min_probs, tl.min(probs_blk))
                max_probs = tl.maximum(max_probs, tl.max(probs_blk))
                tl.store(PROBS_ROW + offs_n, probs_blk, mask=mask_n)

            max_range = max_probs
            min_range = min_probs

            num_iters = 0
            p_pivots_sum_0 = 0.0
            while p_pivot == -float('inf') and num_iters < 32:
                p_pivot_0 = (max_range - min_range) * 1.0 / 2.0 + min_range
                p_pivots_sum_0 = 0.0

                min_larger_0 = 1.0
                max_smaller_0 = 0.0
                second_max_smaller_0 = 0.0

                for i in range(0, search_iters):
                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask_n = offs_n < search_range
                    probs_blk = tl.load(PROBS_ROW + offs_n, mask=mask_n, other=0.0)

                    masked_larger_0 = tl.where(probs_blk > p_pivot_0, probs_blk, 1.0)
                    min_larger_0 = tl.minimum(min_larger_0, tl.min(masked_larger_0))
                    masked_smaller_0 = probs_blk * (probs_blk < p_pivot_0)
                    max_smaller_0 = tl.maximum(max_smaller_0, tl.max(masked_smaller_0))
                    masked_second_smaller_0 = probs_blk * (probs_blk < max_smaller_0)
                    second_max_smaller_0 = tl.maximum(second_max_smaller_0, tl.max(masked_second_smaller_0))
                    p_pivots_sum_0 += tl.sum(probs_blk * (probs_blk > p_pivot_0))

                # Check if any of the pivots are equal to k
                if tl.abs(p_pivots_sum_0 - p) < 1e-6:
                    p_pivot = p_pivot_0 
                elif p_pivots_sum_0 > p:
                    if p_pivots_sum_0 - min_larger_0 < p:
                        p_pivot = p_pivot_0
                    min_range = p_pivot_0
                elif p_pivots_sum_0 < p:
                    if p_pivots_sum_0 + max_smaller_0 > p:
                        p_pivot = second_max_smaller_0
                    max_range = p_pivot_0
                
                num_iters += 1
                if num_iters >= 32 or tl.abs(min_range - max_range) < 1e-6:
                    p_pivot = p_pivot_0
                
                if row_id == 0:
                    tl.store(DEBUG_PTR + num_iters * 17 + 0, p_pivots_sum_0)
                    tl.store(DEBUG_PTR + num_iters * 17 + 1, p_pivot_0)
                    tl.store(DEBUG_PTR + num_iters * 17 + 2, min_probs)
                    tl.store(DEBUG_PTR + num_iters * 17 + 3, max_probs)
                    tl.store(DEBUG_PTR + num_iters * 17 + 4, min_range)
                    tl.store(DEBUG_PTR + num_iters * 17 + 5, max_range)
                    tl.store(DEBUG_PTR + num_iters * 17 + 6, num_iters)
                    tl.store(DEBUG_PTR + num_iters * 17 + 7, sum_exp_logits)
                    tl.store(DEBUG_PTR + num_iters * 17 + 8, p_pivot)
                    tl.store(DEBUG_PTR + num_iters * 17 + 9, tl.log(p_pivot * sum_exp_logits))
                    tl.store(DEBUG_PTR + num_iters * 17 + 10, min_larger_0)
                    tl.store(DEBUG_PTR + num_iters * 17 + 11, max_smaller_0)
                    tl.store(DEBUG_PTR + num_iters * 17 + 12, second_max_smaller_0)
            # Subtract a small value to include the nearest smaller value
            # If the nearest smaller value very small, it may cause numerical instability
            if row_id == 0:
                tl.store(DEBUG_PTR + num_iters * 17 + 13, p_pivots_sum_0)
                tl.store(DEBUG_PTR + num_iters * 17 + 14, p_pivot)
                tl.store(DEBUG_PTR + num_iters * 17 + 15, num_iters)
            p_pivot = tl.log(p_pivot * sum_exp_logits) + max_logit
            if row_id == 0:
                tl.store(DEBUG_PTR + num_iters * 17 + 16, p_pivot)
                tl.store(DEBUG_PTR + num_iters * 17 + 17, p)

            # Transform p_pivot into equivalent logit
            # p_pivot = tl.log(p_pivot * sum_exp_logits)

        # Sixth pass: Apply mask
        if k_pivot != -float('inf') or p_pivot != -float('inf'):
            pivot = tl.maximum(k_pivot, p_pivot)
            for i in range(0, NUM_TILES):
                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask_n = offs_n < N
                logits_blk = tl.load(LOGITS_ROW + offs_n, mask=mask_n)
                logits_blk = tl.where(logits_blk > pivot, logits_blk, -float('inf'))
                tl.store(LOGITS_ROW + offs_n, logits_blk, mask=mask_n)

def triton_apply_top_k_top_p(
    logits: torch.Tensor,
    k: Optional[torch.Tensor],
    p: Optional[torch.Tensor],
) -> torch.Tensor:
    batch_size, vocab_size = logits.shape
    BLOCK_SIZE = 4096
    NUM_PROGRAMS = 128
    NUM_TILES = (vocab_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    probs = torch.full((NUM_PROGRAMS, vocab_size), -float('inf'), device=logits.device)
    debug = torch.full((32, 18), -float('inf'), device=logits.device)
    print(b_str("Launch params:") + f"logits.shape: {logits.shape}, probs.shape: {probs.shape}, "
          f"k.shape: {k.shape if k is not None else None}, p.shape: {p.shape if p is not None else None}, "
          f"batch_size: {batch_size}, vocab_size: {vocab_size}, BLOCK_SIZE: {BLOCK_SIZE}, NUM_TILES: {NUM_TILES}")
    # print(f"Input logits: {logits}")
    if p is None and k is not None:
        _topk_kernel[(NUM_PROGRAMS,)](logits, probs, k, batch_size, 
                                      vocab_size, BLOCK_SIZE, NUM_TILES)
    else:
        _topk_topp_kernel[(NUM_PROGRAMS,)](logits, probs, k, p, batch_size, 
                                           vocab_size, BLOCK_SIZE, NUM_TILES, debug)
    print(f"debug: {debug[:, :17]}")
    # print(f"Output logits: {logits}")
    # print(f"Output probs: {probs}")
    return logits

@torch.compile
def compiled_apply_top_k_top_p(
    logits: torch.Tensor,
    k: Optional[torch.Tensor],
    p: Optional[torch.Tensor],
) -> torch.Tensor:
    return original_apply_top_k_top_p(logits, k, p)

def apply_top_k_top_p(
    logits: torch.Tensor,
    k: Optional[torch.Tensor],
    p: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply top-k and top-p masks to the logits.

    If a top-p is used, this function will sort the logits tensor,
    which can be slow for large batches.

    The logits tensor may be updated in-place.
    """
    # logits = torch.full_like(logits, -10.0)
    # logits[:, :11] = torch.arange(1, 12, dtype=torch.float32, device=logits.device)
    input_logits = logits.clone()
    print(f"input_logits: {input_logits[:12, :12]}")
    original_logits = original_apply_top_k_top_p(input_logits, k, p)
    # original_probs = torch.softmax(input_logits, dim=-1)

    batch_size, vocab_size = logits.shape
    print(g_str("apply_top_k_top_p") + f" logits.shape: {batch_size} x {vocab_size}, p is None: {p is None}, k is None: {k is None}")
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    # logits = original_apply_top_k_top_p(logits, k, p)
    # logits = compiled_apply_top_k_top_p(logits, k, p)
    logits = triton_apply_top_k_top_p(logits, k, p)
        
    torch.cuda.synchronize()
    time_taken = time.time() - start_time
    print(y_str(f"apply_top_k_top_p done in {time_taken} seconds"))

    if not torch.allclose(logits, original_logits):
        print(r_str("Error: logits are not close"))
    # print(f"logits: {logits[:12, :12]}")
    # print(f"original_logits: {original_logits[:12, :12]}")

    start_time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(start_time))
    out_dir = "./sampler_input_output"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/llama8b_{start_time_str}.pt"
    torch.save({"input_logits": input_logits, "p": p, "k": k, "output_logits": logits}, out_path)
    return logits


def apply_top_k_only(
    logits: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    """
    Apply top-k mask to the logits.

    This implementation doesn't involve sorting the entire vocab.

    The logits tensor may be updated in-place.
    """
    no_top_k_mask = k == logits.shape[1]
    # Set non-top-k rows to 1 so that we can gather.
    k = k.masked_fill(no_top_k_mask, 1)
    max_top_k = k.max()
    # topk.values tensor has shape [batch_size, max_top_k].
    # Convert top k to 0-based index in range [0, max_top_k).
    k_index = k.sub_(1).unsqueeze(1)
    top_k_mask = logits.topk(max_top_k, dim=1).values.gather(1, k_index.long())
    # Handle non-topk rows.
    top_k_mask.masked_fill_(no_top_k_mask.unsqueeze(1), -float("inf"))
    logits.masked_fill_(logits < top_k_mask, -float("inf"))
    return logits


def random_sample(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """Randomly sample from the probabilities.

    We use this function instead of torch.multinomial because torch.multinomial
    causes CPU-GPU synchronization.
    """
    q = torch.empty_like(probs)
    # NOTE(woosuk): To batch-process the requests without their own seeds,
    # which is the common case, we first assume that every request does
    # not have its own seed. Then, we overwrite the values for the requests
    # that have their own seeds.
    if len(generators) != probs.shape[0]:
        q.exponential_()
    if generators:
        # TODO(woosuk): This can be slow because we handle each request
        # one by one. Optimize this.
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)
    return probs.div_(q).argmax(dim=-1).view(-1)


def flashinfer_sample(
    logits: torch.Tensor,
    k: Optional[torch.Tensor],
    p: Optional[torch.Tensor],
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """Sample from the logits using FlashInfer.

    Statistically, this function is equivalent to the `random_sample` function.
    However, this function is faster because it avoids sorting the logits tensor
    via rejection sampling.

    NOTE: The outputs of this function do not necessarily match the outputs of
    the `random_sample` function. It only guarantees that the outputs are
    statistically equivalent.

    NOTE: This function includes CPU-GPU synchronization, while `random_sample`
    does not. Call this function at the end of the forward pass to minimize
    the synchronization overhead.
    """
    assert not (k is None and p is None)
    if k is None:
        # Top-p only.
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        next_token_ids = flashinfer.sampling.top_p_sampling_from_probs(
            probs, p, deterministic=True)
    elif p is None:
        # Top-k only.
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        next_token_ids = flashinfer.sampling.top_k_sampling_from_probs(
            probs, k, deterministic=True)
    else:
        # Both top-k and top-p.
        next_token_ids = flashinfer.sampling.top_k_top_p_sampling_from_logits(
            logits, k, p, deterministic=True)

    return next_token_ids.view(-1)
