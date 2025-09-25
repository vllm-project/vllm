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
def _topk_topp_kernel(LOGITS, PROBS, K, P, B, 
                      N: tl.constexpr,
                      BLOCK_SIZE: tl.constexpr,
                      NUM_TILES: tl.constexpr,
                      NUM_PIVOTS: tl.constexpr):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for row_id in tl.range(pid, B, num_programs):
        k = tl.load(K + row_id)
        p = tl.load(P + row_id)
        if not (k == N and p == 1.0): # All tokens are valid
            max_logit = -float('inf')
            min_logit = float('inf')

            max_prob = 0.0
            min_prob = 1.0

            LOGITS_ROW = LOGITS + row_id * N
            PROBS_ROW = PROBS + row_id * N

            # First pass: compute max and min logits (for numerical stability)
            for i in range(0, NUM_TILES):
                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask_n = offs_n < N
                logits_blk = tl.load(LOGITS_ROW + offs_n, mask=mask_n, other=0.0)

                max_logit = tl.maximum(max_logit, tl.max(logits_blk))
                min_logit = tl.minimum(min_logit, tl.min(logits_blk))
            
            # Second pass: compute probabilities using softmax
            # (This requires the max for numerical stability)
            exp_logits_sum = 0.0
            for i in range(0, NUM_TILES):
                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask_n = offs_n < N
                logits_blk = tl.load(LOGITS_ROW + offs_n, mask=mask_n, other=-float('inf'))
                
                logits_tile_stable = logits_blk - max_logit  # Numerical stability
                exp_logits = tl.exp(logits_tile_stable)
                exp_logits_sum += tl.sum(exp_logits)
                tl.store(PROBS_ROW + offs_n, exp_logits, mask=mask_n)

            # Third pass: compute probabilities and update max and min probabilities
            for i in range(0, NUM_TILES):
                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask_n = offs_n < N
                probs_blk = tl.load(PROBS_ROW + offs_n, mask=mask_n, other=0.0)
                probs_blk = probs_blk / exp_logits_sum
                max_prob = tl.maximum(max_prob, tl.max(probs_blk))
                min_prob = tl.minimum(min_prob, tl.min(probs_blk))
                tl.store(PROBS_ROW + offs_n, probs_blk, mask=mask_n)

            # Fourth passes: Search for pivots
            num_iters = 0
            k_pivot = -float('inf')
            k_pivots = tl.full((NUM_PIVOTS,), -float('inf'), dtype=tl.float32)
            
            p_pivot = 0.0
            p_pivots = tl.full((NUM_PIVOTS,), -float('inf'), dtype=tl.float32)
                
            while (k_pivot == -float('inf') or p_pivot == 0.0) and num_iters < 32:
                k_pivots = (max_logit - min_logit) * tl.arange(1, NUM_PIVOTS + 1) / NUM_PIVOTS + min_logit
                p_pivots = (max_prob - min_prob) * tl.arange(1, NUM_PIVOTS + 1) / NUM_PIVOTS + min_prob
                
                k_pivots_num = tl.full((NUM_PIVOTS,), 0, dtype=tl.uint32)
                p_pivots_sum = tl.full((NUM_PIVOTS,), 0.0, dtype=tl.float32)
                for i in range(0, NUM_TILES):
                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask_n = offs_n < N
                    logits_blk = tl.load(LOGITS_ROW + offs_n, mask=mask_n, other=-float('inf'))
                    probs_blk = tl.load(PROBS_ROW + offs_n, mask=mask_n, other=0.0)

                    logits_expanded = logits_blk[None, :] # shape: 1 x BLOCK_SIZE
                    k_pivots_expanded = k_pivots[:, None] # shape: NUM_PIVOTS x 1
                    larger_mask = logits_expanded > k_pivots_expanded # shape: NUM_PIVOTS x BLOCK_SIZE
                    k_pivots_num += tl.sum(larger_mask, axis=1) # shape: NUM_PIVOTS

                    probs_expanded = probs_blk[None, :] # shape: 1 x BLOCK_SIZE
                    p_pivots_expanded = p_pivots[:, None] # shape: NUM_PIVOTS x 1
                    larger_mask = probs_expanded > p_pivots_expanded # shape: NUM_PIVOTS x BLOCK_SIZE
                    larger_probs = tl.where(larger_mask, probs_expanded, 0.0) # shape: NUM_PIVOTS x BLOCK_SIZE
                    p_pivots_sum += tl.sum(larger_probs, axis=1) # shape: NUM_PIVOTS

                exact_match_k = k_pivots_num == k
                if tl.sum(exact_match_k) > 0:
                    matches = tl.where(exact_match_k, k_pivots, float('inf'))
                    k_pivot = tl.min(matches)
                else:
                    smaller_mask = k_pivots_num < k
                    if tl.sum(smaller_mask) > 0:
                        matches = tl.where(smaller_mask, k_pivots, float('inf'))
                        max_logit = tl.min(matches)
                    larger_mask = k_pivots_num > k
                    if tl.sum(larger_mask) > 0:
                        matches = tl.where(larger_mask, k_pivots, -float('inf'))
                        min_logit = tl.max(matches)

                exact_match_p = tl.abs(p_pivots_sum - p) < 1e-6
                if tl.sum(exact_match_p) > 0:
                    matches = tl.where(exact_match_p, p_pivots, float('inf'))
                    p_pivot = tl.min(matches)
                else:
                    smaller_mask = p_pivots_sum < p
                    if tl.sum(smaller_mask) > 0:
                        matches = tl.where(smaller_mask, p_pivots, float('inf'))
                        max_prob = tl.min(matches)
                    larger_mask = p_pivots_sum > p
                    if tl.sum(larger_mask) > 0:
                        matches = tl.where(larger_mask, p_pivots, -float('inf'))
                        min_prob = tl.max(matches)
                # For the case where sum of existing probabilities does not hit p
                if min_prob == max_prob:
                    p_pivot = min_prob

                num_iters += 1

            # Fifth pass: Apply top-k and top-p masks
            for i in range(0, NUM_TILES):
                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask_n = offs_n < N
                logits_blk = tl.load(LOGITS_ROW + offs_n, mask=mask_n)
                probs_blk = tl.load(PROBS_ROW + offs_n, mask=mask_n)
                logits_blk = tl.where(logits_blk > k_pivot, logits_blk, -float('inf'))
                logits_blk = tl.where(probs_blk > p_pivot, logits_blk, -float('inf'))
                tl.store(LOGITS_ROW + offs_n, logits_blk, mask=mask_n)

@triton.jit
def _topk_kernel(LOGITS, PROBS, K, P, B, 
                 N: tl.constexpr,
                 BLOCK_SIZE: tl.constexpr,
                 NUM_TILES: tl.constexpr):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for row_id in tl.range(pid, B, num_programs):
        k = tl.load(K + row_id)
        if not (k == N): # All tokens are valid
            max_logit = -float('inf')
            min_logit = float('inf')

            LOGITS_ROW = LOGITS + row_id * N
            PROBS_ROW = PROBS + row_id * N

            # Zeroth pass: Compute avg and std from a sample block
            offs = tl.arange(0, BLOCK_SIZE)
            mask_n = offs < N
            logits_blk = tl.load(LOGITS_ROW + offs, mask=mask_n, other=0.0)
            avg_logit = tl.sum(logits_blk) / N
            sq_avg_logit = tl.sum(logits_blk * logits_blk) / N
            std_logit = tl.sqrt(sq_avg_logit - avg_logit * avg_logit)

            outlier_pivot = avg_logit + 3 * std_logit
            num_outliers = tl.zeros((), dtype=tl.uint32)
            # First pass: compute max and min logits and gather outliers
            for i in range(0, NUM_TILES):
                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask_n = offs_n < N
                logits_blk = tl.load(LOGITS_ROW + offs_n, mask=mask_n, other=avg_logit)

                max_logit = tl.maximum(max_logit, tl.max(logits_blk))
                min_logit = tl.minimum(min_logit, tl.min(logits_blk))
                outlier_mask = (logits_blk > outlier_pivot) & mask_n
                num_blk_outliers = tl.sum(outlier_mask)
                cumulative_pos = tl.cast(tl.cumsum(outlier_mask) - 1 + num_outliers, tl.int32) 
                num_outliers += num_blk_outliers
                write_pos = tl.where(outlier_mask, cumulative_pos, -1)
                tl.store(PROBS_ROW + write_pos, logits_blk, mask=mask_n)
                
            if num_outliers > k:
                # min_logit = outlier_pivot 
                search_addr = PROBS_ROW
                search_range = tl.cast(num_outliers, tl.int32)
                search_iters = tl.cast((num_outliers + BLOCK_SIZE - 1) // BLOCK_SIZE, tl.int32)
            else:
                search_addr = LOGITS_ROW
                search_range = N
                search_iters = NUM_TILES

            # Second passes: Quaternary search for pivots (nlog_4(n))
            k_pivot = -float('inf')
            num_iters = 0

            while k_pivot == -float('inf') and num_iters < 18:
                k_pivot_0 = (max_logit - min_logit) * 1.0 / 4.0 + min_logit
                k_pivot_1 = (max_logit - min_logit) * 2.0 / 4.0 + min_logit
                k_pivot_2 = (max_logit - min_logit) * 3.0 / 4.0 + min_logit
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
                    min_logit = k_pivot_2
                elif k_pivots_num_1 > k:
                    min_logit = k_pivot_1
                elif k_pivots_num_0 > k:
                    min_logit = k_pivot_0
                if k_pivots_num_0 < k:
                    max_logit = k_pivot_0
                elif k_pivots_num_1 < k:
                    max_logit = k_pivot_1
                elif k_pivots_num_2 < k:
                    max_logit = k_pivot_2
                
                num_iters += 1
                if num_iters >= 18:
                    k_pivot = k_pivot_0

            # Third pass: Apply top-k mask
            for i in range(0, NUM_TILES):
                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask_n = offs_n < N
                logits_blk = tl.load(LOGITS_ROW + offs_n, mask=mask_n)
                logits_blk = tl.where(logits_blk > k_pivot, logits_blk, -float('inf'))
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
    NUM_PIVOTS = 16 # Multi pivot search for smaller number of scans
    probs = torch.full_like(logits, -float('inf'))
    print(f"Input logits: {logits}")
    if p is None:
        _topk_kernel[(NUM_PROGRAMS,)](logits, probs, k, p, batch_size, 
                                      vocab_size, BLOCK_SIZE, NUM_TILES)
    else:
        _topk_topp_kernel[(NUM_PROGRAMS,)](logits, probs, k, p, batch_size, 
                                        vocab_size, BLOCK_SIZE, NUM_TILES, NUM_PIVOTS)
    print(f"Output logits: {logits}")
    print(f"Output probs: {probs}")
    return logits, probs

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
    input_logits = logits.clone()
    original_logits = original_apply_top_k_top_p(input_logits, k, p)
    original_probs = torch.softmax(input_logits, dim=-1)

    batch_size, vocab_size = logits.shape
    print(g_str("apply_top_k_top_p") + f" logits.shape: {batch_size} x {vocab_size}, p is None: {p is None}, k is None: {k is None}")
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    # logits = original_apply_top_k_top_p(logits, k, p)
    # logits = compiled_apply_top_k_top_p(logits, k, p)
    logits, probs = triton_apply_top_k_top_p(logits, k, p)
        
    torch.cuda.synchronize()
    time_taken = time.time() - start_time
    print(y_str(f"apply_top_k_top_p done in {time_taken} seconds"))

    # if not torch.allclose(probs, original_probs):
    #     print(r_str("Error: probs are not close"))
    #     print(f"probs: {probs}")
    #     print(f"original_probs: {original_probs}")

    if not torch.allclose(logits, original_logits):
        print(r_str("Error: logits are not close"))
        print(f"logits: {logits}")
        print(f"original_logits: {original_logits}")
        diff = (logits - original_logits).abs().flatten()
        diff_nonzero = diff[diff > 1e-6]
        print(f"diff_nonzero: {diff_nonzero}")

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
