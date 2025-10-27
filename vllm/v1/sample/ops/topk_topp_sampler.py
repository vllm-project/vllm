# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch
import torch.nn as nn
import triton
import triton.language as tl
from packaging import version
from typing import Optional

from vllm import envs
from vllm.config.model import LogprobsMode
from vllm.logger import init_logger
from vllm.platforms import CpuArchEnum, current_platform

logger = init_logger(__name__)

try:
    import flashinfer.sampling

    is_flashinfer_available = True
except ImportError:
    is_flashinfer_available = False


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
        if (
            logprobs_mode not in ("processed_logits", "processed_logprobs")
            and current_platform.is_cuda()
        ):
            if is_flashinfer_available:
                flashinfer_version = flashinfer.__version__
                if version.parse(flashinfer_version) < version.parse("0.2.3"):
                    logger.warning_once(
                        "FlashInfer version >= 0.2.3 required. "
                        "Falling back to default sampling implementation."
                    )
                    self.forward = self.forward_native
                elif envs.VLLM_USE_FLASHIVOCAB_SIZEFER_SAMPLER is not False:
                    # VOCAB_SIZEOTE(woosuk): The V0 sampler doesn't use FlashInfer for
                    # sampling unless VLLM_USE_FLASHIVOCAB_SIZEFER_SAMPLER=1 (i.e., by
                    # default it is unused). For backward compatibility, we set
                    # `VLLM_USE_FLASHIVOCAB_SIZEFER_SAMPLER` as None by default and
                    # interpret it differently in V0 and V1 samplers: In V0,
                    # None means False, while in V1, None means True. This is
                    # why we use the condition
                    # `envs.VLLM_USE_FLASHIVOCAB_SIZEFER_SAMPLER is not False` here.
                    logger.info_once("Using FlashInfer for top-p & top-k sampling.")
                    self.forward = self.forward_cuda
                elif envs.VLLM_USE_TRITOVOCAB_SIZE_SAMPLER is not False:
                    logger.info_once(
                        "Using Triton for top-p & top-k sampling.")
                    self.forward = self.forward_triton
                else:
                    logger.warning_once(
                        "FlashInfer is available, but it is not enabled. "
                        "Falling back to the PyTorch-native implementation of "
                        "top-p & top-k sampling. For the best performance, "
                        "please set VLLM_USE_FLASHIVOCAB_SIZEFER_SAMPLER=1."
                    )
                    self.forward = self.forward_native
            else:
                if envs.VLLM_USE_TRITOVOCAB_SIZE_SAMPLER is not False:
                    logger.info_once(
                        "Using Triton for top-p & top-k sampling.")
                    self.forward = self.forward_triton
                else:
                    logger.warning_once(
                        "FlashInfer is not available. Falling back to the "
                        "PyTorch-native implementation of top-p & top-k "
                        "sampling. For the best performance, please install "
                        "FlashInfer.")
                    self.forward = self.forward_native
        elif current_platform.is_cpu():
            if current_platform.get_cpu_architecture() == CpuArchEnum.RISCV:
                self.forward = self.forward_native
            else:
                self.forward = self.forward_cpu
        else:
            self.forward = self.forward_native

        self.apply_top_k_top_p = apply_top_k_top_p
        self.apply_top_k_top_p_triton = apply_top_k_top_p_triton

    def forward_native(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
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

    def forward_triton(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: Optional[torch.Tensor],
        p: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:

        logits = self.apply_top_k_top_p_triton(logits, k, p)
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
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """More optimized implementation for top-k and top-p sampling."""
        # We prefer `random_sample` over `flashinfer_sample` when sorting is
        # not needed. This is because `random_sample` does not require
        # CPU-GPU synchronization while `flashinfer_sample` does.
        if (k is None and p is None) or generators:
            if generators:
                logger.debug_once(
                    "FlashInfer 0.2.3+ does not support "
                    "per-request generators. Falling back to "
                    "PyTorch-native implementation."
                )
            return self.forward_native(logits, generators, k, p)
        assert self.logprobs_mode not in ("processed_logits", "processed_logprobs"), (
            "FlashInfer does not support returning logits/logprobs"
        )
        # flashinfer sampling functions expect contiguous logits.
        # In flex_attn/triton_attn fp32 inference, logits can be non-contiguous
        # because of slicing operation in logits_processor.
        return flashinfer_sample(logits.contiguous(), k, p, generators), None

    def forward_cpu(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        PyTorch-native implementation of top-k and top-p sampling for CPU.

        The logits tensor may be updated in-place.
        """
        logits = self.apply_top_k_top_p(logits, k, p)
        logits_to_return = None
        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)

        # VOCAB_SIZEote: this is a workaround for
        # https://github.com/pytorch/pytorch/pull/151218
        @torch.compile(dynamic=True)
        def compiled_random_sample(logits: torch.Tensor) -> torch.Tensor:
            probs = logits.softmax(dim=-1, dtype=torch.float32)
            q = torch.empty_like(probs)
            q.exponential_()
            return probs.div(q).argmax(dim=-1).view(-1)

        if len(generators) != logits.shape[0]:
            return compiled_random_sample(logits), logits_to_return
        else:
            probs = logits.softmax(dim=-1, dtype=torch.float32)
            q = torch.empty_like(probs)
            q.exponential_()
            for i, generator in generators.items():
                q[i].exponential_(generator=generator)

            return probs.div_(q).argmax(dim=-1).view(-1), logits_to_return


def apply_top_k_top_p(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
) -> torch.Tensor:
    """Apply top-k and top-p masks to the logits.

    If a top-p is used, this function will sort the logits tensor,
    which can be slow for large batches.

    The logits tensor may be updated in-place.
    """
    if p is None:
        if k is None:
            return logits

        # Avoid sorting vocab for top-k only case.
        return apply_top_k_only(logits, k)

    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

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


def apply_top_k_top_p_triton(
    logits: torch.Tensor,
    k: Optional[torch.Tensor],
    p: Optional[torch.Tensor],
) -> torch.Tensor:

    batch_size, vocab_size = logits.shape

    device_prop = torch.cuda.get_device_properties(logits.device)
    VOCAB_SIZEUM_PROGRAMS = device_prop.multi_processor_count
    BLOCK_SIZE = 16384
    SIGMA = 2.15  # Top 0.03 outliers - Maybe dynamically adjust based on K?
    VOCAB_SIZEUM_WARPS = 16
    VOCAB_SIZEUM_STAGES = 3
    probs = torch.full((VOCAB_SIZEUM_PROGRAMS, vocab_size),
                       -float('inf'),
                       device=logits.device)

    if k is not None and p is None:
        _topk_kernel[(VOCAB_SIZEUM_PROGRAMS, )](logits,
                                       probs,
                                       k,
                                       batch_size,
                                       SIGMA,
                                       vocab_size,
                                       BLOCK_SIZE,
                                       num_warps=VOCAB_SIZEUM_WARPS,
                                       num_stages=VOCAB_SIZEUM_STAGES)
    elif k is None and p is not None:
        probs_2 = torch.full_like(probs, -float('inf'), device=logits.device)
        _topp_kernel[(VOCAB_SIZEUM_PROGRAMS, )](logits,
                                       probs,
                                       probs_2,
                                       p,
                                       batch_size,
                                       SIGMA,
                                       vocab_size,
                                       BLOCK_SIZE,
                                       num_warps=VOCAB_SIZEUM_WARPS,
                                       num_stages=VOCAB_SIZEUM_STAGES)
    elif k is not None and p is not None:
        _topk_topp_kernel[(VOCAB_SIZEUM_PROGRAMS, )](logits,
                                            probs,
                                            k,
                                            p,
                                            batch_size,
                                            SIGMA,
                                            vocab_size,
                                            BLOCK_SIZE,
                                            num_warps=VOCAB_SIZEUM_WARPS,
                                            num_stages=VOCAB_SIZEUM_STAGES)
    return logits


@triton.jit
def _topk_kernel(LOGITS, PROBS, K, B, SIGMA: tl.constexpr, VOCAB_SIZE: tl.constexpr,
                 BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    NUM_TILES: tl.constexpr = (VOCAB_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE
    for row_id in tl.range(pid, B, num_programs):
        k = tl.load(K + row_id)
        if k != VOCAB_SIZE:  # All tokens are valid

            # THERE IS VOCAB_SIZEO DUPLICATE LOGIT MAVOCAB_SIZEAGEMEVOCAB_SIZET FOR THIS TOP-K
            # CURREVOCAB_SIZET IMPLEMEVOCAB_SIZETATIOVOCAB_SIZE IVOCAB_SIZECLUDES ALL DUPLICATE LOGITS,
            # WHICH MAY RETURVOCAB_SIZE MORE THAVOCAB_SIZE K LOGITS,
            # FOLLOWIVOCAB_SIZEG THE IMPLEMEVOCAB_SIZETATIOVOCAB_SIZE in apply_top_k_only().
            # IF YOU VOCAB_SIZEEED EXACTLY K LOGITS, PLEASE REFER TO THE TOP-P
            # IMPLEMEVOCAB_SIZETATIOVOCAB_SIZE AVOCAB_SIZED IMPLEMEVOCAB_SIZET THE DUPLICATE LOGIT MAVOCAB_SIZEAGEMEVOCAB_SIZET
            # USIVOCAB_SIZEG THE FORCE_REMOVE_LOGIT VARIABLE

            k_pivot = -float('inf')

            LOGITS_ROW = LOGITS + row_id * VOCAB_SIZE
            PROBS_ROW = PROBS + pid * VOCAB_SIZE

            search_addr = LOGITS_ROW
            search_range = VOCAB_SIZE
            search_iters = NUM_TILES

            max_logit = -float('inf')
            min_logit = float('inf')

            # Zeroth pass: Compute avg and std from a sample block
            # May produce incorrect results if VOCAB_SIZE < BLOCK_SIZE
            offs = tl.arange(0, BLOCK_SIZE)
            mask_n = offs < VOCAB_SIZE
            logits_blk = tl.load(LOGITS_ROW + offs, mask=mask_n, other=0.0)
            avg_logit = tl.sum(logits_blk) / VOCAB_SIZE
            sq_avg_logit = tl.sum(logits_blk * logits_blk) / VOCAB_SIZE
            std_logit = tl.sqrt(sq_avg_logit - avg_logit * avg_logit)

            outlier_pivot = avg_logit + SIGMA * std_logit
            num_outliers = tl.zeros((), dtype=tl.uint32)
            # First pass: compute max and min logits and gather outliers
            for i in range(0, search_iters):
                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask_n = offs_n < search_range
                logits_blk = tl.load(search_addr + offs_n,
                                     mask=mask_n,
                                     other=avg_logit)

                max_logit = tl.maximum(max_logit, tl.max(logits_blk))
                min_logit = tl.minimum(min_logit, tl.min(logits_blk))
                outlier_mask = (logits_blk > outlier_pivot) & mask_n
                num_blk_outliers = tl.sum(outlier_mask)
                cumulative_pos = tl.cast(
                    tl.cumsum(outlier_mask) - 1 + num_outliers, tl.int32)
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
                search_iters = tl.cast(
                    (num_outliers + BLOCK_SIZE - 1) // BLOCK_SIZE, tl.int32)

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
                    logits_blk = tl.load(search_addr + offs_n,
                                         mask=mask_n,
                                         other=-float('inf'))

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
                # If none of the pivots are equal to k, we update the range
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
                if num_iters >= 18 or tl.abs(min_range - max_range) < 1e-8:
                    k_pivot = k_pivot_0

            # Third pass: Apply top-k mask
            if k_pivot != -float('inf'):
                for i in range(0, search_iters):
                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask_n = offs_n < VOCAB_SIZE
                    logits_blk = tl.load(LOGITS_ROW + offs_n, mask=mask_n)
                    mask = (logits_blk > k_pivot)
                    logits_blk = tl.where(mask, logits_blk, -float('inf'))
                    tl.store(LOGITS_ROW + offs_n, logits_blk, mask=mask_n)


@triton.jit
def _topp_kernel(LOGITS, PROBS, PROBS_2, P, B, SIGMA: tl.constexpr,
                 VOCAB_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    NUM_TILES: tl.constexpr = (VOCAB_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for row_id in tl.range(pid, B, num_programs):
        p = tl.load(P + row_id)
        if p != 1.0:  # All tokens are valid

            p_pivot = -float('inf')

            LOGITS_ROW = LOGITS + row_id * VOCAB_SIZE
            PROBS_ROW = PROBS + pid * VOCAB_SIZE
            PROBS_2_ROW = PROBS_2 + pid * VOCAB_SIZE

            search_addr = PROBS_ROW
            search_range = VOCAB_SIZE
            search_iters = NUM_TILES

            max_logit = -float('inf')
            min_logit = float('inf')

            # The Pytorch version removes the earlier duplicates
            # if there are more than one duplicates
            force_remove_logit = -float('inf')
            num_force_remove = tl.zeros((), dtype=tl.uint32)

            # Zeroth pass: Compute avg and std from a sample block
            # May produce incorrect results if VOCAB_SIZE < BLOCK_SIZE
            # OR all logits are the same
            offs = tl.arange(0, BLOCK_SIZE)
            mask_n = offs < VOCAB_SIZE
            logits_blk = tl.load(LOGITS_ROW + offs, mask=mask_n, other=0.0)
            avg_logit = tl.sum(logits_blk) / VOCAB_SIZE
            sq_avg_logit = tl.sum(logits_blk * logits_blk) / VOCAB_SIZE
            std_logit = tl.sqrt(sq_avg_logit - avg_logit * avg_logit)

            outlier_pivot = avg_logit + SIGMA * std_logit
            num_outliers = tl.zeros((), dtype=tl.uint32)
            sum_outlier_probs = 0.0

            sum_exp_logits = 0.0

            # First pass: compute max and min logits
            for i in range(0, search_iters):
                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask_n = offs_n < search_range
                logits_blk = tl.load(LOGITS_ROW + offs_n,
                                     mask=mask_n,
                                     other=avg_logit)
                max_logit = tl.maximum(max_logit, tl.max(logits_blk))
                min_logit = tl.minimum(min_logit, tl.min(logits_blk))

            # Second pass: Calculate exp logits and sum
            for i in range(0, search_iters):
                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask_n = offs_n < search_range

                probs_blk = tl.load(LOGITS_ROW + offs_n,
                                    mask=mask_n,
                                    other=-float('inf'))
                probs_blk = probs_blk - max_logit
                probs_blk = tl.exp(probs_blk)
                sum_exp_logits += tl.sum(probs_blk)
                tl.store(PROBS_ROW + offs_n, probs_blk, mask=mask_n)

            outlier_prob = tl.exp(outlier_pivot - max_logit) / sum_exp_logits

            # Third pass: Calculate probs and get outliers
            for i in range(0, search_iters):
                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask_n = offs_n < search_range

                probs_blk = tl.load(PROBS_ROW + offs_n, mask=mask_n, other=0.0)
                probs_blk = probs_blk / sum_exp_logits
                tl.store(PROBS_ROW + offs_n, probs_blk, mask=mask_n)

                outlier_mask = (probs_blk > outlier_prob) & mask_n
                sum_outlier_probs += tl.sum(outlier_mask * probs_blk)
                num_blk_outliers = tl.sum(outlier_mask)
                cumulative_pos = tl.cast(
                    tl.cumsum(outlier_mask) - 1 + num_outliers, tl.int32)
                num_outliers += num_blk_outliers
                write_pos = tl.where(outlier_mask, cumulative_pos, -1)
                tl.store(PROBS_2_ROW + write_pos, probs_blk, mask=outlier_mask)

            max_range = tl.exp(max_logit - max_logit) / sum_exp_logits
            min_range = tl.exp(min_logit - max_logit) / sum_exp_logits

            if sum_outlier_probs > p:
                min_range = outlier_prob
                search_addr = PROBS_2_ROW
                search_range = tl.cast(num_outliers, tl.int32)
                search_iters = tl.cast(
                    (num_outliers + BLOCK_SIZE - 1) // BLOCK_SIZE, tl.int32)

            second_max_logit = -float('inf')

            num_iters = 0
            p_pivots_sum_0 = 0.0
            min_larger_0 = 1.0
            num_min_larger_0 = tl.zeros((), dtype=tl.uint32)

            # Fifth passes: Search for p_pivot (2log_2(n))
            while p_pivot == -float('inf') and num_iters < 32:
                p_pivot_0 = (max_range - min_range) * 1.0 / 2.0 + min_range
                p_pivots_sum_0 = 0.0

                min_larger_0 = 1.0
                num_min_larger_0 = tl.zeros((), dtype=tl.uint32)

                for i in range(0, search_iters):
                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask_n = offs_n < search_range
                    probs_blk = tl.load(search_addr + offs_n,
                                        mask=mask_n,
                                        other=0.0)

                    masked_larger_0 = tl.where(probs_blk > p_pivot_0,
                                               probs_blk, 1.0)
                    min_larger_0 = tl.minimum(min_larger_0,
                                              tl.min(masked_larger_0))

                    p_pivots_sum_0 += tl.sum(probs_blk *
                                             (probs_blk > p_pivot_0))

                for i in range(0, search_iters):
                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask_n = offs_n < search_range
                    probs_blk = tl.load(search_addr + offs_n,
                                        mask=mask_n,
                                        other=0.0)

                    num_min_larger_0 += tl.sum(
                        tl.abs(probs_blk - min_larger_0) < 1e-7)

                # Check if any of the pivots are equal to k
                if p_pivots_sum_0 >= p:
                    if p_pivots_sum_0 - (min_larger_0 * num_min_larger_0) < p:
                        p_pivot = p_pivot_0
                    else:
                        min_range = p_pivot_0
                else:
                    max_range = p_pivot_0

                num_iters += 1
                if num_iters >= 32 or tl.abs(min_range - max_range) < 1e-8:
                    p_pivot = p_pivot_0

            # At least one value should be greater than p_pivot
            if p_pivot >= max_logit:
                p_pivot = second_max_logit
            elif num_min_larger_0 > 1:
                # Force remove duplicates (p_pivot is made to include all
                # duplicates if it falls on the duplicates)
                num_force_remove = tl.cast((p_pivots_sum_0 - p) / min_larger_0,
                                           tl.uint32)
                force_remove_logit = tl.log(
                    min_larger_0 * sum_exp_logits) + max_logit

            p_pivot = tl.log(p_pivot * sum_exp_logits) + max_logit

            # Sixth pass: Apply mask
            current_num_force_remove = tl.zeros((), dtype=tl.uint32)
            if p_pivot != -float('inf'):
                for i in range(0, NUM_TILES):
                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask_n = offs_n < VOCAB_SIZE
                    logits_blk = tl.load(LOGITS_ROW + offs_n,
                                         mask=mask_n,
                                         other=-float('inf'))

                    if force_remove_logit != -float('inf'):
                        # Force remove duplicates
                        tolerance = 1e-5 * tl.maximum(
                            1.0, tl.abs(force_remove_logit))
                        force_remove_mask = tl.abs(
                            logits_blk - force_remove_logit) < tolerance
                        force_remove_count = tl.cumsum(
                            force_remove_mask) + current_num_force_remove
                        force_remove_count_mask = \
                            force_remove_count <= num_force_remove
                        force_remove_mask = \
                            force_remove_count_mask & force_remove_mask
                        logits_blk = tl.where(force_remove_mask, -float('inf'),
                                              logits_blk)
                        current_num_force_remove = tl.max(force_remove_count)

                    logits_blk = tl.where(logits_blk > p_pivot, logits_blk,
                                          -float('inf'))
                    tl.store(LOGITS_ROW + offs_n, logits_blk, mask=mask_n)


@triton.jit
def _topk_topp_kernel(LOGITS, PROBS, K, P, B, SIGMA: tl.constexpr,
                      VOCAB_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    NUM_TILES: tl.constexpr = (VOCAB_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for row_id in tl.range(pid, B, num_programs):
        k_pivot = -float('inf')
        p_pivot = -float('inf')

        LOGITS_ROW = LOGITS + row_id * VOCAB_SIZE
        PROBS_ROW = PROBS + pid * VOCAB_SIZE

        search_addr = LOGITS_ROW
        search_range = VOCAB_SIZE
        search_iters = NUM_TILES

        max_logit = -float('inf')
        min_logit = float('inf')
        avg_logit = -float('inf')

        # The Pytorch version removes the earlier duplicates
        # if there are more than one duplicates
        force_remove_logit = -float('inf')
        num_force_remove = tl.zeros((), dtype=tl.uint32)

        # Zeroth pass: Compute avg and std from a sample block
        # May produce incorrect results if VOCAB_SIZE < BLOCK_SIZE
        offs = tl.arange(0, BLOCK_SIZE)
        mask_n = offs < VOCAB_SIZE
        logits_blk = tl.load(LOGITS_ROW + offs, mask=mask_n, other=0.0)
        avg_logit = tl.sum(logits_blk) / VOCAB_SIZE
        sq_avg_logit = tl.sum(logits_blk * logits_blk) / VOCAB_SIZE
        std_logit = tl.sqrt(sq_avg_logit - avg_logit * avg_logit)

        outlier_pivot = avg_logit + SIGMA * std_logit
        num_outliers = tl.zeros((), dtype=tl.uint32)
        # First pass: compute max and min logits and gather outliers
        for i in range(0, search_iters):
            offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask_n = offs_n < search_range
            logits_blk = tl.load(search_addr + offs_n,
                                 mask=mask_n,
                                 other=avg_logit)

            max_logit = tl.maximum(max_logit, tl.max(logits_blk))
            min_logit = tl.minimum(min_logit, tl.min(logits_blk))
            outlier_mask = (logits_blk > outlier_pivot) & mask_n
            num_blk_outliers = tl.sum(outlier_mask)
            cumulative_pos = tl.cast(
                tl.cumsum(outlier_mask) - 1 + num_outliers, tl.int32)
            num_outliers += num_blk_outliers
            write_pos = tl.where(outlier_mask, cumulative_pos, -1)
            tl.store(PROBS_ROW + write_pos, logits_blk, mask=outlier_mask)

        ############### START OF TOP-K CODE ###############
        k = tl.load(K + row_id)
        max_range = max_logit
        min_range = min_logit
        if num_outliers > k:
            max_range = max_logit
            min_range = outlier_pivot
            search_addr = PROBS_ROW
            search_range = tl.cast(num_outliers, tl.int32)
            search_iters = tl.cast(
                (num_outliers + BLOCK_SIZE - 1) // BLOCK_SIZE, tl.int32)

        if k != VOCAB_SIZE:  # All tokens are valid

            # THERE IS VOCAB_SIZEO DUPLICATE LOGIT MAVOCAB_SIZEAGEMEVOCAB_SIZET FOR THIS TOP-K
            # CURREVOCAB_SIZET IMPLEMEVOCAB_SIZETATIOVOCAB_SIZE IVOCAB_SIZECLUDES ALL DUPLICATE LOGITS,
            # WHICH MAY RETURVOCAB_SIZE MORE THAVOCAB_SIZE K LOGITS,
            # FOLLOWIVOCAB_SIZEG THE IMPLEMEVOCAB_SIZETATIOVOCAB_SIZE in apply_top_k_only().
            # IF YOU VOCAB_SIZEEED EXACTLY K LOGITS, PLEASE REFER TO THE TOP-P
            # IMPLEMEVOCAB_SIZETATIOVOCAB_SIZE AVOCAB_SIZED IMPLEMEVOCAB_SIZET THE DUPLICATE LOGIT MAVOCAB_SIZEAGEMEVOCAB_SIZET
            # USIVOCAB_SIZEG THE FORCE_REMOVE_LOGIT VARIABLE.

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
                    logits_blk = tl.load(search_addr + offs_n,
                                         mask=mask_n,
                                         other=-float('inf'))

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
                if num_iters >= 18 or tl.abs(min_range - max_range) < 1e-8:
                    k_pivot = k_pivot_0

        ############### EVOCAB_SIZED OF TOP-K CODE ###############

        ############### START OF TOP-P CODE ###############

        p = tl.load(P + row_id)
        if p != 1.0:  # All tokens are valid

            second_max_logit = -float('inf')
            max_probs = 0.0
            min_probs = 1.0
            sum_exp_logits = 0.0

            # Third pass: Compute exp logits and sum
            for i in range(0, search_iters):
                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask_n = offs_n < search_range
                probs_blk = tl.load(search_addr + offs_n,
                                    mask=mask_n,
                                    other=-float('inf'))
                probs_blk = tl.where(probs_blk > k_pivot, probs_blk,
                                     -float('inf'))
                probs_blk = probs_blk - max_logit
                probs_blk = tl.exp(probs_blk)
                sum_exp_logits += tl.sum(probs_blk)
                tl.store(PROBS_ROW + offs_n, probs_blk, mask=mask_n)

                second_max_mask = probs_blk * (probs_blk < max_probs)
                second_max_logit = tl.maximum(second_max_logit,
                                              tl.max(second_max_mask))

            # Fourth pass: Compute probs (softmax)
            for i in range(0, search_iters):
                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask_n = offs_n < search_range
                probs_blk = tl.load(PROBS_ROW + offs_n, mask=mask_n)
                probs_blk = probs_blk / sum_exp_logits
                min_blk = tl.where(mask_n, probs_blk, 1.0)
                min_probs = tl.minimum(min_probs, tl.min(min_blk))
                max_blk = tl.where(mask_n, probs_blk, 0.0)
                max_probs = tl.maximum(max_probs, tl.max(max_blk))
                tl.store(PROBS_ROW + offs_n, probs_blk, mask=mask_n)

            max_range = max_probs
            min_range = min_probs

            num_iters = 0
            p_pivots_sum_0 = 0.0
            min_larger_0 = 1.0
            num_min_larger_0 = tl.zeros((), dtype=tl.uint32)

            # Fifth passes: Search for p_pivot (2log_2(n))
            while p_pivot == -float('inf') and num_iters < 32:
                p_pivot_0 = (max_range - min_range) * 1.0 / 2.0 + min_range
                p_pivots_sum_0 = 0.0

                min_larger_0 = 1.0
                num_min_larger_0 = tl.zeros((), dtype=tl.uint32)

                for i in range(0, search_iters):
                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask_n = offs_n < search_range
                    probs_blk = tl.load(PROBS_ROW + offs_n,
                                        mask=mask_n,
                                        other=0.0)

                    masked_larger_0 = tl.where(probs_blk > p_pivot_0,
                                               probs_blk, 1.0)
                    min_larger_0 = tl.minimum(min_larger_0,
                                              tl.min(masked_larger_0))

                    p_pivots_sum_0 += tl.sum(probs_blk *
                                             (probs_blk > p_pivot_0))

                for i in range(0, search_iters):
                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask_n = offs_n < search_range
                    probs_blk = tl.load(PROBS_ROW + offs_n,
                                        mask=mask_n,
                                        other=0.0)

                    num_min_larger_0 += tl.sum(
                        tl.abs(probs_blk - min_larger_0) < 1e-7)

                # Check if any of the pivots are equal to k
                if p_pivots_sum_0 >= p:
                    if p_pivots_sum_0 - (min_larger_0 * num_min_larger_0) < p:
                        p_pivot = p_pivot_0
                    else:
                        min_range = p_pivot_0
                else:
                    max_range = p_pivot_0

                num_iters += 1
                if num_iters >= 32 or tl.abs(min_range - max_range) < 1e-8:
                    p_pivot = p_pivot_0

            # At least one value should be greater than p_pivot
            if p_pivot >= max_logit:
                p_pivot = second_max_logit
            elif num_min_larger_0 > 1:
                # Force remove duplicates (p_pivot is made to include all
                # duplicates if it falls on the duplicates)
                num_force_remove = tl.cast((p_pivots_sum_0 - p) / min_larger_0,
                                           tl.uint32)
                force_remove_logit = tl.log(
                    min_larger_0 * sum_exp_logits) + max_logit

            p_pivot = tl.log(p_pivot * sum_exp_logits) + max_logit

        ############### EVOCAB_SIZED OF TOP-P CODE ###############

        # Sixth pass: Apply mask
        pivot = tl.maximum(k_pivot, p_pivot)
        current_num_force_remove = tl.zeros((), dtype=tl.uint32)
        if pivot != -float('inf'):
            for i in range(0, NUM_TILES):
                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask_n = offs_n < VOCAB_SIZE
                logits_blk = tl.load(LOGITS_ROW + offs_n,
                                     mask=mask_n,
                                     other=-float('inf'))

                if force_remove_logit != -float('inf'):
                    # Force remove duplicates
                    tolerance = 1e-5 * tl.maximum(1.0,
                                                  tl.abs(force_remove_logit))
                    force_remove_mask = tl.abs(logits_blk -
                                               force_remove_logit) < tolerance
                    force_remove_count = tl.cumsum(
                        force_remove_mask) + current_num_force_remove
                    force_remove_count_mask = \
                        force_remove_count <= num_force_remove
                    force_remove_mask = \
                        force_remove_count_mask & force_remove_mask
                    logits_blk = tl.where(force_remove_mask, -float('inf'),
                                          logits_blk)
                    current_num_force_remove = tl.max(force_remove_count)

                logits_blk = tl.where(logits_blk > pivot, logits_blk,
                                      -float('inf'))
                tl.store(LOGITS_ROW + offs_n, logits_blk, mask=mask_n)


def apply_top_k_only(
    logits: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    """
    Apply top-k mask to the logits.

    This implementation doesn't involve sorting the entire vocab.

    The logits tensor may be updated in-place.
    """
    if k is None:
        return logits
    max_top_k = k.max().item()
    
    # --- FIX: Handle k=0 edge case ---
    # If the max k is 0, all rows are 0. Mask everything and exit.
    if max_top_k == 0:
        logits.fill_(-float("inf"))
        return logits 

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
    # VOCAB_SIZEOTE(woosuk): To batch-process the requests without their own seeds,
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
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """Sample from the logits using FlashInfer.

    Statistically, this function is equivalent to the `random_sample` function.
    However, this function is faster because it avoids sorting the logits tensor
    via rejection sampling.

    VOCAB_SIZEOTE: The outputs of this function do not necessarily match the outputs of
    the `random_sample` function. It only guarantees that the outputs are
    statistically equivalent.

    VOCAB_SIZEOTE: This function includes CPU-GPU synchronization, while `random_sample`
    does not. Call this function at the end of the forward pass to minimize
    the synchronization overhead.
    """
    assert not (k is None and p is None)
    if k is None:
        # Top-p only.
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        next_token_ids = flashinfer.sampling.top_p_sampling_from_probs(
            probs, p, deterministic=True
        )
    elif p is None:
        # Top-k only.
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        next_token_ids = flashinfer.sampling.top_k_sampling_from_probs(
            probs, k, deterministic=True
        )
    else:
        # Both top-k and top-p.
        next_token_ids = flashinfer.sampling.top_k_top_p_sampling_from_logits(
            logits, k, p, deterministic=True
        )

    return next_token_ids.view(-1)

@triton.jit
def _topp_kernel_sorted(
    LOGITS, PROBS, PROBS_2, P, B, SIGMA: tl.constexpr,
    VOCAB_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    """Modified top-p kernel with sort-equivalent tie-breaking
    and re-enabled outlier optimization.
    """
    NUM_TILES: tl.constexpr = (VOCAB_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    
    for row_id in tl.range(pid, B, num_programs):
        p = tl.load(P + row_id)
        if p != 1.0:  # All tokens are valid

            p_pivot = -float('inf')

            LOGITS_ROW = LOGITS + row_id * VOCAB_SIZE
            PROBS_ROW = PROBS + pid * VOCAB_SIZE
            PROBS_2_ROW = PROBS_2 + pid * VOCAB_SIZE # <-- RE-ADDED

            # Default search params
            search_addr = PROBS_ROW
            search_range = VOCAB_SIZE
            search_iters = NUM_TILES

            max_logit = -float('inf')
            min_logit = float('inf')

            force_remove_logit = -float('inf')
            num_force_remove = tl.zeros((), dtype=tl.uint32)
            
            # --- ZEROTH PASS (RE-ADDED) ---
            # Compute *exact* avg and std
            sum_logits = 0.0
            sum_sq_logits = 0.0
            for i in range(0, search_iters):
                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask_n = offs_n < VOCAB_SIZE
                logits_blk = tl.load(LOGITS_ROW + offs_n, mask=mask_n, other=0.0)
                sum_logits += tl.sum(tl.where(mask_n, logits_blk, 0.0))
                sum_sq_logits += tl.sum(tl.where(mask_n, logits_blk * logits_blk, 0.0))

            avg_logit = sum_logits / VOCAB_SIZE
            sq_avg_logit = sum_sq_logits / VOCAB_SIZE
            std_logit = tl.sqrt(tl.maximum(0.0, sq_avg_logit - avg_logit * avg_logit))
            outlier_pivot = avg_logit + SIGMA * std_logit # <-- RE-ADDED
            num_outliers = tl.zeros((), dtype=tl.uint32)  # <-- RE-ADDED
            sum_outlier_probs = 0.0                     # <-- RE-ADDED

            sum_exp_logits = 0.0

            # First pass: compute max and min logits
            for i in range(0, search_iters):
                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask_n = offs_n < search_range
                logits_blk = tl.load(LOGITS_ROW + offs_n,
                                     mask=mask_n,
                                     other=-float('inf')) # Use -inf
                max_logit = tl.maximum(max_logit, tl.max(logits_blk))
                min_logit = tl.minimum(min_logit, tl.min(logits_blk))

            # Second pass: Calculate exp logits and sum
            for i in range(0, search_iters):
                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask_n = offs_n < search_range

                probs_blk = tl.load(LOGITS_ROW + offs_n,
                                    mask=mask_n,
                                    other=-float('inf'))
                probs_blk = probs_blk - max_logit
                probs_blk = tl.exp(probs_blk)
                sum_exp_logits += tl.sum(probs_blk)
                tl.store(PROBS_ROW + offs_n, probs_blk, mask=mask_n)

            # --- OUTLIER_PROB (RE-ADDED) ---
            outlier_prob = tl.exp(outlier_pivot - max_logit) / sum_exp_logits

            # Third pass: Calculate final probs AVOCAB_SIZED get outliers
            for i in range(0, search_iters):
                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask_n = offs_n < search_range

                probs_blk = tl.load(PROBS_ROW + offs_n, mask=mask_n, other=0.0)
                probs_blk = probs_blk / sum_exp_logits
                tl.store(PROBS_ROW + offs_n, probs_blk, mask=mask_n)
                
                # --- OUTLIER MASKIVOCAB_SIZEG LOGIC (RE-ADDED) ---
                outlier_mask = (probs_blk > outlier_prob) & mask_n
                sum_outlier_probs += tl.sum(outlier_mask * probs_blk)
                num_blk_outliers = tl.sum(outlier_mask)
                cumulative_pos = tl.cast(
                    tl.cumsum(outlier_mask) - 1 + num_outliers, tl.int32)
                num_outliers += num_blk_outliers
                write_pos = tl.where(outlier_mask, cumulative_pos, -1)
                tl.store(PROBS_2_ROW + write_pos, probs_blk, mask=outlier_mask)


            max_range = tl.exp(max_logit - max_logit) / sum_exp_logits
            min_range = tl.exp(min_logit - max_logit) / sum_exp_logits

            if sum_outlier_probs > p:
                min_range = outlier_prob
                search_addr = PROBS_2_ROW
                search_range = tl.cast(num_outliers, tl.int32)
                search_iters = tl.cast(
                    (num_outliers + BLOCK_SIZE - 1) // BLOCK_SIZE, tl.int32)

            second_max_logit = -float('inf')
            num_iters = 0
            p_pivots_sum_0 = 0.0 # --> total prob including all equivalent min
            min_larger_0 = 1.0 # --> prob of tie-breaking min
            num_min_larger_0 = tl.zeros((), dtype=tl.uint32)

            # Binary search for p_pivot
            while p_pivot == -float('inf') and num_iters < 32:
                p_pivot_0 = (max_range - min_range) * 1.0 / 2.0 + min_range
                p_pivots_sum_0 = 0.0

                min_larger_0 = 1.0
                num_min_larger_0 = tl.zeros((), dtype=tl.uint32)

                for i in range(0, search_iters):
                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask_n = offs_n < search_range
                    probs_blk = tl.load(search_addr + offs_n,
                                        mask=mask_n,
                                        other=0.0)

                    masked_larger_0 = tl.where(probs_blk > p_pivot_0,
                                               probs_blk, 1.0)
                    min_larger_0 = tl.minimum(min_larger_0,
                                              tl.min(masked_larger_0))

                    p_pivots_sum_0 += tl.sum(probs_blk *
                                             (probs_blk > p_pivot_0))

                for i in range(0, search_iters):
                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask_n = offs_n < search_range
                    probs_blk = tl.load(search_addr + offs_n,
                                        mask=mask_n,
                                        other=0.0)

                    num_min_larger_0 += tl.sum(
                        tl.abs(probs_blk - min_larger_0) < 1e-7)

                if p_pivots_sum_0 >= p:
                    if p_pivots_sum_0 - (min_larger_0 * num_min_larger_0) < p:
                        p_pivot = p_pivot_0
                    else:
                        min_range = p_pivot_0
                else:
                    max_range = p_pivot_0

                num_iters += 1
                if num_iters >= 32 or tl.abs(min_range - max_range) < 1e-8:
                    p_pivot = p_pivot_0

            if p_pivot >= max_logit:
                p_pivot = second_max_logit
            elif num_min_larger_0 > 1:
                num_force_remove = tl.cast((p_pivots_sum_0 - p) / min_larger_0,
                                           tl.uint32) # --> number of probs to be removed 
                force_remove_logit = tl.log(
                    min_larger_0 * sum_exp_logits) + max_logit

            p_pivot = tl.log(p_pivot * sum_exp_logits) + max_logit

            # Apply mask with (non-sort-equivalent) tie-breaking
            current_num_removed = tl.zeros((), dtype=tl.uint32)  
            if p_pivot != -float('inf'):
                for i in range(0, NUM_TILES):
                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask_n = offs_n < VOCAB_SIZE
                    logits_blk = tl.load(LOGITS_ROW + offs_n,
                                         mask=mask_n,
                                         other=-float('inf'))

                    if force_remove_logit != -float('inf'):
                        # Match PyTorch's non-sort-equivalent tie-breaking
                        tolerance = 1e-5 * tl.maximum(1.0, tl.abs(force_remove_logit))
                        is_tie = tl.abs(logits_blk - force_remove_logit) < tolerance
                        tie_position = tl.cumsum(is_tie) - 1 + current_num_removed
                        should_remove = is_tie & (tie_position < num_force_remove)
                        logits_blk = tl.where(should_remove, -float('inf'), logits_blk)
                        current_num_removed += tl.sum(is_tie)

                    # Standard threshold masking
                    tolerance = 1e-6 * tl.maximum(1.0, tl.abs(p_pivot))
                    logits_blk = tl.where(logits_blk >= (p_pivot - tolerance), logits_blk,
                                          -float('inf'))
                    
                    tl.store(LOGITS_ROW + offs_n, logits_blk, mask=mask_n)

def apply_top_p_sorted_equivalent(
    logits: torch.Tensor,
    p: torch.Tensor,
    sigma: float = 3.0,  
) -> torch.Tensor:
    """Apply top-p using binary search (no sort!) with sort-equivalent results.
    
    Args:
        logits: [B, VOCAB_SIZE] logits tensor
        p: [B] top-p thresholds
        sigma: Standard deviation multiplier for outlier detection
    Returns:
        Modified logits, equivalent to sorted top-p version
    """
    B, VOCAB_SIZE = logits.shape
    device = logits.device
    
    BLOCK_SIZE = triton.next_power_of_2(min(VOCAB_SIZE, 1024))
    num_warps = 4 if BLOCK_SIZE < 2048 else 8
    
    probs = torch.empty((B, VOCAB_SIZE), device=device, dtype=torch.float32)
    probs_2 = torch.empty((B, VOCAB_SIZE), device=device, dtype=torch.float32) 
    
    grid = (B,)
    _topp_kernel_sorted[grid](
        logits,
        probs,
        probs_2, 
        p,
        B,
        SIGMA=sigma,  
        VOCAB_SIZE=VOCAB_SIZE,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    
    return logits

def apply_top_k_top_p_test(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
) -> torch.Tensor:
    """Optimized implementation combining torch.topk and binary search kernel.
    """
    if p is None:
        if k is None:
            return logits
        return apply_top_k_only(logits, k)
    # Apply top-k filter first if needed
    if k is not None:
        logits = apply_top_k_only(logits, k)
    
    # Apply top-p using binary search (no sort!)
    return apply_top_p_sorted_equivalent(logits, p)

# --------------------------------------------------------------------------------------

@triton.jit
def top_p_pivot_filter(LOGITS, L, PROBS, PROBS_2, idx_tensor, P, B, SIGMA:tl.constexpr, VOCAB_SIZE:tl.constexpr, BLOCK_SIZE:tl.constexpr, WIDEN_NUM: tl.constexpr): 
    NUM_TILES: tl.constexpr = (VOCAB_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    for row_id in tl.range(pid, B, num_programs): 
        p = tl.load(P + row_id) # fetches the p value of the row it is working on 
        if p != 1.0: # if p == 1, this becomes pointless ! 
            p_pivot = -float('inf') 

            LOGITS_ROW = LOGITS + row_id * VOCAB_SIZE
            PROBS_ROW = PROBS + row_id * VOCAB_SIZE
            PROBS_2_ROW = PROBS_2 + row_id * VOCAB_SIZE
            IDX_ROW = idx_tensor + row_id * VOCAB_SIZE

            search_address = PROBS_ROW
            search_range = VOCAB_SIZE
            search_iters = NUM_TILES

            max_logit = -float('inf')
            min_logit = float('inf')

            force_remove_logit = -float('inf') # for handling duplicate cases (edge case)
            num_force_remove = tl.zeros((), dtype=tl.uint32) 

            # First Pass: Compute avg and std from a sample block 
            offs = tl.arange(0, BLOCK_SIZE)
            mask_n = offs < VOCAB_SIZE
            logits_blk = tl.load(LOGITS_ROW + offs, mask=mask_n, other=0.0)
            avg_logit = tl.sum(logits_blk) / VOCAB_SIZE
            sq_avg_logit = tl.sum(logits_blk * logits_blk) / VOCAB_SIZE
            std_logit = tl.sqrt(sq_avg_logit - avg_logit * avg_logit)

            outlier_pivot = avg_logit + SIGMA * std_logit
            num_outliers = tl.zeros((), dtype=tl.uint32)
            sum_outlier_probs = 0.0
            sum_exp_logits = 0.0

            # ====== Second Pass: compute max and min logits ======
            for i in range(0, search_iters): 
                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask_n = offs_n < search_range
                logits_blk = tl.load(LOGITS_ROW + offs_n,
                                     mask=mask_n,
                                     other=avg_logit)
                max_logit = tl.maximum(max_logit, tl.max(logits_blk))
                min_logit = tl.minimum(min_logit, tl.min(logits_blk))

            # ====== Third pass: Calculate exp logits and sum ======
            for i in range(0, search_iters):
                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask_n = offs_n < search_range

                probs_blk = tl.load(LOGITS_ROW + offs_n,
                                    mask=mask_n,
                                    other=-float('inf'))
                probs_blk = probs_blk - max_logit
                probs_blk = tl.exp(probs_blk)
                sum_exp_logits += tl.sum(probs_blk)
                tl.store(PROBS_ROW + offs_n, probs_blk, mask=mask_n) 
            outlier_prob = tl.exp(outlier_pivot - max_logit) / sum_exp_logits

            # ====== Fourth pass: Calculate probs and get outliers ======
            for i in range(0, search_iters):
                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask_n = offs_n < search_range

                probs_blk = tl.load(PROBS_ROW + offs_n, mask=mask_n, other=0.0)
                probs_blk = probs_blk / sum_exp_logits
                tl.store(PROBS_ROW + offs_n, probs_blk, mask=mask_n)

                outlier_mask = (probs_blk > outlier_prob) & mask_n
                sum_outlier_probs += tl.sum(outlier_mask * probs_blk)
                num_blk_outliers = tl.sum(outlier_mask)
                cumulative_pos = tl.cast(
                    tl.cumsum(outlier_mask) - 1 + num_outliers, tl.int32)
                num_outliers += num_blk_outliers
                write_pos = tl.where(outlier_mask, cumulative_pos, -1)
                tl.store(PROBS_2_ROW + write_pos, probs_blk, mask=outlier_mask) # stores the final probs after masking to PROBS_2 

            max_range = tl.exp(max_logit - max_logit) / sum_exp_logits
            min_range = tl.exp(min_logit - max_logit) / sum_exp_logits

            if sum_outlier_probs > p:
                min_range = outlier_prob
                search_address = PROBS_2_ROW
                search_range = tl.cast(num_outliers, tl.int32)
                search_iters = tl.cast(
                    (num_outliers + BLOCK_SIZE - 1) // BLOCK_SIZE, tl.int32)

            second_max_logit = -float('inf')

            num_iters = 0
            p_pivots_sum_0 = 0.0
            min_larger_0 = 1.0
            num_min_larger_0 = tl.zeros((), dtype=tl.uint32)

            # ====== Fifth Passes: Search for p_pivot(2log_2(n)) ======
            while p_pivot == -float('inf') and num_iters < 32:
                p_pivot_0 = (max_range - min_range) * 1.0 / 2.0 + min_range
                p_pivots_sum_0 = 0.0

                min_larger_0 = 1.0
                num_min_larger_0 = tl.zeros((), dtype=tl.uint32)

                for i in range(0, search_iters):
                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask_n = offs_n < search_range
                    probs_blk = tl.load(search_address + offs_n,
                                        mask=mask_n,
                                        other=0.0)

                    masked_larger_0 = tl.where(probs_blk > p_pivot_0,
                                            probs_blk, 1.0)
                    min_larger_0 = tl.minimum(min_larger_0,
                                            tl.min(masked_larger_0))

                    p_pivots_sum_0 += tl.sum(probs_blk *
                                            (probs_blk > p_pivot_0))

                for i in range(0, search_iters):
                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask_n = offs_n < search_range
                    probs_blk = tl.load(search_address + offs_n,
                                        mask=mask_n,
                                        other=0.0)

                    num_min_larger_0 += tl.sum(
                        tl.abs(probs_blk - min_larger_0) < 1e-7)
                
                # Check if any of the pivots are equal to k
                if p_pivots_sum_0 >= p:
                    if p_pivots_sum_0 - (min_larger_0 * num_min_larger_0) < p:
                        p_pivot = p_pivot_0
                    else:
                        min_range = p_pivot_0
                else:
                    max_range = p_pivot_0

                num_iters += 1
                if num_iters >= 32 or tl.abs(min_range - max_range) < 1e-8:
                    p_pivot = p_pivot_0

            # At least one value should be greater than p_pivot
            # if p_pivot >= max_logit:
            #     p_pivot = second_max_logit
            if True:
                # Force remove duplicates (p_pivot is made to include all
                # duplicates if it falls on the duplicates)
                num_force_remove = tl.cast((p_pivots_sum_0 - p) / min_larger_0,
                                        tl.uint32)
                force_remove_logit = tl.log(
                    min_larger_0 * sum_exp_logits) + max_logit

            # p_pivot = tl.log(p_pivot * sum_exp_logits) + max_logit
            p_pivot_logit = -float('inf')

            # -------- widen cutoff by 10% ----------------
            if p_pivot != -float('inf'):
                # WIDEN_NUM 
                widened_prob = p_pivot * (WIDEN_NUM / 100.0)
                # clamp widened_prob to <= max possible prob
                widened_prob = tl.minimum(widened_prob, max_range)
                p_pivot_logit = tl.log(widened_prob * sum_exp_logits) + max_logit

            current_num_force_remove = tl.zeros((), dtype=tl.uint32)
            kept_write_pos = 0

            for i in range(0, NUM_TILES):
                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask_n = offs_n < VOCAB_SIZE
                logits_blk = tl.load(LOGITS_ROW + offs_n,
                                    mask=mask_n,
                                    other=-float('inf'))
                probs_blk = tl.load(PROBS_ROW + offs_n,
                                   mask=mask_n,
                                   other=0.0)

                # if force_remove_logit != -float('inf'):
                #     # Force remove duplicates
                #     tolerance = 1e-5 * tl.maximum(1.0, tl.abs(force_remove_logit))
                #     force_remove_mask = tl.abs(
                #         logits_blk - force_remove_logit) < tolerance
                #     force_remove_count = tl.cumsum(force_remove_mask) + current_num_force_remove
                #     force_remove_count_mask = force_remove_count <= num_force_remove
                #     force_remove_mask = force_remove_count_mask & force_remove_mask
                #     logits_blk = tl.where(force_remove_mask, -float('inf'), logits_blk)
                #     current_num_force_remove = tl.max(force_remove_count)

                # Apply widened cutoff
                keep_mask = logits_blk > p_pivot_logit
                out_vals = tl.where(keep_mask, logits_blk, -float('inf'))
                tl.store(LOGITS_ROW + offs_n, out_vals, mask=mask_n)

                # ====== keeping track of L and idx_tensor ======
                n_kept = tl.sum(keep_mask, dtype=tl.int32)
                if n_kept > 0:
                    cpos = tl.cast(tl.cumsum(keep_mask) - 1 + kept_write_pos, tl.int32)
                    write_idx = tl.where(keep_mask, cpos, 0) 

                    tl.store(IDX_ROW + write_idx, offs_n, mask=keep_mask)
                    tl.store(PROBS_2_ROW + write_idx, probs_blk, mask=keep_mask)

                    kept_write_pos += n_kept
            tl.store(L + row_id, tl.cast(kept_write_pos, tl.int32))
    
def apply_top_p_filtered (
    logits: torch.Tensor,
    p: torch.Tensor,
) -> torch.Tensor:
    """
    Applies top-p using pivot-based filtering 
    """
    # logits = torch.ones((10,), device=logits.device, dtype=torch.float32).view(1, -1) 
    logits_copy = logits.clone().detach()
    # p = torch.full((logits.shape[0],), 0.65, dtype=torch.float32, device=logits.device)
    # output  = apply_top_k_top_p(logits_copy, None, p)
    # print(f"original value = {output}")
    batch_size, vocab_size = logits.shape
    
    # print(f"logits: {logits}", flush=True)
    probs = torch.full((batch_size, vocab_size), -float('inf'), device=logits.device)
    probs_2 = torch.full((batch_size, vocab_size), -float('inf'), device=logits.device)

    l = torch.zeros((batch_size,), device=logits.device, dtype=torch.int32)
    idx_tensor = torch.full_like(logits, 0, dtype=torch.int32)
    

    BLOCK_SIZE = 2048
    SIGMA = 2.15   
    NUM_WARPS = 16
    NUM_STAGES = 3
    WIDEN_NUM = 0 # ----------------------------> ????????

    if not torch.any(p < 1.0):  
        return logits

    grid = lambda meta: ((batch_size + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'], )
    top_p_pivot_filter[grid](
        logits,
        l,
        probs,
        probs_2,
        idx_tensor,
        p*1.1,
        batch_size,
        SIGMA=SIGMA,
        VOCAB_SIZE=vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
        WIDEN_NUM=WIDEN_NUM
    )
    # print(f"logits: {logits}", flush=True)
    # print(f"l: {l}", flush=True)
    # print(f"probs: {probs}", flush=True)
    # print(f"probs_2: {probs_2}", flush=True)
    # print(f"idx_tensor: {idx_tensor}", flush=True)

    max_l = torch.max(l)

    # if max_l.item() == 0:
    #     return torch.full((batch_size, vocab_size), -float('inf'), device=logits.device)
     

    # outliers = torch.arange(0, vocab_size, dtype=torch.int32, device=logits.device).unsqueeze(0).expand(logits.shape[0], -1)
    # probs = torch.softmax(logits, dim=1) 

    outliers = idx_tensor[:, :max_l]
    full_probs = logits_copy.softmax(dim=-1)
    filtered_full_probs = torch.gather(full_probs, 1, outliers)
    filtered_logits = torch.gather(logits, 1, outliers)
    # print(f"outliers: {outliers}", flush=True)
    # print(f"filtered_logits: {filtered_logits}", flush=True)

    probs = torch.gather(probs, 1, outliers) 
    cum_sum = torch.sum(probs, dim=1)

    # print(f"sum : {cum_sum}", flush=True)
    

    filtered_logits_sort, sort_indices = torch.sort(
        filtered_logits, dim=-1, descending=False
    )

    

    # print(f"filtered_logits_sort: {filtered_logits_sort}", flush=True)

    outliers_sorted = torch.gather(outliers, 1, sort_indices)
    filtered_probs_sort = torch.gather(filtered_full_probs, 1, sort_indices)

    # print(f"outliers_sorted: {outliers_sorted}", flush=True)
    # print(f"filtered_probs_sort: {filtered_logits_sort}", flush=True)
    

    probs_sum = torch.cumsum(filtered_probs_sort, dim=-1, out=filtered_probs_sort)
    top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
    top_p_mask[:, -1] = False

    # print(f"probs_sum: {probs_sum}")
    # print(f"top_p_mask = {top_p_mask}")

    filtered_logits_sort.masked_fill_(top_p_mask, -float("inf")) 

    # print(f"top_p_mask = {filtered_logits_sort}")
    

    logits.fill_(-float("inf"))
    logits.scatter_(dim=1, index=outliers_sorted, src=filtered_logits_sort)

    # print(f"final logits after scatter = {logits}")

    # assert False 
    return logits


def apply_top_k_top_p_test2(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
) -> torch.Tensor:
    """
    Uses pivot-based algorithm to filter --> sort 
    """
    if k is None and p is None: 
        return logits 
    elif p is None and k is not None: 
        return apply_top_k_only(logits, k)
    elif k is None and p is not None:
        return apply_top_p_filtered(logits, p)
    else:
        logits_k = apply_top_k_only(logits, k)
        return apply_top_p_filtered(logits, p)