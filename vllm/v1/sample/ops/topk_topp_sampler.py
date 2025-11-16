# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch
import torch.nn as nn
import triton
import triton.language as tl
from packaging import version

from vllm import envs
from vllm.config.model import LogprobsMode
from vllm.logger import init_logger
from vllm.platforms import CpuArchEnum, current_platform

logger = init_logger(__name__)


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
            if envs.VLLM_USE_FLASHINFER_SAMPLER:
                # Users must opt in explicitly via VLLM_USE_FLASHINFER_SAMPLER=1.
                logger.info_once(
                    "Using FlashInfer for top-p & top-k sampling.",
                    scope="global",
                )
                self.forward = self.forward_cuda
            else:
                logger.debug_once(
                    "FlashInfer top-p/top-k sampling is available but disabled "
                    "by default. Set VLLM_USE_FLASHINFER_SAMPLER=1 to opt in "
                    "after verifying accuracy for your workloads."
                )
                self.forward = self.forward_native

            if envs.VLLM_USE_TRITON_SAMPLER:
                logger.info_once("Using Triton for top-p & top-k sampling.")
                if envs.VLLM_USE_FLASHINFER_SAMPLER:
                    logger.info_once(
                        "Overriding FlashInfer with Triton for top-p & top-k sampling."
                    )
                self.forward = self.forward_triton
            else:
                logger.warning_once(
                    "Triton top-p/top-k sampling is available but disabled "
                    "by default. Set VLLM_USE_TRITON_SAMPLER=1 to opt in "
                    "after verifying accuracy for your workloads."
                )
                self.forward = self.forward_native
        elif current_platform.is_cpu():
            arch = current_platform.get_cpu_architecture()
            # Fall back to native implementation for POWERPC and RISCV.
            # On PowerPC argmax produces incorrect output with torch.compile.
            # PR: https://github.com/vllm-project/vllm/pull/26987
            if arch in (CpuArchEnum.RISCV, CpuArchEnum.POWERPC):
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
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
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

        if len(generators) != logits.shape[0]:
            return compiled_random_sample(logits), logits_to_return
        else:
            probs = logits.softmax(dim=-1, dtype=torch.float32)
            q = torch.empty_like(probs)
            q.exponential_()
            for i, generator in generators.items():
                q[i].exponential_(generator=generator)

            return probs.div_(q).argmax(dim=-1).view(-1), logits_to_return


# Note: this is a workaround for
# https://github.com/pytorch/pytorch/pull/151218
@torch.compile(dynamic=True)
def compiled_random_sample(logits: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    q = torch.empty_like(probs)
    q.exponential_()
    return probs.div(q).argmax(dim=-1).view(-1)


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
    k: torch.Tensor | None,
    p: torch.Tensor | None,
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
    import flashinfer

    if version.parse(flashinfer.__version__) < version.parse("0.2.3"):
        raise ImportError(
            "FlashInfer version >= 0.2.3 required for top-k and top-p sampling. "
        )

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


# fmt: off
_PERCENTILE_TO_STD_TABLE = [
     2.576,  2.326,  2.054,  1.881,  1.751,
     1.645,  1.555,  1.476,  1.405,  1.341,
     1.282,  1.227,  1.175,  1.126,  1.080,
     1.036,  0.994,  0.954,  0.915,  0.878,
     0.842,  0.806,  0.772,  0.739,  0.706,
     0.674,  0.643,  0.613,  0.583,  0.553,
     0.524,  0.496,  0.468,  0.440,  0.412,
     0.385,  0.358,  0.332,  0.305,  0.279,
     0.253,  0.228,  0.202,  0.176,  0.151,
     0.126,  0.100,  0.075,  0.050,  0.025,
     0.000, -0.025, -0.050, -0.075, -0.100,
    -0.126, -0.151, -0.176, -0.202, -0.228,
    -0.253, -0.279, -0.305, -0.332, -0.358,
    -0.385, -0.412, -0.440, -0.468, -0.496,
    -0.524, -0.553, -0.583, -0.613, -0.643,
    -0.674, -0.706, -0.739, -0.772, -0.806,
    -0.842, -0.878, -0.915, -0.954, -0.994,
    -1.036, -1.080, -1.126, -1.175, -1.227,
    -1.282, -1.341, -1.405, -1.476, -1.555,
    -1.645, -1.751, -1.881, -2.054, -2.326
]
# fmt: on


def apply_top_k_top_p_triton(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
) -> torch.Tensor:
    """
    Uses pivot-based algorithm to filter --> sort
    """

    if k is None and p is None:
        return logits
    if p is None and k is not None:
        return apply_top_k_only_triton(logits, k)
    # Fallback to torch for small batch sizes for top-p
    if logits.shape[0] < 16 or logits.shape[1] < 32768:
        return apply_top_k_top_p(logits, k, p)
    return apply_top_k_top_p_filtered(logits, k, p)


@triton.jit
def _topk_triton_kernel(
    LOGITS,
    OUTPUT,
    PERCENTILE_TO_STD_TABLE,
    K,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    NUM_TILES: tl.constexpr = (VOCAB_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE
    k = tl.load(K + row_id)

    if k != VOCAB_SIZE:
        # THERE IS NO DUPLICATE LOGIT MANAGEMENT FOR THIS TOP-K KERNEL
        # CURRENT IMPLEMENTATION INCLUDES ALL DUPLICATE LOGITS,
        # WHICH MAY RETURN MORE THAN K LOGITS.
        # THIS FOLLOWS THE IMPLEMENTATION IN apply_top_k_only().

        LOGITS_ROW = LOGITS + row_id * VOCAB_SIZE
        OUTPUT_ROW = OUTPUT + row_id * VOCAB_SIZE
        search_addr = LOGITS_ROW
        search_range = VOCAB_SIZE
        search_iters = NUM_TILES

        max_logit = -float("inf")
        min_logit = float("inf")

        # Zeroth pass: Compute avg and std from a sample block
        # May produce incorrect results if VOCAB_SIZE < BLOCK_SIZE
        offs = tl.arange(0, BLOCK_SIZE)
        mask_n = offs < VOCAB_SIZE
        num_valid = tl.sum(mask_n)
        logits_blk = tl.load(LOGITS_ROW + offs, mask=mask_n, other=0.0)
        avg_logit = tl.sum(logits_blk) / num_valid
        sq_avg_logit = tl.sum(logits_blk * logits_blk) / num_valid
        std_logit = tl.sqrt(sq_avg_logit - avg_logit * avg_logit)

        percentile = tl.cast(k * 2.0 / VOCAB_SIZE * 100, tl.uint32) + 1
        percentile = tl.minimum(percentile, 99)
        sigma = tl.load(PERCENTILE_TO_STD_TABLE + percentile)
        outlier_pivot = avg_logit + sigma * std_logit
        num_outliers = tl.zeros((), dtype=tl.uint32)

        # First pass: compute max and min logits and gather outliers
        for i in range(0, search_iters):
            offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask_n = offs_n < search_range
            logits_blk = tl.load(search_addr + offs_n, mask=mask_n, other=avg_logit)

            max_logit = tl.maximum(max_logit, tl.max(logits_blk))
            min_logit = tl.minimum(min_logit, tl.min(logits_blk))

            outlier_mask = (logits_blk > outlier_pivot) & mask_n
            num_blk_outliers = tl.sum(outlier_mask)
            cumulative_pos = tl.cast(
                tl.cumsum(outlier_mask) - 1 + num_outliers, tl.int32
            )
            num_outliers += num_blk_outliers
            write_pos = tl.where(outlier_mask, cumulative_pos, -1)
            tl.store(OUTPUT_ROW + write_pos, logits_blk, mask=outlier_mask)

        max_range = max_logit
        min_range = min_logit
        if num_outliers > k:
            max_range = max_logit
            min_range = outlier_pivot
            search_addr = OUTPUT_ROW
            search_range = tl.cast(num_outliers, tl.int32)
            search_iters = tl.cast(
                (num_outliers + BLOCK_SIZE - 1) // BLOCK_SIZE, tl.int32
            )

        # Second passes: Quaternary search for pivots (nlog_4(n))
        num_iters = 0
        k_pivot = -float("inf")
        while k_pivot == -float("inf"):
            k_pivot_0 = (max_range - min_range) * 1.0 / 4.0 + min_range
            k_pivot_1 = (max_range - min_range) * 2.0 / 4.0 + min_range
            k_pivot_2 = (max_range - min_range) * 3.0 / 4.0 + min_range
            k_pivots_num_0 = tl.zeros((), dtype=tl.uint32)
            k_pivots_num_1 = tl.zeros((), dtype=tl.uint32)
            k_pivots_num_2 = tl.zeros((), dtype=tl.uint32)

            for i in range(0, search_iters):
                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask_n = offs_n < search_range
                logits_blk = tl.load(
                    search_addr + offs_n, mask=mask_n, other=-float("inf")
                )

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
            if num_iters >= 32 or tl.abs(min_range - max_range) < 1e-16:
                k_pivot = k_pivot_0

        # Third pass: Apply top-k mask
        if k_pivot != -float("inf"):
            for i in range(0, NUM_TILES):
                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask_n = offs_n < VOCAB_SIZE
                logits_blk = tl.load(LOGITS_ROW + offs_n, mask=mask_n)
                mask = logits_blk > k_pivot
                logits_blk = tl.where(mask, logits_blk, -float("inf"))
                tl.store(OUTPUT_ROW + offs_n, logits_blk, mask=mask_n)


def apply_top_k_only_triton(
    logits: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    """
    Apply top-k mask to the logits using Triton.

    The logits tensor will be updated out-of-place.
    """
    if k is None:
        return logits

    batch_size, vocab_size = logits.shape
    NUM_PROGRAMS = batch_size  # Non-persistent kernel
    BLOCK_SIZE = 8192
    NUM_WARPS = 16
    NUM_STAGES = 3
    output = torch.full(logits.shape, -float("inf"), device=logits.device)
    PERCENTILE_TO_STD_TABLE = torch.tensor(
        _PERCENTILE_TO_STD_TABLE, device=logits.device
    )

    _topk_triton_kernel[(NUM_PROGRAMS,)](
        logits,
        output,
        PERCENTILE_TO_STD_TABLE,
        k,
        vocab_size,
        BLOCK_SIZE,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )

    return output


@triton.jit
def top_k_top_p_filter(
    LOGITS,
    DO_TOP_K,
    K,
    P,
    P_FIL,
    BUFFER,
    BATCH_SIZE,
    SUM_EXCLUDED_PROBS,
    FILTERED_LOGITS,
    FILTERED_INDICES,
    FILTERED_PROBS,
    PERCENTILE_TO_STD_TABLE,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    NUM_TILES: tl.constexpr = (VOCAB_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    for row_id in tl.range(pid, BATCH_SIZE, num_programs):
        LOGITS_ROW = LOGITS + row_id * VOCAB_SIZE
        BUFFER_ROW = BUFFER + pid * VOCAB_SIZE

        search_addr = LOGITS_ROW
        search_range = VOCAB_SIZE
        search_iters = NUM_TILES

        max_logit = -float("inf")
        min_logit = float("inf")

        # Zeroth pass: Compute avg and std from a sample block
        offs = tl.arange(0, BLOCK_SIZE)
        mask_n = offs < VOCAB_SIZE
        num_mask = tl.sum(mask_n)
        logits_blk = tl.load(LOGITS_ROW + offs, mask=mask_n, other=0.0)
        avg_logit = tl.sum(logits_blk) / num_mask
        sq_avg_logit = tl.sum(logits_blk * logits_blk) / num_mask
        std_logit = tl.sqrt(sq_avg_logit - avg_logit * avg_logit)

        percentile = tl.cast(P_FIL * 2.0 / VOCAB_SIZE * 100 + 4, tl.uint32)
        percentile = tl.minimum(percentile, 99)
        sigma = tl.load(PERCENTILE_TO_STD_TABLE + percentile)
        outlier_pivot = avg_logit + sigma * std_logit
        num_outliers = tl.zeros((), dtype=tl.uint32)

        # First pass: compute max and min logits and gather outliers
        for i in range(0, search_iters):
            offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask_n = offs_n < search_range
            logits_blk = tl.load(search_addr + offs_n, mask=mask_n, other=avg_logit)
            probs_blk = tl.load(BUFFER_ROW + offs_n, mask=mask_n, other=0.0)

            max_logit = tl.maximum(max_logit, tl.max(logits_blk))
            min_logit = tl.minimum(min_logit, tl.min(logits_blk))

            outlier_mask = (logits_blk > outlier_pivot) & mask_n
            num_blk_outliers = tl.sum(outlier_mask)
            cumulative_pos = tl.cast(
                tl.cumsum(outlier_mask) - 1 + num_outliers, tl.int32
            )
            num_outliers += num_blk_outliers

            write_idx = tl.where(outlier_mask, cumulative_pos, -1)
            tl.store(BUFFER_ROW + write_idx, logits_blk, mask=outlier_mask)

        k_max_range = max_logit
        k_min_range = min_logit
        p_fil_max_range = max_logit
        p_fil_min_range = min_logit

        if num_outliers > P_FIL:
            search_addr = BUFFER_ROW
            search_range = tl.cast(num_outliers, tl.int32)
            search_iters = tl.cast(
                (num_outliers + BLOCK_SIZE - 1) // BLOCK_SIZE, tl.int32
            )
            k_min_range = outlier_pivot
            p_fil_min_range = outlier_pivot

        k = tl.load(K + row_id)

        # Second passes: Quaternary search for pivots (nlog_4(n))
        num_iters = 0
        k_pivot = -float("inf")
        p_fil_pivot = -float("inf")
        while k_pivot == -float("inf") or p_fil_pivot == -float("inf"):
            k_pivot_0 = (k_max_range - k_min_range) * 1.0 / 4.0 + k_min_range
            k_pivot_1 = (k_max_range - k_min_range) * 2.0 / 4.0 + k_min_range
            k_pivot_2 = (k_max_range - k_min_range) * 3.0 / 4.0 + k_min_range
            k_pivots_num_0 = tl.zeros((), dtype=tl.uint32)
            k_pivots_num_1 = tl.zeros((), dtype=tl.uint32)
            k_pivots_num_2 = tl.zeros((), dtype=tl.uint32)

            p_fil_pivot_0 = (
                p_fil_max_range - p_fil_min_range
            ) * 1.0 / 4.0 + p_fil_min_range
            p_fil_pivot_1 = (
                p_fil_max_range - p_fil_min_range
            ) * 2.0 / 4.0 + p_fil_min_range
            p_fil_pivot_2 = (
                p_fil_max_range - p_fil_min_range
            ) * 3.0 / 4.0 + p_fil_min_range
            p_fil_pivots_num_0 = tl.zeros((), dtype=tl.uint32)
            p_fil_pivots_num_1 = tl.zeros((), dtype=tl.uint32)
            p_fil_pivots_num_2 = tl.zeros((), dtype=tl.uint32)

            for i in range(0, search_iters):
                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask_n = offs_n < search_range
                logits_blk = tl.load(
                    search_addr + offs_n, mask=mask_n, other=-float("inf")
                )

                k_pivots_num_0 += tl.sum(logits_blk > k_pivot_0)
                k_pivots_num_1 += tl.sum(logits_blk > k_pivot_1)
                k_pivots_num_2 += tl.sum(logits_blk > k_pivot_2)

                p_fil_pivots_num_0 += tl.sum(logits_blk > p_fil_pivot_0)
                p_fil_pivots_num_1 += tl.sum(logits_blk > p_fil_pivot_1)
                p_fil_pivots_num_2 += tl.sum(logits_blk > p_fil_pivot_2)

            # Check if any of the pivots are equal to k
            if k_pivot == -float("inf"):
                if k_pivots_num_0 == k:
                    k_pivot = k_pivot_0
                elif k_pivots_num_1 == k:
                    k_pivot = k_pivot_1
                elif k_pivots_num_2 == k:
                    k_pivot = k_pivot_2
                # If none of the pivots are equal to k, we update the range
                elif k_pivots_num_2 > k:
                    k_min_range = k_pivot_2
                elif k_pivots_num_1 > k:
                    k_min_range = k_pivot_1
                elif k_pivots_num_0 > k:
                    k_min_range = k_pivot_0
                if k_pivots_num_0 < k:
                    k_max_range = k_pivot_0
                elif k_pivots_num_1 < k:
                    k_max_range = k_pivot_1
                elif k_pivots_num_2 < k:
                    k_max_range = k_pivot_2

            # Check if any of the pivots are equal to P_FIL
            if p_fil_pivot == -float("inf"):
                if p_fil_pivots_num_0 == P_FIL:
                    p_fil_pivot = p_fil_pivot_0
                elif p_fil_pivots_num_1 == P_FIL:
                    p_fil_pivot = p_fil_pivot_1
                elif p_fil_pivots_num_2 == P_FIL:
                    p_fil_pivot = p_fil_pivot_2
                # If none of the pivots are equal to P_FIL, we update the range
                elif p_fil_pivots_num_2 > P_FIL:
                    p_fil_min_range = p_fil_pivot_2
                elif p_fil_pivots_num_1 > P_FIL:
                    p_fil_min_range = p_fil_pivot_1
                elif p_fil_pivots_num_0 > P_FIL:
                    p_fil_min_range = p_fil_pivot_0
                if p_fil_pivots_num_0 < P_FIL:
                    p_fil_max_range = p_fil_pivot_0
                elif p_fil_pivots_num_1 < P_FIL:
                    p_fil_max_range = p_fil_pivot_1
                elif p_fil_pivots_num_2 < P_FIL:
                    p_fil_max_range = p_fil_pivot_2

            num_iters += 1
            if num_iters >= 32 or (
                (tl.abs(k_min_range - k_max_range) < 1e-16 and k_pivot != -float("inf"))
                and (
                    tl.abs(p_fil_min_range - p_fil_max_range) < 1e-16
                    and p_fil_pivot != -float("inf")
                )
            ):
                if k_pivot == -float("inf"):
                    k_pivot = k_pivot_0
                if p_fil_pivot == -float("inf"):
                    p_fil_pivot = p_fil_pivot_0

        # Third pass: Calculate exp logits and sum with top-k mask
        if not DO_TOP_K or k == VOCAB_SIZE:
            k_pivot = -float("inf")

        sum_exp_logits = tl.zeros((), dtype=tl.float32)
        for i in range(0, NUM_TILES):
            offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask_n = offs_n < VOCAB_SIZE
            logits_blk = tl.load(LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf"))

            top_k_mask = logits_blk > k_pivot
            logits_blk = tl.where(top_k_mask, logits_blk, -float("inf"))

            probs_blk = logits_blk - max_logit
            probs_blk = tl.exp(probs_blk)
            sum_exp_logits += tl.sum(probs_blk)
            tl.store(BUFFER_ROW + offs_n, probs_blk, mask=mask_n)

        # Fourth pass: Calculate softmax
        for i in range(0, NUM_TILES):
            offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask_n = offs_n < VOCAB_SIZE
            probs_blk = tl.load(BUFFER_ROW + offs_n, mask=mask_n, other=0.0)
            probs_blk = probs_blk / sum_exp_logits
            tl.store(BUFFER_ROW + offs_n, probs_blk, mask=mask_n)

        # Fifth pass : Gather filtered values with top-k mask
        write_pos = tl.zeros((), dtype=tl.int32)
        sum_excluded_probs = tl.zeros((), dtype=tl.float32)
        FILTERED_LOGITS_ROW = FILTERED_LOGITS + row_id * P_FIL
        FILTERED_INDICES_ROW = FILTERED_INDICES + row_id * P_FIL
        FILTERED_PROBS_ROW = FILTERED_PROBS + row_id * P_FIL
        for i in range(0, NUM_TILES):
            offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask_n = offs_n < VOCAB_SIZE
            logits_blk = tl.load(LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf"))
            probs_blk = tl.load(BUFFER_ROW + offs_n, mask=mask_n, other=0.0)

            keep_mask = (logits_blk > p_fil_pivot) & mask_n
            cpos = tl.cumsum(keep_mask) - 1 + write_pos
            f_mask = keep_mask & (cpos < P_FIL)
            write_idx = tl.where(f_mask, cpos, 0)

            top_k_mask = logits_blk > k_pivot
            logits_blk = tl.where(top_k_mask, logits_blk, -float("inf"))

            # Gather filtered values
            tl.store(FILTERED_LOGITS_ROW + write_idx, logits_blk, mask=f_mask)
            tl.store(FILTERED_INDICES_ROW + write_idx, offs_n, mask=f_mask)
            tl.store(FILTERED_PROBS_ROW + write_idx, probs_blk, mask=f_mask)

            sum_excluded_probs += tl.sum(probs_blk * ((~f_mask) & mask_n))
            write_pos += tl.sum(f_mask, dtype=tl.int32)
        tl.store(SUM_EXCLUDED_PROBS + row_id, sum_excluded_probs)


def apply_top_k_top_p_filtered(
    logits: torch.Tensor,
    k: torch.Tensor,
    p: torch.Tensor,
) -> torch.Tensor:
    """
    Applies top p using pivot based filtering
    """
    batch_size, vocab_size = logits.shape

    # If k is too large, speedup is not significant as the filtered set is large.
    max_k = k.max().item() if k is not None else 0
    # Probability value is not guaranteed to be equivalent to the PyTorch implementation
    # in the distribution tail due to floating point non-associativity.
    # We avoid high p values to avoid this accuracy issue.
    max_p = p.max().item() if p is not None else 0
    if max_k > vocab_size / 10 or max_p > 0.95:
        return apply_top_k_top_p(logits, k, p)

    BLOCK_SIZE = 8192
    device_prop = torch.cuda.get_device_properties(logits.device)
    NUM_PROGRAMS = device_prop.multi_processor_count  # Persistent kernel
    NUM_WARPS = 16
    NUM_STAGES = 3
    buffer = torch.empty(
        (NUM_PROGRAMS, vocab_size), device=logits.device, dtype=torch.float32
    )
    p_filter = (
        min(int(max_k * 1.2), vocab_size - 1) if k is not None else int(vocab_size / 32)
    )
    filtered_logits = torch.full(
        (batch_size, p_filter), -float("inf"), device=logits.device
    )
    filtered_indices = torch.full(
        (batch_size, p_filter), p_filter, dtype=torch.int64, device=logits.device
    )
    filtered_probs = torch.full(
        (batch_size, p_filter), -float("inf"), device=logits.device
    )
    sum_excluded_probs = torch.zeros(
        (batch_size,), device=logits.device, dtype=torch.float32
    )
    PERCENTILE_TO_STD_TABLE = torch.tensor(
        _PERCENTILE_TO_STD_TABLE, device=logits.device
    )

    top_k_top_p_filter[(NUM_PROGRAMS,)](
        logits,
        (k is not None),
        k if k is not None else filtered_indices,
        p,
        p_filter,
        buffer,
        batch_size,
        sum_excluded_probs,
        filtered_logits,
        filtered_indices,
        filtered_probs,
        PERCENTILE_TO_STD_TABLE,
        VOCAB_SIZE=vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )

    if torch.any(sum_excluded_probs >= p):
        return apply_top_k_top_p(logits, k, p)

    logits_sort, sort_indices = filtered_logits.sort(dim=-1, descending=False)
    logits_sort_indices = torch.gather(filtered_indices, -1, sort_indices)
    sorted_probs = torch.gather(filtered_probs, -1, sort_indices)

    sorted_probs[:, 0] = sorted_probs[:, 0] + sum_excluded_probs
    probs_sum = torch.cumsum(sorted_probs, dim=-1)
    top_p_mask = probs_sum <= (1 - p.unsqueeze(dim=-1))
    top_p_mask[:, -1] = False
    logits_sort.masked_fill_(top_p_mask, -float("inf"))

    logits.fill_(-float("inf"))
    logits.scatter_(dim=1, index=logits_sort_indices, src=logits_sort)
    return logits
