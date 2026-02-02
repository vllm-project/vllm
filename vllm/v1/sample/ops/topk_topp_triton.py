# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Combined Top-K and Top-P Triton kernels.

Based on the paper "Qrita: High-performance Top-k and Top-p Algorithm for GPUs 
using Pivot-based Truncation and Selection" By Park et al.

"""

import torch

from vllm.triton_utils import tl, triton


_TRITON_TABLE_CACHE: dict[
    tuple[torch.device, torch.dtype], (torch.Tensor, torch.Tensor)
] = {}
_TRITON_BUFFER_CACHE: dict[
    tuple[torch.device, torch.dtype, int, int], torch.Tensor
] = {}  

_NORMAL_CDF_TO_SIGMA_TABLE = [
  3.656,  3.650,  3.650,  3.650,  3.626,  3.626,  3.626,  3.514,  3.514,  3.503, 
  3.503,  3.434,  3.434,  3.428,  3.428,  3.387,  3.380,  3.380,  3.376,  3.373, 
  3.373,  3.356,  3.354,  3.354,  3.291,  3.249,  3.234,  3.214,  3.198,  3.198, 
  3.185,  3.177,  3.177,  3.165,  3.164,  3.161,  3.138,  3.120,  3.115,  3.113, 
  3.093,  3.066,  3.054,  3.043,  3.037,  3.023,  2.993,  2.991,  2.976,  2.970, 
  2.952,  2.946,  2.932,  2.908,  2.902,  2.895,  2.886,  2.874,  2.861,  2.844, 
  2.836,  2.810,  2.801,  2.790,  2.784,  2.779,  2.767,  2.757,  2.745,  2.733, 
  2.723,  2.716,  2.693,  2.678,  2.671,  2.656,  2.649,  2.629,  2.611,  2.595, 
  2.592,  2.585,  2.574,  2.550,  2.543,  2.534,  2.521,  2.518,  2.497,  2.485, 
  2.468,  2.450,  2.441,  2.430,  2.412,  2.402,  2.389,  2.383,  2.377,  2.364, 
  2.349,  2.338,  2.332,  2.319,  2.310,  2.301,  2.282,  2.274,  2.266,  2.250, 
  2.242,  2.236,  2.226,  2.215,  2.207,  2.196,  2.179,  2.171,  2.162,  2.147, 
  2.135,  2.121,  2.109,  2.095,  2.085,  2.073,  2.063,  2.045,  2.030,  2.016, 
  2.003,  1.992,  1.983,  1.972,  1.960,  1.949,  1.940,  1.928,  1.912,  1.897, 
  1.881,  1.869,  1.854,  1.838,  1.824,  1.807,  1.792,  1.779,  1.764,  1.751, 
  1.739,  1.726,  1.711,  1.697,  1.685,  1.668,  1.652,  1.636,  1.622,  1.603, 
  1.585,  1.568,  1.551,  1.534,  1.513,  1.499,  1.480,  1.464,  1.441,  1.422, 
  1.394,  1.373,  1.347,  1.320,  1.296,  1.270,  1.246,  1.219,  1.190,  1.163, 
  1.135,  1.104,  1.073,  1.041,  1.006,  0.969,  0.931,  0.894,  0.851,  0.806, 
  0.757,  0.702,  0.643,  0.574,  0.498,  0.405,  0.288,  0.134, -0.110, -3.813 
]

_PERCENTILE_TO_STD_TABLE = [
  2.576,  2.319,  2.178,  2.064,  1.968,  1.892,  1.819,  1.757,  1.708,  1.659, 
  1.616,  1.568,  1.526,  1.492,  1.456,  1.420,  1.382,  1.342,  1.309,  1.280, 
  1.249,  1.221,  1.193,  1.169,  1.145,  1.121,  1.095,  1.073,  1.050,  1.030, 
  1.008,  0.987,  0.966,  0.945,  0.926,  0.910,  0.891,  0.871,  0.854,  0.837, 
  0.819,  0.803,  0.784,  0.767,  0.753,  0.734,  0.719,  0.702,  0.690,  0.675, 
  0.658,  0.640,  0.625,  0.609,  0.595,  0.578,  0.564,  0.550,  0.537,  0.521, 
  0.509,  0.495,  0.481,  0.466,  0.453,  0.439,  0.424,  0.410,  0.397,  0.383, 
  0.370,  0.356,  0.343,  0.330,  0.316,  0.302,  0.289,  0.274,  0.261,  0.247, 
  0.235,  0.223,  0.209,  0.196,  0.184,  0.172,  0.159,  0.149,  0.137,  0.124, 
  0.112,  0.100,  0.086,  0.074,  0.062,  0.050,  0.035,  0.023,  0.009, -0.003, 
 -0.015, -0.027, -0.039, -0.052, -0.063, -0.074, -0.085, -0.097, -0.109, -0.122, 
 -0.134, -0.147, -0.158, -0.171, -0.184, -0.196, -0.210, -0.223, -0.235, -0.248, 
 -0.261, -0.275, -0.289, -0.302, -0.317, -0.328, -0.341, -0.353, -0.368, -0.382, 
 -0.396, -0.410, -0.426, -0.439, -0.452, -0.465, -0.480, -0.493, -0.507, -0.521, 
 -0.537, -0.551, -0.568, -0.582, -0.597, -0.614, -0.628, -0.643, -0.658, -0.673, 
 -0.691, -0.706, -0.721, -0.738, -0.754, -0.769, -0.789, -0.808, -0.824, -0.838, 
 -0.857, -0.877, -0.893, -0.912, -0.929, -0.947, -0.965, -0.983, -1.003, -1.027, 
 -1.050, -1.070, -1.092, -1.117, -1.139, -1.162, -1.189, -1.216, -1.241, -1.272, 
 -1.300, -1.330, -1.367, -1.404, -1.441, -1.485, -1.523, -1.564, -1.607, -1.658, 
 -1.710, -1.778, -1.832, -1.901, -1.978, -2.068, -2.174, -2.325, -2.577, -3.813 
]
# fmt: on

@triton.jit
def _topk_topp_kernel(
    LOGITS, 
    BUFFER, 
    PERCENTILE_TO_STD_TABLE,
    NORMAL_CDF_TO_SIGMA_TABLE,
    K,
    P, 
    BATCH_SIZE,
    VOCAB_SIZE: tl.constexpr, 
    MASK_VALUE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_TRUNC: tl.constexpr,
    TOPK_ENABLED: tl.constexpr,
    TOPP_ENABLED: tl.constexpr,
):
    NUM_TILES: tl.constexpr = (VOCAB_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for row_id in tl.range(pid, BATCH_SIZE, num_programs):

        LOGITS_ROW = LOGITS + row_id * VOCAB_SIZE
        BUFFER_ROW = BUFFER + pid * VOCAB_SIZE

        final_pivot = -float("inf")
        duplicate_logit = float("inf")
        num_duplicate_logit = tl.zeros((), dtype=tl.uint32)
        num_keep = tl.zeros((), dtype=tl.uint32)
        num_kept = tl.zeros((), dtype=tl.uint32)

        max_logit = -float('inf')
        min_logit = float('inf')

        if TOPK_ENABLED:
            k = tl.load(K + row_id)
            if k != VOCAB_SIZE: 

                # Zeroth pass: Compute avg and std from a sample block
                offs = tl.arange(0, BLOCK_SIZE)
                mask_n = offs < VOCAB_SIZE
                num_valid = tl.sum(mask_n)
                logits_blk0 = tl.load(LOGITS_ROW + offs, mask=mask_n, other=0.0)
                avg_logit = tl.sum(logits_blk0) / num_valid
                sq_avg_logit = tl.sum(logits_blk0 * logits_blk0) / num_valid
                std_logit = tl.sqrt(sq_avg_logit - avg_logit * avg_logit)
                std_logit = tl.maximum(std_logit, 0.0)

                # Calculate outlier pivot t for Gaussian sigma-truncation
                percentile = tl.cast(k / VOCAB_SIZE * 200, tl.uint32)
                percentile = tl.minimum(percentile, 199)
                sigma = tl.load(PERCENTILE_TO_STD_TABLE + percentile)
                sigma = sigma + tl.abs(sigma) * -0.25
                outlier_pivot = avg_logit + std_logit * sigma
                num_outliers = tl.zeros((), dtype=tl.uint32)

                # First pass: compute max and min logits and gather outliers
                for i in range(0, NUM_TILES):
                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask_n = offs_n < VOCAB_SIZE
                    logits_blk = tl.load(LOGITS_ROW + offs_n, mask=mask_n, other=avg_logit)

                    max_logit = tl.maximum(max_logit, tl.max(logits_blk))
                    min_logit = tl.minimum(min_logit, tl.min(logits_blk))

                    outlier_mask = (logits_blk > outlier_pivot) & mask_n
                    cumulative_pos = tl.cast(
                        tl.cumsum(outlier_mask) - 1 + num_outliers, tl.int32)
                    num_outliers += tl.sum(outlier_mask)
                    write_pos = tl.where(outlier_mask, cumulative_pos, -1)
                    tl.store(BUFFER_ROW + write_pos, logits_blk, mask=outlier_mask)

                # Second passes: Quaternary search for pivots (nlog_4(batch_size))
                num_iters = 0
                k_pivot = float("inf")
                k_pivots_num = tl.zeros((), dtype=tl.uint32)
                min_larger = float("inf")
                num_min_larger = tl.zeros((), dtype=tl.uint32)
                if num_outliers > k:
                    max_range = max_logit
                    min_range = outlier_pivot
                    search_range = tl.cast(num_outliers, tl.int32)
                    search_iters = tl.cast(
                        (num_outliers + BLOCK_SIZE_TRUNC - 1) // BLOCK_SIZE_TRUNC, tl.int32
                    )
                    while k_pivot == float("inf"):
                        k_pivot_0 = (max_range - min_range) * 1.0 / 4.0 + min_range
                        k_pivots_num_0 = tl.zeros((), dtype=tl.uint32)
                        min_larger_0 = float("inf")
                        num_min_larger_0 = tl.zeros((), dtype=tl.uint32)

                        k_pivot_1 = (max_range - min_range) * 2.0 / 4.0 + min_range
                        k_pivots_num_1 = tl.zeros((), dtype=tl.uint32)
                        min_larger_1 = float("inf")
                        num_min_larger_1 = tl.zeros((), dtype=tl.uint32)

                        k_pivot_2 = (max_range - min_range) * 3.0 / 4.0 + min_range
                        k_pivots_num_2 = tl.zeros((), dtype=tl.uint32)
                        min_larger_2 = float("inf")
                        num_min_larger_2 = tl.zeros((), dtype=tl.uint32)

                        # First pass: Calculate k_pivots_num and min_larger
                        for i in range(0, search_iters):
                            offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(0, BLOCK_SIZE_TRUNC)
                            mask_n_2 = offs_n < search_range
                            logits_blk2 = tl.load(
                                BUFFER_ROW + offs_n, mask=mask_n_2, other=-float("inf")
                            )

                            k_pivots_num_0 += tl.sum(logits_blk2 > k_pivot_0)
                            k_pivots_num_1 += tl.sum(logits_blk2 > k_pivot_1)
                            k_pivots_num_2 += tl.sum(logits_blk2 > k_pivot_2)

                            min_larger_0 = tl.minimum(min_larger_0, tl.min(logits_blk2))
                            min_larger_1 = tl.minimum(min_larger_1, tl.min(logits_blk2))
                            min_larger_2 = tl.minimum(min_larger_2, tl.min(logits_blk2))

                        # Second pass: Calculate num_min_larger
                        for i in range(0, search_iters):
                            offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(0, BLOCK_SIZE_TRUNC)
                            mask_n_2 = offs_n < search_range
                            logits_blk2 = tl.load(
                                BUFFER_ROW + offs_n, mask=mask_n_2, other=-float("inf")
                            )

                            num_min_larger_0 += tl.sum(tl.abs(logits_blk2 - min_larger_0) < 1e-15)
                            num_min_larger_1 += tl.sum(tl.abs(logits_blk2 - min_larger_1) < 1e-15)
                            num_min_larger_2 += tl.sum(tl.abs(logits_blk2 - min_larger_2) < 1e-15)

                        # Check if any of the pivots satisfy termination condition
                        if k_pivots_num_0 >= k:
                            if k_pivots_num_0 - (min_larger_0 * num_min_larger_0) < k:
                                k_pivot = k_pivot_0
                                k_pivots_num = k_pivots_num_0
                                min_larger = min_larger_0
                                num_min_larger = num_min_larger_0
                        if k_pivots_num_1 >= k:
                            if k_pivots_num_1 - (min_larger_1 * num_min_larger_1) < k:
                                k_pivot = k_pivot_1
                                k_pivots_num = k_pivots_num_1
                                min_larger = min_larger_1
                                num_min_larger = num_min_larger_1
                        if k_pivots_num_2 >= k:
                            if k_pivots_num_2 - (min_larger_2 * num_min_larger_2) < k:
                                k_pivot = k_pivot_2
                                k_pivots_num = k_pivots_num_2
                                min_larger = min_larger_2
                                num_min_larger = num_min_larger_2
                        
                        # Update range
                        if k_pivots_num_2 > k:
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
                        if num_iters >= 24 or tl.abs(min_range - max_range) < 1e-15:
                            k_pivot = (max_range + min_range) / 2.0
                else:
                    # If top-k outlier gathering failed, search whole logit space
                    max_range = max_logit
                    min_range = min_logit
                    while k_pivot == float("inf"):
                        k_pivot_0 = (max_range - min_range) * 1.0 / 4.0 + min_range
                        k_pivots_num_0 = tl.zeros((), dtype=tl.uint32)
                        min_larger_0 = float("inf")
                        num_min_larger_0 = tl.zeros((), dtype=tl.uint32)

                        k_pivot_1 = (max_range - min_range) * 2.0 / 4.0 + min_range
                        k_pivots_num_1 = tl.zeros((), dtype=tl.uint32)
                        min_larger_1 = float("inf")
                        num_min_larger_1 = tl.zeros((), dtype=tl.uint32)

                        k_pivot_2 = (max_range - min_range) * 3.0 / 4.0 + min_range
                        k_pivots_num_2 = tl.zeros((), dtype=tl.uint32)
                        min_larger_2 = float("inf")
                        num_min_larger_2 = tl.zeros((), dtype=tl.uint32)

                        # First pass: Calculate k_pivots_num and min_larger
                        for i in range(0, NUM_TILES):
                            offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                            mask_n = offs_n < VOCAB_SIZE
                            logits_blk2 = tl.load(
                                LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                            )

                            k_pivots_num_0 += tl.sum(logits_blk2 > k_pivot_0)
                            k_pivots_num_1 += tl.sum(logits_blk2 > k_pivot_1)
                            k_pivots_num_2 += tl.sum(logits_blk2 > k_pivot_2)

                            min_larger_0 = tl.minimum(min_larger_0, tl.min(logits_blk2))
                            min_larger_1 = tl.minimum(min_larger_1, tl.min(logits_blk2))
                            min_larger_2 = tl.minimum(min_larger_2, tl.min(logits_blk2))

                        # Second pass: Calculate num_min_larger
                        for i in range(0, NUM_TILES):
                            offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                            mask_n = offs_n < VOCAB_SIZE
                            logits_blk2 = tl.load(
                                LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                            )

                            num_min_larger_0 += tl.sum(tl.abs(logits_blk2 - min_larger_0) < 1e-15)
                            num_min_larger_1 += tl.sum(tl.abs(logits_blk2 - min_larger_1) < 1e-15)
                            num_min_larger_2 += tl.sum(tl.abs(logits_blk2 - min_larger_2) < 1e-15)

                        # Check if any of the pivots satisfy termination condition
                        if k_pivots_num_0 >= k:
                            if k_pivots_num_0 - (min_larger_0 * num_min_larger_0) < k:
                                k_pivot = k_pivot_0
                                k_pivots_num = k_pivots_num_0
                                min_larger = min_larger_0
                                num_min_larger = num_min_larger_0
                        if k_pivots_num_1 >= k:
                            if k_pivots_num_1 - (min_larger_1 * num_min_larger_1) < k:
                                k_pivot = k_pivot_1
                                k_pivots_num = k_pivots_num_1
                                min_larger = min_larger_1
                                num_min_larger = num_min_larger_1
                        if k_pivots_num_2 >= k:
                            if k_pivots_num_2 - (min_larger_2 * num_min_larger_2) < k:
                                k_pivot = k_pivot_2
                                k_pivots_num = k_pivots_num_2
                                min_larger = min_larger_2
                                num_min_larger = num_min_larger_2

                        # Update range                    
                        if k_pivots_num_2 > k:
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
                        if num_iters >= 24 or tl.abs(min_range - max_range) < 1e-15:
                            k_pivot = (max_range + min_range) / 2.0

                duplicate_logit = min_larger
                num_duplicate_logit = num_min_larger
                num_keep = num_duplicate_logit - (k_pivots_num - k)
                num_kept = tl.zeros((), dtype=tl.uint32)
                                
                if TOPP_ENABLED:
                    #### TOP-P SAMPLING AFTER TOP-K ####
                    p = tl.load(P + row_id)
                    if p < 1.0:
                        min_logit = k_pivot         
                        sum_exp_logits = 0.0
                        num_outliers_2 = tl.zeros((), dtype=tl.uint32)
                        search_range = tl.cast(num_outliers, tl.int32)
                        search_iters = tl.cast(
                            (num_outliers + BLOCK_SIZE_TRUNC - 1) // BLOCK_SIZE_TRUNC, tl.int32
                        )

                        # Third pass: Calculate exp logits and sum, gather top-k outliers
                        if num_outliers > k:
                            for i in range(0, search_iters):
                                offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(0, BLOCK_SIZE_TRUNC)
                                mask_n_2 = offs_n < search_range

                                probs_blk = tl.load(BUFFER_ROW + offs_n,
                                                    mask=mask_n_2,
                                                    other=-float('inf'))
                                
                                outlier_mask = (probs_blk > min_logit) & mask_n_2

                                # Duplicate logit handling
                                if num_keep < num_duplicate_logit:
                                    duplicate_mask = tl.abs(probs_blk - duplicate_logit) < 1e-15
                                    duplicate_count = tl.cumsum(duplicate_mask) + num_kept
                                    duplicate_keep_mask = (duplicate_count <= num_keep) & duplicate_mask
                                    duplicate_remove_mask = duplicate_mask & ~duplicate_keep_mask
                                    outlier_mask = outlier_mask & (~duplicate_remove_mask)
                                    num_kept += tl.sum(duplicate_keep_mask)

                                probs_blk = tl.where(outlier_mask, probs_blk, -float('inf'))
                                probs_blk = probs_blk - max_logit
                                probs_blk = tl.exp(probs_blk)
                                sum_exp_logits += tl.sum(probs_blk)
                                
                            # Fourth pass: Calculate BUFFER and get outliers
                            for i in range(0, search_iters):
                                offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(0, BLOCK_SIZE_TRUNC)
                                mask_n_2 = offs_n < search_range

                                probs_blk = tl.load(BUFFER_ROW + offs_n,
                                                    mask=mask_n_2,
                                                    other=-float('inf'))
                                
                                outlier_mask = (probs_blk > min_logit) & mask_n_2
                                probs_blk = probs_blk - max_logit
                                probs_blk = tl.exp(probs_blk)
                                probs_blk = probs_blk / sum_exp_logits
                                tl.store(BUFFER_ROW + offs_n, probs_blk, mask=mask_n_2)
                        else:
                            # If top-k outlier gathering failed, retry gathering using top-k pivot 
                            for i in range(0, NUM_TILES):
                                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                                mask_n = offs_n < VOCAB_SIZE

                                probs_blk = tl.load(LOGITS_ROW + offs_n,
                                                    mask=mask_n,
                                                    other=-float('inf'))
                                
                                outlier_mask = (probs_blk > min_logit) & mask_n

                                # Duplicate logit handling
                                duplicate_mask = tl.abs(probs_blk - duplicate_logit) < 1e-15
                                duplicate_count = tl.cumsum(duplicate_mask) + num_kept
                                duplicate_keep_mask = (duplicate_count <= num_keep) & duplicate_mask
                                duplicate_remove_mask = duplicate_mask & ~duplicate_keep_mask
                                outlier_mask = outlier_mask & (~duplicate_remove_mask)
                                num_kept += tl.sum(duplicate_keep_mask)

                                probs_blk = tl.where(outlier_mask, probs_blk, -float('inf'))
                                probs_blk = probs_blk - max_logit
                                probs_blk = tl.exp(probs_blk)
                                sum_exp_logits += tl.sum(probs_blk)
                                
                                cumulative_pos = tl.cast(
                                    tl.cumsum(outlier_mask) - 1 + num_outliers_2, tl.int32)
                                num_outliers_2 += tl.sum(outlier_mask)
                                write_pos = tl.where(outlier_mask, cumulative_pos, -1)
                                tl.store(BUFFER_ROW + write_pos, probs_blk, mask=outlier_mask)
                                    
                            search_range = tl.cast(num_outliers_2, tl.int32)
                            search_iters = tl.cast(
                                (num_outliers_2 + BLOCK_SIZE_TRUNC - 1) // BLOCK_SIZE_TRUNC, tl.int32)

                            # Fourth pass: Calculate BUFFER and get outliers
                            for i in range(0, search_iters):
                                offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(0, BLOCK_SIZE_TRUNC)
                                mask_n_2 = offs_n < search_range

                                probs_blk = tl.load(BUFFER_ROW + offs_n, mask=mask_n_2, other=0.0)
                                probs_blk = probs_blk / sum_exp_logits
                                tl.store(BUFFER_ROW + offs_n, probs_blk, mask=mask_n_2)

                                
                        max_range = tl.exp(max_logit - max_logit) / sum_exp_logits
                        min_range = tl.exp(min_logit - max_logit) / sum_exp_logits
                        
                        p_pivot = 1.0
                        num_iters = 0
                        min_larger_prob = 1.0
                        num_min_larger = tl.zeros((), dtype=tl.uint32)
                        p_pivots_sum = 0.0
                        
                        # Fifth passes: Search for p_pivot
                        while p_pivot == 1.0:
                            p_pivot_0 = (max_range - min_range) * 1.0 / 4.0 + min_range
                            p_pivots_sum_0 = 0.0
                            min_larger_0 = 1.0
                            num_min_larger_0 = tl.zeros((), dtype=tl.uint32)
                            
                            p_pivot_1 = (max_range - min_range) * 2.0 / 4.0 + min_range
                            p_pivots_sum_1 = 0.0
                            min_larger_1 = 1.0
                            num_min_larger_1 = tl.zeros((), dtype=tl.uint32)

                            p_pivot_2 = (max_range - min_range) * 3.0 / 4.0 + min_range
                            p_pivots_sum_2 = 0.0
                            min_larger_2 = 1.0
                            num_min_larger_2 = tl.zeros((), dtype=tl.uint32)


                            # First pass: Calculate p_pivots_sum and min_larger
                            for i in range(0, search_iters):
                                offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(0, BLOCK_SIZE_TRUNC)
                                mask_n_2 = offs_n < search_range
                                probs_blk = tl.load(BUFFER_ROW + offs_n, mask=mask_n_2, other=0.0)

                                p_pivots_sum_0 += tl.sum(probs_blk * (probs_blk > p_pivot_0))
                                masked_larger_0 = tl.where(probs_blk > p_pivot_0, probs_blk, 1.0)
                                min_larger_0 = tl.minimum(min_larger_0, tl.min(masked_larger_0))
                                
                                p_pivots_sum_1 += tl.sum(probs_blk * (probs_blk > p_pivot_1))
                                masked_larger_1 = tl.where(probs_blk > p_pivot_1, probs_blk, 1.0)
                                min_larger_1 = tl.minimum(min_larger_1, tl.min(masked_larger_1))

                                p_pivots_sum_2 += tl.sum(probs_blk * (probs_blk > p_pivot_2))
                                masked_larger_2 = tl.where(probs_blk > p_pivot_2, probs_blk, 1.0)
                                min_larger_2 = tl.minimum(min_larger_2, tl.min(masked_larger_2))


                            # Second pass: Calculate num_min_larger
                            for i in range(0, search_iters):
                                offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(0, BLOCK_SIZE_TRUNC)
                                mask_n_2 = offs_n < search_range
                                probs_blk = tl.load(BUFFER_ROW + offs_n, mask=mask_n_2, other=0.0)

                                num_min_larger_0 += tl.sum(tl.abs(probs_blk - min_larger_0) < 1e-15)
                                num_min_larger_1 += tl.sum(tl.abs(probs_blk - min_larger_1) < 1e-15)
                                num_min_larger_2 += tl.sum(tl.abs(probs_blk - min_larger_2) < 1e-15)

                            # Check if any of the pivots satisfy termination condition
                            if p_pivots_sum_2 >= p:
                                if p_pivots_sum_2 - (min_larger_2 * num_min_larger_2) < p:
                                    p_pivot = p_pivot_2
                                    min_larger_prob = min_larger_2
                                    num_min_larger = num_min_larger_2
                                    p_pivots_sum = p_pivots_sum_2
                            if p_pivots_sum_1 >= p:
                                if p_pivots_sum_1 - (min_larger_1 * num_min_larger_1) < p:
                                    p_pivot = p_pivot_1
                                    min_larger_prob = min_larger_1
                                    num_min_larger = num_min_larger_1
                                    p_pivots_sum = p_pivots_sum_1
                            if p_pivots_sum_0 >= p:
                                if p_pivots_sum_0 - (min_larger_0 * num_min_larger_0) < p:
                                    p_pivot = p_pivot_0
                                    min_larger_prob = min_larger_0
                                    num_min_larger = num_min_larger_0
                                    p_pivots_sum = p_pivots_sum_0
                            
                            # Update range
                            if p_pivots_sum_2 > p:
                                min_range = p_pivot_2
                            elif p_pivots_sum_1 > p:
                                min_range = p_pivot_1
                            elif p_pivots_sum_0 > p:
                                min_range = p_pivot_0

                            if p_pivots_sum_0 < p:
                                max_range = p_pivot_0
                            elif p_pivots_sum_1 < p:
                                max_range = p_pivot_1
                            elif p_pivots_sum_2 < p:
                                max_range = p_pivot_2

                            num_iters += 1
                            if (max_range - min_range) < 1e-15 or num_iters >= 24:
                                p_pivot = (max_range + min_range) / 2.0

                        duplicate_logit = tl.log(min_larger_prob * sum_exp_logits) + max_logit
                        num_duplicate_logit = num_min_larger
                        num_keep = num_duplicate_logit - tl.cast((p_pivots_sum - p) / min_larger_prob, tl.uint32)
                        num_kept = tl.zeros((), dtype=tl.uint32)

                        # Top-k + Top-p path
                        final_pivot = tl.log(p_pivot * sum_exp_logits) + max_logit
                else:
                    # Top-k only path
                    final_pivot = k_pivot
            elif TOPP_ENABLED:
                #### STANDALONE TOP-P SAMPLING ####
                p = tl.load(P + row_id)
                if p < 1.0:
                    # Zeroth pass: Compute avg and std from a sample block
                    offs = tl.arange(0, BLOCK_SIZE)
                    mask_n = offs < VOCAB_SIZE
                    num_valid = tl.sum(mask_n)
                    logits_blk0 = tl.load(LOGITS_ROW + offs, mask=mask_n, other=0.0)
                    avg_logit = tl.sum(logits_blk0) / num_valid
                    sq_avg_logit = tl.sum(logits_blk0 * logits_blk0) / num_valid
                    std_logit = tl.sqrt(sq_avg_logit - avg_logit * avg_logit)
                    std_logit = tl.maximum(std_logit, 0.0)
                    max_sample = avg_logit + std_logit * 10.0
                    sum_exp_logits = 0.0

                    # First pass: compute max and min logits and sum_exp_logits
                    for i in range(0, NUM_TILES):
                        offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                        mask_n = offs_n < VOCAB_SIZE
                        logits_blk = tl.load(LOGITS_ROW + offs_n,
                                            mask=mask_n,
                                            other=avg_logit)
                        max_logit = tl.maximum(max_logit, tl.max(logits_blk))
                        min_logit = tl.minimum(min_logit, tl.min(logits_blk))
                        
                        probs_blk = tl.exp(logits_blk - max_sample)
                        probs_blk = tl.where(mask_n, probs_blk, 0.0)
                        sum_exp_logits += tl.sum(probs_blk)

                    idx = tl.cast(p * 200, tl.int32)
                    idx = tl.maximum(0, tl.minimum(idx, 199))
                    sigma = tl.load(NORMAL_CDF_TO_SIGMA_TABLE + idx)
                    sigma = sigma + tl.abs(sigma) * -0.25
                    outlier_pivot = avg_logit + std_logit * sigma

                    outlier_prob = tl.exp(outlier_pivot - max_sample) / sum_exp_logits
                    sum_outlier_probs = 0.0
                    num_outliers = tl.zeros((), dtype=tl.uint32)

                    # Second pass: Calculate softmax and gather outliers
                    for i in range(0, NUM_TILES):
                        offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                        mask_n = offs_n < VOCAB_SIZE

                        probs_blk = tl.load(LOGITS_ROW + offs_n, mask=mask_n, other=-float('inf'))
                        probs_blk = tl.exp(probs_blk - max_sample)
                        probs_blk = probs_blk / sum_exp_logits

                        outlier_mask = (probs_blk > outlier_prob) & mask_n
                        sum_outlier_probs += tl.sum(outlier_mask * probs_blk)
                        cumulative_pos = tl.cast(
                            tl.cumsum(outlier_mask) - 1 + num_outliers, tl.int32)
                        num_outliers += tl.sum(outlier_mask)
                        write_pos = tl.where(outlier_mask, cumulative_pos, -1)
                        tl.store(BUFFER_ROW + write_pos, probs_blk, mask=outlier_mask)
                        

                    max_range = tl.exp(max_logit - max_sample) / sum_exp_logits
                    min_range = tl.exp(min_logit - max_sample) / sum_exp_logits
                    
                    p_pivot = 1.0
                    num_iters = 0
                    min_larger_prob = 1.0
                    num_min_larger = tl.zeros((), dtype=tl.uint32)
                    p_pivots_sum = 0.0
                    
                    # Third pass: Search for p_pivot
                    if sum_outlier_probs > p:
                        min_range = outlier_prob
                        search_range = tl.cast(num_outliers, tl.int32)
                        search_iters = tl.cast(
                            (num_outliers + BLOCK_SIZE_TRUNC - 1) // BLOCK_SIZE_TRUNC, tl.int32)

                        while p_pivot == 1.0:
                            p_pivot_0 = (max_range - min_range) * 1.0 / 4.0 + min_range
                            p_pivots_sum_0 = 0.0
                            min_larger_0 = 1.0
                            num_min_larger_0 = tl.zeros((), dtype=tl.uint32)
                            
                            p_pivot_1 = (max_range - min_range) * 2.0 / 4.0 + min_range
                            p_pivots_sum_1 = 0.0
                            min_larger_1 = 1.0
                            num_min_larger_1 = tl.zeros((), dtype=tl.uint32)

                            p_pivot_2 = (max_range - min_range) * 3.0 / 4.0 + min_range
                            p_pivots_sum_2 = 0.0
                            min_larger_2 = 1.0
                            num_min_larger_2 = tl.zeros((), dtype=tl.uint32)


                            # First pass: Calculate p_pivots_sum and min_larger
                            for i in range(0, search_iters):
                                offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(0, BLOCK_SIZE_TRUNC)
                                mask_n_2 = offs_n < search_range
                                probs_blk = tl.load(BUFFER_ROW + offs_n, mask=mask_n_2, other=0.0)

                                p_pivots_sum_0 += tl.sum(probs_blk * (probs_blk > p_pivot_0))
                                masked_larger_0 = tl.where(probs_blk > p_pivot_0, probs_blk, 1.0)
                                min_larger_0 = tl.minimum(min_larger_0, tl.min(masked_larger_0))
                                
                                p_pivots_sum_1 += tl.sum(probs_blk * (probs_blk > p_pivot_1))
                                masked_larger_1 = tl.where(probs_blk > p_pivot_1, probs_blk, 1.0)
                                min_larger_1 = tl.minimum(min_larger_1, tl.min(masked_larger_1))

                                p_pivots_sum_2 += tl.sum(probs_blk * (probs_blk > p_pivot_2))
                                masked_larger_2 = tl.where(probs_blk > p_pivot_2, probs_blk, 1.0)
                                min_larger_2 = tl.minimum(min_larger_2, tl.min(masked_larger_2))


                            # Second pass: Calculate num_min_larger
                            for i in range(0, search_iters):
                                offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(0, BLOCK_SIZE_TRUNC)
                                mask_n_2 = offs_n < search_range
                                probs_blk = tl.load(BUFFER_ROW + offs_n, mask=mask_n_2, other=0.0)

                                num_min_larger_0 += tl.sum(tl.abs(probs_blk - min_larger_0) < 1e-15)
                                num_min_larger_1 += tl.sum(tl.abs(probs_blk - min_larger_1) < 1e-15)
                                num_min_larger_2 += tl.sum(tl.abs(probs_blk - min_larger_2) < 1e-15)

                            # Check if any of the pivots satisfy termination condition
                            if p_pivots_sum_2 >= p:
                                if p_pivots_sum_2 - (min_larger_2 * num_min_larger_2) < p:
                                    p_pivot = p_pivot_2
                                    min_larger_prob = min_larger_2
                                    num_min_larger = num_min_larger_2
                                    p_pivots_sum = p_pivots_sum_2
                            if p_pivots_sum_1 >= p:
                                if p_pivots_sum_1 - (min_larger_1 * num_min_larger_1) < p:
                                    p_pivot = p_pivot_1
                                    min_larger_prob = min_larger_1
                                    num_min_larger = num_min_larger_1
                                    p_pivots_sum = p_pivots_sum_1
                            if p_pivots_sum_0 >= p:
                                if p_pivots_sum_0 - (min_larger_0 * num_min_larger_0) < p:
                                    p_pivot = p_pivot_0
                                    min_larger_prob = min_larger_0
                                    num_min_larger = num_min_larger_0
                                    p_pivots_sum = p_pivots_sum_0
                            
                            # Update range
                            if p_pivots_sum_2 > p:
                                min_range = p_pivot_2
                            elif p_pivots_sum_1 > p:
                                min_range = p_pivot_1
                            elif p_pivots_sum_0 > p:
                                min_range = p_pivot_0

                            if p_pivots_sum_0 < p:
                                max_range = p_pivot_0
                            elif p_pivots_sum_1 < p:
                                max_range = p_pivot_1
                            elif p_pivots_sum_2 < p:
                                max_range = p_pivot_2

                            num_iters += 1
                            if (max_range - min_range) < 1e-15 or num_iters >= 24:
                                p_pivot = (max_range + min_range) / 2.0
                    else:
                        # Re-populate the buffer with full softmax probabilities
                        for i in range(0, NUM_TILES):
                            offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                            mask_n = offs_n < VOCAB_SIZE

                            probs_blk = tl.load(LOGITS_ROW + offs_n, mask=mask_n, other=-float('inf'))
                            probs_blk = tl.exp(probs_blk - max_sample)
                            probs_blk = probs_blk / sum_exp_logits
                            tl.store(BUFFER_ROW + offs_n, probs_blk, mask=mask_n)
                            
                        while p_pivot == 1.0:
                            p_pivot_0 = (max_range - min_range) * 1.0 / 4.0 + min_range
                            p_pivots_sum_0 = 0.0
                            min_larger_0 = 1.0
                            num_min_larger_0 = tl.zeros((), dtype=tl.uint32)
                            
                            p_pivot_1 = (max_range - min_range) * 2.0 / 4.0 + min_range
                            p_pivots_sum_1 = 0.0
                            min_larger_1 = 1.0
                            num_min_larger_1 = tl.zeros((), dtype=tl.uint32)

                            p_pivot_2 = (max_range - min_range) * 3.0 / 4.0 + min_range
                            p_pivots_sum_2 = 0.0
                            min_larger_2 = 1.0
                            num_min_larger_2 = tl.zeros((), dtype=tl.uint32)


                            # First pass: Calculate p_pivots_sum and min_larger
                            for i in range(0, NUM_TILES):
                                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                                mask_n = offs_n < VOCAB_SIZE
                                probs_blk = tl.load(BUFFER_ROW + offs_n, mask=mask_n, other=0.0)

                                p_pivots_sum_0 += tl.sum(probs_blk * (probs_blk > p_pivot_0))
                                masked_larger_0 = tl.where(probs_blk > p_pivot_0, probs_blk, 1.0)
                                min_larger_0 = tl.minimum(min_larger_0, tl.min(masked_larger_0))
                                
                                p_pivots_sum_1 += tl.sum(probs_blk * (probs_blk > p_pivot_1))
                                masked_larger_1 = tl.where(probs_blk > p_pivot_1, probs_blk, 1.0)
                                min_larger_1 = tl.minimum(min_larger_1, tl.min(masked_larger_1))

                                p_pivots_sum_2 += tl.sum(probs_blk * (probs_blk > p_pivot_2))
                                masked_larger_2 = tl.where(probs_blk > p_pivot_2, probs_blk, 1.0)
                                min_larger_2 = tl.minimum(min_larger_2, tl.min(masked_larger_2))


                            # Second pass: Calculate num_min_larger
                            for i in range(0, NUM_TILES):
                                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                                mask_n = offs_n < VOCAB_SIZE
                                probs_blk = tl.load(BUFFER_ROW + offs_n, mask=mask_n, other=0.0)

                                num_min_larger_0 += tl.sum(tl.abs(probs_blk - min_larger_0) < 1e-15)
                                num_min_larger_1 += tl.sum(tl.abs(probs_blk - min_larger_1) < 1e-15)
                                num_min_larger_2 += tl.sum(tl.abs(probs_blk - min_larger_2) < 1e-15)

                            # Check if any of the pivots satisfy termination condition
                            if p_pivots_sum_2 >= p:
                                if p_pivots_sum_2 - (min_larger_2 * num_min_larger_2) < p:
                                    p_pivot = p_pivot_2
                                    min_larger_prob = min_larger_2
                                    num_min_larger = num_min_larger_2
                                    p_pivots_sum = p_pivots_sum_2
                            if p_pivots_sum_1 >= p:
                                if p_pivots_sum_1 - (min_larger_1 * num_min_larger_1) < p:
                                    p_pivot = p_pivot_1
                                    min_larger_prob = min_larger_1
                                    num_min_larger = num_min_larger_1
                                    p_pivots_sum = p_pivots_sum_1
                            if p_pivots_sum_0 >= p:
                                if p_pivots_sum_0 - (min_larger_0 * num_min_larger_0) < p:
                                    p_pivot = p_pivot_0
                                    min_larger_prob = min_larger_0
                                    num_min_larger = num_min_larger_0
                                    p_pivots_sum = p_pivots_sum_0
                            
                            # Update range
                            if p_pivots_sum_2 > p:
                                min_range = p_pivot_2
                            elif p_pivots_sum_1 > p:
                                min_range = p_pivot_1
                            elif p_pivots_sum_0 > p:
                                min_range = p_pivot_0

                            if p_pivots_sum_0 < p:
                                max_range = p_pivot_0
                            elif p_pivots_sum_1 < p:
                                max_range = p_pivot_1
                            elif p_pivots_sum_2 < p:
                                max_range = p_pivot_2

                            num_iters += 1
                            if (max_range - min_range) < 1e-15 or num_iters >= 24:
                                p_pivot = (max_range + min_range) / 2.0

                    duplicate_logit = tl.log(min_larger_prob * sum_exp_logits) + max_logit
                    num_duplicate_logit = num_min_larger
                    num_keep = num_duplicate_logit - tl.cast((p_pivots_sum - p) / min_larger_prob, tl.uint32)
                    num_kept = tl.zeros((), dtype=tl.uint32)
        
                    # Top-p only path
                    final_pivot = tl.log(p_pivot * sum_exp_logits) + max_sample

            # Sixth pass: Apply mask
            if final_pivot != -float("inf"):
                for i in range(0, NUM_TILES):
                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask_n = offs_n < VOCAB_SIZE
                    logits_blk = tl.load(LOGITS_ROW + offs_n, mask=mask_n, other=-float('inf'))
                    keep_mask = (logits_blk > final_pivot) & mask_n

                    # Duplicate logit handling
                    if num_keep < num_duplicate_logit:
                        duplicate_mask = (tl.abs(logits_blk - duplicate_logit) < 1e-15) & mask_n
                        duplicate_count = tl.cumsum(duplicate_mask) + num_kept
                        duplicate_keep_mask = (duplicate_count <= num_duplicate_logit) & duplicate_mask
                        duplicate_remove_mask = duplicate_mask & ~duplicate_keep_mask
                        num_kept += tl.sum(duplicate_keep_mask)
                        keep_mask = keep_mask & (~duplicate_remove_mask)
                        
                    logits_blk = tl.where(keep_mask, logits_blk, MASK_VALUE)
                    tl.store(LOGITS_ROW + offs_n, logits_blk, mask=mask_n)

def apply_top_k_top_p_triton(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    mask_value: float = float("-inf"),
) -> torch.Tensor:
    """
    Apply combined top-k and top-p masking using Triton.

    Top-k is applied first (by logit value), then top-p is applied
    to the remaining k values (by probability).

    Args:
        logits: [batch_size, vocab_size] float32 tensor, modified in-place
        k: [batch_size] int32 tensor of top-k values per row, or None to disable top-k
        p: [batch_size] float32 tensor of top-p values per row (0 to 1),
            or None to disable top-p
        mask_value: Value for masked positions (default: -inf)

    Returns:
        The logits tensor (modified in-place)
    """
    assert logits.ndim == 2
    assert logits.dtype == torch.float32
    assert logits.is_cuda

    batch_size, vocab_size = logits.shape
    num_sm = torch.cuda.get_device_properties(logits.device).multi_processor_count
    NUM_PROGRAMS = min(num_sm, batch_size)

    topk_enabled = k is not None
    topp_enabled = p is not None

    if batch_size == 0 or not (topk_enabled or topp_enabled):
        return logits

    if k is not None:
        assert k.ndim == 1 and k.shape[0] == batch_size and k.is_cuda
        k_ptr = k.to(torch.int32)
    else:
        k_ptr = logits  # Dummy pointer (won't be read)

    if p is not None:
        assert p.ndim == 1 and p.shape[0] == batch_size and p.is_cuda
        p_ptr = p.to(torch.float32)
    else:
        p_ptr = logits  # Dummy pointer (won't be read)

    num_sm = torch.cuda.get_device_properties(logits.device).multi_processor_count
    NUM_PROGRAMS = min(num_sm, batch_size)

    buf_key = (logits.device, logits.dtype, NUM_PROGRAMS, vocab_size)
    buffer = _TRITON_BUFFER_CACHE.get(buf_key)
    if buffer is None or buffer.numel() < NUM_PROGRAMS * vocab_size:
        buffer = torch.empty(
            (NUM_PROGRAMS, vocab_size), device=logits.device, dtype=logits.dtype
        )
        _TRITON_BUFFER_CACHE[buf_key] = buffer

    # Cache percentile table per device.
    tbl_key = (logits.device, torch.float32)
    tables = _TRITON_TABLE_CACHE.get(tbl_key)
    if tables is None:
        normal_cdf_to_sigma_table = torch.tensor(
            _NORMAL_CDF_TO_SIGMA_TABLE, device=logits.device, dtype=torch.float32
        )
        percentile_to_std_table = torch.tensor(
            _PERCENTILE_TO_STD_TABLE, device=logits.device, dtype=torch.float32
        )
        _TRITON_TABLE_CACHE[tbl_key] = (normal_cdf_to_sigma_table, percentile_to_std_table)
    else:
        normal_cdf_to_sigma_table, percentile_to_std_table = tables


    _topk_topp_kernel[(NUM_PROGRAMS,)](
        logits,
        buffer,
        percentile_to_std_table,
        normal_cdf_to_sigma_table,
        k_ptr,
        p_ptr,
        BATCH_SIZE=batch_size,
        MASK_VALUE=mask_value,
        VOCAB_SIZE=vocab_size,
        BLOCK_SIZE=8192,
        BLOCK_SIZE_TRUNC=4096,
        TOPK_ENABLED=topk_enabled,
        TOPP_ENABLED=topp_enabled,
    )

    return logits
