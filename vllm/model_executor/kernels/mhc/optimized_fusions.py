# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optimized fused MHC kernels to reduce HBM traffic.

This module contains additional kernel fusions beyond the existing
mhc_fused_post_pre_tilelang to further optimize memory-bound operations.

Optimizations implemented:
1. mhc_post_hc_head_fused: Fuses MHC post + HC head (final layer)
2. mhc_post_hc_head_norm_fused: Adds RMSNorm fusion on top of #1
3. mhc_post_mean_fused: Fuses MHC post + mean reduction (aux layers)
"""

import math
from typing import Any

import torch

from vllm.platforms import current_platform
from vllm.utils.import_utils import has_tilelang

if not has_tilelang():
    raise ImportError(
        "tilelang is required for optimized mhc fusions but is not installed. "
        "Install it with `pip install tilelang`."
    )

import tilelang
import tilelang.language as T

ENABLE_PDL = current_platform.is_arch_support_pdl() and current_platform.is_cuda()

pass_configs: dict[tilelang.PassConfigKey, Any] = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}

if current_platform.is_cuda():
    pass_configs[tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL] = 10


@tilelang.jit(pass_configs=pass_configs)
def mhc_post_hc_head_fused_tilelang(
    comb_mix,
    residual_in,
    post_mix,
    x_in,
    fn,
    hc_scale,
    hc_base,
    out,
    hc: int,
    hidden: int,
    rms_eps: float,
    hc_eps: float,
    n_thr: int = 128,
    h_blk: int = 1024,
):
    """Fused MHC post + HC head kernel.

    This kernel fuses two operations that were previously separate:
    1. MHC post: residual_cur = post_mix * x + comb_mix @ residual
    2. HC head: compress hc_mult residual channels to single output

    By keeping the intermediate residual_cur in shared memory instead of
    writing to HBM, we eliminate one full HBM round-trip.

    Memory traffic comparison:
    - Original: Write residual_cur (HBM) + Read residual_cur (HBM) = 2x
    - Fused: Keep in shared memory = 0x (only final output written)

    Args:
        comb_mix: [num_tokens, hc, hc] combination mixing weights
        residual_in: [num_tokens, hc, hidden] input residual streams
        post_mix: [num_tokens, hc] post layer mixing weights
        x_in: [num_tokens, hidden] layer output
        fn: [hc, hc * hidden] HC head projection matrix
        hc_scale: [1] HC head scale factor
        hc_base: [hc] HC head bias
        out: [num_tokens, hidden] output tensor
        hc: number of HC channels (typically 4)
        hidden: hidden dimension size
        rms_eps: epsilon for RMS normalization
        hc_eps: epsilon for HC head gates
    """
    num_tokens = T.dynamic("num_tokens")
    hc_dim = hc * hidden
    h_block = math.gcd(h_blk, hidden)
    n_h = hidden // h_block

    comb_mix: T.Tensor[[num_tokens, hc, hc], T.float32]  # type: ignore[no-redef, valid-type]
    residual_in: T.Tensor[[num_tokens, hc, hidden], T.bfloat16]  # type: ignore[no-redef, valid-type]
    post_mix: T.Tensor[[num_tokens, hc], T.float32]  # type: ignore[no-redef, valid-type]
    x_in: T.Tensor[[num_tokens, hidden], T.bfloat16]  # type: ignore[no-redef, valid-type]
    fn: T.Tensor[[hc, hc_dim], T.float32]  # type: ignore[no-redef, valid-type]
    hc_scale: T.Tensor[[1], T.float32]  # type: ignore[no-redef, valid-type]
    hc_base: T.Tensor[[hc], T.float32]  # type: ignore[no-redef, valid-type]
    out: T.Tensor[[num_tokens, hidden], T.bfloat16]  # type: ignore[no-redef, valid-type]

    with T.Kernel(num_tokens, threads=n_thr) as i_n:
        if ENABLE_PDL:
            T.pdl_sync()

        # Load mixing weights into registers/shared memory
        a_local = T.alloc_fragment((hc, hc), T.float32)
        c_local = T.alloc_fragment(hc, T.float32)
        T.copy(comb_mix[i_n, 0, 0], a_local)
        T.copy(post_mix[i_n, 0], c_local)

        # =====================================================================
        # FUSED PASS 1: Compute MHC post AND accumulate HC head statistics
        # =====================================================================
        # Instead of writing residual_cur to HBM, we:
        # 1. Store it in shared memory
        # 2. Simultaneously accumulate sqrsum and mixes for HC head

        residual_cur_shared = T.alloc_shared((hc, hidden), T.bfloat16)
        sqrsum_r = T.alloc_reducer((1,), T.float32, replication="all")
        mixes_r = T.alloc_reducer((hc,), T.float32, replication="all")
        T.fill(sqrsum_r, 0.0)
        T.fill(mixes_r, 0.0)

        for i0_h in T.serial(n_h):
            # Load inputs for this block
            b_shared = T.alloc_shared((hc, h_block), T.bfloat16)
            d_shared = T.alloc_shared(h_block, T.bfloat16)
            T.copy(residual_in[i_n, 0, i0_h * h_block], b_shared)
            T.copy(x_in[i_n, i0_h * h_block], d_shared)

            b_local = T.alloc_fragment((hc, h_block), T.float32)
            d_local = T.alloc_fragment(h_block, T.float32)
            T.copy(b_shared, b_local)
            T.copy(d_shared, d_local)

            # Compute MHC post for this block
            x_local = T.alloc_fragment((hc, h_block), T.float32)
            for i_hco, i1_h in T.Parallel(hc, h_block):
                x_local[i_hco, i1_h] = c_local[i_hco] * d_local[i1_h]
                for i_hci in T.vectorized(hc):
                    x_local[i_hco, i1_h] += a_local[i_hci, i_hco] * b_local[i_hci, i1_h]

            # Store to shared memory (not HBM!)
            for i_hco, i1_h in T.Parallel(hc, h_block):
                residual_cur_shared[i_hco, i0_h * h_block + i1_h] = T.bfloat16(
                    x_local[i_hco, i1_h]
                )

            # Simultaneously accumulate HC head statistics
            for m_c in T.serial(hc):
                for k in T.Parallel(h_block):
                    val = x_local[m_c, k]
                    sqrsum_r[0] += val * val

                for m_m in T.unroll(hc):
                    fn_local = T.alloc_fragment(h_block, T.float32)
                    T.copy(fn[m_m, m_c * hidden + i0_h * h_block], fn_local)
                    for k in T.Parallel(h_block):
                        mixes_r[m_m] += x_local[m_c, k] * fn_local[k]

        T.finalize_reducer(sqrsum_r)
        T.finalize_reducer(mixes_r)

        # =====================================================================
        # PASS 2: Compute HC head output from shared memory
        # =====================================================================
        pre_mix_shared = T.alloc_shared(hc, T.float32)
        rsqrt_val = T.alloc_fragment(1, T.float32)
        rsqrt_val[0] = T.rsqrt(sqrsum_r[0] / hc_dim + rms_eps)

        for m in T.Parallel(hc):
            pre_mix_shared[m] = (
                T.sigmoid(mixes_r[m] * rsqrt_val[0] * hc_scale[0] + hc_base[m]) + hc_eps
            )

        # Apply weighted sum to produce final output
        for i0_h in T.Pipelined(n_h, num_stages=2):
            xs = T.alloc_shared((hc, h_block), T.bfloat16)
            xl = T.alloc_fragment((hc, h_block), T.float32)

            # Load from shared memory (fast!) instead of HBM
            for i_hc, i1_h in T.Parallel(hc, h_block):
                xs[i_hc, i1_h] = residual_cur_shared[i_hc, i0_h * h_block + i1_h]
            T.copy(xs, xl)

            ol = T.alloc_fragment(h_block, T.float32)
            T.clear(ol)
            for i_hc in T.serial(hc):
                pre = pre_mix_shared[i_hc]
                for i1_h in T.Parallel(h_block):
                    ol[i1_h] += pre * xl[i_hc, i1_h]

            T.copy(ol, out[i_n, i0_h * h_block], disable_tma=True)

        if ENABLE_PDL:
            T.pdl_trigger()


@tilelang.jit(pass_configs=pass_configs)
def mhc_post_hc_head_norm_fused_tilelang(
    comb_mix,
    residual_in,
    post_mix,
    x_in,
    fn,
    hc_scale,
    hc_base,
    norm_weight,
    norm_eps,
    out,
    hc: int,
    hidden: int,
    rms_eps: float,
    hc_eps: float,
    n_thr: int = 128,
    h_blk: int = 1024,
):
    """Fused MHC post + HC head + RMSNorm kernel.

    Three-way fusion: MHC post → HC head → RMSNorm

    Memory traffic comparison:
    - Original: 3 writes + 2 reads = 5 HBM operations
    - Fused: 1 write = 1 HBM operation (80% reduction!)

    Args:
        Same as mhc_post_hc_head_fused_tilelang, plus:
        norm_weight: [hidden] RMSNorm weight
        norm_eps: RMSNorm epsilon
    """
    num_tokens = T.dynamic("num_tokens")
    hc_dim = hc * hidden
    h_block = math.gcd(h_blk, hidden)
    n_h = hidden // h_block

    comb_mix: T.Tensor[[num_tokens, hc, hc], T.float32]  # type: ignore[no-redef, valid-type]
    residual_in: T.Tensor[[num_tokens, hc, hidden], T.bfloat16]  # type: ignore[no-redef, valid-type]
    post_mix: T.Tensor[[num_tokens, hc], T.float32]  # type: ignore[no-redef, valid-type]
    x_in: T.Tensor[[num_tokens, hidden], T.bfloat16]  # type: ignore[no-redef, valid-type]
    fn: T.Tensor[[hc, hc_dim], T.float32]  # type: ignore[no-redef, valid-type]
    hc_scale: T.Tensor[[1], T.float32]  # type: ignore[no-redef, valid-type]
    hc_base: T.Tensor[[hc], T.float32]  # type: ignore[no-redef, valid-type]
    norm_weight: T.Tensor[[hidden], T.bfloat16]  # type: ignore[no-redef, valid-type]
    out: T.Tensor[[num_tokens, hidden], T.bfloat16]  # type: ignore[no-redef, valid-type]

    with T.Kernel(num_tokens, threads=n_thr) as i_n:
        if ENABLE_PDL:
            T.pdl_sync()

        a_local = T.alloc_fragment((hc, hc), T.float32)
        c_local = T.alloc_fragment(hc, T.float32)
        T.copy(comb_mix[i_n, 0, 0], a_local)
        T.copy(post_mix[i_n, 0], c_local)

        residual_cur_shared = T.alloc_shared((hc, hidden), T.bfloat16)
        sqrsum_r = T.alloc_reducer((1,), T.float32, replication="all")
        mixes_r = T.alloc_reducer((hc,), T.float32, replication="all")
        T.fill(sqrsum_r, 0.0)
        T.fill(mixes_r, 0.0)

        # Pass 1: MHC post + accumulate HC head stats (same as before)
        for i0_h in T.serial(n_h):
            b_shared = T.alloc_shared((hc, h_block), T.bfloat16)
            d_shared = T.alloc_shared(h_block, T.bfloat16)
            T.copy(residual_in[i_n, 0, i0_h * h_block], b_shared)
            T.copy(x_in[i_n, i0_h * h_block], d_shared)

            b_local = T.alloc_fragment((hc, h_block), T.float32)
            d_local = T.alloc_fragment(h_block, T.float32)
            T.copy(b_shared, b_local)
            T.copy(d_shared, d_local)

            x_local = T.alloc_fragment((hc, h_block), T.float32)
            for i_hco, i1_h in T.Parallel(hc, h_block):
                x_local[i_hco, i1_h] = c_local[i_hco] * d_local[i1_h]
                for i_hci in T.vectorized(hc):
                    x_local[i_hco, i1_h] += a_local[i_hci, i_hco] * b_local[i_hci, i1_h]

            for i_hco, i1_h in T.Parallel(hc, h_block):
                residual_cur_shared[i_hco, i0_h * h_block + i1_h] = T.bfloat16(
                    x_local[i_hco, i1_h]
                )

            for m_c in T.serial(hc):
                for k in T.Parallel(h_block):
                    val = x_local[m_c, k]
                    sqrsum_r[0] += val * val

                for m_m in T.unroll(hc):
                    fn_local = T.alloc_fragment(h_block, T.float32)
                    T.copy(fn[m_m, m_c * hidden + i0_h * h_block], fn_local)
                    for k in T.Parallel(h_block):
                        mixes_r[m_m] += x_local[m_c, k] * fn_local[k]

        T.finalize_reducer(sqrsum_r)
        T.finalize_reducer(mixes_r)

        # Pass 2: HC head to produce intermediate output in shared memory
        hc_head_output_shared = T.alloc_shared(hidden, T.bfloat16)
        pre_mix_shared = T.alloc_shared(hc, T.float32)
        rsqrt_val = T.alloc_fragment(1, T.float32)
        rsqrt_val[0] = T.rsqrt(sqrsum_r[0] / hc_dim + rms_eps)

        for m in T.Parallel(hc):
            pre_mix_shared[m] = (
                T.sigmoid(mixes_r[m] * rsqrt_val[0] * hc_scale[0] + hc_base[m]) + hc_eps
            )

        # Accumulate RMSNorm variance while computing HC head output
        # Use reducer to avoid data race
        norm_sqrsum_r = T.alloc_reducer((1,), T.float32, replication="all")
        T.fill(norm_sqrsum_r, 0.0)

        for i0_h in T.serial(n_h):
            xs = T.alloc_shared((hc, h_block), T.bfloat16)
            xl = T.alloc_fragment((hc, h_block), T.float32)

            for i_hc, i1_h in T.Parallel(hc, h_block):
                xs[i_hc, i1_h] = residual_cur_shared[i_hc, i0_h * h_block + i1_h]
            T.copy(xs, xl)

            ol = T.alloc_fragment(h_block, T.float32)
            T.clear(ol)
            for i_hc in T.serial(hc):
                pre = pre_mix_shared[i_hc]
                for i1_h in T.Parallel(h_block):
                    ol[i1_h] += pre * xl[i_hc, i1_h]

            # Store to shared memory and accumulate squared sum
            for i1_h in T.Parallel(h_block):
                hc_head_output_shared[i0_h * h_block + i1_h] = T.bfloat16(ol[i1_h])
                norm_sqrsum_r[0] += ol[i1_h] * ol[i1_h]

        T.finalize_reducer(norm_sqrsum_r)

        # Pass 3: Apply RMSNorm and write final output
        rsqrt_norm = T.alloc_fragment(1, T.float32)
        rsqrt_norm[0] = T.rsqrt(norm_sqrsum_r[0] / hidden + norm_eps)

        for i0_h in T.Pipelined(n_h, num_stages=2):
            w_shared = T.alloc_shared(h_block, T.bfloat16)
            w_local = T.alloc_fragment(h_block, T.float32)
            T.copy(norm_weight[i0_h * h_block], w_shared)
            T.copy(w_shared, w_local)

            ol = T.alloc_fragment(h_block, T.float32)
            for i1_h in T.Parallel(h_block):
                ol[i1_h] = (
                    hc_head_output_shared[i0_h * h_block + i1_h]
                    * rsqrt_norm[0]
                    * w_local[i1_h]
                )

            T.copy(ol, out[i_n, i0_h * h_block], disable_tma=True)

        if ENABLE_PDL:
            T.pdl_trigger()


@tilelang.jit(pass_configs=pass_configs)
def mhc_post_hc_head_norm_fused_tilelang_mtp(
    comb_mix,
    residual_in,
    post_mix,
    x_in,
    fn,
    hc_scale,
    hc_base,
    norm_weight,
    norm_eps,
    out,
    mtp_output,
    hc: int,
    hidden: int,
    rms_eps: float,
    hc_eps: float,
    n_thr: int = 128,
    h_blk: int = 1024,
):
    """Fused MHC post + HC head + RMSNorm kernel with MTP residual stash.

    Same as mhc_post_hc_head_norm_fused_tilelang, but additionally writes
    the pre-hc_head residual (output of MHC post) to mtp_output for the
    MTP draft model.

    Memory traffic comparison vs original (separate ops):
    - Original: 3 writes + 2 reads = 5 HBM operations
    - Fused + MTP: 2 writes (mtp_output + final output) = 2 HBM operations
    - Still saves 60% HBM traffic compared to the original 5 ops.

    Args:
        Same as mhc_post_hc_head_norm_fused_tilelang, plus:
        mtp_output: [num_tokens, hc * hidden] pre-hc_head residual for MTP
    """
    num_tokens = T.dynamic("num_tokens")
    hc_dim = hc * hidden
    h_block = math.gcd(h_blk, hidden)
    n_h = hidden // h_block

    comb_mix: T.Tensor[[num_tokens, hc, hc], T.float32]  # type: ignore[no-redef, valid-type]
    residual_in: T.Tensor[[num_tokens, hc, hidden], T.bfloat16]  # type: ignore[no-redef, valid-type]
    post_mix: T.Tensor[[num_tokens, hc], T.float32]  # type: ignore[no-redef, valid-type]
    x_in: T.Tensor[[num_tokens, hidden], T.bfloat16]  # type: ignore[no-redef, valid-type]
    fn: T.Tensor[[hc, hc_dim], T.float32]  # type: ignore[no-redef, valid-type]
    hc_scale: T.Tensor[[1], T.float32]  # type: ignore[no-redef, valid-type]
    hc_base: T.Tensor[[hc], T.float32]  # type: ignore[no-redef, valid-type]
    norm_weight: T.Tensor[[hidden], T.bfloat16]  # type: ignore[no-redef, valid-type]
    out: T.Tensor[[num_tokens, hidden], T.bfloat16]  # type: ignore[no-redef, valid-type]
    mtp_output: T.Tensor[[num_tokens, hc_dim], T.bfloat16]  # type: ignore[no-redef, valid-type]

    with T.Kernel(num_tokens, threads=n_thr) as i_n:
        if ENABLE_PDL:
            T.pdl_sync()

        a_local = T.alloc_fragment((hc, hc), T.float32)
        c_local = T.alloc_fragment(hc, T.float32)
        T.copy(comb_mix[i_n, 0, 0], a_local)
        T.copy(post_mix[i_n, 0], c_local)

        residual_cur_shared = T.alloc_shared((hc, hidden), T.bfloat16)
        sqrsum_r = T.alloc_reducer((1,), T.float32, replication="all")
        mixes_r = T.alloc_reducer((hc,), T.float32, replication="all")
        T.fill(sqrsum_r, 0.0)
        T.fill(mixes_r, 0.0)

        # Pass 1: MHC post + accumulate HC head stats
        for i0_h in T.serial(n_h):
            b_shared = T.alloc_shared((hc, h_block), T.bfloat16)
            d_shared = T.alloc_shared(h_block, T.bfloat16)
            T.copy(residual_in[i_n, 0, i0_h * h_block], b_shared)
            T.copy(x_in[i_n, i0_h * h_block], d_shared)

            b_local = T.alloc_fragment((hc, h_block), T.float32)
            d_local = T.alloc_fragment(h_block, T.float32)
            T.copy(b_shared, b_local)
            T.copy(d_shared, d_local)

            x_local = T.alloc_fragment((hc, h_block), T.float32)
            for i_hco, i1_h in T.Parallel(hc, h_block):
                x_local[i_hco, i1_h] = c_local[i_hco] * d_local[i1_h]
                for i_hci in T.vectorized(hc):
                    x_local[i_hco, i1_h] += a_local[i_hci, i_hco] * b_local[i_hci, i1_h]

            for i_hco, i1_h in T.Parallel(hc, h_block):
                residual_cur_shared[i_hco, i0_h * h_block + i1_h] = T.bfloat16(
                    x_local[i_hco, i1_h]
                )

            for m_c in T.serial(hc):
                for k in T.Parallel(h_block):
                    val = x_local[m_c, k]
                    sqrsum_r[0] += val * val

                for m_m in T.unroll(hc):
                    fn_local = T.alloc_fragment(h_block, T.float32)
                    T.copy(fn[m_m, m_c * hidden + i0_h * h_block], fn_local)
                    for k in T.Parallel(h_block):
                        mixes_r[m_m] += x_local[m_c, k] * fn_local[k]

        T.finalize_reducer(sqrsum_r)
        T.finalize_reducer(mixes_r)

        # Pass 2: HC head to produce intermediate output in shared memory
        hc_head_output_shared = T.alloc_shared(hidden, T.bfloat16)
        pre_mix_shared = T.alloc_shared(hc, T.float32)
        rsqrt_val = T.alloc_fragment(1, T.float32)
        rsqrt_val[0] = T.rsqrt(sqrsum_r[0] / hc_dim + rms_eps)

        for m in T.Parallel(hc):
            pre_mix_shared[m] = (
                T.sigmoid(mixes_r[m] * rsqrt_val[0] * hc_scale[0] + hc_base[m]) + hc_eps
            )

        norm_sqrsum_r = T.alloc_reducer((1,), T.float32, replication="all")
        T.fill(norm_sqrsum_r, 0.0)

        for i0_h in T.serial(n_h):
            xs = T.alloc_shared((hc, h_block), T.bfloat16)
            xl = T.alloc_fragment((hc, h_block), T.float32)

            for i_hc, i1_h in T.Parallel(hc, h_block):
                xs[i_hc, i1_h] = residual_cur_shared[i_hc, i0_h * h_block + i1_h]
            T.copy(xs, xl)

            ol = T.alloc_fragment(h_block, T.float32)
            T.clear(ol)
            for i_hc in T.serial(hc):
                pre = pre_mix_shared[i_hc]
                for i1_h in T.Parallel(h_block):
                    ol[i1_h] += pre * xl[i_hc, i1_h]

            for i1_h in T.Parallel(h_block):
                hc_head_output_shared[i0_h * h_block + i1_h] = T.bfloat16(ol[i1_h])
                norm_sqrsum_r[0] += ol[i1_h] * ol[i1_h]

        T.finalize_reducer(norm_sqrsum_r)

        # Pass 3: Write pre-hc_head residual to MTP output
        for i0_h in T.serial(n_h):
            for i_hco, i1_h in T.Parallel(hc, h_block):
                mtp_output[i_n, i_hco * hidden + i0_h * h_block + i1_h] = \
                    residual_cur_shared[i_hco, i0_h * h_block + i1_h]

        # Pass 4: Apply RMSNorm and write final output
        rsqrt_norm = T.alloc_fragment(1, T.float32)
        rsqrt_norm[0] = T.rsqrt(norm_sqrsum_r[0] / hidden + norm_eps)

        for i0_h in T.Pipelined(n_h, num_stages=2):
            w_shared = T.alloc_shared(h_block, T.bfloat16)
            w_local = T.alloc_fragment(h_block, T.float32)
            T.copy(norm_weight[i0_h * h_block], w_shared)
            T.copy(w_shared, w_local)

            ol = T.alloc_fragment(h_block, T.float32)
            for i1_h in T.Parallel(h_block):
                ol[i1_h] = (
                    hc_head_output_shared[i0_h * h_block + i1_h]
                    * rsqrt_norm[0]
                    * w_local[i1_h]
                )

            T.copy(ol, out[i_n, i0_h * h_block], disable_tma=True)

        if ENABLE_PDL:
            T.pdl_trigger()


@tilelang.jit(pass_configs=pass_configs)
def mhc_post_mean_fused_tilelang(
    comb_mix,
    residual_in,
    post_mix,
    x_in,
    out_full,
    out_mean,
    hc: int,
    hidden: int,
    n_thr: int = 128,
    h_blk: int = 1024,
):
    """Fused MHC post + mean(dim=1) kernel for aux layers.

    In aux_hidden_state layers, mhc_post output is immediately followed by
    .mean(dim=1). This kernel fuses both operations.

    Memory traffic comparison:
    - Original: Write full [hc, hidden] + Read full [hc, hidden] = 2x
    - Fused: Write full [hc, hidden] + Write mean [hidden] = 1.25x (for hc=4)

    Args:
        comb_mix: [num_tokens, hc, hc]
        residual_in: [num_tokens, hc, hidden]
        post_mix: [num_tokens, hc]
        x_in: [num_tokens, hidden]
        out_full: [num_tokens, hc, hidden] full MHC post output
        out_mean: [num_tokens, hidden] mean across hc dimension
    """
    num_tokens = T.dynamic("num_tokens")
    h_block = math.gcd(h_blk, hidden)
    n_h = hidden // h_block

    comb_mix: T.Tensor[[num_tokens, hc, hc], T.float32]  # type: ignore[no-redef, valid-type]
    residual_in: T.Tensor[[num_tokens, hc, hidden], T.bfloat16]  # type: ignore[no-redef, valid-type]
    post_mix: T.Tensor[[num_tokens, hc], T.float32]  # type: ignore[no-redef, valid-type]
    x_in: T.Tensor[[num_tokens, hidden], T.bfloat16]  # type: ignore[no-redef, valid-type]
    out_full: T.Tensor[[num_tokens, hc, hidden], T.bfloat16]  # type: ignore[no-redef, valid-type]
    out_mean: T.Tensor[[num_tokens, hidden], T.bfloat16]  # type: ignore[no-redef, valid-type]

    with T.Kernel(num_tokens, threads=n_thr) as i_n:
        if ENABLE_PDL:
            T.pdl_sync()

        a_local = T.alloc_fragment((hc, hc), T.float32)
        c_local = T.alloc_fragment(hc, T.float32)
        T.copy(comb_mix[i_n, 0, 0], a_local)
        T.copy(post_mix[i_n, 0], c_local)

        for i0_h in T.Serial(n_h):
            b_shared = T.alloc_shared((hc, h_block), T.bfloat16)
            d_shared = T.alloc_shared(h_block, T.bfloat16)
            T.copy(residual_in[i_n, 0, i0_h * h_block], b_shared)
            T.copy(x_in[i_n, i0_h * h_block], d_shared)

            b_local = T.alloc_fragment((hc, h_block), T.float32)
            d_local = T.alloc_fragment(h_block, T.float32)
            T.copy(b_shared, b_local)
            T.copy(d_shared, d_local)

            # Compute MHC post
            x_local = T.alloc_fragment((hc, h_block), T.float32)
            for i_hco, i1_h in T.Parallel(hc, h_block):
                x_local[i_hco, i1_h] = c_local[i_hco] * d_local[i1_h]
                for i_hci in T.vectorized(hc):
                    x_local[i_hco, i1_h] += a_local[i_hci, i_hco] * b_local[i_hci, i1_h]

            # Write full output
            T.copy(x_local, out_full[i_n, 0, i0_h * h_block])

            # Simultaneously compute mean across hc dimension using reducer
            mean_r = T.alloc_reducer((h_block,), T.float32, replication="none")
            T.fill(mean_r, 0.0)

            for i_hc in T.serial(hc):
                for i1_h in T.Parallel(h_block):
                    mean_r[i1_h] += x_local[i_hc, i1_h]

            T.finalize_reducer(mean_r)

            for i1_h in T.Parallel(h_block):
                out_mean[i_n, i0_h * h_block + i1_h] = T.bfloat16(mean_r[i1_h] / hc)

        if ENABLE_PDL:
            T.pdl_trigger()
