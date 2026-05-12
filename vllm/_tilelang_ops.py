# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from functools import cache
from typing import TYPE_CHECKING

import torch

from vllm.platforms import current_platform
from vllm.utils.import_utils import has_tilelang
from vllm.utils.math_utils import cdiv

# tilelang is only available on CUDA platforms
if TYPE_CHECKING or current_platform.is_cuda():
    if not has_tilelang():
        raise ImportError(
            "tilelang is required for mhc but is not installed. Install it with "
            "`pip install tilelang`."
        )
    import tilelang
    import tilelang.language as T
else:
    tilelang = None  # type: ignore[assignment]
    T = None  # type: ignore[assignment]


@cache
def compute_num_split(block_k: int, k: int | None, grid_size: int) -> int:
    device_props = torch.cuda.get_device_properties(0)
    n_sms = device_props.multi_processor_count
    split_k = n_sms // grid_size
    if k is not None:
        # avoid split_k for small k
        num_block_k = cdiv(k, block_k)
        split_k = min(split_k, num_block_k // 4)
    split_k = max(split_k, 1)
    return split_k


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    },
)
def mhc_pre_big_fuse_tilelang(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    residual,
    post_mix,
    comb_mix,
    layer_input,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 16,
    hc_mult: int = 4,
):
    """Deeply fused kernels, everything other than gemm & sqrsum in mHC pre block."""
    num_tokens = T.dynamic("num_tokens")
    hc_mult3 = hc_mult * (2 + hc_mult)
    hidden_block = math.gcd(512, hidden_size)

    gemm_out_mul: T.Tensor[[n_splits, num_tokens, hc_mult3], T.float32]  # type: ignore[no-redef, valid-type]
    gemm_out_sqrsum: T.Tensor[[n_splits, num_tokens], T.float32]  # type: ignore[no-redef, valid-type]
    hc_scale: T.Tensor[[3], T.float32]  # type: ignore[no-redef, valid-type]
    hc_base: T.Tensor[[hc_mult3], T.float32]  # type: ignore[no-redef, valid-type]
    residual: T.Tensor[[num_tokens, hc_mult, hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]
    # outputs
    post_mix: T.Tensor[[num_tokens, hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
    comb_mix: T.Tensor[[num_tokens, hc_mult * hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
    layer_input: T.Tensor[[num_tokens, hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]

    with T.Kernel(num_tokens, threads=96) as i:
        T.pdl_sync()
        ##################################################################
        # _pre_norm_fn_fwd_norm
        rms = T.alloc_fragment(1, T.float32)
        mixes = T.alloc_fragment(hc_mult3, T.float32)
        T.clear(mixes)
        rms[0] = 0
        for i_split in T.serial(n_splits):
            rms[0] += gemm_out_sqrsum[i_split, i]
        rms[0] = T.rsqrt(rms[0] / (hc_mult * hidden_size) + rms_eps)
        for j in T.Parallel(hc_mult3):
            mixes[j] = 0
            for i_split in T.serial(n_splits):
                mixes[j] += gemm_out_mul[i_split, i, j]
            mixes[j] *= rms[0]
        mixes_shared = T.alloc_shared(hc_mult3, T.float32)
        T.copy(mixes, mixes_shared)

        if T.get_thread_binding() < 32:
            ##################################################################
            # _pre_split_mixes_fwd (post & comb)
            cm = T.alloc_fragment((hc_mult, hc_mult), T.float32)
            for j in T.Parallel(hc_mult):
                post_mix[i, j] = (
                    T.sigmoid(
                        mixes_shared[j + hc_mult] * hc_scale[1] + hc_base[j + hc_mult]
                    )
                    * hc_post_mult_value
                )
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = (
                    mixes_shared[j * hc_mult + k + hc_mult * 2] * hc_scale[2]
                    + hc_base[j * hc_mult + k + hc_mult * 2]
                )

            ##################################################################
            # _sinkhorn_fwd
            row_sum = T.alloc_fragment(hc_mult, T.float32)
            col_sum = T.alloc_fragment(hc_mult, T.float32)

            # comb = comb.softmax(-1) + eps
            row_max = T.alloc_fragment(hc_mult, T.float32)
            T.reduce_max(cm, row_max, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = T.exp(cm[j, k] - row_max[j])
            T.reduce_sum(cm, row_sum, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / row_sum[j] + hc_sinkhorn_eps

            # comb = comb / (comb.sum(-2) + eps)
            T.reduce_sum(cm, col_sum, dim=0)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            for _ in T.serial(sinkhorn_repeat - 1):
                # comb = comb / (comb.sum(-1) + eps)
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (row_sum[j] + hc_sinkhorn_eps)

                # comb = comb / (comb.sum(-2) + eps)
                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            # save comb_mix to global memory
            for j, k in T.Parallel(hc_mult, hc_mult):
                comb_mix[i, j * hc_mult + k] = cm[j, k]
        else:
            ##################################################################
            # _pre_split_mixes_fwd (pre)
            pre_mix_shared = T.alloc_shared(hc_mult, T.float32)
            for j in T.Parallel(hc_mult):
                pre_mix_shared[j] = (
                    T.sigmoid(
                        mixes_shared[j] * hc_scale[0] + hc_base[j],
                    )
                    + hc_pre_eps
                )
            ###################################################################
            # _pre_apply_mix_fwd
            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                xs = T.alloc_shared((hc_mult, hidden_block), T.float32)
                xl = T.alloc_fragment((hc_mult, hidden_block), T.float32)
                T.copy(residual[i, 0, i0_h * hidden_block], xs)
                T.copy(xs, xl)

                ol = T.alloc_fragment(hidden_block, T.float32)
                T.clear(ol)

                for i_hc in T.serial(hc_mult):
                    pre = pre_mix_shared[i_hc]
                    for i1_h in T.Parallel(hidden_block):
                        ol[i1_h] += pre * xl[i_hc, i1_h]

                T.copy(ol, layer_input[i, i0_h * hidden_block])
        T.pdl_trigger()


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    },
)
def mhc_fused_tilelang(
    comb_mix,
    residual_in,
    post_mix,
    x_in,
    weight_t,
    yp_out,
    rp_out,
    residual_out,
    hc: int,
    hidden: int,
    n_out: int,
    n_thr: int = 256,
    h_blk: int = 256,
    tile_n: int = 1,
    split_k: int = 1,
) -> tilelang.JITKernel:
    """Fused mhc post-mapping + pre-norm GEMM FMA"""
    m = T.dynamic("num_tokens")
    split_k = T.dynamic("split_k")
    h = hidden
    h_blk = math.gcd(hidden, h_blk)
    h_per_split = h // split_k
    n_tiles = n_out // tile_n

    comb_mix: T.Tensor((m, hc, hc), T.float32)  # type: ignore[no-redef, valid-type]
    residual_in: T.Tensor((m, hc, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
    post_mix: T.Tensor((m, hc), T.float32)  # type: ignore[no-redef, valid-type]
    x_in: T.Tensor((m, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
    weight_t: T.Tensor((n_out, hc, h), T.float32)  # type: ignore[no-redef, valid-type]
    yp_out: T.Tensor((split_k, m, n_out), T.float32)  # type: ignore[no-redef, valid-type]
    rp_out: T.Tensor((split_k, m), T.float32)  # type: ignore[no-redef, valid-type]
    residual_out: T.Tensor((m, hc, h), T.bfloat16)  # type: ignore[no-redef, valid-type]

    h_iters = h_per_split // n_thr
    num_warps = n_thr // 32

    with T.Kernel(m, n_tiles, split_k, threads=n_thr) as (i_n, i_nt, i_ks):
        tid = T.get_thread_binding()
        warp_id = T.get_warp_idx()
        lane = T.get_lane_idx()

        s_warp = T.alloc_shared((num_warps, tile_n + 1), T.float32)
        s_post = T.alloc_shared((hc,), T.float32)
        s_comb = T.alloc_shared((hc, hc), T.float32)

        pm = T.alloc_local((hc,), T.float32)
        cm = T.alloc_local((hc, hc), T.float32)
        acc = T.alloc_local((tile_n,), T.float32)
        sqr = T.alloc_local((1,), T.float32)
        new_r = T.alloc_local((hc,), T.float32)

        T.clear(acc)
        T.clear(sqr)
        h_split_start = i_ks * h_per_split

        T.pdl_sync()

        T.copy(post_mix[i_n, 0], s_post)
        T.copy(comb_mix[i_n, 0, 0], s_comb)

        for j in T.unroll(hc):
            pm[j] = s_post[j]
        for j in T.unroll(hc):
            for k in T.unroll(hc):
                cm[k, j] = s_comb[k, j]

        # Each thread owns h_iters elements of the k-split's h slice.
        for it in T.serial(h_iters):
            h_idx = h_split_start + it * n_thr + tid

            # Compute new residual from layer output and past residual
            for j in T.unroll(hc):
                new_r[j] = pm[j] * x_in[i_n, h_idx]
                for k in T.unroll(hc):
                    new_r[j] += cm[k, j] * residual_in[i_n, k, h_idx]

            # populate residual_out and compute sqr sum
            if i_nt == 0:
                for j in T.unroll(hc):
                    residual_out[i_n, j, h_idx] = new_r[j]
                    sqr[0] += new_r[j] * new_r[j]

            # Per-thread FMA into acc[n]
            for n in T.unroll(tile_n):
                for j in T.unroll(hc):
                    acc[n] += weight_t[i_nt * tile_n + n, j, h_idx] * new_r[j]

        for n in T.unroll(tile_n):
            acc[n] = T.warp_reduce_sum(acc[n])
        if i_nt == 0:
            sqr[0] = T.warp_reduce_sum(sqr[0])

        # Cross-warp reduce via shared mem
        if lane == 0:
            for n in T.unroll(tile_n):
                s_warp[warp_id, n] = acc[n]
            if i_nt == 0:
                s_warp[warp_id, tile_n] = sqr[0]
        T.sync_threads()

        # Warp 0 does the final cross-warp sum and writes outputs
        if warp_id == 0:
            if lane < tile_n:
                v = T.alloc_var(T.float32, init=0.0)
                for w in T.unroll(num_warps):
                    v += s_warp[w, lane]
                yp_out[i_ks, i_n, i_nt * tile_n + lane] = v

            if i_nt == 0 and lane == 0:
                v2 = T.alloc_var(T.float32, init=0.0)
                for w in T.unroll(num_warps):
                    v2 += s_warp[w, tile_n]
                rp_out[i_ks, i_n] = v2

        T.pdl_trigger()


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    },
)
def mhc_post_tilelang(
    a,
    b,
    c,
    d,
    x,
    hc: int,
    hidden: int,
    n_thr: int = 128,
    h_blk: int = 1024,
) -> tilelang.JITKernel:
    # rename for shorter code
    n = T.dynamic("num_tokens")
    h = hidden

    h_blk = math.gcd(hidden, h_blk)
    a: T.Tensor((n, hc, hc), T.float32)  # type: ignore[no-redef, valid-type]
    b: T.Tensor((n, hc, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
    c: T.Tensor((n, hc), T.float32)  # type: ignore[no-redef, valid-type]
    d: T.Tensor((n, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
    x: T.Tensor((n, hc, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
    with T.Kernel(n, threads=n_thr) as i_n:
        x_shared = T.alloc_shared((hc, h_blk), T.bfloat16)
        b_shared = T.alloc_shared((hc, h_blk), T.bfloat16)
        d_shared = T.alloc_shared(h_blk, T.bfloat16)

        x_local = T.alloc_fragment((hc, h_blk), T.float32)
        b_local = T.alloc_fragment((hc, h_blk), T.float32)
        d_local = T.alloc_fragment(h_blk, T.float32)

        a_local = T.alloc_fragment((hc, hc), T.float32)
        c_local = T.alloc_fragment(hc, T.float32)
        T.pdl_sync()
        T.copy(a[i_n, 0, 0], a_local)
        T.copy(c[i_n, 0], c_local)

        for i0_h in T.Pipelined(T.ceildiv(h, h_blk), num_stages=2):
            T.copy(b[i_n, 0, i0_h * h_blk], b_shared)
            T.copy(d[i_n, i0_h * h_blk], d_shared)

            T.copy(b_shared, b_local)
            T.copy(d_shared, d_local)
            for i_hco, i1_h in T.Parallel(hc, h_blk):
                x_local[i_hco, i1_h] = c_local[i_hco] * d_local[i1_h]
                for i_hci in T.serial(hc):
                    x_local[i_hco, i1_h] += a_local[i_hci, i_hco] * b_local[i_hci, i1_h]
            T.copy(x_local, x_shared)

            T.copy(x_shared, x[i_n, 0, i0_h * h_blk])
        T.pdl_trigger()


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    },
)
def hc_head_fuse_tilelang(
    residual,
    fn,
    hc_scale,
    hc_base,
    out,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    hc_mult: int = 4,
    n_thr: int = 128,
    h_blk: int = 1024,
):
    """Two-pass fused kernel for hc_head.

    Pass 1: accumulate per-token squared sum and hc_mult dot-products
            (projections onto fn rows) using cross-thread reducers.
    Pass 2: apply sigmoid-gated weighted sum of residual channels to output.

    Avoids materialising mixes / rsqrt / pre tensors to global memory.
    """
    num_tokens = T.dynamic("num_tokens")
    hc_dim = hc_mult * hidden_size
    h_block = math.gcd(h_blk, hidden_size)
    n_h = hidden_size // h_block

    residual: T.Tensor[[num_tokens, hc_mult, hidden_size], T.bfloat16]  # type: ignore[no-redef,valid-type]
    fn: T.Tensor[[hc_mult, hc_dim], T.float32]  # type: ignore[no-redef,valid-type]
    hc_scale: T.Tensor[[1], T.float32]  # type: ignore[no-redef,valid-type]
    hc_base: T.Tensor[[hc_mult], T.float32]  # type: ignore[no-redef,valid-type]
    out: T.Tensor[[num_tokens, hidden_size], T.bfloat16]  # type: ignore[no-redef,valid-type]

    with T.Kernel(num_tokens, threads=n_thr) as i:
        T.pdl_sync()

        # ------------------------------------------------------------------
        # Pass 1 – for each residual channel m_c and h_block:
        #   • accumulate squared sum (for RMS norm denominator)
        #   • accumulate hc_mult dot-products with fn rows
        # ------------------------------------------------------------------
        sqrsum_r = T.alloc_reducer((1,), T.float32, replication="all")
        mixes_r = T.alloc_reducer((hc_mult,), T.float32, replication="all")
        T.fill(sqrsum_r, 0.0)
        T.fill(mixes_r, 0.0)

        for m_c in T.serial(hc_mult):
            for i_h in T.serial(n_h):
                x_local = T.alloc_fragment(h_block, T.float32)
                T.copy(residual[i, m_c, i_h * h_block], x_local)

                for k in T.Parallel(h_block):
                    sqrsum_r[0] += x_local[k] * x_local[k]

                for m_m in T.unroll(hc_mult):
                    fn_local = T.alloc_fragment(h_block, T.float32)
                    T.copy(fn[m_m, m_c * hidden_size + i_h * h_block], fn_local)
                    for k in T.Parallel(h_block):
                        mixes_r[m_m] += x_local[k] * fn_local[k]

        T.finalize_reducer(sqrsum_r)
        T.finalize_reducer(mixes_r)

        # ------------------------------------------------------------------
        # Compute pre_mix = sigmoid(mix * rsqrt * scale + base) + eps
        # ------------------------------------------------------------------
        pre_mix_shared = T.alloc_shared(hc_mult, T.float32)
        rsqrt_val = T.alloc_fragment(1, T.float32)
        rsqrt_val[0] = T.rsqrt(sqrsum_r[0] / hc_dim + rms_eps)
        for m in T.Parallel(hc_mult):
            pre_mix_shared[m] = (
                T.sigmoid(mixes_r[m] * rsqrt_val[0] * hc_scale[0] + hc_base[m]) + hc_eps
            )

        # ------------------------------------------------------------------
        # Pass 2 – apply_mix: pipelined weighted sum over residual channels
        # ------------------------------------------------------------------
        for i0_h in T.Pipelined(n_h, num_stages=2):
            xs = T.alloc_shared((hc_mult, h_block), T.bfloat16)
            xl = T.alloc_fragment((hc_mult, h_block), T.float32)
            T.copy(residual[i, 0, i0_h * h_block], xs, disable_tma=True)
            T.copy(xs, xl)

            ol = T.alloc_fragment(h_block, T.float32)
            T.clear(ol)
            for i_hc in T.serial(hc_mult):
                pre = pre_mix_shared[i_hc]
                for i1_h in T.Parallel(h_block):
                    ol[i1_h] += pre * xl[i_hc, i1_h]

            T.copy(ol, out[i, i0_h * h_block], disable_tma=True)

        T.pdl_trigger()
