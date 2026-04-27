# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from functools import cache
from typing import TYPE_CHECKING

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import use_dsv4_reference_kernels
from vllm.utils.import_utils import has_tilelang
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import direct_register_custom_op

# tilelang is only available on CUDA platforms
if TYPE_CHECKING or current_platform.is_cuda_alike():
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


@triton.jit
def _mhc_sinkhorn_kernel(
    in_ptr,  # comb_logits (T, hc, hc) fp32
    out_ptr,  # comb_mix (T, hc, hc) fp32
    n_tokens,
    eps: tl.constexpr,
    hc: tl.constexpr,
    iterations: tl.constexpr,
):
    """Fused softmax + iterative Sinkhorn (row+col) normalisation.

    Replaces:
        comb_mix = softmax(logits, -1) + eps
        comb_mix /= comb_mix.sum(-2, keepdim=True) + eps
        for _ in range(iterations - 1):
            comb_mix /= comb_mix.sum(-1, keepdim=True) + eps
            comb_mix /= comb_mix.sum(-2, keepdim=True) + eps

    One program per token; comb is hc x hc held in registers throughout."""
    pid = tl.program_id(0)
    if pid >= n_tokens:
        return

    base = in_ptr + pid * hc * hc
    rows = tl.arange(0, hc)
    cols = tl.arange(0, hc)
    cm = tl.load(base + rows[:, None] * hc + cols[None, :]).to(tl.float32)

    # Row-wise softmax
    row_max = tl.max(cm, axis=1)
    cm = cm - row_max[:, None]
    cm = tl.exp(cm)
    row_sum = tl.sum(cm, axis=1)
    cm = cm / row_sum[:, None] + eps

    # First column-norm
    col_sum = tl.sum(cm, axis=0)
    cm = cm / (col_sum[None, :] + eps)

    # iterations - 1 more row-then-column normalisations
    for _ in range(iterations - 1):
        row_sum = tl.sum(cm, axis=1)
        cm = cm / (row_sum[:, None] + eps)
        col_sum = tl.sum(cm, axis=0)
        cm = cm / (col_sum[None, :] + eps)

    out_base = out_ptr + pid * hc * hc
    tl.store(out_base + rows[:, None] * hc + cols[None, :], cm)


def _mhc_softmax_sinkhorn_triton(
    comb_logits: torch.Tensor,
    eps: float,
    iterations: int,
) -> torch.Tensor:
    """Fused softmax + Sinkhorn iterations on (T, hc, hc) input. Replaces
    the 1 + 2*(iterations-1) PyTorch sum/div pairs with one Triton kernel
    that holds the (small) hc x hc matrix in registers throughout."""
    n_tokens, hc, hc2 = comb_logits.shape
    assert hc == hc2
    out = torch.empty_like(comb_logits)
    _mhc_sinkhorn_kernel[(n_tokens,)](
        comb_logits.contiguous(),
        out,
        n_tokens,
        eps=eps,
        hc=hc,
        iterations=iterations,
        num_warps=1,
    )
    return out


@triton.jit
def _mhc_post_per_token(
    a_ptr,  # comb_res_mix (T, hc, hc) fp32
    b_ptr,  # residual (T, hc, h) bf16
    c_ptr,  # post_layer_mix (T, hc, 1) fp32 — last dim length 1
    d_ptr,  # x (T, h) bf16
    out_ptr,  # out (T, hc, h) bf16
    n_tokens,
    hc: tl.constexpr,
    h: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Fused mHC post:
        out[t, j, k] = sum_i comb[t, i, j] * residual[t, i, k]
                     + post[t, j] * x[t, k]
    One program per (token, h_block). Loads `comb` (hc x hc) and
    `post` (hc) once; streams `residual` and `x` over h-tiles."""
    pid_t = tl.program_id(0).to(tl.int64)
    pid_hb = tl.program_id(1).to(tl.int64)

    h_off = pid_hb * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_off < h

    a_off = (
        a_ptr
        + pid_t * hc * hc
        + tl.arange(0, hc)[:, None] * hc
        + tl.arange(0, hc)[None, :]
    )
    a = tl.load(a_off).to(tl.float32)  # (hc, hc) fp32

    c_off = c_ptr + pid_t * hc + tl.arange(0, hc)
    c = tl.load(c_off).to(tl.float32)  # (hc,) fp32

    d_off = d_ptr + pid_t * h + h_off
    d = tl.load(d_off, mask=h_mask, other=0.0).to(tl.float32)  # (BLOCK_H,)

    # b: (hc, BLOCK_H) bf16 → fp32
    b_off = b_ptr + pid_t * hc * h + tl.arange(0, hc)[:, None] * h + h_off[None, :]
    b = tl.load(b_off, mask=h_mask[None, :], other=0.0).to(tl.float32)

    # out[j, k] = sum_i a[i, j] * b[i, k] + c[j] * d[k]
    # tl.dot requires K >= 16; hc is typically 4. 3D-broadcast + reduce
    # turns the contraction into a few FMAs in registers (Triton can't
    # index a 2D tensor by a constexpr scalar, so static_range over rows
    # is unavailable).
    a_T = tl.trans(a)  # (hc, hc): a_T[j, i] = a[i, j]
    prod = a_T[:, :, None] * b[None, :, :]  # (hc, hc, BLOCK_H)
    mixed = tl.sum(prod, axis=1)  # (hc, BLOCK_H)
    post_term = c[:, None] * d[None, :]
    result = (mixed + post_term).to(tl.bfloat16)

    out_off = out_ptr + pid_t * hc * h + tl.arange(0, hc)[:, None] * h + h_off[None, :]
    tl.store(out_off, result, mask=h_mask[None, :])


def _mhc_post_triton(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """SM80 fused Triton mhc_post. Replaces the multi-op PyTorch reference."""
    outer_shape = residual.shape[:-2]
    hc = residual.shape[-2]
    h = residual.shape[-1]
    n_tokens = residual.numel() // (hc * h)

    res = residual.reshape(n_tokens, hc, h).contiguous()
    comb = comb_res_mix.reshape(n_tokens, hc, hc).to(torch.float32).contiguous()
    post = post_layer_mix.reshape(n_tokens, hc).to(torch.float32).contiguous()
    x_flat = x.reshape(n_tokens, h).contiguous()

    out = torch.empty_like(res)

    BLOCK_H = 256 if h >= 256 else 128
    grid = (n_tokens, triton.cdiv(h, BLOCK_H))
    _mhc_post_per_token[grid](
        comb,
        res,
        post,
        x_flat,
        out,
        n_tokens,
        hc=hc,
        h=h,
        BLOCK_H=BLOCK_H,
    )
    return out.view(*outer_shape, hc, h)


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


def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass for mHC pre block.

    Args:
        residual: shape (..., hc_mult, hidden_size), dtype torch.bfloat16
        fn: shape (hc_mult3, hc_mult * hidden_size), dtype torch.float32
        hc_scale: shape (3,), dtype torch.float32
        hc_base: shape (hc_mult3,), dtype torch.float32
        rms_eps: RMS normalization epsilon
        hc_pre_eps: pre-mix epsilon
        hc_sinkhorn_eps: sinkhorn epsilon
        hc_post_mult_value: post-mix multiplier value
        sinkhorn_repeat: number of sinkhorn iterations
        n_splits: split-k factor;

    Returns:
        post_mix: shape (..., hc_mult), dtype torch.float32
        comb_mix: shape (..., hc_mult, hc_mult), dtype torch.float32
        layer_input: shape (..., hidden_size), dtype torch.bfloat16
    """

    # Validate shapes
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2

    hc_hidden_size = hc_mult * hidden_size
    assert fn.shape[0] == hc_mult3
    assert fn.shape[1] == hc_hidden_size
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    outer_shape = residual.shape[:-2]

    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    fn_flat = fn

    if use_dsv4_reference_kernels():
        # SM80/ROCm reference path: pure-PyTorch fused implementation that
        # bypasses both DeepGEMM `tf32_hc_prenorm_gemm` (Hopper+ only) and
        # the tilelang follow-up. Lifted from PR 40871's ROCm branch.
        x = residual_flat.view(num_tokens, hc_mult * hidden_size).to(torch.float32)
        mixes = torch.matmul(x, fn_flat.t())
        sqrsum = x.square().sum(dim=-1, keepdim=True)
        mixes = mixes * torch.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)

        pre_logits = mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
        pre_mix = torch.sigmoid(pre_logits) + hc_pre_eps

        post_logits = (
            mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1]
            + hc_base[hc_mult : 2 * hc_mult]
        )
        post_mix = torch.sigmoid(post_logits) * hc_post_mult_value

        comb_logits = mixes[:, 2 * hc_mult :].view(
            num_tokens, hc_mult, hc_mult
        ) * hc_scale[2] + hc_base[2 * hc_mult :].view(1, hc_mult, hc_mult)
        # Fused softmax + Sinkhorn iterations (replaces 1 + 2*(iters-1)
        # PyTorch sum/div pairs with one Triton kernel — primary mhc_pre
        # CPU-launch bottleneck on SM80 with sinkhorn_repeat=20).
        comb_mix = _mhc_softmax_sinkhorn_triton(
            comb_logits, hc_sinkhorn_eps, sinkhorn_repeat
        )

        layer_input = torch.sum(
            pre_mix.unsqueeze(-1) * residual_flat.to(torch.float32), dim=1
        ).to(torch.bfloat16)
        return (
            post_mix.view(*outer_shape, hc_mult, 1),
            comb_mix.view(*outer_shape, hc_mult, hc_mult),
            layer_input.view(*outer_shape, hidden_size),
        )

    # these number are from deepgemm kernel impl
    block_k = 64
    block_m = 64
    n_splits = compute_num_split(block_k, hc_hidden_size, cdiv(num_tokens, block_m))

    post_mix = torch.empty(
        num_tokens,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix = torch.empty(
        num_tokens,
        hc_mult2,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input = torch.empty(
        num_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device=residual.device,
    )

    gemm_out_mul = torch.empty(
        n_splits,
        num_tokens,
        hc_mult3,
        dtype=torch.float32,
        device=residual.device,
    )
    gemm_out_sqrsum = torch.empty(
        n_splits,
        num_tokens,
        dtype=torch.float32,
        device=residual.device,
    )

    from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm

    tf32_hc_prenorm_gemm(
        residual_flat.view(num_tokens, hc_mult * hidden_size),
        fn_flat,
        gemm_out_mul,
        gemm_out_sqrsum,
        n_splits,
    )

    mhc_pre_big_fuse_tilelang(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual_flat,
        post_mix,
        comb_mix,
        layer_input,
        hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
        hc_mult,
    )

    post_mix = post_mix.view(*outer_shape, hc_mult, 1)
    comb_mix = comb_mix.view(*outer_shape, hc_mult, hc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size)

    return post_mix, comb_mix, layer_input


def _mhc_pre_fake(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]

    # Create empty tensors with correct shapes for meta device / shape inference
    post_mix = torch.empty(
        *outer_shape,
        hc_mult,
        1,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix = torch.empty(
        *outer_shape,
        hc_mult,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input = torch.empty(
        *outer_shape,
        hidden_size,
        dtype=torch.bfloat16,
        device=residual.device,
    )

    return post_mix, comb_mix, layer_input


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


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    if use_dsv4_reference_kernels():
        # SM80/ROCm fused Triton path; faster than the multi-op PyTorch
        # einsum + broadcast reference and matches the upstream tilelang
        # kernel's output to bf16 precision.
        return _mhc_post_triton(x, residual, post_layer_mix, comb_res_mix)
    out = torch.empty_like(residual)
    mhc_post_tilelang(
        comb_res_mix,
        residual,
        post_layer_mix.squeeze(-1),
        x,
        out,
        residual.shape[-2],
        residual.shape[-1],
    )
    return out


def _mhc_post_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(residual)


direct_register_custom_op(
    op_name="mhc_pre",
    op_func=mhc_pre,
    mutates_args=[],
    fake_impl=_mhc_pre_fake,
)
direct_register_custom_op(
    op_name="mhc_post",
    op_func=mhc_post,
    mutates_args=[],
    fake_impl=_mhc_post_fake,
)
