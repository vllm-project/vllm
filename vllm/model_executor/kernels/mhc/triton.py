# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn.functional as F
from torch import Tensor

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _rmsnorm_nw_kernel(
    x_ptr,
    out_ptr,
    stride_row,
    D,
    eps,
    RBLOCK: tl.constexpr,
):
    """Weight-free RMSNorm Triton kernel: out = x * rsqrt(mean(x², -1) + eps)."""
    row = tl.program_id(0)
    cols = tl.arange(0, RBLOCK)
    mask = cols < D

    x = tl.load(
        x_ptr + row * stride_row + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)

    var = tl.sum(x * x, 0) / D
    rstd = tl.rsqrt(var + eps)

    out = (x * rstd).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + row * D + cols, out, mask=mask, eviction_policy="evict_first")


def rmsnorm_nw(x: Tensor, eps: float) -> Tensor:
    """Weight-free RMSNorm over the last dimension.

    Treats *x* as ``[num_rows, D]`` where ``num_rows = product(shape[:-1])``.
    Returns a contiguous tensor with the same shape and dtype as *x*.
    """
    orig_shape = x.shape
    D = orig_shape[-1]
    x_2d = x.reshape(-1, D)
    num_rows = x_2d.shape[0]

    out = torch.empty_like(x_2d)
    RBLOCK = triton.next_power_of_2(D)

    _rmsnorm_nw_kernel[(num_rows,)](
        x_2d,
        out,
        x_2d.stride(0),
        D,
        eps,
        RBLOCK=RBLOCK,
        num_warps=1 if RBLOCK <= 512 else (4 if RBLOCK <= 4096 else 8),
    )
    return out.view(orig_shape)


@triton.jit
def _hc_head_reduce_store_kernel(
    pre_ptr,
    x_ptr,
    out_ptr,
    hidden_size: tl.constexpr,
    hc_mult: tl.constexpr,
    pre_stride_t: tl.constexpr,
    pre_stride_m: tl.constexpr,
    x_stride_t: tl.constexpr,
    x_stride_m: tl.constexpr,
    x_stride_h: tl.constexpr,
    out_stride_t: tl.constexpr,
    out_stride_h: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1)
    offsets = block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offsets < hidden_size

    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for mix_idx in tl.static_range(0, hc_mult):
        pre = tl.load(pre_ptr + token_idx * pre_stride_t + mix_idx * pre_stride_m).to(
            tl.float32
        )
        x = tl.load(
            x_ptr
            + token_idx * x_stride_t
            + mix_idx * x_stride_m
            + offsets * x_stride_h,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        acc += pre * x

    tl.store(
        out_ptr + token_idx * out_stride_t + offsets * out_stride_h,
        acc,
        mask=mask,
    )


def hc_collapse_triton(x: Tensor, pre_mix: Tensor) -> Tensor:
    """Collapse BF16 residual streams with FP32 pre-mix coefficients."""
    assert x.ndim == 3 and x.dtype == torch.bfloat16
    num_tokens, hc_mult, hidden_size = x.shape
    assert pre_mix.shape == (num_tokens, hc_mult)
    assert pre_mix.dtype == torch.float32
    out = torch.empty(num_tokens, hidden_size, dtype=x.dtype, device=x.device)
    if num_tokens == 0:
        return out

    block_h = 1024
    _hc_head_reduce_store_kernel[(num_tokens, triton.cdiv(hidden_size, block_h))](
        pre_mix,
        x,
        out,
        hidden_size,
        hc_mult,
        pre_mix.stride(0),
        pre_mix.stride(1),
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK_H=block_h,
        num_warps=4,
        # Preserve the separate FP32 multiply and sum in the Torch reference.
        enable_fp_fusion=False,
    )
    return out


def _hc_collapse_triton_fake(x: Tensor, pre_mix: Tensor) -> Tensor:
    return torch.empty(x.shape[0], x.shape[2], dtype=x.dtype, device=x.device)


@triton.jit
def _mhc_pre_mix_kernel(
    gemm_ptr,
    sqrsum_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    out_ptr,
    splitk,
    hc_mult: tl.constexpr,
    gemm_stride_k,
    gemm_stride_t,
    gemm_stride_j,
    sqrsum_stride_k,
    sqrsum_stride_t,
    out_stride_t,
    out_stride_j,
    inv_hc_hidden,
    rms_eps,
    hc_pre_eps,
    SPLITK_BLOCK: tl.constexpr,
    HC_BLOCK: tl.constexpr,
):
    """Recover the pre-mix gate from a split-k mHC pre GEMM output."""
    token_idx = tl.program_id(0).to(tl.int64)
    ks = tl.arange(0, SPLITK_BLOCK)
    js = tl.arange(0, HC_BLOCK)
    kmask = ks < splitk
    jmask = js < hc_mult

    # Only the first hc_mult GEMM columns feed the pre gate.
    gemm = tl.load(
        gemm_ptr
        + ks[:, None] * gemm_stride_k
        + token_idx * gemm_stride_t
        + js[None, :] * gemm_stride_j,
        mask=kmask[:, None] & jmask[None, :],
        other=0.0,
    ).to(tl.float32)
    mixes = tl.sum(gemm, 0)

    sqrsum = tl.load(
        sqrsum_ptr + ks * sqrsum_stride_k + token_idx * sqrsum_stride_t,
        mask=kmask,
        other=0.0,
    ).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(sqrsum, 0) * inv_hc_hidden + rms_eps)

    scale = tl.load(hc_scale_ptr).to(tl.float32)
    base = tl.load(hc_base_ptr + js, mask=jmask, other=0.0).to(tl.float32)

    pre = tl.sigmoid(mixes * rstd * scale + base) + hc_pre_eps
    tl.store(out_ptr + token_idx * out_stride_t + js * out_stride_j, pre, mask=jmask)


def mhc_pre_mix_triton(
    gemm_out: Tensor,
    sqrsum: Tensor,
    hc_scale: Tensor,
    hc_base: Tensor,
    hc_mult: int,
    hc_hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
) -> Tensor:
    """Pre-mix gate for the delayed mHC pre, from AITER's split-k GEMM output.

    AITER's ``mhc_pre_big_fuse`` consumes the unreduced ``[splitk, tokens,
    hc_mult3]`` GEMM output and the matching row square-sums, but only returns
    the post and comb gates. The delayed formulation also needs the pre gate,
    to carry into the next sublayer seam. It is the same slice of the same
    numbers, so recover it here rather than repeating the projection.
    """
    assert gemm_out.ndim == 3 and gemm_out.dtype == torch.float32
    assert sqrsum.ndim == 2 and sqrsum.dtype == torch.float32
    splitk, num_tokens = gemm_out.shape[0], gemm_out.shape[1]
    assert sqrsum.shape == (splitk, num_tokens)

    out = torch.empty(num_tokens, hc_mult, dtype=torch.float32, device=gemm_out.device)
    if num_tokens == 0:
        return out

    _mhc_pre_mix_kernel[(num_tokens,)](
        gemm_out,
        sqrsum,
        hc_scale,
        hc_base,
        out,
        splitk,
        hc_mult,
        gemm_out.stride(0),
        gemm_out.stride(1),
        gemm_out.stride(2),
        sqrsum.stride(0),
        sqrsum.stride(1),
        out.stride(0),
        out.stride(1),
        1.0 / hc_hidden_size,
        rms_eps,
        hc_pre_eps,
        SPLITK_BLOCK=triton.next_power_of_2(splitk),
        HC_BLOCK=triton.next_power_of_2(hc_mult),
        num_warps=1,
        # Match the separate FP32 multiply and add in the Torch reference.
        enable_fp_fusion=False,
    )
    return out


def _mhc_pre_mix_triton_fake(
    gemm_out: Tensor,
    sqrsum: Tensor,
    hc_scale: Tensor,
    hc_base: Tensor,
    hc_mult: int,
    hc_hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
) -> Tensor:
    return torch.empty(
        gemm_out.shape[1], hc_mult, dtype=torch.float32, device=gemm_out.device
    )


def hc_head_reduce_triton_kernel(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    out: torch.Tensor,
    norm_eps: float,
    hc_eps: float,
) -> None:
    x_flat = x.flatten(-2)
    x_normed = rmsnorm_nw(x_flat, norm_eps)
    mixes = F.linear(x_normed.float(), hc_fn)
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps

    hidden_size = x.shape[-1]
    hc_mult = x.shape[-2]
    block_h = 1024
    _hc_head_reduce_store_kernel[(x.shape[0], (hidden_size + block_h - 1) // block_h)](
        pre,
        x,
        out,
        hidden_size,
        hc_mult,
        pre.stride(0),
        pre.stride(1),
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK_H=block_h,
        num_warps=4,
    )


def _hc_head_triton(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    out: torch.Tensor,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    hc_mult: int,
) -> None:
    """Fill pre-allocated `out` (T, H) in-place with the hc_head result."""
    if hs_flat.shape[0] == 0:
        return

    hc_head_reduce_triton_kernel(
        hs_flat,
        fn,
        hc_scale,
        hc_base,
        out,
        rms_eps,
        hc_eps,
    )
    return


direct_register_custom_op(
    op_name="hc_head_triton",
    op_func=_hc_head_triton,
    mutates_args=["out"],
)


direct_register_custom_op(
    op_name="hc_collapse_triton",
    op_func=hc_collapse_triton,
    mutates_args=[],
    fake_impl=_hc_collapse_triton_fake,
)


direct_register_custom_op(
    op_name="mhc_pre_mix_triton",
    op_func=mhc_pre_mix_triton,
    mutates_args=[],
    fake_impl=_mhc_pre_mix_triton_fake,
)
