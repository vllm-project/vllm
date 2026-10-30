# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W8A16 GEMM for RDNA (gfx11/gfx12).

Weights are stored once as symmetric int8 [N, K], with either one scale per
output row (scales [N]) or one per group of K columns (scales [N, K // group]);
the group size is implied by the scale shape. Small batches run on the HIP
skinny GEMM (``wvSplitK_int8``); everything it cannot take (more than
``MAX_SKINNY_BATCH_SIZE`` rows, or activations that do not fit in LDS) runs on a
Triton kernel that reads the same int8 buffer, so no fp16/bf16 copy of the
weight is kept or materialised per call.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from .rdna_hybrid_w4a16 import (
    LDS_CAPACITY_ELEMENTS,
    MAX_SKINNY_BATCH_SIZE,
    _on_gfx1151,
    _profile_scope,
    _weight_pad_bytes,
)


@triton.jit
def _w8a16_kernel(
    a_ptr,  # [M, K] fp16/bf16
    b_ptr,  # [N, K] int8
    scale_ptr,  # [N], or [N, K // GROUP_SIZE] if GROUP_SIZE, fp16/bf16
    bias_ptr,  # [N] fp16/bf16, unused unless HAS_BIAS
    c_ptr,  # [M, N] fp16/bf16
    M,
    N,
    K,
    stride_am,
    stride_bn,
    stride_cm,
    GROUP_SIZE: tl.constexpr,  # 0 for one scale per output row
    HAS_BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SCALE_K: tl.constexpr,  # K columns per dot: BLOCK_K, or one group
):
    # M varies fastest so every M tile of a weight tile runs back to back and
    # the weight is read from DRAM once. Grouping several N tiles per band to
    # also reuse activations measured no faster on gfx1151 (<0.3%).
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    # vocab x hidden and tokens x vocab both reach 2^31, so tile bases are
    # int64 while the in-tile offsets stay int32.
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, SCALE_K)
    # Clamp the ragged edge instead of masking so the K loop loads unmasked.
    rows_a = tl.minimum(offs_m, M - 1 - m0)
    rows_b = tl.minimum(offs_n, N - 1 - n0)

    a_ptrs = (
        a_ptr
        + m0.to(tl.int64) * stride_am
        + rows_a[:, None] * stride_am
        + offs_k[None, :]
    )
    b_ptrs = (
        b_ptr
        + n0.to(tl.int64) * stride_bn
        + rows_b[:, None] * stride_bn
        + offs_k[None, :]
    )
    if GROUP_SIZE:
        num_groups = K // GROUP_SIZE
        s_ptrs = scale_ptr + (n0 + rows_b) * num_groups

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        for kk in tl.static_range(0, BLOCK_K, SCALE_K):
            if EVEN_K:
                a = tl.load(a_ptrs + kk)
                b = tl.load(b_ptrs + kk)
            else:
                k_mask = offs_k < K - k * BLOCK_K - kk
                a = tl.load(a_ptrs + kk, mask=k_mask[None, :], other=0.0)
                b = tl.load(b_ptrs + kk, mask=k_mask[None, :], other=0)
            if a.dtype == tl.bfloat16:
                # gfx11 has no fp32->bf16 convert, so the default
                # round-to-nearest is emulated in VALU and caps the GEMM at
                # ~24 TFLOPs on gfx1151; truncating is exact for |v| <= 127 and
                # reaches ~35.
                b = b.to(tl.float32).to(tl.bfloat16, fp_downcast_rounding="rtz")
            else:
                b = b.to(a.dtype)
            if GROUP_SIZE:
                # Scaling the fp32 partial sum instead of dequantizing the
                # weight avoids rounding every q * scale to fp16/bf16, which
                # in bf16 costs as much accuracy as the int8 quantization.
                group = (k * BLOCK_K + kk) // GROUP_SIZE
                s = tl.load(s_ptrs + group, mask=group < num_groups, other=0.0)
                acc += tl.dot(a, tl.trans(b)) * s.to(tl.float32)[None, :]
            else:
                acc = tl.dot(a, tl.trans(b), acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K

    n_idx = n0 + offs_n
    n_mask = n_idx < N
    if not GROUP_SIZE:
        # A per-row scale factors out of the K sum.
        scale = tl.load(scale_ptr + n_idx, mask=n_mask, other=0.0)
        acc *= scale.to(tl.float32)[None, :]
    if HAS_BIAS:
        acc += tl.load(bias_ptr + n_idx, mask=n_mask, other=0.0).to(tl.float32)[None, :]

    c_ptrs = (
        c_ptr
        + m0.to(tl.int64) * stride_cm
        + offs_m[:, None] * stride_cm
        + n_idx[None, :]
    )
    c_mask = (offs_m[:, None] < M - m0) & n_mask[None, :]
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


# (max M, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages), swept on gfx1151
# over lm_head shapes 151936x{2560,4096} and 262144x5376. Up to M=64 the GEMM
# is bound by the int8 weight read (~210-230 GB/s); past M~256 it is bound by
# Triton's WMMA throughput (~33 TFLOPs).
_TILE_CONFIGS = (
    (16, 16, 32, 256, 2, 2),
    (32, 32, 32, 256, 4, 1),
    (64, 64, 32, 128, 2, 1),
    (128, 128, 64, 128, 4, 1),
    (256, 128, 64, 64, 4, 1),
)
_LARGE_M_CONFIG = (128, 128, 32, 4, 1)
# Per-group scaling adds a second fp32 accumulator per tile, with which the
# per-channel tiles run up to 5x slower from M=32 on; swept at group size 32
# over 151936x2560 and 262144x5376.
_GROUPED_TILE_CONFIGS = (
    (16, 16, 32, 256, 2, 2),
    (32, 32, 32, 128, 2, 2),
    (64, 64, 64, 128, 4, 2),
)
_GROUPED_LARGE_M_CONFIG = (128, 64, 256, 4, 1)


def _get_config(num_rows: int, grouped: bool = False) -> dict[str, int]:
    configs = _GROUPED_TILE_CONFIGS if grouped else _TILE_CONFIGS
    tile: tuple[int, ...] = _GROUPED_LARGE_M_CONFIG if grouped else _LARGE_M_CONFIG
    for max_m, *bucket_tile in configs:
        if num_rows <= max_m:
            tile = tuple(bucket_tile)
            break
    block_m, block_n, block_k, num_warps, num_stages = tile
    return dict(
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def empty_int8_weight(n: int, k: int, device: torch.device) -> torch.Tensor:
    """Allocate int8 [N, K] weights the way both W8A16 kernels read them fastest.

    On gfx1151 rows whose byte stride sits on the vector-cache cliff are padded
    by one cache line (see ``rdna_hybrid_w4a16._weight_pad_bytes``); the
    returned tensor has shape [N, K] with the padded row stride. Writing codes
    straight into it avoids holding a second int8 copy of the weight at load.
    """
    pad = _weight_pad_bytes(k) if _on_gfx1151() else 0
    return torch.empty((n, k + pad), dtype=torch.int8, device=device)[:, :k]


def triton_w8a16_gemm(
    a: torch.Tensor,
    b_q: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    config: dict[str, int] | None = None,
) -> torch.Tensor:
    """``a @ dequant(b_q).T + bias``, without materialising ``dequant(b_q)``.

    Args:
        a:      Activations [M, K], float16 or bfloat16, rows contiguous.
        b_q:    Symmetric int8 weights [N, K], rows contiguous (may be padded).
        scale:  Per-output-channel scales [N], or per-group scales
                [N, K // group_size], same dtype as ``a``.
        bias:   Optional bias [N], same dtype as ``a``.
        config: Tile config override, for tuning; defaults to ``_get_config``.

    Returns:
        Output [M, N] in the dtype of ``a``.

    """
    M, K = a.shape
    N = b_q.shape[0]
    assert b_q.dtype == torch.int8 and b_q.shape == (N, K), (
        f"b_q must be int8 [N, {K}], got {b_q.dtype} {tuple(b_q.shape)}"
    )
    assert a.stride(1) == 1 and b_q.stride(1) == 1, "rows must be contiguous"
    group_size = 0 if scale.dim() == 1 else K // scale.shape[1]
    assert scale.shape == ((N,) if group_size == 0 else (N, K // group_size)), (
        f"scale must be [N] or [N, K // group_size], got {tuple(scale.shape)}"
    )
    assert scale.is_contiguous()
    assert bias is None or (bias.shape == (N,) and bias.is_contiguous())

    c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    if M == 0:
        return c

    cfg = config if config is not None else _get_config(M, group_size > 0)
    block_k = cfg["BLOCK_K"]
    scale_k = min(block_k, group_size) if group_size else block_k
    assert block_k % scale_k == 0 and group_size % scale_k == 0
    grid = (triton.cdiv(M, cfg["BLOCK_M"]) * triton.cdiv(N, cfg["BLOCK_N"]),)
    _w8a16_kernel[grid](
        a,
        b_q,
        scale,
        bias if bias is not None else scale,
        c,
        M,
        N,
        K,
        a.stride(0),
        b_q.stride(0),
        c.stride(0),
        GROUP_SIZE=group_size,
        HAS_BIAS=bias is not None,
        EVEN_K=K % block_k == 0,
        SCALE_K=scale_k,
        **cfg,
    )
    return c


def _rdna_w8a16_apply_impl(
    x_2d: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    bias: torch.Tensor | None,
    cu_count: int,
) -> torch.Tensor:
    M, K = x_2d.shape
    N = w_q.shape[0]
    # wvSplitK_int8 stages every activation row in LDS and reads the weight
    # in 16-byte chunks.
    use_skinny = (
        M <= MAX_SKINNY_BATCH_SIZE
        and K * M <= LDS_CAPACITY_ELEMENTS
        and K % 16 == 0
        and w_q.stride(0) % 16 == 0
    )
    name = "wvsplitk_int8" if use_skinny else "triton_w8a16"
    with _profile_scope(f"{name} {M}x{N}x{K}"):
        if use_skinny:
            import vllm._custom_ops as ops

            return ops.wvSplitK_int8(w_q, x_2d.contiguous(), w_s, cu_count, bias)
        return triton_w8a16_gemm(x_2d, w_q, w_s, bias)


def _rdna_w8a16_apply_fake(
    x_2d: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    bias: torch.Tensor | None,
    cu_count: int,
) -> torch.Tensor:
    return x_2d.new_empty((x_2d.shape[0], w_q.shape[0]))


direct_register_custom_op(
    op_name="rdna_w8a16_apply",
    op_func=_rdna_w8a16_apply_impl,
    mutates_args=[],
    fake_impl=_rdna_w8a16_apply_fake,
)
