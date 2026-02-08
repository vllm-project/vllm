# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import helion
import helion.language as hl
import torch

from vllm.utils.import_utils import has_helion

if not has_helion():
    raise ImportError(
        "Helion kernel requires helion to be installed. "
        "Install it with: pip install helion"
    )


def is_weak_contiguous(x: torch.Tensor):
    strides = x.stride()
    sizes = x.shape
    is_not_transpose = strides[0] == 1 and (strides[1] >= max(1, sizes[0]))
    is_transpose = strides[1] == 1 and (strides[0] >= max(1, sizes[1]))
    return is_transpose or is_not_transpose


@helion.kernel(
    autotune_effort="none",
    static_shapes=False,
    allow_warp_specialize=True,
    autotune_baseline_atol=0.0,
    autotune_baseline_rtol=0.0,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def scaled_mm(
    a: torch.Tensor,  # [M, K]
    b: torch.Tensor,  # [K, N]
    scale_a: torch.Tensor,  # [1]/[1, 1]/[M]/[M, 1]
    scale_b: torch.Tensor,  # [1]/[1, 1]/[N]/[N, 1]
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,  # [N]
) -> torch.Tensor:
    M, K = a.shape
    N = b.shape[1]
    hl.specialize(K)
    hl.specialize(N)

    assert N > 0 and K > 0 and M > 0
    assert b.shape[0] == K
    assert a.dtype == b.dtype

    scale_a = scale_a.reshape(-1, 1) if scale_a.dim() <= 1 else scale_a
    scale_b = scale_b.reshape(-1, 1) if scale_b.dim() <= 1 else scale_b

    assert scale_a.dtype == scale_b.dtype and scale_a.is_floating_point()
    assert scale_a.shape[1] == 1 and (scale_a.shape[0] == 1 or scale_a.shape[0] == M)
    assert scale_b.shape[1] == 1 and (scale_b.shape[0] == 1 or scale_b.shape[0] == N)
    assert out_dtype.is_floating_point
    assert is_weak_contiguous(a)
    assert is_weak_contiguous(b)
    has_bias = bias is not None
    if has_bias:
        assert bias.numel() == N and bias.dtype == out_dtype

    c = torch.empty((M, N), dtype=out_dtype, device=a.device)
    accumulator_dtype = torch.float32 if a.is_floating_point() else torch.int32

    for tile_m, tile_n in hl.tile([M, N]):
        accumulator = hl.zeros([tile_m, tile_n], accumulator_dtype)
        for tile_k in hl.tile(K):
            accumulator = hl.dot(
                a[tile_m, tile_k],
                b[tile_k, tile_n],
                acc=accumulator,
                out_dtype=accumulator_dtype,
            )

        scale_a_mask = (tile_m.index < scale_a.shape[0])[:, None]
        scale_a_blk = torch.where(scale_a_mask, scale_a[tile_m, :], scale_a[0, 0])
        accumulator = scale_a_blk * accumulator.to(torch.float32)

        scale_b_mask = (tile_n.index < scale_b.shape[0])[:, None]
        scale_b_blk = torch.where(scale_b_mask, scale_b[tile_n, :], scale_b[0, 0])
        accumulator = scale_b_blk.T * accumulator.to(torch.float32)

        c_blk = accumulator.to(out_dtype)

        if has_bias:
            c_blk += bias[tile_n]

        c[tile_m, tile_n] = c_blk

    return c
