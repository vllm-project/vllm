# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 gate/up GEMM with native-rounded SiLU-multiply epilogue."""

import torch

from vllm.triton_utils import tl, tldevice, triton


@triton.jit
def native_round_silu_mul(gate, up):
    # Match the native boundary: linear -> BF16, SiLU -> BF16, product -> BF16.
    gate = gate.to(tl.bfloat16).to(tl.float32)
    up = up.to(tl.bfloat16).to(tl.float32)
    activated = tl.div_rn(gate, 1.0 + tldevice.exp(-gate))
    activated = activated.to(tl.bfloat16).to(tl.float32)
    return (activated * up).to(tl.bfloat16)


@triton.jit
def tile_ids(M, N, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    pm, pn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    group_size = 8 * pn
    group = pid // group_size
    first = group * 8
    actual = tl.minimum(pm - first, 8)
    return first + (pid % group_size) % actual, (pid % group_size) // actual


@triton.jit(do_not_specialize=["M"])
def _gated_linear_silu_kernel(
    X,
    W,
    Y,
    M,
    K: tl.constexpr,
    DFF: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    WS0: tl.constexpr,
    WS1: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pm, pn = tile_ids(M, DFF, BM, BN)
    rm = pm * BM + tl.arange(0, BM)
    pair = 2 * pn * BN + tl.arange(0, 2 * BN)
    raw_row = pair // 2 + (pair % 2) * DFF
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, 2 * BN), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        kk = block * BK + rk
        a = tl.load(
            X + rm[:, None] * XS0 + kk[None, :] * XS1,
            mask=(rm[:, None] < M) & (kk[None, :] < K),
            other=0,
        )
        b = tl.load(
            W + raw_row[None, :] * WS0 + kk[:, None] * WS1,
            mask=(pair[None, :] < 2 * DFF) & (kk[:, None] < K),
            other=0,
        )
        acc = tl.dot(a, b, acc)
    gate, up = tl.split(tl.reshape(acc, (BM, BN, 2)))
    rn = pn * BN + tl.arange(0, BN)
    tl.store(
        Y + rm[:, None] * DFF + rn[None, :],
        native_round_silu_mul(gate, up),
        mask=(rm[:, None] < M) & (rn[None, :] < DFF),
    )


def gated_linear_silu(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Map BF16 ``x[M,K]`` and raw ``weight[2I,K]`` to a fresh BF16 ``out[M,I]``.

    The caller validates dtype and supported shapes. Gate/up rows are read in
    paired order without transforming or retaining a copy of the weight.
    """
    m, k = x.shape
    d = weight.shape[0] // 2
    out = torch.empty((m, d), dtype=x.dtype, device=x.device)
    _gated_linear_silu_kernel[(triton.cdiv(m, 64) * triton.cdiv(d, 64),)](
        x,
        weight,
        out,
        M=m,
        K=k,
        DFF=d,
        XS0=x.stride(0),
        XS1=x.stride(1),
        WS0=weight.stride(0),
        WS1=weight.stride(1),
        BM=64,
        BN=64,
        BK=32,
        num_warps=4,
        num_stages=3,
        enable_fp_fusion=False,
    )
    return out
