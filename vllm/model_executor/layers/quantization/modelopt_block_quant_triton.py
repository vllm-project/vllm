# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for ModelOpt IQ2_XXS / Q8_0 block weights on NVIDIA SM75+."""

import functools

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

# ggml `iq2xxs_grid`: entry i packs the 8 magnitudes (8, 25 or 43) of row i.
# fmt: off
_IQ2XXS_GRID = (
    0x0808080808080808, 0x080808080808082B, 0x0808080808081919, 0x0808080808082B08,
    0x0808080808082B2B, 0x0808080808190819, 0x0808080808191908, 0x08080808082B0808,
    0x08080808082B082B, 0x08080808082B2B08, 0x08080808082B2B2B, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192B08, 0x08080808192B0819,
    0x08080808192B1908, 0x080808082B080808, 0x080808082B08082B, 0x080808082B082B2B,
    0x080808082B2B082B, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192B081908, 0x080808192B192B08,
    0x0808082B08080808, 0x0808082B0808082B, 0x0808082B082B082B, 0x0808082B2B08082B,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082B0819,
    0x08081908082B1908, 0x0808190819080808, 0x080819081908082B, 0x0808190819082B08,
    0x08081908192B0808, 0x080819082B080819, 0x080819082B081908, 0x080819082B190808,
    0x080819082B2B1908, 0x0808191908080808, 0x080819190808082B, 0x0808191908082B08,
    0x08081919082B0808, 0x080819191908192B, 0x08081919192B2B19, 0x080819192B080808,
    0x080819192B190819, 0x0808192B08082B19, 0x0808192B08190808, 0x0808192B19080808,
    0x0808192B2B081908, 0x0808192B2B2B1908, 0x08082B0808080808, 0x08082B0808081919,
    0x08082B0808082B08, 0x08082B0808191908, 0x08082B08082B2B08, 0x08082B0819080819,
    0x08082B0819081908, 0x08082B0819190808, 0x08082B081919082B, 0x08082B082B082B08,
    0x08082B1908081908, 0x08082B1919080808, 0x08082B2B0808082B, 0x08082B2B08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082B0819,
    0x0819080819080808, 0x08190808192B0808, 0x081908082B081908, 0x081908082B190808,
    0x081908082B191919, 0x0819081908080808, 0x0819081908082B08, 0x08190819082B0808,
    0x0819081919190808, 0x0819081919192B2B, 0x081908192B080808, 0x0819082B082B1908,
    0x0819082B19081919, 0x0819190808080808, 0x0819190808082B08, 0x08191908082B0808,
    0x08191908082B1919, 0x0819190819082B19, 0x081919082B080808, 0x0819191908192B08,
    0x08191919192B082B, 0x0819192B08080808, 0x0819192B0819192B, 0x08192B0808080819,
    0x08192B0808081908, 0x08192B0808190808, 0x08192B0819080808, 0x08192B082B080819,
    0x08192B1908080808, 0x08192B1908081919, 0x08192B192B2B0808, 0x08192B2B19190819,
    0x082B080808080808, 0x082B08080808082B, 0x082B080808082B2B, 0x082B080819081908,
    0x082B0808192B0819, 0x082B08082B080808, 0x082B08082B08082B, 0x082B0819082B2B19,
    0x082B081919082B08, 0x082B082B08080808, 0x082B082B0808082B, 0x082B190808080819,
    0x082B190808081908, 0x082B190808190808, 0x082B190819080808, 0x082B19081919192B,
    0x082B191908080808, 0x082B191919080819, 0x082B1919192B1908, 0x082B192B2B190808,
    0x082B2B0808082B08, 0x082B2B08082B0808, 0x082B2B082B191908, 0x082B2B2B19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192B08,
    0x19080808082B0819, 0x19080808082B1908, 0x1908080819080808, 0x1908080819082B08,
    0x190808081919192B, 0x19080808192B0808, 0x190808082B080819, 0x190808082B081908,
    0x190808082B190808, 0x1908081908080808, 0x19080819082B0808, 0x19080819192B0819,
    0x190808192B080808, 0x190808192B081919, 0x1908082B08080819, 0x1908082B08190808,
    0x1908082B19082B08, 0x1908082B1919192B, 0x1908082B192B2B08, 0x1908190808080808,
    0x1908190808082B08, 0x19081908082B0808, 0x190819082B080808, 0x190819082B192B19,
    0x190819190819082B, 0x19081919082B1908, 0x1908192B08080808, 0x19082B0808080819,
    0x19082B0808081908, 0x19082B0808190808, 0x19082B0819080808, 0x19082B0819081919,
    0x19082B1908080808, 0x19082B1919192B08, 0x19082B19192B0819, 0x19082B192B08082B,
    0x19082B2B19081919, 0x19082B2B2B190808, 0x1919080808080808, 0x1919080808082B08,
    0x1919080808190819, 0x1919080808192B19, 0x19190808082B0808, 0x191908082B080808,
    0x191908082B082B08, 0x1919081908081908, 0x191908191908082B, 0x191908192B2B1908,
    0x1919082B2B190819, 0x191919082B190808, 0x191919082B19082B, 0x1919191908082B2B,
    0x1919192B08080819, 0x1919192B19191908, 0x19192B0808080808, 0x19192B0808190819,
    0x19192B0808192B19, 0x19192B08192B1908, 0x19192B1919080808, 0x19192B2B08082B08,
    0x192B080808081908, 0x192B080808190808, 0x192B080819080808, 0x192B0808192B2B08,
    0x192B081908080808, 0x192B081919191919, 0x192B082B08192B08, 0x192B082B192B0808,
    0x192B190808080808, 0x192B190808081919, 0x192B191908190808, 0x192B19190819082B,
    0x192B19192B081908, 0x192B2B081908082B, 0x2B08080808080808, 0x2B0808080808082B,
    0x2B08080808082B2B, 0x2B08080819080819, 0x2B0808082B08082B, 0x2B08081908081908,
    0x2B08081908192B08, 0x2B08081919080808, 0x2B08082B08190819, 0x2B08190808080819,
    0x2B08190808081908, 0x2B08190808190808, 0x2B08190808191919, 0x2B08190819080808,
    0x2B081908192B0808, 0x2B08191908080808, 0x2B0819191908192B, 0x2B0819192B191908,
    0x2B08192B08082B19, 0x2B08192B19080808, 0x2B08192B192B0808, 0x2B082B080808082B,
    0x2B082B1908081908, 0x2B082B2B08190819, 0x2B19080808081908, 0x2B19080808190808,
    0x2B190808082B1908, 0x2B19080819080808, 0x2B1908082B2B0819, 0x2B1908190819192B,
    0x2B1908192B080808, 0x2B19082B19081919, 0x2B19190808080808, 0x2B191908082B082B,
    0x2B19190819081908, 0x2B19191919190819, 0x2B192B082B080819, 0x2B192B19082B0808,
    0x2B2B08080808082B, 0x2B2B080819190808, 0x2B2B08082B081919, 0x2B2B081908082B19,
    0x2B2B082B08080808, 0x2B2B190808192B08, 0x2B2B2B0819190808, 0x2B2B2B1908081908,
)
# fmt: on

_CODEC_ID = {"q8_0": 0, "iq2_xxs": 1}


def repack_blocks(raw: torch.Tensor, codec: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Split ggml blocks into an aligned payload and fp16 block scales.

    Args:
        raw: uint8 [..., NB, 66] (IQ2_XXS) or [..., NB, 34] (Q8_0).
        codec: "iq2_xxs" or "q8_0".

    Returns:
        qweight: int32 [..., NB * 16] (IQ2_XXS) or int8 [..., NB * 32] (Q8_0).
        scales: fp16 [..., NB].

    """
    scales = raw[..., :2].contiguous().view(torch.float16).squeeze(-1)
    payload = raw[..., 2:].contiguous().flatten(-2)
    return payload.view(torch.int32 if codec == "iq2_xxs" else torch.int8), scales


@functools.cache
def iq2_lut(device: torch.device) -> torch.Tensor:
    """int32 [256 * 8 + 256]: the grid's fp32 magnitudes (8, 25 or 43) as bits,
    then each entry's 8 magnitudes as 2-bit codes. Build at load time.
    """
    grid = torch.tensor(_IQ2XXS_GRID, dtype=torch.int64, device=device)
    shifts = torch.arange(8, device=device)
    mags = (grid[:, None] >> (8 * shifts)) & 0xFF
    codes = ((mags > 8).long() + (mags > 25).long()) << (2 * shifts)
    return torch.cat([mags.float().view(torch.int32).flatten(), codes.sum(-1).int()])


@triton.jit
def _decode_tile(
    qw_ptr,
    sc_ptr,
    lut_ptr,
    offs_n,
    n_mask,
    k0,
    stride_qn,
    stride_sn,
    CODEC: tl.constexpr,
    GATHER: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """Weights [offs_n, k0:k0+BK] as an fp32 [BN, BK] tile; BK % 32 == 0.

    GATHER loads IQ2 magnitudes from the fp32 table, which vectorizes in the
    blocked layouts of the GEMV and dequant kernels. In `tl.dot` operand layouts
    it doesn't, and unpacking 2-bit codes is faster.
    """
    R: tl.constexpr = BK // 32
    rr = k0 // 32 + tl.arange(0, R)
    row = offs_n.to(tl.int64)[:, None]
    if CODEC == 1:
        # Per record: scale. Per group of 8: grid index and sign mask.
        d = tl.load(
            sc_ptr + row * stride_sn + (rr // 8)[None, :],
            mask=n_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        idx = tl.load(
            qw_ptr + row * stride_qn + (2 * rr)[None, :], mask=n_mask[:, None], other=0
        )
        aux = tl.load(
            qw_ptr + row * stride_qn + (2 * rr + 1)[None, :],
            mask=n_mask[:, None],
            other=0,
        )
        rs = d * (0.5 + ((aux >> 28) & 15).to(tl.float32)) * 0.25
        g = tl.arange(0, 4)[None, None, :]
        gidx = (idx[:, :, None] >> (8 * g)) & 255
        s = (aux[:, :, None] >> (7 * g)) & 127
        p = s ^ (s >> 4)
        p = p ^ (p >> 2)
        p = p ^ (p >> 1)
        smask = s | ((p & 1) << 7)
        j = tl.arange(0, 8)[None, None, None, :]
        if GATHER:
            mag = tl.load(lut_ptr + gidx[:, :, :, None] * 8 + j)
            mag = mag.to(tl.float32, bitcast=True)
        else:
            codes = tl.load(lut_ptr + 2048 + gidx)
            c = (codes[:, :, :, None] >> (2 * j)) & 3
            mag = tl.where(c == 0, 8.0, tl.where(c == 1, 25.0, 43.0))
        neg = (smask[:, :, :, None] >> j) & 1
        w = tl.where(neg != 0, -mag, mag) * rs[:, :, None, None]
        w = tl.reshape(w, (BN, BK))
    else:
        d = tl.load(
            sc_ptr + row * stride_sn + rr[None, :], mask=n_mask[:, None], other=0.0
        ).to(tl.float32)
        q = tl.load(
            qw_ptr + row * stride_qn + (k0 + tl.arange(0, BK))[None, :],
            mask=n_mask[:, None],
            other=0,
        ).to(tl.float32)
        w = tl.reshape(tl.reshape(q, (BN, R, 32)) * d[:, :, None], (BN, BK))
    return w


@triton.jit(do_not_specialize=["M"])
def _gemm_kernel(
    x_ptr,
    qw_ptr,
    sc_ptr,
    lut_ptr,
    out_ptr,
    M,
    N,
    k_per_split,
    stride_xm,
    stride_qn,
    stride_sn,
    CODEC: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """out[pid_k] = x[:, split] @ W[:, split]^T; out is [SPLIT, M, N]."""
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_n = tl.cdiv(N, BN)
    offs_m = (pid // num_n) * BM + tl.arange(0, BM)
    offs_n = (pid % num_n) * BN + tl.arange(0, BN)
    m_mask = offs_m < M
    n_mask = offs_n < N
    x_rows = x_ptr + offs_m.to(tl.int64)[:, None] * stride_xm
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(pid_k * k_per_split, (pid_k + 1) * k_per_split, BK):
        x = tl.load(
            x_rows + (k0 + tl.arange(0, BK))[None, :], mask=m_mask[:, None], other=0.0
        )
        w = _decode_tile(
            qw_ptr, sc_ptr, lut_ptr, offs_n, n_mask, k0, stride_qn, stride_sn,
            CODEC, False, BN, BK,
        )  # fmt: skip
        acc += tl.dot(x, tl.trans(w.to(x.dtype)))
    out = (pid_k * M + offs_m.to(tl.int64))[:, None] * N + offs_n[None, :]
    tl.store(
        out_ptr + out,
        acc.to(out_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


@triton.jit(do_not_specialize=["M"])
def _gemv_kernel(
    x_ptr,
    qw_ptr,
    sc_ptr,
    lut_ptr,
    out_ptr,
    M,
    N,
    k_per_split,
    stride_xm,
    stride_qn,
    stride_sn,
    CODEC: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """Small M without tensor cores, decoding each tile once for all BM rows:
    out[pid_k] = x[:, split] @ W[:, split]^T; out is [SPLIT, M, N]."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    m_mask = offs_m < M
    n_mask = offs_n < N
    x_rows = x_ptr + offs_m.to(tl.int64)[:, None] * stride_xm
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(pid_k * k_per_split, (pid_k + 1) * k_per_split, BK):
        w = _decode_tile(
            qw_ptr, sc_ptr, lut_ptr, offs_n, n_mask, k0, stride_qn, stride_sn,
            CODEC, True, BN, BK,
        )  # fmt: skip
        if BM == 1:  # the 3-D form below is ~35% slower for one row
            x = tl.load(x_ptr + k0 + tl.arange(0, BK)).to(tl.float32)
            acc += tl.sum(w * x[None, :], axis=1)[None, :]
        else:
            x = tl.load(
                x_rows + (k0 + tl.arange(0, BK))[None, :],
                mask=m_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(w[None, :, :] * x[:, None, :], axis=2)
    out = (pid_k * M + offs_m.to(tl.int64))[:, None] * N + offs_n[None, :]
    tl.store(
        out_ptr + out,
        acc.to(out_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


@triton.jit
def _dequant_kernel(
    qw_ptr, sc_ptr, lut_ptr, out_ptr, N, K, stride_qn, stride_sn,
    CODEC: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):  # fmt: skip
    offs_n = tl.program_id(0) * BN + tl.arange(0, BN)
    n_mask = offs_n < N
    k0 = tl.program_id(1) * BK
    w = _decode_tile(
        qw_ptr, sc_ptr, lut_ptr, offs_n, n_mask, k0, stride_qn, stride_sn,
        CODEC, True, BN, BK,
    )  # fmt: skip
    out = offs_n.to(tl.int64)[:, None] * K + (k0 + tl.arange(0, BK))[None, :]
    tl.store(out_ptr + out, w.to(out_ptr.dtype.element_ty), mask=n_mask[:, None])


@triton.jit(do_not_specialize=["EM", "num_valid_tokens"])
def _moe_gemm_kernel(
    a_ptr,
    qw_ptr,
    sc_ptr,
    lut_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_qe,
    stride_qn,
    stride_se,
    stride_sn,
    stride_cm,
    top_k: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    CODEC: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """`fused_moe_kernel` with B decoded from packed blocks of expert e."""
    pid = tl.program_id(0)
    num_n = tl.cdiv(N, BN)
    pid_m = pid // num_n
    if pid_m * BM >= tl.load(num_tokens_post_padded_ptr):
        return
    offs_token = tl.load(sorted_token_ids_ptr + pid_m * BM + tl.arange(0, BM))
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    e = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    offs_n = (pid % num_n) * BN + tl.arange(0, BN)
    n_mask = offs_n < N
    a_rows = a_ptr + (offs_token // top_k)[:, None] * stride_am
    qw_ptr += e * stride_qe
    sc_ptr += e * stride_se
    if tl.constexpr(BM == 64 and BN == 32 and BK == 128):
        # Sparse expert tiles avoid multiplying 63 padding rows by decoded weights.
        first = tl.load(sorted_token_ids_ptr + pid_m * BM).to(tl.int64)
        second = tl.load(sorted_token_ids_ptr + pid_m * BM + 1)
        if second >= num_valid_tokens:
            row = a_ptr + (first // top_k) * stride_am
            single = tl.zeros((BN,), dtype=tl.float32)
            for k0 in range(0, K, BK):
                # Match the tensor-core path's weight rounding.
                single_w = (
                    _decode_tile(
                        qw_ptr,
                        sc_ptr,
                        lut_ptr,
                        offs_n,
                        n_mask,
                        k0,
                        stride_qn,
                        stride_sn,
                        CODEC,
                        True,
                        BN,
                        BK,
                    )
                    .to(a_ptr.dtype.element_ty)
                    .to(tl.float32)
                )
                single_x = tl.load(row + k0 + tl.arange(0, BK)).to(tl.float32)
                single += tl.sum(single_w * single_x[None, :], axis=1)
            if MUL_ROUTED_WEIGHT:
                single *= tl.load(topk_weights_ptr + first).to(tl.float32)
            tl.store(
                c_ptr + first * stride_cm + offs_n,
                single.to(c_ptr.dtype.element_ty),
                n_mask,
            )
            return
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(
            a_rows + (k0 + tl.arange(0, BK))[None, :],
            mask=token_mask[:, None],
            other=0.0,
        )
        w = _decode_tile(
            qw_ptr, sc_ptr, lut_ptr, offs_n, n_mask, k0, stride_qn, stride_sn,
            CODEC, False, BN, BK,
        )  # fmt: skip
        acc += tl.dot(a, tl.trans(w.to(a.dtype)))
    if MUL_ROUTED_WEIGHT:
        weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        acc *= weight.to(tl.float32)[:, None]
    tl.store(
        c_ptr + offs_token[:, None] * stride_cm + offs_n[None, :],
        acc.to(c_ptr.dtype.element_ty),
        mask=token_mask[:, None] & n_mask[None, :],
    )


@triton.jit(do_not_specialize=["num_valid_tokens"])
def _moe_gemv_kernel(
    a_ptr,
    qw_ptr,
    sc_ptr,
    lut_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    N,
    K,
    num_valid_tokens,
    stride_am,
    stride_qe,
    stride_qn,
    stride_se,
    stride_sn,
    stride_cm,
    top_k: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    CODEC: tl.constexpr,
    DIRECT_ROUTING: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    APPLY_RELU2: tl.constexpr = False,
):
    """One routed row per program: c[slot] = a[slot // top_k] @ W[e]^T.

    For decode, when there are fewer routed rows than experts: no tensor cores,
    so no rows are decoded that nobody asked for, and the LUT gather decode
    beats the 2-bit codes used by `_moe_gemm_kernel`'s `tl.dot` layout.
    """
    slot = tl.program_id(0)
    if DIRECT_ROUTING:
        offs_token = slot.to(tl.int64)
    else:
        offs_token = tl.load(sorted_token_ids_ptr + slot).to(tl.int64)
    valid = offs_token < num_valid_tokens
    # A padding slot reads row 0 and an expert that exists; only its store is off.
    row = tl.where(valid, offs_token, 0) // top_k
    e = tl.load(expert_ids_ptr + slot).to(tl.int64)
    offs_n = tl.program_id(1) * BN + tl.arange(0, BN)
    n_mask = offs_n < N
    a_row = a_ptr + row * stride_am
    qw_ptr += e * stride_qe
    sc_ptr += e * stride_se
    acc = tl.zeros((BN,), dtype=tl.float32)
    for k0 in range(0, K, BK):
        w = _decode_tile(
            qw_ptr, sc_ptr, lut_ptr, offs_n, n_mask, k0, stride_qn, stride_sn,
            CODEC, True, BN, BK,
        )  # fmt: skip
        x = tl.load(a_row + k0 + tl.arange(0, BK)).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    if MUL_ROUTED_WEIGHT:
        weight = tl.load(topk_weights_ptr + offs_token, mask=valid, other=0.0)
        acc *= weight.to(tl.float32)
    if APPLY_RELU2:
        # Preserve the intermediate stored by the unfused first GEMM.
        value = tl.maximum(acc.to(c_ptr.dtype.element_ty).to(tl.float32), 0.0)
        acc = value * value
    tl.store(
        c_ptr + offs_token * stride_cm + offs_n,
        acc.to(c_ptr.dtype.element_ty),
        mask=valid & n_mask,
    )


@triton.jit(do_not_specialize=["total", "split"])
def _splitk_reduce_kernel(p_ptr, out_ptr, total, split, BN: tl.constexpr):
    """Sum the split-K partials: p [split, total] fp32 -> [total] in out's dtype."""
    offs = tl.program_id(0) * BN + tl.arange(0, BN)
    mask = offs < total
    acc = tl.load(p_ptr + offs, mask=mask, other=0.0)
    for s in range(1, split):
        acc += tl.load(p_ptr + s * total + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _q8_embedding_kernel(
    ids_ptr, qw_ptr, sc_ptr, out_ptr, H, stride_qn, stride_sn, stride_om,
    BH: tl.constexpr,
):  # fmt: skip
    t = tl.program_id(0)
    offs = tl.program_id(1) * BH + tl.arange(0, BH)
    mask = offs < H
    row = tl.load(ids_ptr + t).to(tl.int64)
    q = tl.load(qw_ptr + row * stride_qn + offs, mask=mask).to(tl.float32)
    d = tl.load(sc_ptr + row * stride_sn + offs // 32, mask=mask).to(tl.float32)
    tl.store(
        out_ptr + t.to(tl.int64) * stride_om + offs,
        (q * d).to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def _linear_config(
    M: int, N: int, K: int, codec: str, num_sms: int, smem: int, major: int
) -> dict[str, int]:
    if (
        major == 9
        and codec == "q8_0"
        and M == 1
        and (N, K) in ((256, 4096), (1024, 4096), (4096, 4096), (4096, 1024))
    ):
        # Hopper decode: wider K tiles avoid a separate split-K reduction.
        return {
            "BM": 1,
            "BN": 4,
            "BK": min(K, 1024) if N == 4096 else K,
            "SPLIT": 1,
            "num_warps": 4,
            "num_stages": 1,
        }
    if M == 1 or (codec == "iq2_xxs" and M <= 4):
        # GEMV; for IQ2 it beats tl.dot up to 4 rows.
        bm = triton.next_power_of_2(M)
        bn, bk, warps = (8, 256, 1) if codec == "iq2_xxs" else (32, 256, 4)
    elif M <= 16:
        bm, bn, bk, warps = 16, 64, 128, 4
    elif M <= 64:
        bm, bn, bk, warps = 32 if M <= 32 else 64, 64, 128, 4
    else:
        bm, bn, bk, warps = 128 if smem >= 160 * 1024 else 64, 128, 64, 8
    stages = 2
    if major < 8 and bm >= 16:
        # Turing has no tensor-core tl.dot; these FMA tiles don't spill on SM75.
        bm, bn, bk, warps, stages = min(bm, 64), 64, 32, 8, 1
    bk = next(b for b in (256, 128, 64, 32) if b <= bk and K % b == 0)
    # Deterministic split-K (fp32 partials) until the grid covers the GPU a few
    # times. GEMV tiles are decode-bound, so give them many more CTAs; IQ2 spends
    # the extra parallelism on unpacking the grid.
    tiles = triton.cdiv(M, bm) * triton.cdiv(N, bn)
    target = (8 if codec == "q8_0" else 32) * num_sms if M <= 4 else 2 * num_sms
    k_steps = K // bk
    split = 1
    while split < 16 and tiles * split < target and k_steps % (2 * split) == 0:
        split *= 2
    if split == 1:
        # No power of two divides the K step count, so the grid stays narrow and
        # stalls on decode latency: shared_down has 5376 / 256 = 21 steps. Take
        # any divisor instead, which is still a deterministic fp32 reduction.
        for d in range(min(16, k_steps), 1, -1):
            if k_steps % d == 0 and tiles * d < 2 * target:
                split = d
                break
    return {
        "BM": bm,
        "BN": bn,
        "BK": bk,
        "SPLIT": split,
        "num_warps": warps,
        "num_stages": stages,
    }


def launch_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    lut: torch.Tensor | None,
    codec: str,
    cfg: dict[str, int],
) -> torch.Tensor:
    """`modelopt_block_linear` with an explicit `_linear_config`-style config."""
    M, K = x.shape
    N = qweight.shape[0]
    split = cfg["SPLIT"]
    out = (
        x.new_empty((M, N))
        if split == 1
        else torch.empty((split, M, N), dtype=torch.float32, device=x.device)
    )
    launch = {"num_warps": cfg["num_warps"], "num_stages": cfg["num_stages"]}
    if cfg["BM"] < 16:  # tl.dot needs BM >= 16
        assert cfg["BM"] >= M
        _gemv_kernel[(triton.cdiv(N, cfg["BN"]), split)](
            x, qweight, scales, lut, out, M, N, K // split, x.stride(0),
            qweight.stride(0), scales.stride(0), CODEC=_CODEC_ID[codec],
            BM=cfg["BM"], BN=cfg["BN"], BK=cfg["BK"], **launch,
        )  # fmt: skip
    else:
        grid = (triton.cdiv(M, cfg["BM"]) * triton.cdiv(N, cfg["BN"]), split)
        _gemm_kernel[grid](
            x, qweight, scales, lut, out, M, N, K // split, x.stride(0),
            qweight.stride(0), scales.stride(0), CODEC=_CODEC_ID[codec],
            BM=cfg["BM"], BN=cfg["BN"], BK=cfg["BK"], **launch,
        )  # fmt: skip
    if split == 1:
        return out
    y = x.new_empty((M, N))
    total = M * N
    _splitk_reduce_kernel[(triton.cdiv(total, 1024),)](
        out, y, total, split, BN=1024, num_warps=4
    )  # fmt: skip
    return y


def dequantize_blocks(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    lut: torch.Tensor | None,
    codec: str,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """[N, K] in `dtype` from 2-D `repack_blocks` output."""
    N, nb = scales.shape
    K = nb * (256 if codec == "iq2_xxs" else 32)
    out = torch.empty((N, K), dtype=dtype, device=qweight.device)
    bk = next(b for b in (256, 128, 64, 32) if K % b == 0)
    bn, warps = (8, 1) if codec == "iq2_xxs" else (32, 4)
    _dequant_kernel[(triton.cdiv(N, bn), K // bk)](
        qweight, scales, lut, out, N, K, qweight.stride(0), scales.stride(0),
        CODEC=_CODEC_ID[codec], BN=bn, BK=bk, num_warps=warps,
    )  # fmt: skip
    return out


# From this many tokens, dequantize once and use cuBLAS, as AWQ does: the fused
# GEMM decodes W once per M tile. Capping the copy at 128 MiB keeps lm_head on
# the fused GEMM. The copy is a write and a read of 2 bytes per weight, so the
# crossover tracks memory bandwidth. Values are (HBM, GDDR). On GDDR the fused
# IQ2_XXS GEMM doubles in cost past one 64-row M tile, hence 65.
DEQUANT_MIN_M = {"iq2_xxs": (8, 65), "q8_0": (64, 256)}
DEQUANT_MAX_NUMEL = 1 << 26


@functools.cache
def _is_hbm(device: int) -> bool:
    # Bus width as the bandwidth class: HBM is at least 1024 bits wide.
    return torch.cuda.get_device_properties(device).memory_bus_width >= 1024


def dequant_min_m(codec: str, device: int) -> int:
    """Token count from which `modelopt_block_linear` dequantizes for cuBLAS."""
    if torch.cuda.get_device_properties(device).major < 8:
        # Turing's fused GEMM runs on FMA while cuBLAS keeps tensor cores.
        # 32 is a roofline estimate for T4.
        return 32
    return DEQUANT_MIN_M[codec][not _is_hbm(device)]


def _modelopt_block_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    lut: torch.Tensor | None,
    codec: str,
    weight_bf16: torch.Tensor | None = None,
) -> torch.Tensor:
    """Packed block GEMM; picks the kernel from M inside the op.

    Args:
        x: bf16 or fp16 [M, K].
        qweight: int32 [N, K / 16] (IQ2_XXS) or int8 [N, K] (Q8_0).
        scales: fp16 [N, K / block_size].
        lut: `iq2_lut` for IQ2_XXS, None for Q8_0.
        codec: "iq2_xxs" or "q8_0".
        weight_bf16: Optional cached dequantized weights for the BF16 GEMM path.

    Returns:
        [M, N] in x's dtype.

    """
    M, K = x.shape
    if M == 0:
        return x.new_empty((0, qweight.shape[0]))
    if (
        dequant_min_m(codec, x.device.index) <= M
        and qweight.shape[0] * K <= DEQUANT_MAX_NUMEL
    ):
        weight = (
            weight_bf16
            if weight_bf16 is not None
            else dequantize_blocks(qweight, scales, lut, codec, x.dtype)
        )
        return x @ weight.T
    if x.stride(1) != 1:
        x = x.contiguous()
    props = torch.cuda.get_device_properties(x.device)
    cfg = _linear_config(
        M,
        qweight.shape[0],
        K,
        codec,
        props.multi_processor_count,
        props.shared_memory_per_block_optin,
        props.major,
    )
    return launch_linear(x, qweight, scales, lut, codec, cfg)


def _modelopt_block_linear_fake(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    lut: torch.Tensor | None,
    codec: str,
    weight_bf16: torch.Tensor | None = None,
) -> torch.Tensor:
    return x.new_empty((x.shape[0], qweight.shape[0]))


def _modelopt_block_embedding(
    ids: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Gather and decode Q8_0 rows: ids [T] -> [T, H] in `dtype`."""
    out = torch.empty(
        (ids.numel(), qweight.shape[1]), dtype=dtype, device=qweight.device
    )
    if ids.numel():
        _q8_embedding_kernel[(ids.numel(), triton.cdiv(out.shape[1], 1024))](
            ids, qweight, scales, out, out.shape[1], qweight.stride(0),
            scales.stride(0), out.stride(0), BH=1024,
        )  # fmt: skip
    return out


def _modelopt_block_embedding_fake(
    ids: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    return qweight.new_empty((ids.numel(), qweight.shape[1]), dtype=dtype)


direct_register_custom_op(
    "modelopt_block_linear",
    _modelopt_block_linear,
    fake_impl=_modelopt_block_linear_fake,
)
direct_register_custom_op(
    "modelopt_block_embedding",
    _modelopt_block_embedding,
    fake_impl=_modelopt_block_embedding_fake,
)


def moe_gemv_config(M: int, E: int, top_k: int, codec: str) -> dict[str, int] | None:
    """Tiles for the decode path, or None when `moe_config`'s tiles win.

    One program per routed row: no `moe_align_block_size` padding and no
    tensor-core transposes, which suits a small decode step. In serving, on
    both HBM and GDDR GPUs, `moe_config`'s padded `tl.dot` tiles win above 8
    rows, even though a standalone kernel sweep keeps the GEMV ahead longer.
    """
    if M > 8:
        return None
    bn, bk, warps = (8, 256, 1) if codec == "iq2_xxs" else (32, 256, 4)
    return {"BN": bn, "BK": bk, "num_warps": warps, "num_stages": 2}


def invoke_block_moe_gemv_kernel(
    a: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    lut: torch.Tensor | None,
    c: torch.Tensor,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor | None,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor | None,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, int],
    codec: str,
    apply_relu2: bool = False,
) -> None:
    """`invoke_block_moe_kernel` for one routed row per program.

    Same arguments, except that `sorted_token_ids` / `expert_ids` come from
    `moe_align_block_size` with `block_size=1`: one entry per routed row and no
    padding, so `num_tokens_post_padded` is not needed (the grid is one program
    per row and padding rows store nothing).
    """
    N = qweight.size(1)
    slots = expert_ids.numel() if sorted_token_ids is None else expert_ids.size(0)
    _moe_gemv_kernel[(slots, triton.cdiv(N, config["BN"]))](
        a, qweight, scales, lut, c, topk_weights, sorted_token_ids, expert_ids,
        N, a.size(1), a.size(0) * top_k, a.stride(0), qweight.stride(0),
        qweight.stride(1), scales.stride(0), scales.stride(1), c.stride(-2),
        top_k=top_k, MUL_ROUTED_WEIGHT=mul_routed_weight,
        DIRECT_ROUTING=sorted_token_ids is None,
        APPLY_RELU2=apply_relu2,
        CODEC=_CODEC_ID[codec], BN=config["BN"], BK=config["BK"],
        num_warps=config["num_warps"], num_stages=config["num_stages"],
    )  # fmt: skip


def moe_config(M: int, E: int, top_k: int) -> dict[str, int]:
    """Tile sizes for `invoke_block_moe_kernel`, from tokens per expert."""
    per_expert = M * top_k / E
    if per_expert < 1:
        # Decode: widen N and deepen K instead of pipelining.
        bm, bn, bk = 64, 32, 128
    elif per_expert < 64:
        bm, bn, bk = 64, 128, 64
    else:
        # One 128-row tile per expert avoids decoding its weights twice.
        bm, bn, bk = 128, 128, 64
    # BN * BK <= 8192 keeps every tile under 60 KB of shared memory on SM80-SM120.
    # H200 profiling favors one stage and four warps, including BM128.
    warps = 4
    if not current_platform.has_device_capability(80):
        # Turing has no tensor-core tl.dot; these FMA tiles don't spill on SM75.
        bm, bn, bk, warps = min(bm, 64), 64, 32, 8
    return {
        "BLOCK_SIZE_M": bm,
        "BN": bn,
        "BK": bk,
        "num_warps": warps,
        "num_stages": 1,
    }


def invoke_block_moe_kernel(
    a: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    lut: torch.Tensor | None,
    c: torch.Tensor,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, int],
    codec: str,
) -> None:
    """One grouped GEMM of the fused MoE: c[slot] = a[slot // top_k] @ W[e]^T.

    Args:
        a: bf16 or fp16 [M, K] (GEMM1, top_k > 1) or [M * top_k, K]
            (GEMM2, top_k == 1).
        qweight: packed [E, N, ...] from `repack_blocks`.
        scales: fp16 [E, N, K / block_size].
        lut: `iq2_lut` for IQ2_XXS, None for Q8_0.
        c: [..., N] in a's dtype, written at row `slot` of its flattened
            leading dims.
        topk_weights: [M, top_k] router weights.
        sorted_token_ids: from `moe_align_block_size`.
        expert_ids: from `moe_align_block_size`.
        num_tokens_post_padded: from `moe_align_block_size`.
        mul_routed_weight: scale each row by its router weight.
        top_k: rows of `a` are repeated top_k times.
        config: from `moe_config`, the one passed to `moe_align_block_size`.
        codec: "iq2_xxs" or "q8_0".

    """
    bm = config["BLOCK_SIZE_M"]
    em = sorted_token_ids.size(0)
    if a.size(0) < bm:
        em = min(em, a.size(0) * top_k * bm)
    N = qweight.size(1)
    _moe_gemm_kernel[(triton.cdiv(em, bm) * triton.cdiv(N, config["BN"]),)](
        a, qweight, scales, lut, c, topk_weights, sorted_token_ids, expert_ids,
        num_tokens_post_padded, N, a.size(1), em, a.size(0) * top_k, a.stride(0),
        qweight.stride(0), qweight.stride(1), scales.stride(0), scales.stride(1),
        c.stride(-2), top_k=top_k, MUL_ROUTED_WEIGHT=mul_routed_weight,
        CODEC=_CODEC_ID[codec], BM=bm, BN=config["BN"], BK=config["BK"],
        num_warps=config["num_warps"], num_stages=config["num_stages"],
    )  # fmt: skip
