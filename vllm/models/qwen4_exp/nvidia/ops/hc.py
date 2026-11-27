# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA HyperConnection kernels for Qwen4Exp."""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@triton.jit
def _grouped_gemma_rmsnorm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    stride_x,
    stride_y,
    DIM: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    W_SHARED: tl.constexpr,
    EPS: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    GROUP_DIM: tl.constexpr = DIM // NUM_GROUPS
    BLOCK_SIZE: tl.constexpr = triton.next_power_of_2(GROUP_DIM)

    pid = tl.program_id(0)
    group_id = pid % NUM_GROUPS
    row = (pid // NUM_GROUPS).to(tl.int64)

    offs_g = tl.arange(0, BLOCK_SIZE)
    offsets = group_id * GROUP_DIM + offs_g
    mask = offs_g < GROUP_DIM
    # A [GROUP_DIM] affine is shared; a [DIM] affine follows the grouped
    # checkpoint layout.
    w_offs = offs_g if W_SHARED else offsets

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    x = tl.load(x_ptr + row * stride_x + offsets, mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + w_offs, mask, other=0.0)

    rrms = tl.rsqrt(tl.sum(x * x) / GROUP_DIM + EPS)
    # Gemma's (1 + w) affine is written this way to lower to an FMA.
    y = x * rrms
    y += y * w.to(tl.float32)

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(y_ptr + row * stride_y + offsets, y, mask)


def grouped_gemma_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, num_groups: int
) -> torch.Tensor:
    N, DIM = x.shape
    assert x.stride(1) == 1, "grouped Gemma RMSNorm requires unit inner stride"
    assert weight.is_contiguous(), "grouped Gemma RMSNorm weight must be contiguous"
    assert DIM % num_groups == 0
    group_dim = DIM // num_groups
    assert weight.numel() in (group_dim, DIM)

    y = x.new_empty(x.shape)
    _grouped_gemma_rmsnorm_kernel[(N * num_groups,)](
        x,
        weight,
        y,
        x.stride(0),
        y.stride(0),
        DIM,
        num_groups,
        W_SHARED=weight.numel() == group_dim,
        EPS=eps,
        launch_pdl=current_platform.is_arch_support_pdl(),
    )
    return y


@triton.jit
def _hc_silu_kernel(
    x_ptr,
    y_ptr,
    stride_x,
    stride_y,
    DIM: tl.constexpr,
    HC: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    BLOCK_SIZE: tl.constexpr = triton.next_power_of_2(DIM)

    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < DIM

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    x = tl.load(x_ptr + row * stride_x + offs, mask).to(tl.float32) / HC
    y = x * tl.sigmoid(x)

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(y_ptr + row * stride_y + offs, y, mask)


def hc_silu(x: torch.Tensor, hc_count: int) -> torch.Tensor:
    num_tokens, DIM = x.shape
    assert x.stride(1) == 1

    output = x.new_empty(x.shape)
    _hc_silu_kernel[(num_tokens,)](
        x,
        output,
        x.stride(0),
        output.stride(0),
        DIM=DIM,
        HC=hc_count,
        launch_pdl=current_platform.is_arch_support_pdl(),
    )
    return output


@triton.jit
def _hc_gate_mix_kernel(
    x_ptr,
    g_ptr,
    y_ptr,
    stride_x,
    stride_g,
    stride_y,
    DIM: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    HC_DIM: tl.constexpr = DIM // HC

    row = tl.program_id(0).to(tl.int64)
    tile_id = tl.program_id(1)
    offs_inner = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs_inner < HC_DIM

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    # The constexpr loop is unrolled and keeps one stream live at a time.
    # Materializing [HC, BLOCK_SIZE] more than doubles latency at large M.
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for stream in tl.static_range(HC):
        offsets = stream * HC_DIM + offs_inner
        g = tl.load(g_ptr + row * stride_g + offsets, mask, other=0.0)
        x = tl.load(x_ptr + row * stride_x + offsets, mask, other=0.0)
        acc += tl.sigmoid(g.to(tl.float32)) * x.to(tl.float32)
    acc /= HC

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(y_ptr + row * stride_y + offs_inner, acc, mask)


def hc_gate_mix(x: torch.Tensor, gate: torch.Tensor, hc_count: int) -> torch.Tensor:
    N, DIM = gate.shape
    assert x.shape == gate.shape
    assert DIM % hc_count == 0
    assert x.stride(1) == 1
    assert gate.stride(1) == 1

    HC_DIM = DIM // hc_count
    out = x.new_empty(N, HC_DIM)
    BLOCK_SIZE = 512
    _hc_gate_mix_kernel[(N, triton.cdiv(HC_DIM, BLOCK_SIZE))](
        x,
        gate,
        out,
        x.stride(0),
        gate.stride(0),
        out.stride(0),
        DIM,
        hc_count,
        BLOCK_SIZE,
        launch_pdl=current_platform.is_arch_support_pdl(),
    )
    return out


# On B300 the fused kernel beats the GEMM + hc_gate_mix chain from M ~ 32.
# Decode-sized batches keep the unfused path until they are benchmarked
# end to end with the fused kernel.
MIN_FUSED_UP_M = 512


@triton.jit(do_not_specialize=["num_tokens"])
def _hc_up_gate_mix_sm90_kernel(
    lowrank,
    weight,
    xn,
    output,
    num_tokens,
    stride_lowrank_m: tl.constexpr,
    stride_lowrank_k: tl.constexpr,
    stride_weight_n: tl.constexpr,
    stride_weight_k: tl.constexpr,
    stride_xn_m: tl.constexpr,
    stride_xn_n: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Each output tile needs matching features from all four HC streams.
    HIDDEN: tl.constexpr = 2560
    RANK: tl.constexpr = 320
    HC: tl.constexpr = 4
    GROUP_M: tl.constexpr = 8
    tile = tl.program_id(0)
    num_m = tl.cdiv(num_tokens, BLOCK_M)
    num_h: tl.constexpr = HIDDEN // (BLOCK_N // HC)
    first_m = tile // (GROUP_M * num_h) * GROUP_M
    group_m = tl.minimum(num_m - first_m, GROUP_M)
    in_group = tile % (GROUP_M * num_h)
    tile_m = first_m + in_group % group_m
    tile_h = in_group // group_m

    rows = (tile_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    cols = tl.arange(0, BLOCK_N)
    features = tile_h * (BLOCK_N // HC) + cols % (BLOCK_N // HC)
    gate_cols = cols // (BLOCK_N // HC) * HIDDEN + features
    k = tl.arange(0, BLOCK_K)
    acc = tl.full((BLOCK_M, BLOCK_N), 0, tl.float32)
    for block in range(RANK // BLOCK_K):
        offset = block * BLOCK_K + k
        a = tl.load(
            lowrank
            + rows[:, None] * stride_lowrank_m
            + offset[None, :] * stride_lowrank_k,
            rows[:, None] < num_tokens,
            other=0,
        )
        w = tl.load(
            weight
            + gate_cols[:, None] * stride_weight_n
            + offset[None, :] * stride_weight_k,
        )
        acc = tl.dot(a, tl.trans(w), acc)

    # Match F.linear's bf16 output and the original stream accumulation order.
    gate = tl.sigmoid(acc.to(tl.bfloat16).to(tl.float32))
    x = tl.load(
        xn + rows[:, None] * stride_xn_m + gate_cols[None, :] * stride_xn_n,
        rows[:, None] < num_tokens,
        other=0,
    ).to(tl.float32)
    gate = tl.permute(tl.reshape(gate, (BLOCK_M, HC, BLOCK_N // HC)), (0, 2, 1))
    x = tl.permute(tl.reshape(x, (BLOCK_M, HC, BLOCK_N // HC)), (0, 2, 1))
    g_even, g_odd = tl.split(tl.reshape(gate, (BLOCK_M, BLOCK_N // HC, 2, 2)))
    x_even, x_odd = tl.split(tl.reshape(x, (BLOCK_M, BLOCK_N // HC, 2, 2)))
    g0, g2 = tl.split(g_even)
    g1, g3 = tl.split(g_odd)
    x0, x2 = tl.split(x_even)
    x1, x3 = tl.split(x_odd)
    mixed = tl.fma(g3, x3, tl.fma(g2, x2, tl.fma(g1, x1, g0 * x0))) / HC
    out_cols = tile_h * (BLOCK_N // HC) + tl.arange(0, BLOCK_N // HC)
    tl.store(
        output + rows[:, None] * HIDDEN + out_cols[None, :],
        mixed,
        rows[:, None] < num_tokens,
    )


def hc_up_gate_mix_supported(w_up: torch.Tensor, hc_count: int) -> bool:
    return (
        (
            current_platform.is_device_capability(90)
            or (
                current_platform.is_device_capability_family(100)
                and hasattr(torch.ops._C, "qwen4_exp_hc_up_gate_mix")
            )
        )
        and hc_count == 4
        and w_up.dtype == torch.bfloat16
        and tuple(w_up.shape) == (4 * 2560, 320)
    )


def hc_up_gate_mix(
    lowrank: torch.Tensor, xn: torch.Tensor, w_up: torch.Tensor
) -> torch.Tensor:
    """Fuse the BF16 up projection and gate mix for four streams, H=2560, K=320.

    SM90 rounds the gate to BF16 before sigmoid. SM100 keeps it in FP32.
    """
    if current_platform.is_device_capability(90):
        assert lowrank.shape == (xn.shape[0], 320)
        assert xn.ndim == 2 and xn.shape[1] == 10240
        assert w_up.shape == (10240, 320)
        assert lowrank.dtype == xn.dtype == w_up.dtype == torch.bfloat16
        assert lowrank.device == xn.device == w_up.device
        out = xn.new_empty(xn.shape[0], 2560)
        block_m, block_n, block_k = 64, 128, 32
        grid = (triton.cdiv(xn.shape[0], block_m) * (xn.shape[1] // block_n),)
        _hc_up_gate_mix_sm90_kernel[grid](
            lowrank,
            w_up,
            xn,
            out,
            xn.shape[0],
            *lowrank.stride(),
            *w_up.stride(),
            *xn.stride(),
            block_m,
            block_n,
            block_k,
            num_warps=4,
            num_stages=4,
        )
        return out
    out = xn.new_empty(xn.shape[0], xn.shape[1] // 4)
    torch.ops._C.qwen4_exp_hc_up_gate_mix(out, lowrank, xn, w_up)
    return out


@triton.jit
def _hc_combine_kernel(
    block_ptr,
    res_ptr,
    inj_ptr,
    out_ptr,
    stride_block,
    stride_res,
    stride_inj,
    stride_out,
    HC_DIM: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    HC_PAD: tl.constexpr = triton.next_power_of_2(HC)

    row = tl.program_id(0).to(tl.int64)
    tile_id = tl.program_id(1)

    offs_inner = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_inner = offs_inner < HC_DIM
    offs_hc = tl.arange(0, HC_PAD)
    mask_hc = offs_hc < HC
    offs = offs_hc[:, None] * HC_DIM + offs_inner[None, :]
    mask = mask_hc[:, None] & mask_inner[None, :]

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    if inj_ptr is not None:
        inj = tl.load(inj_ptr + row * stride_inj + offs_hc, mask_hc, other=0.0)
    block = tl.load(block_ptr + row * stride_block + offs_inner, mask_inner, other=0.0)
    res = tl.load(res_ptr + row * stride_res + offs, mask, other=0.0)

    # Keeping HC as a broadcast dimension is faster here than four separate
    # residual load/store sequences.
    if inj_ptr is not None:
        inj = 2.0 * tl.sigmoid(inj.to(tl.float32) / HC)
        block = block.to(tl.float32)[None, :] * inj[:, None]
    out = res.to(tl.float32) + block.to(tl.float32)

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(out_ptr + row * stride_out + offs, out, mask=mask)


def hc_combine(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor | None,
    hc_count: int,
) -> torch.Tensor:
    N, DIM = residual.shape
    assert DIM % hc_count == 0
    hc_dim = DIM // hc_count
    assert block_output.shape == (N, hc_dim)
    assert residual.stride(1) == 1
    assert block_output.stride(1) == 1
    if injection_logits is not None:
        assert injection_logits.shape == (N, hc_count)
        assert injection_logits.stride(1) == 1

    stride_injection = injection_logits.stride(0) if injection_logits is not None else 0

    out = residual.new_empty(residual.shape)
    BLOCK_SIZE = 512
    _hc_combine_kernel[(N, triton.cdiv(hc_dim, BLOCK_SIZE))](
        block_output,
        residual,
        injection_logits,
        out,
        block_output.stride(0),
        residual.stride(0),
        stride_injection,
        out.stride(0),
        hc_dim,
        hc_count,
        BLOCK_SIZE,
        launch_pdl=current_platform.is_arch_support_pdl(),
    )
    return out


@triton.jit
def _hc_combine_norm_kernel(
    block_ptr,
    res_ptr,
    inj_ptr,
    w_ptr,
    out_ptr,
    y_ptr,
    stride_block,
    stride_res,
    stride_inj,
    stride_out,
    stride_y,
    HC_DIM: tl.constexpr,
    HC: tl.constexpr,
    W_SHARED: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    HC_PAD: tl.constexpr = triton.next_power_of_2(HC)
    NUM_TILES: tl.constexpr = triton.cdiv(HC_DIM, BLOCK_SIZE)
    NUM_TILES_PAD: tl.constexpr = triton.next_power_of_2(NUM_TILES)

    pid = tl.program_id(0)
    row = (pid // HC).to(tl.int64)
    stream = pid % HC
    offs_hc = tl.arange(0, HC_PAD)
    mask_hc = offs_hc < HC
    tile_ids = tl.arange(0, NUM_TILES_PAD)
    offs_inner = tile_ids[:, None] * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    mask_inner = offs_inner < HC_DIM
    offs = stream * HC_DIM + offs_inner
    # Shared norm weights repeat across streams; per-branch weights use the
    # same flattened HC layout as the residual.
    w_offs = offs_inner if W_SHARED else offs

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    # Start the uncached residual load first, then issue the other combine
    # loads before consuming any of them.
    res = tl.load(res_ptr + row * stride_res + offs, mask_inner, other=0.0)
    if inj_ptr is not None:
        inj = tl.load(inj_ptr + row * stride_inj + offs_hc, mask_hc, other=0.0)
    block = tl.load(
        block_ptr + row * stride_block + offs_inner,
        mask_inner,
        other=0.0,
    )
    if inj_ptr is not None:
        inj = 2.0 * tl.sigmoid(inj.to(tl.float32) / HC)
        block = block.to(tl.float32) * tl.sum(tl.where(offs_hc == stream, inj, 0.0))
    # Round the materialized combine result before normalization. This matches
    # the unfused combine -> RMSNorm boundary.
    out = (res.to(tl.float32) + block.to(tl.float32)).to(out_ptr.dtype.element_ty)
    if inj_ptr is None:
        w = tl.load(w_ptr + w_offs, mask_inner, other=0.0)
    tl.store(out_ptr + row * stride_out + offs, out, mask=mask_inner)

    out = out.to(tl.float32)
    sum_sq = tl.sum(tl.sum(out * out, axis=1), axis=0)
    rrms = tl.rsqrt(sum_sq / HC_DIM + EPS)

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()

    if inj_ptr is not None:
        # Loading the weight earlier helps decode but keeps the tile live across
        # the reduction and regresses larger batches, so defer it to the norm.
        w = tl.load(w_ptr + w_offs, mask_inner, other=0.0)
    y = out * rrms
    y += y * w.to(tl.float32)
    tl.store(y_ptr + row * stride_y + offs, y, mask_inner)


def hc_combine_norm(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor | None,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    N, DIM = residual.shape
    assert DIM % hc_count == 0
    hc_dim = DIM // hc_count
    assert block_output.shape == (N, hc_dim)
    assert residual.stride(1) == 1
    assert block_output.stride(1) == 1
    if injection_logits is not None:
        assert injection_logits.shape == (N, hc_count)
        assert injection_logits.stride(1) == 1
    assert norm_weight.is_contiguous()
    assert norm_weight.numel() in (hc_dim, DIM)

    stride_injection = injection_logits.stride(0) if injection_logits is not None else 0

    out = residual.new_empty(residual.shape)
    y = residual.new_empty(residual.shape)
    BLOCK_SIZE = 512
    _hc_combine_norm_kernel[(N * hc_count,)](
        block_output,
        residual,
        injection_logits,
        norm_weight,
        out,
        y,
        block_output.stride(0),
        residual.stride(0),
        stride_injection,
        out.stride(0),
        y.stride(0),
        hc_dim,
        hc_count,
        W_SHARED=norm_weight.numel() == hc_dim,
        EPS=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        launch_pdl=current_platform.is_arch_support_pdl(),
    )
    return out, y


__all__ = [
    "grouped_gemma_rmsnorm",
    "hc_combine",
    "hc_combine_norm",
    "hc_gate_mix",
    "hc_silu",
]
