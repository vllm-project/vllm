# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Softcapped SwiGLU of the Inkling2 experts, fused with the down-projection
input quantization."""

import torch

from vllm.triton_utils import tl, tldevice, triton

from .mxfp8 import _mxfp8_block, _store_mxfp8_block
from .nvfp4 import FP32_MAX, _quantize_block, _store_block

# Up to this many rows, the kernel lets the next one launch (and wait) as soon
# as it starts, which hides its launch; with more, the waiting programs would
# hold back the kernel's own.
LAUNCH_DEPENDENTS_MAX_ROWS = 2048


@triton.jit
def _sigmoid_approx(x):
    return tl.inline_asm_elementwise(
        "{ .reg .f32 t; mul.rn.f32 t, $1, 0fBFB8AA3B; "
        "ex2.approx.ftz.f32 t, t; add.rn.f32 t, t, 0f3F800000; "
        "rcp.approx.ftz.f32 $0, t; }",
        "=f,f",
        [x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _div_rn(x, DIVISOR: tl.constexpr):
    """Bitwise `tl.div_rn(x, DIVISOR)` without the division routine: the
    product with the rounded reciprocal and one FMA correction step (whose
    residual is clamped for infinite `x`) are exact for these divisors on
    every float32 input."""
    tl.static_assert(DIVISOR == 4.0 or DIVISOR == 25.0)
    q = x * (1.0 / DIVISOR)
    if DIVISOR != 4.0:
        residual = tl.maximum(tl.fma(q, DIVISOR, -x), -FP32_MAX)
        q = tl.fma(-residual, 1.0 / DIVISOR, q)
    return q


@triton.jit
def _softcap(x, CAP: tl.constexpr):
    if CAP > 0:
        x = CAP * tl.inline_asm_elementwise(
            "tanh.approx.f32 $0, $1;",
            "=f,f",
            [_div_rn(x, CAP)],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    return x


@triton.jit
def _situ(X, row, start, gamma, N: tl.constexpr, BLOCK: tl.constexpr, GATE_CAP, UP_CAP):
    """fp32 values of the bf16 SiTU of columns `[start, start + BLOCK)`."""
    # Interleaved (gate, up) bf16 pairs as one word each, loaded so that each
    # thread holds whole 16-column blocks: column 16 * b + 4 * i + j at
    # [i, b, j].
    i = tl.arange(0, 4)[:, None, None]
    b = tl.arange(0, BLOCK // 16)[None, :, None]
    j = tl.arange(0, 4)[None, None, :]
    cols = start + 16 * b + 4 * i + j
    pairs = tl.load(X.to(tl.pointer_type(tl.uint32)) + row * N + cols)
    pairs = tl.reshape(tl.permute(pairs, (1, 0, 2)), (BLOCK,))
    gate = (pairs << 16).to(tl.float32, bitcast=True)
    up = (pairs & 0xFFFF0000).to(tl.float32, bitcast=True)
    sigmoid = _sigmoid_approx(gate)
    gate = _softcap(gate, GATE_CAP)
    up = _softcap(up, UP_CAP)
    return (((gate * sigmoid) * up) * gamma).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _quantize_part(
    y,
    Values,
    Scales,
    row,
    start,
    encoder_scale,
    decoder_scale,
    outer_scale,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    TILE: tl.constexpr,
    NVFP4: tl.constexpr,
):
    if NVFP4:
        packed, scales = _quantize_block(
            y, encoder_scale, decoder_scale, outer_scale, BLOCK
        )
        _store_block(Values, Scales, row, start, packed, scales, N, BLOCK, TILE)
    else:
        values, exponent = _mxfp8_block(tl.reshape(y, (BLOCK // 32, 32)))
        _store_mxfp8_block(Values, Scales, row, start, values, exponent, N, BLOCK, TILE)


@triton.jit
def _row_inputs(
    Rows, Gammas, CtaExpert, WAmax, r, num_rows, TILE: tl.constexpr, NVFP4: tl.constexpr
):
    """Row, routing weight and (NVFP4) weight amax of routed row `r`."""
    valid = r < num_rows
    row = tl.load(Rows + r, mask=valid, other=0).to(tl.int64)
    gamma = tl.load(Gammas + row, mask=valid)
    w_amax = 0.0
    if NVFP4:
        expert = tl.load(CtaExpert + row // TILE, mask=valid, other=0)
        w_amax = tl.load(WAmax + expert, mask=valid)
    return row, gamma, w_amax


@triton.jit
def _situ_quantize_kernel(
    X,
    Gammas,
    Values,
    Scales,
    RowScales,
    Rows,
    NumRows,
    CtaExpert,
    WAmax,
    WDenominator,
    N: tl.constexpr,
    GATE_CAP: tl.constexpr,
    UP_CAP: tl.constexpr,
    HEAD: tl.constexpr,
    TAIL: tl.constexpr,
    TILE: tl.constexpr,
    NVFP4: tl.constexpr,
    LAUNCH_DEPENDENTS: tl.constexpr,
):
    # Each program quantizes columns [start, start + HEAD + TAIL) of its rows
    # (NVFP4: whole rows, for the row amax). The routing outputs and weight
    # scales are read before waiting for the gate/up GEMM, which launches this
    # kernel only after its own wait for the routing kernel.
    if LAUNCH_DEPENDENTS:
        tl.extra.cuda.gdc_launch_dependents()
    num_rows = tl.load(NumRows)
    r = tl.program_id(0)
    row, gamma, w_amax = _row_inputs(
        Rows, Gammas, CtaExpert, WAmax, r, num_rows, TILE, NVFP4
    )
    denominator = 0.0
    if NVFP4:
        denominator = 1536.0 * tl.load(WDenominator)
    tl.extra.cuda.gdc_wait()
    start = tl.program_id(1) * (HEAD + TAIL)
    while r < num_rows:
        head = _situ(X, row, start, gamma, N, HEAD, GATE_CAP, UP_CAP)
        tail = head
        if TAIL > 0:
            tail = _situ(X, row, start + HEAD, gamma, N, TAIL, GATE_CAP, UP_CAP)
        encoder_scale = 0.0
        decoder_scale = 0.0
        outer_scale = 0.0
        if NVFP4:
            amax = tl.max(tl.abs(head), 0)
            if TAIL > 0:
                amax = tl.maximum(amax, tl.max(tl.abs(tail), 0))
            tl.store(RowScales + row, (amax * w_amax) * tldevice.rcp_rn(denominator))
            encoder_scale = tl.minimum(tl.div_rn(1536.0, amax), FP32_MAX)
            decoder_scale = tl.div_rn(1.0, encoder_scale)
            outer_scale = amax * (1.0 / 1536.0)
        _quantize_part(
            head,
            Values,
            Scales,
            row,
            start,
            encoder_scale,
            decoder_scale,
            outer_scale,
            N,
            HEAD,
            TILE,
            NVFP4,
        )
        if TAIL > 0:
            _quantize_part(
                tail,
                Values,
                Scales,
                row,
                start + HEAD,
                encoder_scale,
                decoder_scale,
                outer_scale,
                N,
                TAIL,
                TILE,
                NVFP4,
            )
        r += tl.num_programs(0)
        row, gamma, w_amax = _row_inputs(
            Rows, Gammas, CtaExpert, WAmax, r, num_rows, TILE, NVFP4
        )


def swiglu_quantize(
    x: torch.Tensor,
    gammas: torch.Tensor,
    gate_cap: float,
    up_cap: float,
    rows: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tile: int,
    weight_amax: torch.Tensor | None = None,
    weight_denominator: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Quantized (NVFP4 with `weight_amax`, else MXFP8) bf16
    `softcap(gate) * sigmoid(gate) * softcap(up) * gamma`, computed in fp32.

    `x` is [M, 2N] with interleaved (gate, up) columns in the expert layout
    of `tile`-row GEMM tiles. `rows` holds the routed rows (the others are
    left uninitialized), their number and the expert of each tile. Returns the
    values, block scales and, for NVFP4, each row's down-projection output
    scale.
    """
    num_rows, n = x.shape[0], x.shape[1] // 2
    routed, num_routed, cta_expert = rows
    nvfp4 = weight_amax is not None
    max_rows = routed.numel()
    num_warps, maxnreg = (4, None) if max_rows <= 16384 else (2, 72)
    if not nvfp4 and max_rows <= 512:
        # Few rows: a program per 256 columns.
        assert n % 256 == 0
        head, tail, num_warps = 256, 0, 2
    else:
        # Whole rows, as a power-of-two head and tail.
        head = 1 << (n.bit_length() - 1)
        tail = n - head
        assert tail & (tail - 1) == 0 and (tail == 0 or tail >= 32)
        if not nvfp4:
            num_warps, maxnreg = 2, None
    if nvfp4:
        values = torch.empty((num_rows, n // 2), dtype=torch.uint8, device=x.device)
        scales = torch.empty(
            (num_rows, n // 16), dtype=torch.float8_e4m3fn, device=x.device
        )
        row_scales = torch.empty(num_rows, dtype=torch.float32, device=x.device)
    else:
        values = torch.empty((num_rows, n), dtype=torch.float8_e4m3fn, device=x.device)
        scales = torch.empty((num_rows, n // 32), dtype=torch.uint8, device=x.device)
        row_scales = None
    _situ_quantize_kernel[(min(max_rows, 16384), n // (head + tail))](
        x,
        gammas,
        values,
        scales,
        row_scales,
        routed,
        num_routed,
        cta_expert,
        weight_amax,
        weight_denominator,
        N=n,
        GATE_CAP=gate_cap,
        UP_CAP=up_cap,
        HEAD=head,
        TAIL=tail,
        TILE=tile,
        NVFP4=nvfp4,
        LAUNCH_DEPENDENTS=max_rows <= LAUNCH_DEPENDENTS_MAX_ROWS,
        num_warps=num_warps,
        maxnreg=maxnreg,
        enable_fp_fusion=False,
        launch_pdl=True,
    )
    return values, scales, row_scales
