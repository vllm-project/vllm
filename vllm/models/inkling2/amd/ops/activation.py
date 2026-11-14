# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Softcapped SwiGLU of the Inkling2 experts, fused with the down-projection
input quantization (MXFP8)."""

import torch

from vllm.triton_utils import tl, tldevice, triton

from .mxfp8 import _mxfp8_block, _store_mxfp8_block

FP32_MAX = tl.constexpr(3.4028234663852886e38)


@triton.jit
def _sigmoid_approx(x):
    return 1.0 / (1.0 + tl.math.exp2(x * -1.4426950408889634))


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
        x = CAP * tldevice.tanh(_div_rn(x, CAP))
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
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    TILE: tl.constexpr,
):
    values, exponent = _mxfp8_block(tl.reshape(y, (BLOCK // 32, 32)))
    _store_mxfp8_block(Values, Scales, row, start, values, exponent, N, BLOCK, TILE)


@triton.jit
def _row_inputs(Rows, Gammas, r, num_rows):
    """Row and routing weight of routed row `r`."""
    valid = r < num_rows
    row = tl.load(Rows + r, mask=valid, other=0).to(tl.int64)
    return row, tl.load(Gammas + row, mask=valid)


@triton.jit
def _situ_quantize_kernel(
    X,
    Gammas,
    Values,
    Scales,
    Rows,
    NumRows,
    N: tl.constexpr,
    GATE_CAP: tl.constexpr,
    UP_CAP: tl.constexpr,
    HEAD: tl.constexpr,
    TAIL: tl.constexpr,
    TILE: tl.constexpr,
):
    # Each program quantizes columns [start, start + HEAD + TAIL) of its rows.
    num_rows = tl.load(NumRows)
    r = tl.program_id(0)
    row, gamma = _row_inputs(Rows, Gammas, r, num_rows)
    start = tl.program_id(1) * (HEAD + TAIL)
    while r < num_rows:
        head = _situ(X, row, start, gamma, N, HEAD, GATE_CAP, UP_CAP)
        _quantize_part(head, Values, Scales, row, start, N, HEAD, TILE)
        if TAIL > 0:
            tail = _situ(X, row, start + HEAD, gamma, N, TAIL, GATE_CAP, UP_CAP)
            _quantize_part(tail, Values, Scales, row, start + HEAD, N, TAIL, TILE)
        r += tl.num_programs(0)
        row, gamma = _row_inputs(Rows, Gammas, r, num_rows)


def swiglu_quantize(
    x: torch.Tensor,
    gammas: torch.Tensor,
    gate_cap: float,
    up_cap: float,
    rows: tuple[torch.Tensor, torch.Tensor],
    tile: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP8-quantized bf16
    `softcap(gate) * sigmoid(gate) * softcap(up) * gamma`, computed in fp32.

    `x` is [M, 2N] with interleaved (gate, up) columns in the expert layout
    of `tile`-row GEMM tiles. `rows` holds the routed rows (the others are
    left uninitialized) and their number. Returns the values and block
    scales.
    """
    num_rows, n = x.shape[0], x.shape[1] // 2
    routed, num_routed = rows
    max_rows = routed.numel()
    if max_rows <= 512:
        # Few rows: a program per 256 columns.
        assert n % 256 == 0
        head, tail = 256, 0
    else:
        # Whole rows, as a power-of-two head and tail.
        head = 1 << (n.bit_length() - 1)
        tail = n - head
        assert tail & (tail - 1) == 0 and (tail == 0 or tail >= 32)
    values = torch.empty((num_rows, n), dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty((num_rows, n // 32), dtype=torch.uint8, device=x.device)
    _situ_quantize_kernel[(min(max_rows, 16384), n // (head + tail))](
        x,
        gammas,
        values,
        scales,
        routed,
        num_routed,
        N=n,
        GATE_CAP=gate_cap,
        UP_CAP=up_cap,
        HEAD=head,
        TAIL=tail,
        TILE=tile,
        num_warps=2,
        enable_fp_fusion=False,
    )
    return values, scales
