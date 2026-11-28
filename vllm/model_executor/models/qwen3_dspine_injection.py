# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fused adjacent injection adapted from Tencent/DSpine's Triton kernels."""

import math

import torch
import torch.nn.functional as F

from vllm.triton_utils import tl, tldevice, triton


@triton.jit
def _prepare_read(
    States, Normalized, RMS, H: tl.constexpr, EPS: tl.constexpr, WIDTH: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.arange(0, WIDTH)
    value = tl.load(States + row * H + col, col < H, 0).to(tl.float32)
    rms = tl.sqrt(tl.maximum(tl.sum(value * value, 0) / H, 1.1754943508222875e-38))
    tl.store(Normalized + row * H + col, tl.div_rn(value, rms + EPS), col < H)
    tl.store(RMS + row, rms)


@triton.jit
def _normalize_codes(
    Projected,
    CodeNorm,
    AnchorFeatures,
    Codes,
    C: tl.constexpr,
    T: tl.constexpr,
    SCALE: tl.constexpr,
    EPS: tl.constexpr,
    WIDTH: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, WIDTH)
    if row % T == 0:
        features = tl.load(AnchorFeatures + row // T * C + col, col < C, 0)
    else:
        value = tl.load(Projected + row * C + col, col < C, 0).to(tl.float32)
        norm = tl.load(CodeNorm + row)
        features = tl.div_rn(SCALE * value, norm + EPS).to(Codes.dtype.element_ty)
    tl.store(Codes + row * C + col, features, col < C)


@triton.jit
def _feature_message(
    Packed, Bias, Message, R: tl.constexpr, T: tl.constexpr, WIDTH: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.arange(0, WIDTH)
    if row % T == 0:
        message = tl.full((WIDTH,), 0.0, tl.float32)
    else:
        query = tl.load(Packed + row * (3 * R) + col, col < R, 0).to(tl.float32)
        sender_offset = (row - 1) * (3 * R) + col
        key = tl.load(Packed + sender_offset + R, col < R, 0).to(tl.float32)
        value = tl.load(Packed + sender_offset + 2 * R, col < R, 0).to(tl.float32)
        bias = tl.load(Bias + col, col < R, 0).to(tl.float32)
        gate = tl.div_rn(1.0, 1.0 + tldevice.exp(-(query + key + bias)))
        message = gate * value
    tl.store(Message + row * R + col, message, col < R)


@triton.jit
def _feature_finish(
    States, Delta, RMS, Output, H: tl.constexpr, T: tl.constexpr, WIDTH: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.arange(0, WIDTH)
    offset = row * H + col
    state = tl.load(States + offset, col < H, 0).to(tl.float32)
    if row % T != 0:
        delta = tl.load(Delta + offset, col < H, 0).to(tl.float32)
        state = state + tl.load(RMS + row) * delta
    tl.store(Output + offset, state, col < H)


def fused_adjacent_injection(
    states,
    anchor_features,
    readout,
    packed_gate_message,
    gate_bias,
    write,
    block_size,
    eps,
):
    states = states.contiguous()
    rows, hidden = states.shape
    assert rows % block_size == 0
    assert anchor_features.shape[0] == rows // block_size
    normalized = torch.empty_like(states)
    rms = torch.empty(rows, device=states.device, dtype=torch.float32)
    _prepare_read[(rows,)](
        states,
        normalized,
        rms,
        hidden,
        eps,
        triton.next_power_of_2(hidden),
        enable_fp_fusion=False,
    )
    projected = readout(normalized)
    code_width = projected.shape[-1]
    codes = torch.empty_like(projected)
    # Preserve PyTorch's FP32 reduction order when rounding the code to BF16.
    code_norm = projected.float().norm(dim=-1)
    _normalize_codes[(rows,)](
        projected,
        code_norm,
        anchor_features.contiguous(),
        codes,
        code_width,
        block_size,
        math.sqrt(code_width),
        eps,
        triton.next_power_of_2(code_width),
        enable_fp_fusion=False,
    )
    packed = F.linear(codes, packed_gate_message)
    rank = gate_bias.numel()
    message = torch.empty((rows, rank), device=states.device, dtype=states.dtype)
    _feature_message[(rows,)](
        packed,
        gate_bias,
        message,
        rank,
        block_size,
        triton.next_power_of_2(rank),
        enable_fp_fusion=False,
    )
    delta = write(message)
    output = torch.empty_like(states)
    _feature_finish[(rows,)](
        states,
        delta,
        rms,
        output,
        hidden,
        block_size,
        triton.next_power_of_2(hidden),
        enable_fp_fusion=False,
    )
    return output
