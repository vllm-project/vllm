# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill conversion for FlashInfer shared quantized checkpoints."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _quantize_checkpoint(
    source,
    state,
    scales,
    destinations,
    sources,
    source_stride,
    state_stride,
    scale_stride,
    ROWS: tl.constexpr,
    N: tl.constexpr,
    QMAX: tl.constexpr,
    HAS_SOURCES: tl.constexpr,
    INT8: tl.constexpr,
):
    batch, row = tl.program_id(0), tl.program_id(1)
    dst = tl.load(destinations + batch).to(tl.int64)
    if dst <= 0:
        return
    src = tl.load(sources + batch).to(tl.int64) if HAS_SOURCES else batch
    cols = tl.arange(0, N)
    value = tl.load(source + src * source_stride + row * N + cols).to(tl.float32)
    amax = tl.max(tl.abs(value), axis=0)
    encode = tl.where(amax == 0, 1.0, QMAX / amax)
    quantized = tl.minimum(tl.maximum(value * encode, -QMAX), QMAX)
    if INT8:
        quantized = tl.inline_asm_elementwise(
            "cvt.rni.s32.f32 $0, $1;",
            constraints="=r,f",
            args=[quantized],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        ).to(tl.int8)
    else:
        quantized = quantized.to(tl.float8e4nv, fp_downcast_rounding="rtne")
    tl.store(state + dst * state_stride + row * N + cols, quantized)
    tl.store(scales + dst * scale_stride + row, 1.0 / encode)


def quantize_replayssm_state(
    states: torch.Tensor,
    state: torch.Tensor,
    state_scale: torch.Tensor,
    state_indices: torch.Tensor,
    source_indices: torch.Tensor | None = None,
) -> None:
    """Store prefill states with per-row decode scales and nearest rounding."""
    if state.dtype not in (torch.float8_e4m3fn, torch.int8):
        raise ValueError("Shared ReplaySSM requires FP8 or INT8 state")
    rows = state.shape[1] * state.shape[2]
    _quantize_checkpoint[(state_indices.numel(), rows)](
        states,
        state,
        state_scale,
        state_indices.contiguous(),
        source_indices,
        states.stride(0),
        state.stride(0),
        state_scale.stride(0),
        ROWS=rows,
        N=state.shape[3],
        QMAX=127.0 if state.dtype == torch.int8 else 448.0,
        HAS_SOURCES=source_indices is not None,
        INT8=state.dtype == torch.int8,
    )


@triton.jit
def _dequantize_checkpoint(
    state,
    scales,
    indices,
    has_initial,
    output,
    state_stride,
    scale_stride,
    ROWS: tl.constexpr,
    N: tl.constexpr,
):
    batch, row = tl.program_id(0), tl.program_id(1)
    src = tl.load(indices + batch).to(tl.int64)
    valid = (src >= 0) & tl.load(has_initial + batch)
    cols = tl.arange(0, N)
    value = tl.load(
        state + src * state_stride + row * N + cols, mask=valid, other=0.0
    ).to(tl.float32)
    scale = tl.load(scales + src * scale_stride + row, mask=valid, other=0.0)
    tl.store(output + batch * ROWS * N + row * N + cols, value * scale)


def dequantize_replayssm_state(
    state: torch.Tensor,
    state_scale: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial: torch.Tensor,
) -> torch.Tensor:
    """Gather existing checkpoints into the floating-point prefill input."""
    output = torch.empty(
        (state_indices.numel(), *state.shape[1:]),
        dtype=torch.float32,
        device=state.device,
    )
    rows = state.shape[1] * state.shape[2]
    _dequantize_checkpoint[(state_indices.numel(), rows)](
        state,
        state_scale,
        state_indices.contiguous(),
        has_initial,
        output,
        state.stride(0),
        state_scale.stride(0),
        ROWS=rows,
        N=state.shape[3],
    )
    return output
