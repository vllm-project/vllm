# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Iterator
from itertools import product
from typing import TYPE_CHECKING, Any

import torch

from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    TritonWarmupTensor,
    triton_kernel_dispatcher_with_warmup,
)
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import direct_register_custom_op

if TYPE_CHECKING:
    from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
    from vllm.model_executor.layers.linear import LinearBase


def _relu_squared_nvfp4_quant_launch(
    m: int, k: int
) -> tuple[tuple[int], dict[str, Any]]:
    scale_cols = round_up(k // 16, 4)
    scale_size = round_up(m, 128) * scale_cols
    groups = m * (k // 16)
    ctas = max(triton.cdiv(groups, 128), min(triton.cdiv(scale_size, 512), 128))
    return (ctas,), dict(
        M=m,
        K=k,
        NUM_GROUPS=groups,
        PADDED_SCALE_COLS=scale_cols,
        PADDED_SCALE_SIZE=scale_size,
        GROUPS_PER_CTA=128,
        PAD_PER_CTA=triton.next_power_of_2(triton.cdiv(scale_size, ctas)),
        num_warps=4,
        enable_fp_fusion=False,
    )


def _relu_squared_nvfp4_max_rows(k: int) -> int:
    # Keep groups and padded scale size <= 2^30. The power-of-two padding
    # tile then keeps every pid * PAD_PER_CTA offset within signed int32.
    groups_per_row = k // 16
    scale_cols = round_up(groups_per_row, 4)
    return min((1 << 30) // groups_per_row, (1 << 30) // scale_cols // 128 * 128)


def _relu_squared_nvfp4_warmup_rows(k: int, max_tokens: int) -> range:
    # All padding-tile buckets occur in the first 128 rows. For M > 128,
    # CTAs = ceil(M * (K/16) / 128), so the padding/CTA ratio is between
    # its M=128 and M=64 values. The row bound keeps all runtime scalars i32.
    return range(1, min(128, max_tokens, _relu_squared_nvfp4_max_rows(k)) + 1)


def _relu_squared_nvfp4_warmup_inputs(
    module: torch.nn.Module, max_tokens: int
) -> Iterator[dict[str, Any]]:
    # Resolve after weight processing and LoRA replacement, not at construction.
    from vllm.model_executor.layers.fusion.fused_act_quant import (
        _relu_squared_nvfp4_warmup_width,
    )

    k = _relu_squared_nvfp4_warmup_width(module)
    if k is None:
        return
    for m in _relu_squared_nvfp4_warmup_rows(k, max_tokens):
        grid, kwargs = _relu_squared_nvfp4_quant_launch(m, k)
        for input_aligned, scale_aligned in product((True, False), repeat=2):
            yield dict(
                grid=grid,
                x_ptr=TritonWarmupTensor(torch.bfloat16, aligned=input_aligned),
                global_scale_ptr=TritonWarmupTensor(
                    torch.float32, aligned=scale_aligned
                ),
                output_ptr=TritonWarmupTensor(torch.uint8),
                scale_ptr=TritonWarmupTensor(torch.float8_e4m3fn),
                **kwargs,
            )


@triton.jit
def _rcp_ftz(x):
    return tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _max_ignore_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.NONE)


@triton.jit
def _e4m3_satfinite(x):
    packed = tl.inline_asm_elementwise(
        "{ .reg .b16 v; cvt.rn.satfinite.e4m3x2.f32 v, 0f00000000, $1; "
        "cvt.u32.u16 $0, v; }",
        constraints="=r,f",
        args=[x],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    return packed.to(tl.uint8).to(tl.float8e4nv, bitcast=True)


@triton_kernel_dispatcher_with_warmup(warmup_inputs=_relu_squared_nvfp4_warmup_inputs)
@triton.jit(do_not_specialize=["M", "NUM_GROUPS", "PADDED_SCALE_SIZE"])
def _relu_squared_nvfp4_quant_kernel(
    x_ptr,
    global_scale_ptr,
    output_ptr,
    scale_ptr,
    M,
    K: tl.constexpr,
    NUM_GROUPS,
    PADDED_SCALE_COLS: tl.constexpr,
    PADDED_SCALE_SIZE,
    GROUPS_PER_CTA: tl.constexpr,
    PAD_PER_CTA: tl.constexpr,
):
    pid = tl.program_id(0)
    groups = pid * GROUPS_PER_CTA + tl.arange(0, GROUPS_PER_CTA)
    row = groups // (K // 16)
    col = groups % (K // 16)
    if pid * GROUPS_PER_CTA < NUM_GROUPS:
        pair = tl.arange(0, 8)
        index = row[:, None].to(tl.int64) * K + col[:, None] * 16 + pair[None, :] * 2
        valid = groups[:, None] < NUM_GROUPS
        values = tl.load(
            x_ptr
            + row[:, None].to(tl.int64) * K
            + col[:, None] * 16
            + tl.arange(0, 16)[None, :],
            valid,
            other=0,
        )
        lo, hi = tl.split(tl.reshape(values, (GROUPS_PER_CTA, 8, 2)))
        lo, hi = lo.to(tl.float32), hi.to(tl.float32)
        # ReLU propagates NaN; tl.maximum otherwise defaults to maxnum semantics.
        lo = tl.maximum(lo, 0.0, propagate_nan=tl.PropagateNan.ALL)
        hi = tl.maximum(hi, 0.0, propagate_nan=tl.PropagateNan.ALL)
        # Preserve the materialized BF16 ReLU2 boundary before computing amax.
        lo = (lo * lo).to(tl.bfloat16).to(tl.float32)
        hi = (hi * hi).to(tl.bfloat16).to(tl.float32)
        # Match __hmax2 / __hmax in the CUDA quantizer: ignore individual NaNs.
        amax = tl.reduce(_max_ignore_nan(lo, hi), 1, _max_ignore_nan)
        global_scale = tl.load(global_scale_ptr)
        scale = _e4m3_satfinite(global_scale * (amax * _rcp_ftz(6.0)))
        scale_f32 = scale.to(tl.float32)
        inverse = tl.where(
            scale_f32 != 0.0,
            _rcp_ftz(scale_f32 * _rcp_ftz(global_scale)),
            0.0,
        )
        lo = lo * inverse[:, None]
        hi = hi * inverse[:, None]
        packed = tl.inline_asm_elementwise(
            "{ .reg .b8 v; cvt.rn.satfinite.e2m1x2.f32 v, $2, $1; cvt.u32.u8 $0, v; }",
            constraints="=r,f,f",
            args=[lo, hi],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        ).to(tl.uint8)
        tl.store(output_ptr + index // 2, packed, valid)
        scale_index = (
            ((row.to(tl.int64) // 128) * (PADDED_SCALE_COLS // 4) + col // 4) * 512
            + (row % 32) * 16
            + ((row % 128) // 32) * 4
            + col % 4
        )
        tl.store(scale_ptr + scale_index, scale, groups < NUM_GROUPS)

    # Initialize only padding; these stores never overlap the live scale stores.
    # Distribute padding across the same launch, including small-M cases.
    offsets = pid * PAD_PER_CTA + tl.arange(0, PAD_PER_CTA)
    tile = offsets // 512
    padded_row = (
        (tile // (PADDED_SCALE_COLS // 4)) * 128
        + (offsets % 16 // 4) * 32
        + offsets % 512 // 16
    )
    padded_col = (tile % (PADDED_SCALE_COLS // 4)) * 4 + offsets % 4
    is_padding = (padded_row >= M) | (padded_col >= K // 16)
    tl.store(scale_ptr + offsets, 0.0, (offsets < PADDED_SCALE_SIZE) & is_padding)


def relu_squared_nvfp4_quant_out(
    x: torch.Tensor,
    global_scale: torch.Tensor,
    output: torch.Tensor,
    block_scale: torch.Tensor,
) -> None:
    """Write BF16 ReLU2 + NVFP4 into distinct, preallocated output buffers.

    Requires M <= _relu_squared_nvfp4_max_rows(K).
    """
    m, k = x.shape
    # vLLM can reuse compiled graphs without checking Dynamo's shape guards.
    assert m <= _relu_squared_nvfp4_max_rows(k), "NVFP4 row limit exceeded"
    if m == 0:
        return
    grid, kwargs = _relu_squared_nvfp4_quant_launch(m, k)
    _relu_squared_nvfp4_quant_kernel[grid](
        x,
        global_scale,
        output,
        block_scale,
        **kwargs,
    )


def _relu_squared_nvfp4_quant_fake(
    x: torch.Tensor, global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = x.shape
    output = torch.empty((m, k // 2), dtype=torch.uint8, device=x.device)
    block_scale = torch.empty(
        (round_up(m, 128), round_up(k // 16, 4)),
        dtype=torch.float8_e4m3fn,
        device=x.device,
    )
    return output, block_scale


def _relu_squared_nvfp4_quant(
    x: torch.Tensor, global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    output, block_scale = _relu_squared_nvfp4_quant_fake(x, global_scale)
    relu_squared_nvfp4_quant_out(x, global_scale, output, block_scale)
    return output, block_scale


# Keep the M-dependent launch geometry outside Dynamo: vLLM drops shape guards.
direct_register_custom_op(
    op_name="relu_squared_nvfp4_quant",
    op_func=_relu_squared_nvfp4_quant,
    fake_impl=_relu_squared_nvfp4_quant_fake,
)


def relu_squared_nvfp4_quant(
    x: torch.Tensor, linear: LinearBase
) -> QuantizedActivation:
    """Fuse BF16 ReLU2 and block16 NVFP4 with 128x4 swizzled scales."""
    from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
    from vllm.model_executor.layers.quantization.utils.quant_utils import kNvfp4Dynamic

    output, block_scale = torch.ops.vllm.relu_squared_nvfp4_quant(
        x, linear.input_global_scale_inv
    )
    return QuantizedActivation(
        data=output,
        scale=block_scale,
        orig_dtype=x.dtype,
        orig_shape=x.shape,
        quant_key=kNvfp4Dynamic,
    )
