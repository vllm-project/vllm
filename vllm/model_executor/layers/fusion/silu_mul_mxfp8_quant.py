# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch
from packaging.version import InvalidVersion, Version

from vllm.triton_utils import tl, tldevice, triton
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import direct_register_custom_op


def _silu_mxfp8_triton_supported() -> bool:
    try:
        if Version(triton.__version__) < Version("3.8.0"):
            return False
        from triton.backends.nvidia.compiler import CUDAOptions
    except (ImportError, AttributeError, InvalidVersion):
        return False
    # Triton's CUDA backend silently discards unrecognized launch options.
    options = getattr(CUDAOptions, "__dataclass_fields__", {})
    return {"enable_reflect_ftz", "enable_fp_fusion"} <= options.keys()


# Check once outside Dynamo. Numerical validation uses Torch 2.14 / Triton 3.8.
_SILU_MXFP8_TRITON_SUPPORTED = _silu_mxfp8_triton_supported()


def _silu_mxfp8_max_rows(k: int) -> int:
    # Keep group indices and padded scale offsets within signed int32.
    scale_cols = round_up(k // 32, 4)
    return (1 << 30) // scale_cols // 128 * 128


@triton.jit
def _max_ignore_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.NONE)


@triton.jit(do_not_specialize=["M", "NUM_GROUPS", "PADDED_SCALE_SIZE"])
def _silu_mul_mxfp8_quant_kernel(
    x_ptr,
    output_ptr,
    scale_ptr,
    M,
    K: tl.constexpr,
    NUM_GROUPS,
    PADDED_SCALE_COLS: tl.constexpr,
    PADDED_SCALE_SIZE,
    GROUPS_PER_CTA: tl.constexpr,
    COMPILED_NATIVE: tl.constexpr,
):
    pid = tl.program_id(0)
    groups = pid * GROUPS_PER_CTA + tl.arange(0, GROUPS_PER_CTA)
    row = groups // (K // 32)
    col = groups % (K // 32)
    if pid * GROUPS_PER_CTA < NUM_GROUPS:
        lane = tl.arange(0, 32)
        index = row[:, None].to(tl.int64) * (2 * K) + col[:, None] * 32 + lane
        valid = groups[:, None] < NUM_GROUPS
        gate = tl.load(x_ptr + index, valid, other=0).to(tl.float32)
        up = tl.load(x_ptr + index + K, valid, other=0).to(tl.float32)
        if COMPILED_NATIVE:
            # Match the retained default-Inductor source: libdevice exponential,
            # ordinary Triton division, and no intermediate BF16 conversion.
            activated = gate / (1.0 + tldevice.exp(-gate))
        else:
            # Match _C.silu_and_mul's materialized intermediate BF16 boundary.
            activated = tl.div_rn(gate, 1.0 + tldevice.exp(-gate))
            activated = activated.to(tl.bfloat16).to(tl.float32)
            # CUDA adds runtime beta=+0 to up, including signed-zero behavior.
            up = tl.inline_asm_elementwise(
                "add.rn.f32 $0, $1, 0f00000000;",
                constraints="=f,f",
                args=[up],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
        values = (activated * up).to(tl.bfloat16).to(tl.float32)

        # Match FlashInfer CuTe DSL's reciprocal multiply before UE8M0 rounding.
        maximum = tl.reduce(tl.abs(values), 1, _max_ignore_nan)
        normalized_max = maximum * (1.0 / 448.0)
        bits = normalized_max.to(tl.uint32, bitcast=True)
        exponent = (bits >> 23) & 255
        mantissa = bits & 0x7FFFFF
        bump = (mantissa != 0) & ~((exponent == 0) & (mantissa <= 0x400000))
        scale = tl.minimum(exponent + bump.to(tl.uint32), 254)
        scale = tl.where(normalized_max <= 0, 0, scale)
        inverse_bits = tl.where(scale == 0, 0, (254 - scale) << 23)
        inverse = inverse_bits.to(tl.float32, bitcast=True)
        scaled = values * inverse[:, None]
        # Native min/max suppress NaNs before the saturating FP8 conversion.
        clipped = tl.inline_asm_elementwise(
            "min.f32 $0, $1, 0f43E00000; max.f32 $0, $0, 0fC3E00000;",
            constraints="=f,f",
            args=[scaled],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        output_index = row[:, None].to(tl.int64) * K + col[:, None] * 32 + lane
        tl.store(output_ptr + output_index, clipped.to(tl.float8e4nv), valid)
        scale_index = (
            ((row.to(tl.int64) // 128) * (PADDED_SCALE_COLS // 4) + col // 4) * 512
            + (row % 32) * 16
            + ((row % 128) // 32) * 4
            + col % 4
        )
        tl.store(scale_ptr + scale_index, scale.to(tl.uint8), groups < NUM_GROUPS)

    # One CTA owns each padding tile. These writes never touch live scales.
    if pid < PADDED_SCALE_SIZE // 512:
        offsets = pid * 512 + tl.arange(0, 512)
        padded_row = (
            (pid // (PADDED_SCALE_COLS // 4)) * 128
            + (offsets % 16 // 4) * 32
            + offsets % 512 // 16
        )
        padded_col = (pid % (PADDED_SCALE_COLS // 4)) * 4 + offsets % 4
        padding = (padded_row >= M) | (padded_col >= K // 32)
        tl.store(scale_ptr + offsets, 0, padding)


def _silu_mul_mxfp8_quant_fake(
    x: torch.Tensor, rounding_mode: str = "cuda"
) -> tuple[torch.Tensor, torch.Tensor]:
    m, two_k = x.shape
    k = two_k // 2
    return (
        torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device),
        torch.empty(
            (round_up(m, 128) * round_up(k // 32, 4),),
            dtype=torch.uint8,
            device=x.device,
        ),
    )


def _silu_mul_mxfp8_quant(
    x: torch.Tensor, rounding_mode: str = "cuda"
) -> tuple[torch.Tensor, torch.Tensor]:
    assert rounding_mode in ("cuda", "compiled_native"), "Unsupported SiLU rounding"
    assert x.is_cuda and x.dtype == torch.bfloat16 and x.is_contiguous()
    assert x.ndim == 2 and x.shape[1] > 0 and x.shape[1] % 64 == 0
    assert _SILU_MXFP8_TRITON_SUPPORTED, (
        "SiLU MXFP8 fusion requires Triton 3.8+ with CUDA numeric options"
    )
    # Compiled graphs may be reused without Dynamo guards on the row count.
    return _silu_mul_mxfp8_quant_impl(
        x, rounding_mode, _silu_mxfp8_max_rows(x.shape[1] // 2)
    )


def _silu_mul_mxfp8_quant_impl(
    x: torch.Tensor, rounding_mode: str, max_rows: int
) -> tuple[torch.Tensor, torch.Tensor]:
    m, two_k = x.shape
    k = two_k // 2
    output, scale = _silu_mul_mxfp8_quant_fake(x)
    if m == 0:
        return output, scale
    assert 0 < max_rows <= _silu_mxfp8_max_rows(k) and max_rows % 128 == 0
    scale_cols = round_up(k // 32, 4)
    # Whole 128-row scale tiles concatenate in the flat F8_128x4 layout.
    for start in range(0, m, max_rows):
        end = min(start + max_rows, m)
        rows = end - start
        groups = rows * (k // 32)
        scale_size = round_up(rows, 128) * scale_cols
        groups_per_cta = 4 if rows <= 32 else 128
        grid = (max(triton.cdiv(groups, groups_per_cta), scale_size // 512),)
        if m <= max_rows:
            chunk_x, chunk_output, chunk_scale = x, output, scale
        else:
            chunk_x = x[start:end]
            chunk_output = output[start:end]
            chunk_scale = scale[start * scale_cols : round_up(end, 128) * scale_cols]
        _silu_mul_mxfp8_quant_kernel[grid](
            chunk_x,
            chunk_output,
            chunk_scale,
            M=rows,
            K=k,
            NUM_GROUPS=groups,
            PADDED_SCALE_COLS=scale_cols,
            PADDED_SCALE_SIZE=scale_size,
            GROUPS_PER_CTA=groups_per_cta,
            COMPILED_NATIVE=rounding_mode == "compiled_native",
            num_warps=4,
            enable_fp_fusion=rounding_mode == "compiled_native",
            # Inductor disable_ftz=False leaves this CUDA backend default enabled.
            enable_reflect_ftz=True,
        )
    return output, scale


# Keep M-dependent launch geometry outside Dynamo and preserve output ownership.
direct_register_custom_op(
    op_name="silu_mul_mxfp8_quant",
    op_func=_silu_mul_mxfp8_quant,
    fake_impl=_silu_mul_mxfp8_quant_fake,
)
