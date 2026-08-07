# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Literal, cast
import torch
from torch import Tensor

from vllm.kernels.helion.case_key import CaseKey
from vllm.logger import init_logger
from vllm.utils.import_utils import has_helion

if not has_helion():
    raise ImportError(
        "nvfp4_gemv_fp4in Helion kernel requires helion to be installed. "
        "Install it with: pip install helion"
    )

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl
from vllm.kernels.helion.register import register_kernel

logger = init_logger(__name__)

cutlass: Any = None
dsl_user_op: Any

cutlass = cast("Any", None)
dsl_user_op = cast("Any", None)

if TYPE_CHECKING:
    from collections.abc import Callable

    from cutlass._mlir import ir


# --- Layout Math & Validation Helper Functions ---

def swizzle_fp8_scales(scales: Tensor) -> Tensor:
    """Convert logical row-major block scales to PyTorch's SWIZZLE_32_4_4 layout."""
    if scales.dim() == 1:
        logical_scales = scales.reshape(1, scales.shape[0])
    elif scales.dim() == 2:
        logical_scales = scales
    else:
        raise ValueError(f"expected 1D or 2D scales, got {scales.dim()}D")

    rows, cols = logical_scales.shape
    out = torch.zeros(
        swizzled_scale_numel(rows, cols),
        device=logical_scales.device,
        dtype=logical_scales.dtype,
    )
    row = torch.arange(rows, device=logical_scales.device, dtype=torch.int64)[:, None]
    col = torch.arange(cols, device=logical_scales.device, dtype=torch.int64)[None, :]
    offsets = cast("Tensor", swizzled_scale_offsets(row, col, cols))
    out[offsets.reshape(-1)] = logical_scales.reshape(-1)
    return out

def make_fp8_scales(shape: tuple[int, ...], device: torch.device) -> Tensor:
    logical_scales = (torch.rand(shape, device=device, dtype=torch.float32) + 0.5).to(
        torch.float8_e4m3fn
    )
    return swizzle_fp8_scales(logical_scales)

def _dequant_e2m1(nibbles: Tensor) -> Tensor:
    sign = ((nibbles >> 3) & 1).to(torch.float32)
    u = (nibbles & 0x7).to(torch.float32)
    abs_val = torch.where(
        u < 4.0,
        u * 0.5,
        torch.where(u < 6.0, u - 2.0, u * 2.0 - 8.0),
    )
    return abs_val * (1.0 - 2.0 * sign)

def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b

def _round_up(a: int, b: int) -> int:
    return _ceil_div(a, b) * b

def swizzled_scale_numel(rows: int, cols: int) -> int:
    return _round_up(rows, 128) * _round_up(cols, 4)

def swizzled_scale_offsets(row: int, col: int, cols: int) -> int:
    num_col_tiles = _ceil_div(cols, 4)
    tile_offset = ((row // 128) * num_col_tiles + col // 4) * 512
    return tile_offset + (row % 32) * 16 + ((row % 128) // 32) * 4 + col % 4

def _check_swizzled_scales(name: str, scales: Tensor, rows: int, cols: int) -> None:
    expected = swizzled_scale_numel(rows, cols)
    if scales.numel() != expected:
        raise ValueError(
            f"{name} must contain {expected} SWIZZLE_32_4_4 scale values; got {scales.numel()}"
        )

def _as_fp4x2(tensor: Tensor) -> Tensor:
    if tensor.dtype is torch.float4_e2m1fn_x2:
        return tensor
    if tensor.dtype is torch.uint8:
        return tensor.view(torch.float4_e2m1fn_x2)
    raise TypeError(f"Expected uint8 or float4_e2m1fn_x2, got {tensor.dtype}")

def _fp4_storage(tensor: Tensor) -> Tensor:
    if tensor.dtype is torch.float4_e2m1fn_x2:
        return tensor.view(torch.uint8)
    return tensor



DEVICE="cuda" if torch.cuda.is_available() else "cpu"

# --- vLLM Infrastructure (Picker & Input Generation) ---

def generate_nvfp4_gemv_fp4in_inputs() -> dict[CaseKey, tuple[Any, ...]]:
    SHAPES = [(128, 64),(256, 128), (128, 4096),(4096, 4096), (2048, 7168)]
    inputs: dict[CaseKey, tuple[Any, ...]] = {}
    for m, k in SHAPES:
        # Match tensor generation directly to the shape expectation of the picker
        weight = torch.randint(0, 256, (m, k), dtype=torch.uint8, device=DEVICE).view(
            torch.float4_e2m1fn_x2
        )
        x = torch.randint(0, 256, (k,), dtype=torch.uint8, device=DEVICE).view(
            torch.float4_e2m1fn_x2
        )
        weight_scale = make_fp8_scales((m, k // 8), DEVICE)
        x_scale = make_fp8_scales((k // 8,), DEVICE)
        
        # Key dictionary keys must directly reflect the packed metrics evaluated above
        key = CaseKey({"m": m, "k": k})
        inputs[key] = (weight, x, weight_scale, x_scale, 1.0)
        
    return inputs


def pick_nvfp4_gemv_fp4in_config(
    args: tuple[Any, ...], config_keys: list[CaseKey]
) -> CaseKey | None:
    """Pick the best pre-tuned config for the given input shape (M, K)."""
    if not config_keys:
        return None
    
    # Unpack target shapes from the runtime weight matrix tensor
    weight_packed = args[0]
    m = int(weight_packed.shape[0])
    k = int(weight_packed.shape[1])
    #print(f"Picking config for M={m}, K={k}, from {len(config_keys)} available configs.")
    # Check cache first
    query_key = CaseKey({"m": m, "k": k})
    if query_key in config_keys:
        return query_key

    
    # Group available compiled configs by M
    by_m: dict[int, list[int]] = {}
    for key in config_keys:
        if key.is_default():
            continue
        by_m.setdefault(key["m"], []).append(key["k"])

    if not by_m:
        return None

    # Proximity Selection Fallback Matrix Matching:
    # 1. Find the closest M size
    best_m = min(by_m.keys(), key=lambda available_m: abs(available_m - m))
    
    # 2. Find the closest K size for that M (preferring exact or upper bound fallback)
    available_k = sorted(by_m[best_m])
    best_k = next((available_k_val for available_k_val in available_k if available_k_val >= k), available_k[-1])
    #print (f"Selected config for M={m}, K={k}: M={best_m}, K={best_k}")
    return CaseKey({"m": best_m, "k": best_k})

# --- Main Entrypoint Registered to vLLM ---
_pick_cache: dict[tuple[int, int], CaseKey | None] = {}



@register_kernel(
    config_picker=pick_nvfp4_gemv_fp4in_config,
    input_generator=generate_nvfp4_gemv_fp4in_inputs,
)
def nvfp4_gemv_fp4in(
    weight_packed: Tensor,
    x_packed: Tensor,
    weight_scale: Tensor,
    x_scale: Tensor,
    alpha: float = 1.0,
) -> Tensor:    
    weight_fp4x2 = _as_fp4x2(weight_packed)
    x_fp4x2 = _as_fp4x2(x_packed)
    weight_bytes = weight_fp4x2.view(torch.uint8)
    x_bytes = x_fp4x2.view(torch.uint8)
    scale_cols = weight_bytes.shape[1] // 8
    _check_swizzled_scales(
        "weight_scale",
        weight_scale,
        weight_bytes.shape[0],
        scale_cols,
    )
    _check_swizzled_scales("x_scale", x_scale, 1, scale_cols)
    out = torch.empty(
        weight_bytes.shape[0], dtype=torch.bfloat16, device=weight_bytes.device
    )

    weight_fp4x2_reshaped = weight_fp4x2.view(weight_bytes.shape[0], weight_bytes.shape[1] // 8, 8)
    x_fp4x2_reshaped = x_fp4x2.view(weight_bytes.shape[1] // 8, 8)
    weight_scale_flat = weight_scale.reshape(-1)
    x_scale_flat = x_scale.reshape(-1)

    M, K_groups, _ = weight_fp4x2_reshaped.shape
    block_m = hl.register_block_size(1, 8)
    block_k = hl.register_block_size(16, K_groups)

    for tile_m in hl.tile(M, block_size=block_m):
        row = tile_m.begin
        acc = hl.zeros([], dtype=torch.float32)
        for tile_k in hl.tile(K_groups, block_size=block_k):
            contrib = hl.zeros([tile_k], dtype=torch.float32)
            for byte in hl.static_range(8):
                weight_lo, weight_hi = hl.float4_e2m1fn_x2_to_float32(
                    weight_fp4x2_reshaped[row, tile_k, byte]
                )
                x_lo, x_hi = hl.float4_e2m1fn_x2_to_float32(x_fp4x2_reshaped[tile_k, byte])
                contrib = contrib + weight_lo * x_lo + weight_hi * x_hi
            weight_scale_offsets = swizzled_scale_offsets(
                cast("int", row),
                tile_k.index,
                K_groups,
            )
            x_scale_offsets = swizzled_scale_offsets(
                tile_k.index * 0,
                tile_k.index,
                K_groups,
            )
            scale = hl.load(
                weight_scale_flat,
                [weight_scale_offsets],
                extra_mask=tile_k.index < K_groups,
            ).to(torch.float32)
            scale = scale * hl.load(
                x_scale_flat,
                [x_scale_offsets],
                extra_mask=tile_k.index < K_groups,
            ).to(torch.float32)
            acc = acc + (contrib * scale).sum()
        out[row] = (acc * alpha).to(torch.bfloat16)
    return out

# from vllm.utils.torch_utils import direct_register_custom_op

# def _nvfp4_gemv_fp4in_impl(
#     output: torch.Tensor,
#     weight: torch.Tensor,
#     x: torch.Tensor,
#     weight_scale: torch.Tensor,
#     x_scale: torch.Tensor,
#     alpha: float,
# ) -> None:
#     logger.debug(f"[HELION] nvfp4_gemv_fp4in called: weight={weight.shape}, x={x.shape}")
#     result = nvfp4_gemv_fp4in(weight, x, weight_scale, x_scale, alpha=alpha)
#     output.copy_(result)

# def _nvfp4_gemv_fp4in_fake(
#     output: torch.Tensor,
#     weight: torch.Tensor,
#     x: torch.Tensor,
#     weight_scale: torch.Tensor,
#     x_scale: torch.Tensor,
#     alpha: float,
# ) -> None:
#     return

# direct_register_custom_op(
#     "nvfp4_gemv_fp4in",
#     _nvfp4_gemv_fp4in_impl,
#     mutates_args=["output"],
#     fake_impl=_nvfp4_gemv_fp4in_fake,
# )