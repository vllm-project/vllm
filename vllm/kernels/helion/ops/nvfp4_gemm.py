# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor

from vllm.kernels.helion.case_key import CaseKey
from vllm.logger import init_logger
from vllm.utils.import_utils import has_helion

if not has_helion():
    raise ImportError(
        "nvfp4_gemm_fp4in Helion kernel requires helion to be installed. "
        "Install it with: pip install helion"
    )

import helion
import helion.language as hl
from helion._testing import DEVICE

from vllm.kernels.helion.register import register_kernel

logger = init_logger(__name__)

cutlass: Any = None
dsl_user_op: Any

cutlass = cast("Any", None)
dsl_user_op = cast("Any", None)

if TYPE_CHECKING:
    pass


# --- Layout Math & Validation Helper Functions ---


def quantize_fp4_e2m1(x: Tensor) -> Tensor:
    """
    Quantize a float tensor to FP4 E2M1 nibble indices (0-15).

    Each value is rounded to the nearest representable FP4 E2M1 value and
    encoded as a 4-bit index: bit 3 = sign, bits 2-0 = magnitude index.
    """
    sign = (x < 0).to(torch.uint8)
    abs_x = x.abs().clamp(max=6.0)
    boundaries = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device, dtype=abs_x.dtype
    )
    mag_idx = torch.bucketize(abs_x, boundaries).to(torch.uint8)
    return mag_idx | (sign << 3)


def pack_fp4_last_dim(indices: Tensor) -> Tensor:
    """
    Pack pairs of FP4 nibble indices along the last dim into bytes.
    lo and hi are across column, so we pack then col[i] (lo) || col[i+1] (hi)
    Example: A = [[1, 2, 3, 4, 5, 6]], shape [1, 6].
                byte0: lo=1,  hi=2 -> (1 & 0xF) | (2 << 4) = 0x21
                byte1: lo =3, hi=4 -> (3 & 0xF) | (4 << 4) = 0x43
                byte2: lo=5,  hi=6 -> (5 & 0xF) | (6 << 4) = 0x65
    """
    M, K = indices.shape
    assert K % 2 == 0, "K dimension must be even for FP4 packing"
    reshaped = indices.reshape(M, K // 2, 2)
    return ((reshaped[:, :, 0] & 0xF) | (reshaped[:, :, 1] << 4)).to(torch.uint8)


def pack_fp4(indices: Tensor) -> Tensor:
    """
    Pack pairs of FP4 nibble indices along dim 0 into bytes.
    Element at even index goes into the low nibble, odd index into the high nibble.
    lo and hi are across rows, so we pack row[i] (lo) || row[i+1] (hi)
    """
    K, N = indices.shape
    assert K % 2 == 0, "K dimension must be even for FP4 packing"
    reshaped = indices.reshape(K // 2, 2, N).permute(1, 0, 2)
    return ((reshaped[0] & 0xF) | (reshaped[1] << 4)).to(torch.uint8)


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
            f"{name} must contain {expected} SWIZZLE_32_4_4 "
            f"scale values; got {scales.numel()}"
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


# --- vLLM Infrastructure (Picker & Input Generation) ---


def generate_nvfp4_gemm_w4a4_inputs() -> dict[CaseKey, tuple[Any, ...]]:
    m_values = [1, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    kn_pairs = [(128, 64), (256, 128), (128, 4096), (4096, 4096), (2048, 7168)]
    inputs: dict[CaseKey, tuple[Any, ...]] = {}
    for m in m_values:
        for k, n in kn_pairs:
            A = torch.randn(m, k, dtype=torch.bfloat16, device=DEVICE)
            W = torch.randn(k, n, dtype=torch.bfloat16, device=DEVICE)
            # quantize to nibble indices then pack pairs along K into bytes
            A_quantized = quantize_fp4_e2m1(A)
            A_packed = pack_fp4_last_dim(A_quantized).view(torch.float4_e2m1fn_x2)
            W_quantized = quantize_fp4_e2m1(W)
            W_packed = pack_fp4(W_quantized).view(torch.float4_e2m1fn_x2)
            act_scale = make_fp8_scales((m, k // 16), DEVICE)
            weight_scale = make_fp8_scales((n, k // 16), DEVICE)

            key = CaseKey({"m": m, "k": k, "n": n})
            inputs[key] = (A_packed, W_packed, act_scale, weight_scale, 1.0)
    return inputs


def pick_nvfp4_gemm_w4a4_config(
    args: tuple[Any, ...], config_keys: list[CaseKey]
) -> CaseKey | None:
    """Pick the best pre-tuned config for the given input shape (M, K)."""
    if not config_keys:
        return None

    A_packed = args[0]  # [M, K//2]
    B_packed = args[1]  # [K//2, N]
    m = int(A_packed.shape[0])
    k = int(A_packed.shape[1]) * 2  # logical K
    n = int(B_packed.shape[1])

    cache_key = (m, k, n)
    cached = _pick_cache.get(cache_key)
    if cached is not None:
        return cached

    # Group configs by (K, N), then pick best M
    by_kn: dict[tuple[int, int], list[int]] = {}
    for key in config_keys:
        if key.is_default():
            continue
        by_kn.setdefault((key["k"], key["n"]), []).append(key["m"])

    if not by_kn:
        return None

    # Find closest (K, N) pair
    best_kn = min(by_kn.keys(), key=lambda kn: abs(kn[0] - k) + abs(kn[1] - n))

    # Find smallest M >= input M (or largest available)
    available_m = sorted(by_kn[best_kn])
    best_m = next((av for av in available_m if av >= m), available_m[-1])

    result = CaseKey({"m": best_m, "k": best_kn[0], "n": best_kn[1]})
    _pick_cache[cache_key] = result
    return result


# --- Main Entrypoint Registered to vLLM ---
_pick_cache: dict[tuple[int, int], CaseKey | None] = {}


@register_kernel(
    config_picker=pick_nvfp4_gemm_w4a4_config,
    input_generator=generate_nvfp4_gemm_w4a4_inputs,
    helion_settings=helion.Settings(backend="cute"),
)
def nvfp4_gemm_w4a4(
    A_packed: Tensor,  # weight_packed
    B_packed: Tensor,
    act_scale: Tensor,
    weight_scale: Tensor,
    alpha: float = 1.0,
) -> Tensor:
    M, K_bytes_a = A_packed.shape
    K_bytes_b, N = B_packed.shape
    # if K_bytes_a != K_bytes_b:
    #    raise ValueError(
    #        f"A_packed shape {tuple(A_packed.shape)} is incompatible with "
    #        f"B_packed shape {tuple(B_packed.shape)}"
    #    )
    K = K_bytes_a * 2
    # if K % 16 != 0:
    #    raise ValueError(f"K must be divisible by 16, got {K}")
    K_groups = K // 16
    _check_swizzled_scales("act_scale", act_scale, M, K_groups)
    _check_swizzled_scales("weight_scale", weight_scale, N, K_groups)
    a_fp4x2 = _as_fp4x2(A_packed).view(M, K_groups, 8)
    b_fp4x2 = _as_fp4x2(B_packed).view(K_groups, 8, N)
    out = torch.empty(M, N, dtype=torch.bfloat16, device=A_packed.device)
    act_scale = act_scale.reshape(-1)
    weight_scale = weight_scale.reshape(-1)
    M, K_groups, _ = a_fp4x2.shape
    _, _, N = b_fp4x2.shape

    for tile_m, tile_n in hl.tile([M, N]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(K_groups):
            w_scale_offsets = swizzled_scale_offsets(
                tile_n.index[None, :],
                tile_k.index[:, None],
                K_groups,
            )
            w_scale = weight_scale[w_scale_offsets].to(torch.float32)
            a_scale_offsets = swizzled_scale_offsets(
                tile_m.index[:, None],
                tile_k.index[None, :],
                K_groups,
            )
            a_scale = act_scale[a_scale_offsets].to(torch.float32)
            for byte in hl.static_range(8):
                a_lo, a_hi = hl.float4_e2m1fn_x2_to_float32(
                    a_fp4x2[tile_m, tile_k, byte]
                )
                w_lo, w_hi = hl.float4_e2m1fn_x2_to_float32(
                    b_fp4x2[tile_k, byte, tile_n]
                )
                contrib_lo = a_lo.unsqueeze(2) * w_lo.unsqueeze(0)
                contrib_hi = a_hi.unsqueeze(2) * w_hi.unsqueeze(0)
                acc = acc + (
                    (contrib_lo + contrib_hi)
                    * a_scale.unsqueeze(2)
                    * w_scale.unsqueeze(0)
                ).sum(dim=1)
        out[tile_m, tile_n] = (acc * alpha).to(torch.bfloat16)
    return out
