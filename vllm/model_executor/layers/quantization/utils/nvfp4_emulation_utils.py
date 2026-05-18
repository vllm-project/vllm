# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import torch

from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.triton_utils import tl, triton

__all__ = [
    "break_fp4_bytes",
    "dequantize_to_dtype",
    "ref_nvfp4_quant",
]

FLOAT4_E2M1_MAX = scalar_types.float4_e2m1f.max()
FLOAT4_E2M1_MAX_RECIPROCAL = 1 / FLOAT4_E2M1_MAX

kE2M1ToFloat_handle = SimpleNamespace(
    val=torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
)


@triton.jit
def _e2m1_inline(magnitude):
    """Inline E2M1 lookup using binary tree - 3 levels instead of 7 sequential.

    Maps 3-bit magnitude to float: [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    Uses bit decomposition for fewer comparisons.
    """
    # Bit 2 (MSB): separates 0-3 from 4-7
    # Bit 1: separates within groups
    # Bit 0 (LSB): separates within pairs
    b2 = (magnitude >> 2) & 1  # 0 for mag 0-3, 1 for mag 4-7
    b1 = (magnitude >> 1) & 1  # middle bit
    b0 = magnitude & 1  # LSB

    # For mag 0-3: [0.0, 0.5, 1.0, 1.5]
    low_group = tl.where(
        b1 == 1, tl.where(b0 == 1, 1.5, 1.0), tl.where(b0 == 1, 0.5, 0.0)
    )
    # For mag 4-7: [2.0, 3.0, 4.0, 6.0]
    high_group = tl.where(
        b1 == 1, tl.where(b0 == 1, 6.0, 4.0), tl.where(b0 == 1, 3.0, 2.0)
    )
    return tl.where(b2 == 1, high_group, low_group)


@triton.jit
def _dequantize_nvfp4_kernel(
    fp4_ptr,
    scale_ptr,
    global_scale_ptr,
    output_ptr,
    rows_per_batch: tl.constexpr,
    num_blocks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    has_batch_global_scale: tl.constexpr,
    TILE_BLOCKS: tl.constexpr,
):
    """Triton kernel for NVFP4 dequantization (swizzle=False).

    Optimized with 2D tile processing + interleave for coalesced stores.
    """
    BLOCK_PACKED: tl.constexpr = BLOCK_SIZE // 2

    row_idx = tl.program_id(0)
    tile_idx = tl.program_id(1)

    if has_batch_global_scale:
        batch_idx = row_idx // rows_per_batch
        global_scale = tl.load(global_scale_ptr + batch_idx).to(tl.float32)
    else:
        global_scale = tl.load(global_scale_ptr).to(tl.float32)

    fp4_row_offset = row_idx * num_blocks * BLOCK_PACKED
    scale_row_offset = row_idx * num_blocks
    output_row_offset = row_idx * num_blocks * BLOCK_SIZE

    start_block = tile_idx * TILE_BLOCKS

    # Load scales for this tile: [TILE_BLOCKS]
    block_offsets = tl.arange(0, TILE_BLOCKS)
    block_mask = (start_block + block_offsets) < num_blocks

    raw_scales = tl.load(
        scale_ptr + scale_row_offset + start_block + block_offsets,
        mask=block_mask,
        other=0,
    )
    scale_f32 = tl.cast(raw_scales, tl.float8e4nv, bitcast=True).to(tl.float32)
    scale_values = (scale_f32 * global_scale)[:, None]

    # Load [TILE_BLOCKS, BLOCK_PACKED] packed bytes
    packed_offsets = tl.arange(0, BLOCK_PACKED)[None, :]
    byte_indices = (
        fp4_row_offset
        + (start_block + block_offsets[:, None]) * BLOCK_PACKED
        + packed_offsets
    )
    elem_mask = block_mask[:, None]
    raw_bytes = tl.load(fp4_ptr + byte_indices, mask=elem_mask, other=0)

    low_nibble = raw_bytes & 0x0F
    high_nibble = (raw_bytes >> 4) & 0x0F

    # Binary tree E2M1 decode
    low_mag = low_nibble & 0x07
    low_val = _e2m1_inline(low_mag)
    low_sign = (low_nibble >> 3) & 1
    low_result = tl.where(low_sign == 1, -low_val, low_val) * scale_values

    high_mag = high_nibble & 0x07
    high_val = _e2m1_inline(high_mag)
    high_sign = (high_nibble >> 3) & 1
    high_result = tl.where(high_sign == 1, -high_val, high_val) * scale_values

    # Interleave for coalesced contiguous store
    result = tl.interleave(low_result, high_result)

    elem_offsets = tl.arange(0, BLOCK_SIZE)[None, :]
    out_indices = (
        output_row_offset
        + (start_block + block_offsets[:, None]) * BLOCK_SIZE
        + elem_offsets
    )
    tl.store(output_ptr + out_indices, result, mask=block_mask[:, None])


@triton.jit
def _e2m1_lookup(magnitude):
    """Lookup E2M1 float value from 3-bit magnitude."""
    result = tl.where(magnitude == 1, 0.5, 0.0)
    result = tl.where(magnitude == 2, 1.0, result)
    result = tl.where(magnitude == 3, 1.5, result)
    result = tl.where(magnitude == 4, 2.0, result)
    result = tl.where(magnitude == 5, 3.0, result)
    result = tl.where(magnitude == 6, 4.0, result)
    result = tl.where(magnitude == 7, 6.0, result)
    return result


@triton.jit
def _round_to_fp4(x):
    """Round float values to the nearest E2M1 representable value.

    Matches the thresholds in the Python ``cast_to_fp4`` exactly.
    """
    sign = tl.where(x < 0.0, -1.0, 1.0)
    abs_x = tl.abs(x)
    result = tl.where(abs_x > 5.0, 6.0, 0.0)
    result = tl.where((abs_x >= 3.5) & (abs_x <= 5.0), 4.0, result)
    result = tl.where((abs_x > 2.5) & (abs_x < 3.5), 3.0, result)
    result = tl.where((abs_x >= 1.75) & (abs_x <= 2.5), 2.0, result)
    result = tl.where((abs_x > 1.25) & (abs_x < 1.75), 1.5, result)
    result = tl.where((abs_x >= 0.75) & (abs_x <= 1.25), 1.0, result)
    result = tl.where((abs_x > 0.25) & (abs_x < 0.75), 0.5, result)
    return result * sign


@triton.jit
def _nvfp4_quant_dequant_kernel(
    input_ptr,
    output_ptr,
    global_scale_ptr,
    k: tl.constexpr,
    num_blocks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    FP4_MAX_RECIPROCAL: tl.constexpr,
    TILE_BLOCKS: tl.constexpr,
):
    """Fused NVFP4 quantize-dequantize kernel.

    Uses a 2D grid (rows x tiles) to parallelize across both rows
    and quantization groups within a row. Each program handles
    TILE_BLOCKS groups at once using vectorized 2D operations.
    """
    row_idx = tl.program_id(0)
    tile_idx = tl.program_id(1)
    global_scale = tl.load(global_scale_ptr).to(tl.float32)
    row_offset = row_idx * k

    start_block = tile_idx * TILE_BLOCKS
    block_offsets = tl.arange(0, TILE_BLOCKS)
    block_mask = (start_block + block_offsets) < num_blocks

    # Load [TILE_BLOCKS, BLOCK_SIZE] elements
    indices = (
        row_offset
        + (start_block + block_offsets[:, None]) * BLOCK_SIZE
        + tl.arange(0, BLOCK_SIZE)[None, :]
    )
    mask_2d = block_mask[:, None]
    x = tl.load(input_ptr + indices, mask=mask_2d, other=0.0).to(tl.float32)

    # Per-group scale: [TILE_BLOCKS]
    vec_max = tl.max(tl.abs(x), axis=1)
    scale = global_scale * (vec_max * FP4_MAX_RECIPROCAL)
    scale = tl.clamp(scale, -448.0, 448.0)
    scale = scale.to(tl.float8e4nv).to(tl.float32)

    # Safe reciprocal, broadcast to [TILE_BLOCKS, 1]
    output_scale = tl.where(scale == 0.0, 0.0, global_scale / scale)[:, None]

    # Quantize: scale, clamp, round to FP4
    scaled_x = tl.clamp(x * output_scale, -6.0, 6.0)
    fp4_val = _round_to_fp4(scaled_x)

    # Dequantize: fp4_val * (scale / global_scale)
    dequant_scale = (scale / global_scale)[:, None]
    result = fp4_val * dequant_scale

    tl.store(output_ptr + indices, result, mask=mask_2d)


def _triton_nvfp4_quant_dequant(
    x: torch.Tensor,
    global_scale: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Triton-accelerated NVFP4 quantize-dequantize."""
    x_m, x_k = x.shape

    if not torch.compiler.is_compiling():
        assert x_k % block_size == 0, (
            f"Weight shape K={x_k} is not divisible by block_size={block_size}"
        )

    output_dtype = x.dtype
    num_blocks = x_k // block_size

    output = torch.empty(x_m, x_k, dtype=output_dtype, device=x.device)

    tile_blocks = min(64, triton.next_power_of_2(num_blocks))
    num_tiles = (num_blocks + tile_blocks - 1) // tile_blocks
    grid = (x_m, num_tiles)
    _nvfp4_quant_dequant_kernel[grid](
        x,
        output,
        global_scale,
        x_k,
        num_blocks,
        block_size,
        FLOAT4_E2M1_MAX_RECIPROCAL,
        tile_blocks,
    )

    return output


def _triton_dequantize_nvfp4(
    tensor_fp4: torch.Tensor,
    tensor_sf: torch.Tensor,
    global_scale: torch.Tensor,
    dtype: torch.dtype,
    block_size: int = 16,
) -> torch.Tensor:
    """Dequantize NVFP4 using Triton (swizzle=False only).

    Supports both 2D and 3D inputs:
    - 2D: [m, packed_k] -> [m, k]
    - 3D: [dim0, m, packed_k] -> [dim0, m, k]
    """
    assert tensor_fp4.dtype == torch.uint8

    is_3d = tensor_fp4.ndim == 3
    if is_3d:
        dim0, m_per_batch, packed_k = tensor_fp4.shape
        tensor_fp4_2d = tensor_fp4.reshape(-1, packed_k)
        tensor_sf_2d = tensor_sf.reshape(-1, tensor_sf.shape[-1])
        total_rows_flat = dim0 * m_per_batch
    else:
        m_per_batch, packed_k = tensor_fp4.shape
        tensor_fp4_2d = tensor_fp4
        tensor_sf_2d = tensor_sf
        total_rows_flat = m_per_batch

    k = packed_k * 2
    num_blocks = k // block_size

    output = torch.empty(total_rows_flat, k, dtype=dtype, device=tensor_fp4.device)

    # View as uint8 so Triton can load raw bytes and bitcast to float8_e4m3fn
    scale_raw = tensor_sf_2d.contiguous().view(torch.uint8)

    # Shape-adaptive tile sizing: for large row counts (3D), process
    # entire row in one tile. For small row counts (2D), use smaller
    # tiles to increase parallelism across CUs.
    np2 = triton.next_power_of_2(num_blocks)
    if total_rows_flat >= 4096:
        # Many rows: maximize work per CTA, one tile per row
        tile_blocks = np2
        nw = 1
        ns = 2
    elif total_rows_flat >= 2048:
        # Medium-many rows: full row, 2 warps
        tile_blocks = np2
        nw = 2
        ns = 2
    else:
        # Few rows: use moderate tiles for CU utilization
        tile_blocks = min(64, np2)
        nw = 4
        ns = 2
    num_tiles = (num_blocks + tile_blocks - 1) // tile_blocks
    grid = (total_rows_flat, num_tiles)
    _dequantize_nvfp4_kernel[grid](
        tensor_fp4_2d,
        scale_raw,
        global_scale,
        output,
        m_per_batch,
        num_blocks,
        block_size,
        is_3d,
        tile_blocks,
        num_warps=nw,
        num_stages=ns,
    )

    if is_3d:
        output = output.reshape(dim0, m_per_batch, k)

    return output


def break_fp4_bytes(a, dtype):
    assert a.dtype == torch.uint8
    m, n = a.shape
    # Vectorized nibble processing
    a_flat = a.flatten()
    high = (a_flat & 0xF0) >> 4  # Upper nibbles
    low = a_flat & 0x0F  # Lower nibbles
    # Combine nibbles for batch processing
    combined = torch.stack((low, high), dim=1).flatten()
    # Vectorized sign and magnitude extraction
    signs = (combined & 0x08).to(torch.bool)  # Sign bits
    abs_vals = (combined & 0x07).to(torch.long)

    kE2M1 = kE2M1ToFloat_handle.val
    # Device-aware lookup and sign application
    values = kE2M1[abs_vals] * torch.where(signs, -1.0, 1.0)
    # Reshape to final form
    return values.reshape(m, n * 2).to(dtype=dtype)


def convert_swizzled_to_linear(a_sf_swizzled: torch.Tensor, m, k, block_size):
    m_tiles = (m + 128 - 1) // 128
    f = block_size * 4
    k_tiles = (k + f - 1) // f
    tmp = torch.reshape(a_sf_swizzled, (1, m_tiles, k_tiles, 32, 4, 4))
    tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    out = tmp.reshape(m_tiles * 128, k_tiles * f // block_size)
    return out[0:m, 0:k]


def dequantize_to_dtype(
    tensor_fp4: torch.Tensor,
    tensor_sf: torch.Tensor,
    global_scale: torch.Tensor,
    dtype: torch.dtype,
    block_size: int = 16,
    swizzle: bool | None = True,
):
    """Dequantize the fp4 tensor back to high precision.

    Supports both 2D and 3D inputs:
    - 2D: [m, packed_k] -> [m, k]
    - 3D: [dim0, m, packed_k] -> [dim0, m, k]
    """
    # Two fp4 values are packed into one uint8.
    assert tensor_fp4.dtype == torch.uint8

    if not swizzle and current_platform.is_cuda_alike():
        return _triton_dequantize_nvfp4(
            tensor_fp4, tensor_sf, global_scale, dtype, block_size
        )

    # We handle 3D tensors reshaping them to 2D.
    is_3d = tensor_fp4.ndim == 3

    if is_3d:
        dim0, m, packed_k = tensor_fp4.shape
        tensor_fp4 = tensor_fp4.reshape(-1, packed_k)
        tensor_sf = tensor_sf.reshape(-1, tensor_sf.shape[-1])
        global_scale = global_scale[:, None, None]
    else:
        m, packed_k = tensor_fp4.shape

    k = packed_k * 2
    tensor_f32 = break_fp4_bytes(tensor_fp4, torch.float32)
    tensor_f32 = tensor_f32.reshape(-1, k // block_size, block_size)
    tensor_sf = tensor_sf.view(torch.float8_e4m3fn)

    if swizzle:
        tensor_sf = convert_swizzled_to_linear(  # noqa: E501
            tensor_sf, tensor_f32.size(0), k, block_size
        )

    if is_3d:
        tensor_sf = tensor_sf.reshape(dim0, m, k // block_size)
    tensor_sf_dtype = tensor_sf.to(torch.float32) * global_scale

    if is_3d:
        tensor_f32 = tensor_f32.reshape(dim0, m, -1, block_size)

    # scale the tensor
    out = tensor_f32 * tensor_sf_dtype.unsqueeze(-1)
    out = out.reshape(*out.shape[:-2], -1)

    return out.to(dtype)


def get_reciprocal(x):
    if isinstance(x, torch.Tensor):
        # torch.where yields operation not permitted when stream is capturing.
        return 1.0 / (x + (x == 0) * 1e8)
    elif isinstance(x, (float, int)):
        return 0.0 if x == 0 else 1.0 / x
    else:
        raise TypeError("Input must be a float, int, or a torch.Tensor.")


def cast_to_fp4(x):
    sign = torch.sign(x)
    x = torch.abs(x)
    x[(x >= 0.0) & (x <= 0.25)] = 0.0
    x[(x > 0.25) & (x < 0.75)] = 0.5
    x[(x >= 0.75) & (x <= 1.25)] = 1.0
    x[(x > 1.25) & (x < 1.75)] = 1.5
    x[(x >= 1.75) & (x <= 2.5)] = 2.0
    x[(x > 2.5) & (x < 3.5)] = 3.0
    x[(x >= 3.5) & (x <= 5.0)] = 4.0
    x[x > 5.0] = 6.0
    return x * sign


def ref_nvfp4_quant(x, global_scale, block_size):
    assert global_scale.dtype == torch.float32
    assert x.ndim == 2
    m, n = x.shape
    x = torch.reshape(x, (m, n // block_size, block_size))
    vec_max = torch.max(torch.abs(x), dim=-1, keepdim=True)[0].to(torch.float32)
    scale = global_scale * (vec_max * FLOAT4_E2M1_MAX_RECIPROCAL)
    scale = torch.clamp(scale, max=448, min=-448)
    scale = scale.to(torch.float8_e4m3fn).to(torch.float32)
    output_scale = get_reciprocal(scale * get_reciprocal(global_scale))

    scaled_x = x.to(torch.float32) * output_scale
    clipped_x = torch.clamp(scaled_x, -6.0, 6.0).reshape(m, n)
    # both outputs are float32
    return cast_to_fp4(clipped_x), scale.squeeze(-1)


def ref_nvfp4_quant_dequant(
    x: torch.Tensor, global_scale: torch.Tensor, block_size: int
) -> torch.Tensor:
    """
    NVFP4 quantize-dequantize operation.

    `global_scale` is expected to have a single element.
    """
    if current_platform.is_cuda_alike():
        return _triton_nvfp4_quant_dequant(x, global_scale, block_size)

    x_m, x_k = x.shape
    output_dtype = x.dtype

    # quantize input to (FP4 and interleaved block scale)
    x_fp4, x_blockscale = ref_nvfp4_quant(x, global_scale, block_size)

    # dequantize input
    x_fp4 = x_fp4.reshape(x_m, x_k // block_size, block_size)
    x_blockscale = x_blockscale.unsqueeze(-1) / global_scale
    x_dq = (x_fp4 * x_blockscale).reshape(x_m, x_k).to(output_dtype)

    return x_dq


def run_nvfp4_emulations(
    x: torch.Tensor,
    input_global_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_swizzled: torch.Tensor,
    weight_global_scale: torch.Tensor,
    swizzle: bool | None = True,
):
    output_dtype = x.dtype
    group_size = 16

    x_dq = ref_nvfp4_quant_dequant(x, input_global_scale, block_size=group_size)

    # dequantize weight
    w_fp4 = weight.data.view(torch.uint8)
    w_dq = dequantize_to_dtype(
        w_fp4,
        weight_scale_swizzled.data,
        weight_global_scale,
        output_dtype,
        group_size,
        swizzle=swizzle,
    )

    # matmul
    out = torch.matmul(x_dq, w_dq.t())
    return out
