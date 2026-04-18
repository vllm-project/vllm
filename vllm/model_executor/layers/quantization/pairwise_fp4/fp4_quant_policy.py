# SPDX-License-Identifier: Apache-2.0

import torch

from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    FLOAT4_E2M1_MAX,
    ref_nvfp4_quant,
)

# FP4 E2M1 encoding table: float value → 4-bit nibble (magnitude only).
# Sign is encoded in bit 3.
_FP4_ENCODE = {
    0.0: 0b0000,
    0.5: 0b0001,
    1.0: 0b0010,
    1.5: 0b0011,
    2.0: 0b0100,
    3.0: 0b0101,
    4.0: 0b0110,
    6.0: 0b0111,
}

# FP8 E4M3 max value (torch.finfo is not available for float8_e4m3fn,
# so we hardcode the well-known constant).
_FP8_E4M3_MAX: float = 448.0


def estimate_global_scale(
    weight: torch.Tensor,
    block_size: int = 16,
) -> torch.Tensor:
    """Compute a global scale for NVFP4 quantization.

    global_scale = 1.0 / (max(|weight|) / FP4_MAX / FP8_MAX)
    """
    amax = weight.abs().max().float()
    if amax == 0.0:
        return torch.tensor(1.0, dtype=torch.float32)
    # NVFP4 convention: global_scale maps weight range into FP4*FP8 range.
    gs = _FP8_E4M3_MAX * FLOAT4_E2M1_MAX / amax
    return gs.clamp(min=torch.finfo(torch.float32).tiny).to(torch.float32)


def quantize_weight_to_fp4(
    weight: torch.Tensor,
    global_scale: torch.Tensor,
    block_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a BF16/FP16/FP32 weight matrix to packed FP4.

    Returns (packed_weight, block_scales_fp8, global_scale).
    """
    if weight.ndim != 2:
        raise ValueError(
            f"weight must be 2-D, got {weight.ndim}-D"
        )
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    if weight.shape[1] % (block_size * 2) != 0:
        raise ValueError(
            f"weight.shape[1] ({weight.shape[1]}) must be divisible by "
            f"block_size * 2 ({block_size * 2})"
        )

    # ref_nvfp4_quant expects float32 global_scale and 2-D weight.
    fp4_float, block_scales_f32 = ref_nvfp4_quant(
        weight, global_scale.float(), block_size
    )

    packed_weight = pack_fp4_to_uint8(fp4_float)
    block_scales_fp8 = block_scales_f32.to(torch.float8_e4m3fn)

    return packed_weight, block_scales_fp8, global_scale


def pack_fp4_to_uint8(
    fp4_float: torch.Tensor,
) -> torch.Tensor:
    """Pack float tensor with discrete FP4 E2M1 values into uint8.

    Two consecutive values along the last dimension are packed per byte:
      byte = encode(vals[..., 2k]) | (encode(vals[..., 2k+1]) << 4)
    This matches vLLM's ``break_fp4_bytes`` unpacking convention (low
    nibble first, high nibble second).
    """
    # Build the lookup table on the same device as fp4_float.
    # Map fp4 float values (both positive and negative) to 4-bit nibbles.
    # We use a round-trip through a dict for clarity; the tensor is small.
    device = fp4_float.device

    # Positive entries
    lut_pos = torch.zeros(13, dtype=torch.uint8, device=device)  # index by int
    for fval, nibble in _FP4_ENCODE.items():
        # We'll use (round(fval * 2)) as an integer key: 0,1,2,3,4,6,8,12
        key = round(fval * 2)
        lut_pos[key] = nibble

    # Encode: separate sign and magnitude
    sign = (fp4_float < 0).to(torch.uint8) << 3  # 0b1000 for negatives
    mag = fp4_float.abs()

    # Convert magnitudes to integer keys: round(mag * 2)
    mag_keys = (mag * 2).round().long()
    # Lookup nibbles for magnitudes
    nibbles = lut_pos[mag_keys] | sign

    # Pack pairs along the last dimension
    *leading, last = nibbles.shape
    assert last % 2 == 0, f"last dim ({last}) must be even"
    nibbles = nibbles.reshape(*leading, last // 2, 2)
    low = nibbles[..., 0]
    high = nibbles[..., 1]
    packed = (low | (high << 4)).to(torch.uint8)
    return packed
