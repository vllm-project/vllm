# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.logger import init_logger
logger = init_logger(__name__)


def mxfp8_e4m3_quantize_python(data: torch.Tensor):
    assert len(data.shape) == 2, "Only 2d input tensor is supported"
    block_size1 = 32
    block_size0 = 1
    shape_before_padding = data.shape
    # pad data to make its shape a multiple of weight_block_size with the last element of data
    assert data.shape[1] % block_size1 == 0 and data.shape[0] % block_size0 == 0, "Data shape must be a multiple of tile size [1, 32]"

    # FP8
    max_dtype = torch.finfo(torch.float8_e4m3fn).max
    shape_after_padding = data.shape
    blk_m, blk_n = data.shape[0] // block_size0, data.shape[1] // block_size1
    data = data.reshape(blk_m, block_size0, blk_n, block_size1)
    # Permute to (BLK_M, BLK_N, BLOCK_SIZE_M, BLOCK_SIZE_N)
    data = data.permute(0, 2, 1, 3)
    # Flatten to (BLK_M, BLK_N, BLOCK_SIZE_M * BLOCK_SIZE_N)
    data = data.to(torch.float32).contiguous().flatten(start_dim=2)
    # Calculate max absolute value per block
    max_abs = torch.amax(torch.abs(data), dim=-1, keepdim=True)

    # Calculate scales
    descale = max_abs / max_dtype
    exponent = torch.ceil(torch.log2(descale))
    # Post process exponent to be in range of -127 to 127 and to be E8M0 biased
    exponent = torch.clamp(exponent, min=-127, max=127) + 127
    # Convert to uint8 container
    exponent = exponent.to(torch.uint8)
    # Calculate descale_fp to apply to data_hp
    scale_fp = torch.where(
        # If exponent is 0, descale_fp is 1.0 rather than 2^127
        exponent == 0,
        1.0,
        torch.exp2(127 - exponent.to(torch.float32)),
    )
    exponent = exponent.reshape(blk_m, blk_n)

    # Scale and saturate cast the data elements to max of target dtype
    data_lp = torch.clamp(data * scale_fp, min=-1 * max_dtype, max=max_dtype)

    fp_data = data_lp.to(torch.float8_e4m3fn)

    # (BLK_M, BLK_N, BLOCK_SIZE_M * BLOCK_SIZE_N) to (M, N)
    fp_data = (
        fp_data.reshape(blk_m, blk_n, block_size0, block_size1)
        .permute(0, 2, 1, 3)
        .reshape(shape_after_padding)
    )

    # remove the padding
    if data.shape != shape_before_padding:
        fp_data = fp_data[: shape_before_padding[0], : shape_before_padding[1]]

    # Convert to target format, but still in original precision container
    return fp_data, exponent


def mxfp8_e4m3_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        from flashinfer import mxfp8_quantize as mxfp8_e4m3_quantize
    except ImportError as err:
        raise ImportError(
            "The package `flashinfer` is required to do "
            "MX-FP8 quantization. Please install it with"
            "`pip install flashinfer`"
        ) from err

    x_q, x_scales = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=False)
    if x_scales.ndim == 1:
        x_scales = x_scales.view(x.size(0), -1)
    return x_q, x_scales


def _cast_mxfp8_scales_to_bf16(scales: torch.Tensor) -> torch.Tensor:
    """
    Cast MXFP8 scales from uint8 to BF16.
    
    The scales are stored in uint8 format and need to be converted to BF16
    by left-shifting by 7 bits (to form the exponent) and reinterpreting
    as bfloat16.
    
    Args:
        scales: uint8 tensor containing MXFP8 scales
        
    Returns:
        BF16 tensor with the converted scales
    """
    return (scales.to(torch.int16) << 7).view(torch.bfloat16)


def dequant_mxfp8_to_bf16(
    x: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    """
    Dequantize MXFP8 tensor to BF16.
    
    Args:
        x: FP8 E4M3 tensor to dequantize
        scales: uint8 tensor containing MXFP8 scales
        
    Returns:
        BF16 dequantized tensor
    """
    scales_bf16 = _cast_mxfp8_scales_to_bf16(scales)
    # Repeat scales along the last dimension to match the block size
    scales_expanded = scales_bf16.reshape(*x.shape[:-1], -1).repeat_interleave(32, dim=-1)
    return x.to(torch.bfloat16) * scales_expanded
