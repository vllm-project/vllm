# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 quantization utilities.

Contains both flashinfer-based quantization (for fused_moe) and
torch.compile-compatible implementation based on torchao's mx_formats.
"""

import torch

# MXFP8 block size (number of elements sharing one scale)
MXFP8_BLOCK_SIZE = 32


def _ceil_div(a: int, b: int) -> int:
    """Ceiling division."""
    return (a + b - 1) // b


def mxfp8_e4m3_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to MXFP8 using flashinfer.

    Used by fused_moe. For torch.compile-compatible version, use mxfp8_quantize.
    """
    try:
        from flashinfer import mxfp8_quantize as flashinfer_mxfp8_quantize
    except ImportError as err:
        raise ImportError(
            "The package `flashinfer` is required to do "
            "MX-FP8 quantization. Please install it with"
            "`pip install flashinfer`"
        ) from err

    x_q, x_scales = flashinfer_mxfp8_quantize(x, is_sf_swizzled_layout=False)
    if x_scales.ndim == 1:
        x_scales = x_scales.view(x.size(0), -1)
    return x_q, x_scales


# from https://github.com/pytorch/ao/blob/21acb9c63f3ae01365821475df7cd2a2a96a5eb8/torchao/prototype/mx_formats/utils.py#L32
def _to_blocked(input_matrix: torch.Tensor) -> torch.Tensor:
    """
    Rearrange scale matrix to cuBLAS blocked layout.

    See: https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout

    Args:
        input_matrix: Input tensor of shape (H, W)

    Returns:
        Rearranged flattened tensor for cuBLAS consumption.
    """
    rows, cols = input_matrix.shape
    n_row_blocks = _ceil_div(rows, 128)
    n_col_blocks = _ceil_div(cols, 4)

    # Calculate the padded shape
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    padded = input_matrix
    # Always create padded tensor when compiling to avoid dynamic shape issues
    # (This follows torchao's pattern for vLLM compatibility)
    if torch.compiler.is_compiling() or (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros(
            (padded_rows, padded_cols),
            device=input_matrix.device,
            dtype=input_matrix.dtype,
        )
        padded[:rows, :cols] = input_matrix

    # Rearrange the blocks
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten()


# torch.float8_e8m0fnu scale format constants
E8M0_EXPONENT_BIAS = 127

# torch.float8_e4m3fn constants
F8E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0


# from https://github.com/pytorch/ao/blob/c6d21f9b82819a7e3546a267eafb15a5b8cf3951/torchao/prototype/mx_formats/mx_tensor.py#L96
def _to_mx_rceil(
    data_hp: torch.Tensor,
    max_abs: torch.Tensor,
    max_pos: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    MXFP scale factor derivation method described in
    https://docs.nvidia.com/cuda/cublas/#d-block-quantization

    Args:
        data_hp: High precision data.
        max_abs: Maximum absolute value for data_hp along specified
            dimension/block_size.
        max_pos: The maximum value of the low precision data type.

    Returns:
        exponent: The biased exponent with torch.float8_e8m0fnu dtype in uint8
            container.
        data_lp: The targeted low precision data, in high precision container
            (requires cast to low precision data type).
    """
    descale = max_abs / max_pos
    exponent = torch.where(
        torch.isnan(descale),
        0xFF,  # Handle biased exponent for nan
        (
            torch.clamp(
                torch.ceil(torch.log2(descale)),
                min=-E8M0_EXPONENT_BIAS,
                max=E8M0_EXPONENT_BIAS,
            )
            + E8M0_EXPONENT_BIAS
        ).to(torch.uint8),
    )

    descale_fp = torch.where(
        exponent == 0, 1.0, torch.exp2(E8M0_EXPONENT_BIAS - exponent.to(torch.float32))
    )

    # scale and saturated cast the data elements to max of target dtype
    data_lp = torch.clamp(data_hp * descale_fp, min=-1 * max_pos, max=max_pos)
    return exponent, data_lp


# from https://github.com/pytorch/ao/blob/21acb9c63f3ae01365821475df7cd2a2a96a5eb8/torchao/prototype/mx_formats/utils.py#L122
def _hp_data_dims_to_swizzled_scale_dims_mx(
    hp_data_M: int,
    hp_data_K: int,
) -> tuple[int, int]:
    """
    Given the M and K dimensions of a high precision contiguous tensor,
    returns a 2d tuple of the dims of the swizzled mx scale corresponding to
    that tensor.
    """
    # a 128x128 unpacked or 128x64 packed qdata tile corresponds
    # to a swizzled 32x16 scale tile
    scale_M = _ceil_div(hp_data_M, 128) * 32
    scale_K = _ceil_div(hp_data_K, 128) * 16
    return scale_M, scale_K


# from https://github.com/pytorch/ao/blob/21acb9c63f3ae01365821475df7cd2a2a96a5eb8/torchao/prototype/mx_formats/mx_tensor.py#L144
# with the `ScaleCalculationMode.RCEIL` branch
# and the `is_swizzled_scales` branch
def mxfp8_quantize(
    data: torch.Tensor,
    block_size: int = MXFP8_BLOCK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor to MXFP8 format.

    Quantizes to torch.float8_e4m3fn with torch.float8_e8m0fnu block scales.
    This is a torch.compile-compatible implementation based on the OCP
    Microscaling spec and torchao's mx_formats.

    Args:
        data: Input tensor of shape [M, K] where K is divisible by block_size.
        block_size: Number of elements per scale block (default: 32).

    Returns:
        Tuple of (quantized_data, scales) where:
        - quantized_data: torch.float8_e4m3fn tensor of same shape as input
        - scales: torch.float8_e8m0fnu scale tensor in cuBLAS blocked (swizzled)
          layout with shape (scale_M, scale_K) where scale_M and scale_K are
          computed by _hp_data_dims_to_swizzled_scale_dims_mx(M, K)
    """
    assert data.dtype == torch.bfloat16, f"Expected bfloat16, got {data.dtype}"
    assert data.shape[-1] % block_size == 0, (
        f"Last dim {data.shape[-1]} must be divisible by block_size {block_size}"
    )
    assert data.is_contiguous(), "Input must be contiguous"

    orig_shape = data.shape
    M, K = orig_shape[-2], orig_shape[-1]

    # Reshape to [..., num_blocks, block_size]
    data = data.reshape(*orig_shape[:-1], orig_shape[-1] // block_size, block_size)

    # Find max absolute value per block
    max_abs = torch.amax(torch.abs(data), dim=-1, keepdim=True)

    # Cast to float32 for scale calculation
    data = data.to(torch.float32)
    max_abs = max_abs.to(torch.float32)

    # Use rceil method for scale derivation
    scale_e8m0, data_scaled = _to_mx_rceil(data, max_abs, F8E4M3_MAX)

    # Cast to torch.float8_e4m3fn and reshape back to original shape
    data_fp8 = data_scaled.to(torch.float8_e4m3fn).reshape(orig_shape)

    # Reshape scales: remove the keepdim and get [..., num_blocks]
    # Convert to torch.float8_e8m0fnu format (view uint8 as e8m0)
    scale_e8m0 = scale_e8m0.squeeze(-1).view(torch.float8_e8m0fnu)

    # Convert scale to cuBLAS blocked (swizzled) layout for best performance
    # This matches torchao's is_swizzled_scales=True behavior
    scale_shape = (M, K // block_size)
    scale_blocked = _to_blocked(scale_e8m0.view(scale_shape))

    # Reshape to swizzled scale dimensions (matches torchao)
    # Note: We only support 2D inputs [M, K] currently, so leading_dims is empty.
    # For batched inputs (e.g., [B, M, K] for MoE), torchao does:
    #   leading_dims = orig_shape[:-2]
    #   scale.view(*leading_dims, scale_M, scale_K)
    # We will add leading_dims support when we add MoE quantization.
    scale_M, scale_K = _hp_data_dims_to_swizzled_scale_dims_mx(M, K)
    scale_e8m0_blocked = scale_blocked.view(scale_M, scale_K)

    return data_fp8, scale_e8m0_blocked
