# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

# MXFP8 constants
MXFP8_VALUE_DTYPE = torch.float8_e4m3fn
MXFP8_SCALE_DTYPE = torch.uint8
MXFP8_BLOCK_SIZE = 32


def mxfp8_e4m3_quantize(
    x: torch.Tensor, is_sf_swizzled_layout: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        from flashinfer import mxfp8_quantize as mxfp8_e4m3_quantize
    except ImportError as err:
        raise ImportError(
            "The package `flashinfer` is required to do "
            "MX-FP8 quantization. Please install it with"
            "`pip install flashinfer`"
        ) from err

    x_q, x_scales = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=is_sf_swizzled_layout)
    if x_scales.ndim == 1:
        if is_sf_swizzled_layout:
            # TODO: check this, maybe not required?
            # When swizzled, scales are padded: M to multiple of 128, K to multiple of 4
            # We must use the padded dimensions, not the original input dimensions
            def _round_up(val: int, mult: int) -> int:
                return (val + mult - 1) // mult * mult

            M = x.size(0)
            K = x.size(-1) // MXFP8_BLOCK_SIZE
            M_padded = _round_up(M, 128)
            K_padded = _round_up(K, 4)
            x_scales = x_scales.view(M_padded, K_padded)
        else:
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


def dequant_mxfp8_to_bf16(x: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
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
    scales_expanded = scales_bf16.reshape(*x.shape[:-1], -1).repeat_interleave(
        MXFP8_BLOCK_SIZE, dim=-1
    )
    return x.to(torch.bfloat16) * scales_expanded


def mxfp8_e4m3_quantize_fake(
    x: torch.Tensor, is_sf_swizzled_layout: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fake implementation for torch.compile tracing.
    Returns empty tensors with the correct shapes and dtypes.
    """
    # FP8 quantized data has same shape as input
    fp_data = torch.empty_like(x, dtype=MXFP8_VALUE_DTYPE)

    # Compute scale shape: one scale per block of 32 elements along last dim
    block_size = MXFP8_BLOCK_SIZE

    if x.ndim == 2:
        M, N = x.shape
        K = (N + block_size - 1) // block_size
        if is_sf_swizzled_layout:
            # When swizzled, scales are padded: M to multiple of 128, K to multiple of 4
            M_padded = ((M + 127) // 128) * 128
            K_padded = ((K + 3) // 4) * 4
            scales = torch.empty(
                (M_padded, K_padded), dtype=MXFP8_SCALE_DTYPE, device=x.device
            )
        else:
            scales = torch.empty((M, K), dtype=MXFP8_SCALE_DTYPE, device=x.device)
    elif x.ndim == 3:
        B, M, N = x.shape
        K = (N + block_size - 1) // block_size
        if is_sf_swizzled_layout:
            M_padded = ((M + 127) // 128) * 128
            K_padded = ((K + 3) // 4) * 4
            scales = torch.empty(
                (B, M_padded, K_padded), dtype=MXFP8_SCALE_DTYPE, device=x.device
            )
        else:
            scales = torch.empty((B, M, K), dtype=MXFP8_SCALE_DTYPE, device=x.device)
    else:
        # Fallback for other dimensions
        scale_shape = list(x.shape)
        scale_shape[-1] = (x.shape[-1] + block_size - 1) // block_size
        scales = torch.empty(scale_shape, dtype=MXFP8_SCALE_DTYPE, device=x.device)

    return fp_data, scales


direct_register_custom_op(
    op_name="mxfp8_quantize",
    op_func=mxfp8_e4m3_quantize,
    fake_impl=mxfp8_e4m3_quantize_fake,
)


class Mxfp8LinearOp:
    """
    This class executes a MXFP8 linear layer.
    """

    def __init__(self, use_fallback: bool = False):
        self.use_fallback = use_fallback

    def apply(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_fallback:
            return self._apply_fallback(input, weight, weight_scale, out_dtype, bias)
        return self._apply_scaled_mm(input, weight, weight_scale, out_dtype, bias)

    def _apply_fallback(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fallback implementation using manual dequantization for debugging."""
        # weight_scale comes in as float8_e8m0fnu
        # after process_weights_after_loading
        # It may be padded to [N_padded, K/32] and flattened
        # Convert back to uint8 for dequantization
        weight_scale_uint8 = weight_scale.view(MXFP8_SCALE_DTYPE)

        out_features, in_features = weight.shape
        # Number of scale blocks along K dimension
        scale_k = in_features // MXFP8_BLOCK_SIZE

        # Compute padded dimensions (same logic as process_weights_after_loading)
        out_features_padded = (out_features + 127) // 128 * 128

        # Reshape to padded 2D, then slice to get original shape
        weight_scale_2d_padded = weight_scale_uint8.view(out_features_padded, scale_k)
        weight_scale_2d = weight_scale_2d_padded[:out_features, :]

        # Dequantize weight to bf16
        weight_bf16 = dequant_mxfp8_to_bf16(weight, weight_scale_2d)

        # Standard linear operation
        output = torch.nn.functional.linear(input, weight_bf16, bias)
        return output.to(out_dtype)

    def _apply_scaled_mm(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Weights should be mxfp8, weight_scale
        # is pre-processed to float8_e8m0fnu
        assert weight.dtype == MXFP8_VALUE_DTYPE
        # weight_scale is already pre-processed
        # in process_weights_after_loading
        assert weight_scale.dtype == torch.float8_e8m0fnu

        assert out_dtype == torch.bfloat16, "Only bfloat16 is supported as out_dtype"

        # From bf16 to mxfp8
        assert input.dtype == torch.bfloat16

        swizzled = True
        input_mxfp8, input_mxfp8_scales = torch.ops.vllm.mxfp8_quantize(input, swizzled)

        # For Blockwise 1x32 scaling, a and b should be float8,
        # scales should be float8_e8m0fnu and 1D contiguous
        # Use .view() to reinterpret uint8 bytes as float8_e8m0fnu
        # (not .to() which converts values)
        input_mxfp8_scales = input_mxfp8_scales.view(torch.float8_e8m0fnu).flatten()

        output = torch._scaled_mm(
            input_mxfp8,
            weight.t(),
            scale_a=input_mxfp8_scales,
            scale_b=weight_scale,
            out_dtype=out_dtype,
            use_fast_accum=True,
        )

        if bias is not None:
            output = output + bias

        return output
