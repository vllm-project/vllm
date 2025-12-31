# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum

import torch

from vllm.logger import init_logger
from vllm.utils import flashinfer as vllm_flashinfer
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


class Mxfp8Backend(Enum):
    TORCH = "torch"
    FLASHINFER_CUTLASS = "flashinfer-cutlass"


# MXFP8 constants
MXFP8_VALUE_DTYPE = torch.float8_e4m3fn
MXFP8_SCALE_DTYPE = torch.uint8
MXFP8_BLOCK_SIZE = 32

# Minimum dimension size for F8_128x4 block scaling layout
MXFP8_BMM_MIN_DIM = 128


def swizzle_mxfp8_scale(sf: torch.Tensor, M: int, K: int) -> torch.Tensor:
    """Swizzle MXFP8 scales from row-major 2D to F8_128x4 layout."""
    scaling_vector_size = MXFP8_BLOCK_SIZE  # 32 for MXFP8
    factor = scaling_vector_size * 4  # 128

    num_m_tiles = (M + 127) // 128
    num_k_tiles = (K + factor - 1) // factor

    m_padded = num_m_tiles * 128
    k_scale_padded = num_k_tiles * 4

    scale_cols = K // scaling_vector_size
    sf_padded = torch.zeros(
        (m_padded, k_scale_padded), dtype=sf.dtype, device=sf.device
    )
    sf_padded[:M, :scale_cols] = sf

    sf_reshaped = sf_padded.view(num_m_tiles, 4, 32, num_k_tiles, 4)

    sf_swizzled = sf_reshaped.transpose(1, 3)

    return sf_swizzled.contiguous().view(-1)


def unswizzle_mxfp8_scale(sf: torch.Tensor, M: int, K: int) -> torch.Tensor:
    """Unswizzle MXFP8 scales from F8_128x4 to row-major 2D layout."""
    scaling_vector_size = MXFP8_BLOCK_SIZE  # 32 for MXFP8
    factor = scaling_vector_size * 4  # 128

    num_m_tiles = (M + 127) // 128
    num_k_tiles = (K + factor - 1) // factor

    sf_reshaped = sf.view(num_m_tiles, num_k_tiles, 32, 4, 4)

    sf_unswizzle = sf_reshaped.transpose(1, 3)
    sf_unswizzle = sf_unswizzle.reshape(num_m_tiles * 32 * 4, num_k_tiles * 4)

    scale_cols = K // scaling_vector_size
    sf_unswizzle_sliced = sf_unswizzle[:M, :scale_cols]

    return sf_unswizzle_sliced.contiguous()


def _mxfp8_e4m3_quantize_impl(
    x: torch.Tensor, is_sf_swizzled_layout: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    from flashinfer import mxfp8_quantize as flashinfer_mxfp8_quantize

    x_q, x_scales = flashinfer_mxfp8_quantize(
        x, is_sf_swizzled_layout=is_sf_swizzled_layout
    )
    if x_scales.ndim == 1 and x.ndim == 2 and not is_sf_swizzled_layout:
        x_scales = x_scales.view(x.size(0), -1)
    return x_q, x_scales


def mxfp8_e4m3_quantize(
    x: torch.Tensor, is_sf_swizzled_layout: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.mxfp8_quantize(x, is_sf_swizzled_layout)


def dequant_mxfp8_to_bf16(x: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize MXFP8 tensor to BF16."""
    x_float = x.to(torch.float32)

    num_blocks = x.shape[-1] // MXFP8_BLOCK_SIZE
    x_blocked = x_float.view(*x.shape[:-1], num_blocks, MXFP8_BLOCK_SIZE)

    descale = torch.exp2(scales.to(torch.float32) - 127.0)

    dequantized = x_blocked * descale.unsqueeze(-1)

    dequantized = dequantized.view(*x.shape)

    return dequantized.to(torch.bfloat16)


def mxfp8_e4m3_quantize_fake(
    x: torch.Tensor, is_sf_swizzled_layout: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fake implementation for torch.compile tracing."""
    fp_data = torch.empty_like(x, dtype=MXFP8_VALUE_DTYPE)

    block_size = MXFP8_BLOCK_SIZE

    if x.ndim == 2:
        M, N = x.shape
        K = (N + block_size - 1) // block_size
        if is_sf_swizzled_layout:
            M_padded = ((M + 127) // 128) * 128
            K_padded = ((K + 3) // 4) * 4
            scales = torch.empty(
                M_padded * K_padded, dtype=MXFP8_SCALE_DTYPE, device=x.device
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
                B * M_padded * K_padded, dtype=MXFP8_SCALE_DTYPE, device=x.device
            )
        else:
            scales = torch.empty((B, M, K), dtype=MXFP8_SCALE_DTYPE, device=x.device)
    else:
        scale_shape = list(x.shape)
        scale_shape[-1] = (x.shape[-1] + block_size - 1) // block_size
        scales = torch.empty(scale_shape, dtype=MXFP8_SCALE_DTYPE, device=x.device)

    return fp_data, scales


direct_register_custom_op(
    op_name="mxfp8_quantize",
    op_func=_mxfp8_e4m3_quantize_impl,
    fake_impl=mxfp8_e4m3_quantize_fake,
)


class Mxfp8LinearOp:
    def __init__(self, backend: Mxfp8Backend):
        if backend not in Mxfp8Backend:
            raise ValueError(f"Unsupported backend: {backend}")

        self.backend = backend

    def apply(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype,
        bias: torch.Tensor | None = None,
        # Flashinfer-specific metadata (set by process_weights_after_loading)
        out_features: int | None = None,
        in_features: int | None = None,
        weight_scale_2d: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.backend == Mxfp8Backend.FLASHINFER_CUTLASS:
            return self._apply_flashinfer(
                input,
                weight,
                weight_scale,
                out_dtype,
                bias,
                out_features,
                in_features,
                weight_scale_2d,
            )
        return self._apply_torch(input, weight, weight_scale, out_dtype, bias)

    def _apply_torch(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out_features, in_features = weight.shape
        scale_k = in_features // MXFP8_BLOCK_SIZE

        if weight_scale.ndim == 2:
            weight_scale_2d = weight_scale
            if weight_scale_2d.dtype != MXFP8_SCALE_DTYPE:
                weight_scale_2d = weight_scale_2d.view(MXFP8_SCALE_DTYPE)
        else:
            weight_scale_uint8 = weight_scale.view(MXFP8_SCALE_DTYPE)
            out_features_padded = (out_features + 127) // 128 * 128
            weight_scale_2d_padded = weight_scale_uint8.view(
                out_features_padded, scale_k
            )
            weight_scale_2d = weight_scale_2d_padded[:out_features, :]

        weight_bf16 = dequant_mxfp8_to_bf16(weight, weight_scale_2d)

        output = torch.nn.functional.linear(input, weight_bf16, bias)
        return output.to(out_dtype)

    def _apply_flashinfer(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype,
        bias: torch.Tensor | None = None,
        out_features: int | None = None,
        in_features: int | None = None,
        weight_scale_2d: torch.Tensor | None = None,
    ) -> torch.Tensor:
        N, K = weight.shape
        if out_features is not None:
            N = out_features
        if in_features is not None:
            K = in_features

        input_shape = input.shape
        input_2d = input.view(-1, K)
        M_orig = input_2d.shape[0]

        min_dim = MXFP8_BMM_MIN_DIM
        assert min_dim <= K, (
            f"mm_mxfp8 requires K >= {min_dim}, got K={K}. "
            f"in_features is too small for mm_mxfp8."
        )
        assert K % MXFP8_BLOCK_SIZE == 0, (
            f"mm_mxfp8 requires K to be divisible by {MXFP8_BLOCK_SIZE}, got K={K}."
        )
        assert min_dim <= N, (
            f"mm_mxfp8 requires N >= {min_dim}, got N={N}. "
            f"out_features is too small for mm_mxfp8."
        )

        M_padded = ((M_orig + min_dim - 1) // min_dim) * min_dim
        if M_padded != M_orig:
            pad_rows = M_padded - M_orig
            input_2d = torch.nn.functional.pad(input_2d, (0, 0, 0, pad_rows))

        input_mxfp8, input_scale = mxfp8_e4m3_quantize(
            input_2d,
            is_sf_swizzled_layout=True,  # Swizzled for best accuracy
        )

        if not weight.is_contiguous():
            weight = weight.contiguous()

        output = vllm_flashinfer.mm_mxfp8(
            input_mxfp8,
            weight.t(),
            input_scale,
            weight_scale,
            out_dtype=out_dtype,
            backend="cutlass",
        )

        if M_padded != M_orig:
            output = output[:M_orig, :]

        if bias is not None:
            output = output + bias

        output_shape = (*input_shape[:-1], N)
        return output.view(output_shape)
