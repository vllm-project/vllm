# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import Enum

import torch

import vllm.envs as envs
from vllm._custom_ops import (
    cutlass_scaled_fp4_mm,
    cutlass_scaled_mm_supports_fp4,
    scaled_fp4_quant,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    is_fp4_marlin_supported,
    prepare_fp4_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    run_nvfp4_emulations,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm, has_flashinfer
from vllm.utils.math_utils import round_up

logger = init_logger(__name__)


class NvFp4LinearBackend(Enum):
    VLLM_CUTLASS = "cutlass"
    FLASHINFER_CUTLASS = "flashinfer-cutlass"
    FLASHINFER_TRTLLM = "flashinfer-trtllm"
    FLASHINFER_CUDNN = "flashinfer-cudnn"
    FBGEMM = "fbgemm"
    MARLIN = "marlin"
    EMULATION = "emulation"


def select_nvfp4_linear_backend() -> NvFp4LinearBackend:
    """
    Select the best available NVFP4 GEMM backend based on environment
    configuration and platform capabilities.
    """
    backend: NvFp4LinearBackend | None = None

    if envs.VLLM_USE_FBGEMM:
        try:
            import fbgemm_gpu  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Backend fbgemm requires fbgemm.f4f4bf16 operator, "
                "Please install with: pip install fbgemm-gpu-genai"
            ) from exc
        backend = NvFp4LinearBackend.FBGEMM
    elif envs.VLLM_USE_NVFP4_CT_EMULATIONS:
        backend = NvFp4LinearBackend.EMULATION
    elif envs.VLLM_NVFP4_GEMM_BACKEND is None:
        # Auto-select best available backend
        if current_platform.has_device_capability(100) and has_flashinfer():
            backend = NvFp4LinearBackend.FLASHINFER_CUTLASS
        elif cutlass_fp4_supported():
            backend = NvFp4LinearBackend.VLLM_CUTLASS
        elif is_fp4_marlin_supported():
            backend = NvFp4LinearBackend.MARLIN
    else:
        backend = NvFp4LinearBackend(envs.VLLM_NVFP4_GEMM_BACKEND)

    # Validate that the backend is supported
    if backend in (
        NvFp4LinearBackend.FLASHINFER_CUTLASS,
        NvFp4LinearBackend.FLASHINFER_TRTLLM,
        NvFp4LinearBackend.FLASHINFER_CUDNN,
    ):
        assert has_flashinfer(), f"FlashInfer is required for {backend}"
    elif backend == NvFp4LinearBackend.VLLM_CUTLASS:
        assert cutlass_fp4_supported(), f"Cutlass is required for {backend}"
    elif backend == NvFp4LinearBackend.MARLIN:
        assert is_fp4_marlin_supported(), f"Marlin is required for {backend}"
    elif backend is None:
        raise ValueError(
            f"No NVFP4 GEMM backend selected, "
            f"available backends: {list(NvFp4LinearBackend)}"
        )

    logger.info_once(f"Using {backend} for NVFP4 GEMM")
    return backend


def prepare_weights_for_nvfp4_flashinfer_trtllm(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare weights and scales for FlashInfer TRTLLM FP4 GEMM."""
    from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a

    epilogue_tile_m = 128
    shuffled_weight = shuffle_matrix_a(weight.view(torch.uint8), epilogue_tile_m)
    shuffled_weight_scale = (
        shuffle_matrix_sf_a(weight_scale.view(torch.uint8), epilogue_tile_m)
        .reshape(weight_scale.shape)
        .view(torch.float8_e4m3fn)
    )

    return shuffled_weight, shuffled_weight_scale


def prepare_weights_for_nvfp4_cutlass(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Prepare weights and scales for CUTLASS/FlashInfer-CUTLASS FP4 GEMM.
    This involves padding weights for alignment (K and N divisible by 32)
    """
    swizzled_weight_scale = swizzle_blockscale(weight_scale)
    padded_weight, weights_padding_cols = pad_nvfp4_weight_for_cutlass(weight)
    return padded_weight, swizzled_weight_scale, weights_padding_cols


def prepare_weights_for_nvfp4_fbgemm(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare weights and scales for FBGEMM FP4 GEMM."""
    swizzled_weight_scale = swizzle_blockscale(weight_scale)
    swizzled_weight_scale = swizzled_weight_scale.view(-1).view(torch.uint8)
    return weight, swizzled_weight_scale


def convert_to_nvfp4_linear_kernel_format(
    backend: NvFp4LinearBackend,
    layer: torch.nn.Module,
) -> None:
    """Convert layer to NVFP4 linear kernel format."""

    assert layer.weight_scale.dtype == torch.float8_e4m3fn, (
        "Weight Block scale must be represented as FP8-E4M3"
    )

    # Default to no padding
    layer.weights_padding_cols = 0

    if backend == NvFp4LinearBackend.MARLIN:
        prepare_fp4_layer_for_marlin(layer)
    elif backend == NvFp4LinearBackend.FLASHINFER_TRTLLM:
        weight, weight_scale = prepare_weights_for_nvfp4_flashinfer_trtllm(
            layer.weight.data, layer.weight_scale.data
        )
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
    elif backend == NvFp4LinearBackend.FBGEMM:
        weight, weight_scale = prepare_weights_for_nvfp4_fbgemm(
            layer.weight.data, layer.weight_scale.data
        )
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
    elif backend in (
        NvFp4LinearBackend.VLLM_CUTLASS,
        NvFp4LinearBackend.FLASHINFER_CUTLASS,
        NvFp4LinearBackend.FLASHINFER_CUDNN,
    ):
        weight, weight_scale, weights_padding_cols = prepare_weights_for_nvfp4_cutlass(
            layer.weight.data, layer.weight_scale.data
        )
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
        layer.weights_padding_cols = weights_padding_cols


def apply_nvfp4_linear(
    backend: NvFp4LinearBackend,
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Apply NVFP4 linear transformation using the specified backend.
    """
    weight = layer.weight
    weight_scale = layer.weight_scale
    weight_global_scale = layer.weight_global_scale
    input_global_scale_inv = layer.input_global_scale_inv
    alpha = layer.alpha
    output_size = layer.output_size_per_partition
    input_size = layer.input_size_per_partition

    if backend == NvFp4LinearBackend.MARLIN:
        return apply_fp4_marlin_linear(
            input=x,
            weight=weight,
            weight_scale=weight_scale,
            weight_global_scale=weight_global_scale,
            workspace=layer.workspace,
            size_n=output_size,
            size_k=input_size,
            bias=bias,
        )
    elif backend == NvFp4LinearBackend.EMULATION:
        out = run_nvfp4_emulations(
            x=x,
            input_global_scale=input_global_scale_inv,
            weight=weight,
            weight_scale_swizzled=weight_scale,
            weight_global_scale=weight_global_scale,
        )
        if bias is not None:
            out = out + bias
        return out

    output_dtype = x.dtype
    output_shape = [*x.shape[:-1], output_size]

    # Quantize BF16 or FP16 to (FP4 and interleaved block scale)
    x_fp4, x_blockscale = scaled_fp4_quant(
        x, input_global_scale_inv, is_sf_swizzled_layout=True, backend=backend.value
    )

    # Validate dtypes
    assert x_fp4.dtype == torch.uint8
    assert weight.dtype == torch.uint8
    assert x_blockscale.dtype == torch.float8_e4m3fn
    # weight_scale is fp8 for most backends, but uint8 for fbgemm
    assert weight_scale.dtype in (torch.float8_e4m3fn, torch.uint8)
    assert alpha.dtype == torch.float32

    # Pad activations to match weight K-dimension padding
    weights_padding_cols = getattr(layer, "weights_padding_cols", 0)
    x_fp4 = pad_nvfp4_activation_for_cutlass(x_fp4, weights_padding_cols)

    # Prepare args for the matmul
    mm_args = (
        x_fp4,
        weight,
        x_blockscale,
        weight_scale,
        alpha,
        output_dtype,
    )

    # Call the appropriate backend
    if backend.value.startswith("flashinfer-"):
        backend_name = backend.value[len("flashinfer-") :]
        out = flashinfer_scaled_fp4_mm(*mm_args, backend=backend_name)
    elif backend == NvFp4LinearBackend.FBGEMM:
        out = torch.ops.fbgemm.f4f4bf16(
            x_fp4,
            weight,
            x_blockscale.view(-1).view(torch.uint8),
            weight_scale,
            alpha,
            use_mx=False,
        ).to(output_dtype)
    else:
        assert backend == NvFp4LinearBackend.VLLM_CUTLASS
        out = cutlass_scaled_fp4_mm(*mm_args)

    # Slice output to remove N-dimension padding
    out = slice_nvfp4_output(out, output_size)

    if bias is not None:
        out = out + bias

    return out.view(*output_shape)


def swizzle_blockscale(scale: torch.Tensor) -> torch.Tensor:
    """
    Pad and block-interleave the FP4 block-scales so that they match the data
    layout expected by the CUTLASS / FlashInfer kernels.

    Parameters
    ----------
    scale: torch.Tensor

    Returns
    -------
    torch.Tensor
        The swizzled tensor with the same logical shape as *scale*.
    """
    assert scale.dtype == torch.float8_e4m3fn, (
        "swizzle_blockscale expects the input tensor to be in "
        "torch.float8_e4m3fn format."
    )

    scale_ndim = scale.ndim
    if scale_ndim == 2:
        scale = scale.unsqueeze(0)  # (1, M, K)
    assert scale.ndim == 3, "Expected a 2-D or 3-D tensor for block scales."

    B, M, K = scale.shape

    M_padded = round_up(M, 128)
    K_padded = round_up(K, 4)

    padded = torch.zeros(
        (B, M_padded, K_padded), dtype=scale.dtype, device=scale.device
    )
    padded[:B, :M, :K] = scale

    # Reshape / permute to the layout required by the kernel.
    padded = padded.reshape(B, M_padded // 128, 4, 32, K_padded // 4, 4)
    swizzled = padded.permute(0, 1, 4, 3, 2, 5).contiguous().cuda()

    if scale_ndim == 2:
        return swizzled.reshape(M_padded, K_padded)
    return swizzled.reshape(B, M_padded, K_padded)


def cutlass_fp4_supported() -> bool:
    if not current_platform.is_cuda():
        return False
    capability_tuple = current_platform.get_device_capability()
    capability = -1 if capability_tuple is None else capability_tuple.to_int()
    return cutlass_scaled_mm_supports_fp4(capability)


def pad_nvfp4_weight_for_cutlass(
    weight: torch.Tensor,
    alignment: int = 32,
) -> tuple[torch.Tensor, int]:
    """
    Pad packed NVFP4 weights so that both N (rows) and K (columns) satisfy
    the alignment constraints required by CUTLASS / FlashInfer FP4 kernels.

    CUTLASS FP4 kernel requires both K and N matrix dimensions to be divisible
    by 32 for aligned memory access and efficient tensor core operations.
    """
    weight_current_rows = weight.shape[0]

    # Pad N dimension (rows) if not aligned
    if weight_current_rows % alignment != 0:
        total_rows = round_up(weight_current_rows, alignment)
        pad_rows = total_rows - weight_current_rows
        weight = torch.nn.functional.pad(weight, (0, 0, 0, pad_rows)).contiguous()

    # Check K dimension alignment
    # 2 FP4 items are packed per byte in the input dimension
    weight_current_col_bytes = weight.shape[1]
    weight_current_col_elements = weight_current_col_bytes * 2

    weights_padding_bytes = 0
    if weight_current_col_elements % alignment != 0:
        total_cols = round_up(weight_current_col_elements, alignment)
        pad_cols = total_cols - weight_current_col_elements
        # Convert from FP4 element count to bytes (2 FP4 values per byte)
        # pad_cols is always even since alignment=32 and current elements are even
        pad_bytes = pad_cols // 2
        weight = torch.nn.functional.pad(weight, (0, pad_bytes, 0, 0)).contiguous()
        weights_padding_bytes = pad_bytes

    return weight, weights_padding_bytes


def pad_nvfp4_activation_for_cutlass(
    x_fp4: torch.Tensor,
    weights_padding_bytes: int,
) -> torch.Tensor:
    """
    Pad packed FP4 activations to match the K-dimension padding applied to weights.
    The padding is in bytes (tensor dimension), not FP4 elements.
    """
    if weights_padding_bytes > 0:
        return torch.nn.functional.pad(x_fp4, (0, weights_padding_bytes)).contiguous()
    return x_fp4


def slice_nvfp4_output(
    out: torch.Tensor,
    output_size: int,
) -> torch.Tensor:
    """
    Slice the output tensor to remove padding in N dimension if weight was padded.
    """
    if out.shape[-1] != output_size:
        return out[..., :output_size].contiguous()
    return out
