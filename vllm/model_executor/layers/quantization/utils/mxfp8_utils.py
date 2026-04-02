# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum

import torch
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.utils import flashinfer as vllm_flashinfer
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


class Mxfp8LinearBackend(Enum):
    EMULATION = "emulation"
    FLASHINFER_CUTLASS = "flashinfer-cutlass"
    MARLIN = "marlin"


# MXFP8 constants
MXFP8_VALUE_DTYPE = torch.float8_e4m3fn
MXFP8_SCALE_DTYPE = torch.uint8
MXFP8_BLOCK_SIZE = 32


def select_mxfp8_linear_backend() -> Mxfp8LinearBackend:
    """Select the best MXFP8 linear backend for the current device.

    - SM100+ (Blackwell): FLASHINFER_CUTLASS (native MXFP8 W8A8 GEMM)
    - SM80+ (Ampere/Ada): MARLIN (MXFP8 W8A16 GEMM)
    - Otherwise: EMULATION (dequant to BF16 fallback)
    """
    from vllm.platforms import current_platform

    if current_platform.has_device_capability(100):
        return Mxfp8LinearBackend.FLASHINFER_CUTLASS

    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        is_fp8_marlin_supported,
    )

    if is_fp8_marlin_supported():
        return Mxfp8LinearBackend.MARLIN

    return Mxfp8LinearBackend.EMULATION


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


def _mxfp8_e4m3_quantize_torch(
    x: torch.Tensor,
    is_sf_swizzled_layout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Naive MXFP8 quantization.
    For each block of 32 elements along the last dimension, compute a
    shared e8m0 scale (the biased exponent of the block-wise amax)
    and quantize each element to float8_e4m3fn.

    Returns (quantized_values [same shape, fp8], scales uint8).
    Scale shape depends on is_sf_swizzled_layout:
      False -> [..., K//32]  (row-major 2D)
      True  -> [flat swizzled 1D]
    """
    assert x.shape[-1] % MXFP8_BLOCK_SIZE == 0
    orig_shape = x.shape
    num_blocks = x.shape[-1] // MXFP8_BLOCK_SIZE

    x_fp32 = x.to(torch.float32)
    x_blocked = x_fp32.view(*orig_shape[:-1], num_blocks, MXFP8_BLOCK_SIZE)

    amax = x_blocked.abs().amax(dim=-1)
    amax = amax.clamp(min=torch.finfo(torch.float32).tiny)
    scale_biased = torch.floor(torch.log2(amax)) + 127.0
    scale_biased = scale_biased.clamp(0, 254)
    scales_uint8 = scale_biased.to(torch.uint8)

    descale = torch.exp2(scale_biased - 127.0)
    x_scaled = x_blocked / descale.unsqueeze(-1)

    x_fp8 = x_scaled.view(orig_shape).to(MXFP8_VALUE_DTYPE)

    if x.ndim == 2:
        M, K = x.shape
        scales_uint8 = scales_uint8.view(M, -1)
        if is_sf_swizzled_layout:
            scales_uint8 = swizzle_mxfp8_scale(scales_uint8, M=M, K=K)
    elif x.ndim == 3:
        B, M, K = x.shape
        scales_uint8 = scales_uint8.view(B, M, -1)
        if is_sf_swizzled_layout:
            swizzled = []
            for i in range(B):
                swizzled.append(swizzle_mxfp8_scale(scales_uint8[i], M=M, K=K))
            scales_uint8 = torch.cat(swizzled)

    return x_fp8, scales_uint8


def _mxfp8_e4m3_quantize_impl(
    x: torch.Tensor, is_sf_swizzled_layout: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.platforms import current_platform

    if current_platform.has_device_capability(100):
        from flashinfer import mxfp8_quantize as flashinfer_mxfp8_quantize

        x_q, x_scales = flashinfer_mxfp8_quantize(
            x, is_sf_swizzled_layout=is_sf_swizzled_layout
        )
        if x_scales.ndim == 1 and x.ndim == 2 and not is_sf_swizzled_layout:
            x_scales = x_scales.view(x.size(0), -1)
        return x_q, x_scales

    return _mxfp8_e4m3_quantize_torch(x, is_sf_swizzled_layout)


def mxfp8_e4m3_quantize(
    x: torch.Tensor, is_sf_swizzled_layout: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.mxfp8_e4m3_quantize(x, is_sf_swizzled_layout)


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
    op_name="mxfp8_e4m3_quantize",
    op_func=_mxfp8_e4m3_quantize_impl,
    fake_impl=mxfp8_e4m3_quantize_fake,
)


def xpu_mxfp8_quantize(
    x: torch.Tensor, dtype: torch.dtype | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.xpu_mxfp8_quantize(x, dtype)


class Mxfp8LinearOp:
    def __init__(self, backend: Mxfp8LinearBackend):
        if backend not in Mxfp8LinearBackend:
            raise ValueError(f"Unsupported backend: {backend}")


class Mxfp8LinearOp:
    def __init__(self):
        self.backend = select_mxfp8_linear_backend()
        logger.info_once("Using %s backend for MXFP8 GEMM", self.backend)

    def process_weights(self, layer: torch.nn.Module) -> None:
        """Process MXFP8 weights after loading into backend-specific format."""
        if self.backend == Mxfp8LinearBackend.MARLIN:
            self._process_weights_marlin(layer)
        elif self.backend == Mxfp8LinearBackend.FLASHINFER_CUTLASS:
            self._process_weights_flashinfer_cutlass(layer)
        else:
            self._process_weights_emulation(layer)

    def _process_weights_emulation(self, layer: torch.nn.Module) -> None:
        """Keep scales as 2D uint8 for dequant-to-BF16 emulation."""
        weight = layer.weight.data  # [N, K]
        N, K = weight.shape
        scale_k = K // MXFP8_BLOCK_SIZE

        weight_scale = layer.weight_scale.data[:N, :scale_k].contiguous()

        layer.weight = Parameter(weight.contiguous(), requires_grad=False)
        layer.weight_scale = Parameter(weight_scale, requires_grad=False)

    def _process_weights_flashinfer_cutlass(self, layer: torch.nn.Module) -> None:
        """Swizzle scales to F8_128x4 layout for flashinfer CUTLASS."""
        weight = layer.weight.data  # [N, K]
        N, K = weight.shape

        scale_k = K // MXFP8_BLOCK_SIZE
        weight_scale_2d = layer.weight_scale.data[:N, :scale_k].contiguous()
        weight_scale_swizzled = swizzle_mxfp8_scale(weight_scale_2d, M=N, K=K)

        layer.weight = Parameter(weight.contiguous(), requires_grad=False)
        layer.weight_scale = Parameter(
            weight_scale_swizzled.contiguous(), requires_grad=False
        )

    def _process_weights_marlin(self, layer: torch.nn.Module) -> None:
        """Repack MXFP8 weights and scales into Marlin kernel format."""
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
            prepare_mxfp8_layer_for_marlin,
        )

        prepare_mxfp8_layer_for_marlin(layer)

    def _apply_emulation(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if weight_scale.dtype != MXFP8_SCALE_DTYPE:
            raise ValueError(
                f"TORCH backend requires {MXFP8_SCALE_DTYPE} weight_scale dtype, "
                f"got {weight_scale.dtype}."
            )
        if weight_scale.ndim != 2:
            raise ValueError(
                f"TORCH backend requires 2D weight_scale, got {weight_scale.ndim}D. "
                f"Ensure process_weights_after_loading was called."
            )

        weight_bf16 = dequant_mxfp8_to_bf16(weight, weight_scale)

        output = torch.nn.functional.linear(input, weight_bf16, bias)
        return output.to(out_dtype)

    def _apply_flashinfer_cutlass(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        N, K = weight.shape

        input_shape = input.shape
        input_2d = input.view(-1, K)
        M_orig = input_2d.shape[0]

        # Minimum dimension size for F8_128x4 block scaling layout
        min_dim = 128

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

    def _apply_marlin(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype,
        bias: torch.Tensor | None = None,
        *,
        workspace: torch.Tensor,
        size_n: int,
        size_k: int,
    ) -> torch.Tensor:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
            apply_mxfp8_marlin_linear,
        )

        return apply_mxfp8_marlin_linear(
            input=input,
            weight=weight,
            weight_scale=weight_scale,
            workspace=workspace,
            size_n=size_n,
            size_k=size_k,
            bias=bias,
        )

    def apply(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype,
        bias: torch.Tensor | None = None,
        *,
        workspace: torch.Tensor | None = None,
        size_n: int = 0,
        size_k: int = 0,
    ) -> torch.Tensor:
        if self.backend == Mxfp8LinearBackend.EMULATION:
            return self._apply_emulation(input, weight, weight_scale, out_dtype, bias)

        if self.backend == Mxfp8LinearBackend.MARLIN:
            assert workspace is not None
            return self._apply_marlin(
                input,
                weight,
                weight_scale,
                out_dtype,
                bias,
                workspace=workspace,
                size_n=size_n,
                size_k=size_k,
            )

        assert self.backend == Mxfp8LinearBackend.FLASHINFER_CUTLASS
        return self._apply_flashinfer_cutlass(
            input, weight, weight_scale, out_dtype, bias
        )
