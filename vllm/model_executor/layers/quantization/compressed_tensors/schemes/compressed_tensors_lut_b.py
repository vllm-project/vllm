# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch
import torch.nn.functional as F
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)

from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import _get_linear_backend
from vllm.model_executor.layers.quantization.utils.lut_b_utils import (
    LUT_B_BLOCK_K,
    LUT_B_BLOCK_N,
    LUT_B_CODEBOOK_SIZE,
    LUT_B_PACKED_TILE_BYTES,
    dequantize_lut_b,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    dequant_mxfp8_to_bf16,
    mxfp8_e4m3_quantize,
)
from vllm.model_executor.parameter import BlockQuantScaleParameter
from vllm.model_executor.utils import replace_parameter

from .compressed_tensors_scheme import CompressedTensorsScheme

__all__ = ["CompressedTensorsLutB"]

logger = init_logger(__name__)


class CompressedTensorsLutB(CompressedTensorsScheme):
    """Compressed-tensors LUT-B linear weights with MXFP8 activation emulation."""

    def __init__(self, weight_quant: QuantizationArgs):
        if (
            weight_quant.type != QuantizationType.CODEBOOK
            or weight_quant.num_bits != 3
            or weight_quant.strategy != QuantizationStrategy.BLOCK
            or weight_quant.block_structure != [LUT_B_BLOCK_N, LUT_B_BLOCK_K]
            or not weight_quant.symmetric
            or weight_quant.dynamic
            or weight_quant.scale_dtype not in (None, torch.uint8)
        ):
            raise ValueError(
                "LUT-B linear requires 3-bit symmetric codebook weights with "
                f"block_structure={[LUT_B_BLOCK_N, LUT_B_BLOCK_K]} and "
                "E8M0 scales."
            )
        backend = _get_linear_backend(quantization="lut_b")
        if backend not in ("auto", "emulation"):
            raise ValueError(
                f"linear_backend={backend!r} is not supported for compressed-"
                "tensors LUT-B linear; expected 'auto' or 'emulation'."
            )
        logger.warning_once(
            "Using emulated compressed-tensors LUT-B linear: checkpoint weights "
            "are expanded to BF16 and activations are MXFP8 quantize-dequantized."
        )

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        if input_size_per_partition % LUT_B_BLOCK_K or any(
            size % LUT_B_BLOCK_N for size in output_partition_sizes
        ):
            raise ValueError(
                "LUT-B linear requires input partitions divisible by "
                f"{LUT_B_BLOCK_K} and each output partition divisible by "
                f"{LUT_B_BLOCK_N}; got input={input_size_per_partition}, "
                f"outputs={output_partition_sizes}."
            )

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.output_partition_sizes = output_partition_sizes
        layer.params_dtype = params_dtype
        layer.weight_block_size = [LUT_B_BLOCK_N, LUT_B_BLOCK_K]

        tile_shape = (
            output_size_per_partition // LUT_B_BLOCK_N,
            input_size_per_partition // LUT_B_BLOCK_K,
        )
        for suffix, tail, dtype in (
            ("packed", (LUT_B_PACKED_TILE_BYTES,), torch.uint8),
            ("codebook", (LUT_B_CODEBOOK_SIZE,), torch.float8_e4m3fn),
            ("scale", (), torch.uint8),
        ):
            param = BlockQuantScaleParameter(
                data=torch.empty(*tile_shape, *tail, dtype=dtype),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter(f"weight_{suffix}", param)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = dequantize_lut_b(
            layer.weight_packed,
            layer.weight_codebook,
            layer.weight_scale,
            out_dtype=layer.params_dtype,
        )
        replace_parameter(layer, "weight", weight)
        del layer.weight_packed
        del layer.weight_codebook
        del layer.weight_scale

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_shape = x.shape
        x_q, x_scales = mxfp8_e4m3_quantize(x.reshape(-1, x_shape[-1]))
        x_qdq = dequant_mxfp8_to_bf16(x_q, x_scales).to(x.dtype).view(x_shape)
        return F.linear(x_qdq, layer.weight, bias)
