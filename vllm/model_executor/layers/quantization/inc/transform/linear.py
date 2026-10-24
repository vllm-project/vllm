# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.platforms import current_platform

from . import ark_ops  # noqa: F401  (registers the ARK fused rotation+quant op)
from .hadamard import HadamardTransform

__all__ = [
    "INCLinearTransformMethod",
    "ArkFusedMxfp4TransformMethod",
    "require_supported_platform",
    "build_linear_transform_method",
]


class INCLinearTransformMethod(LinearMethodBase):
    """Apply an input transform before a wrapped linear method."""

    def __init__(
        self,
        quant_method: LinearMethodBase,
        block_size: int,
        rotation: HadamardTransform | None = None,
    ) -> None:
        self.quant_method = quant_method
        self.block_size = block_size
        self.rotation = rotation

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        self._validate_block_size(input_size_per_partition, input_size)
        self.quant_method.create_weights(
            layer=layer,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            input_size=input_size,
            output_size=output_size,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.quant_method.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.rotation is not None
        x = self.rotation(x)
        return self.quant_method.apply(layer, x, bias)

    def _validate_block_size(
        self, input_size_per_partition: int, input_size: int
    ) -> None:
        if input_size_per_partition % self.block_size == 0:
            return
        if input_size_per_partition != input_size:
            raise ValueError(
                f"AutoRound rotation block_size={self.block_size} "
                "does not evenly divide this layer's tensor-parallel shard "
                f"width={input_size_per_partition} (full input_size="
                f"{input_size}). Hadamard rotation blocks that span "
                "multiple tensor-parallel shards are not supported; "
                "either use tensor_parallel_size=1 for this checkpoint, "
                "or re-quantize with a rotation block_size that evenly "
                "divides the per-shard width."
            )
        raise ValueError(
            f"Linear input width {input_size_per_partition} is not "
            f"divisible by AutoRound rotation block_size={self.block_size}"
        )


class ArkFusedMxfp4TransformMethod(INCLinearTransformMethod):
    """Apply fused ARK Hadamard, MXFP4 quantization, and FP4 GEMM."""

    def __init__(self, quant_method: LinearMethodBase, block_size: int) -> None:
        super().__init__(quant_method, block_size, rotation=None)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_fp4, x_blockscale = torch.ops.vllm.inc_ark_mxfp4_hadamard_quant(x)
        output = torch.ops._xpu_C.fp4_gemm(
            x_fp4.view(torch.float4_e2m1fn_x2),
            layer.weight,
            x_blockscale.view(torch.float8_e8m0fnu),
            layer.weight_scale,
            x.dtype,
            bias,
        )
        return output.reshape(*x.shape[:-1], layer.output_size_per_partition)


def require_supported_platform() -> None:
    """Raise if the platform does not support Hadamard transforms."""
    if not (current_platform.is_xpu() or current_platform.is_cuda()):
        raise NotImplementedError(
            "AutoRound Hadamard rotation requires CUDA Hadacore or XPU ARK"
        )


def build_linear_transform_method(
    quant_method: LinearMethodBase,
    block_size: int,
) -> LinearMethodBase:
    """Wrap a linear method with the platform-specific Hadamard transform."""
    require_supported_platform()
    if current_platform.is_xpu():
        return ArkFusedMxfp4TransformMethod(quant_method, block_size)
    return INCLinearTransformMethod(
        quant_method, block_size, HadamardTransform(block_size)
    )
