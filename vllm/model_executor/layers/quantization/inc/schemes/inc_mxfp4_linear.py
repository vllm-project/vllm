# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any

import torch
from torch.nn.parameter import Parameter

import vllm._custom_ops as ops
from vllm.model_executor.kernels.linear import init_mxfp4_linear_kernel
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Dynamic
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
)
from vllm.platforms import current_platform

from . import inc_ark_ops  # noqa: F401
from .inc_scheme import INCLinearScheme

if TYPE_CHECKING:
    from ..config_parser import INCLayerConfig


class INCMxfp4LinearMethod(INCLinearScheme):
    """MXFP4 (W4A4) linear method for AutoRound checkpoints.

    E2M1 weights packed two per byte with per-group E8M0 scales
    (group_size=32, no global scale). The platform kernel is selected by
    ``init_mxfp4_linear_kernel`` (FlashInfer / Marlin on CUDA, ``fp4_gemm``
    on XPU). XPU rotation uses ARK's fused XMX Hadamard and MXFP4 kernel.
    """

    def __init__(
        self,
        layer_config: "INCLayerConfig",
        rotation_block_size: int | None = None,
    ) -> None:
        if not isinstance(layer_config.group_size, int):
            raise ValueError(
                "INC MXFP4 requires scalar group_size, "
                f"but found group_size={layer_config.group_size!r}."
            )
        self.group_size = layer_config.group_size or 32
        self.rotation_block_size = rotation_block_size
        self.kernel = init_mxfp4_linear_kernel(activation_quant_key=kMxfp4Dynamic)

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        del output_size
        if self.rotation_block_size is not None and (
            input_size_per_partition % self.rotation_block_size
        ):
            if input_size_per_partition != input_size:
                raise ValueError(
                    f"AutoRound rotation block_size={self.rotation_block_size} "
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
                "divisible by AutoRound rotation "
                f"block_size={self.rotation_block_size}"
            )
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.params_dtype = params_dtype
        weight_loader = extra_weight_attrs.get("weight_loader")

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_packed", weight)

        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.group_size,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight = Parameter(layer.weight_packed.data, requires_grad=False)
        del layer.weight_packed
        self.kernel.process_weights_after_loading(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.rotation_block_size is not None:
            if current_platform.is_xpu() and x.is_xpu:
                return self._apply_xpu_rotation(layer, x, bias)
            original_shape = x.shape
            x = x.unflatten(-1, (-1, self.rotation_block_size)).contiguous().clone()
            x = ops.hadacore_transform(x)
            x = x.flatten(-2, -1).reshape(original_shape)
        return self.kernel.apply_weights(layer, x, bias)

    def _apply_xpu_rotation(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
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
