# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch

from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW4A4Fp4,
)
from vllm.platforms import current_platform

from .inc_scheme import INCScheme

if TYPE_CHECKING:
    from ..config_parser import INCLayerConfig
    from ..inc import INCConfig


@register_weight_loader_v2_supported_method
class INCNvFp4LinearMethod(LinearMethodBase):
    """Load AutoRound NVFP4 tensors with the native compressed-tensors kernel."""

    def __init__(self, require_shared_scales: bool = False) -> None:
        self.scheme = CompressedTensorsW4A4Fp4()
        self.require_shared_scales = require_shared_scales

    @classmethod
    def get_min_capability(cls) -> int:
        return CompressedTensorsW4A4Fp4.get_min_capability()

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
        self.scheme.create_weights(
            layer=layer,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            params_dtype=params_dtype,
            weight_loader=extra_weight_attrs.get("weight_loader"),
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.require_shared_scales:
            for name in ("weight_global_scale", "input_global_scale"):
                scale = getattr(layer, name)
                if not torch.equal(scale, scale[0].expand_as(scale)):
                    raise ValueError(
                        "AutoRound NVFP4 Hadamard requires shared global scales "
                        f"for fused projections, but {name} differs. "
                        "Re-export with shared Q/K/V global scales."
                    )
        self.scheme.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.scheme.apply_weights(layer, x, bias)


class INCNvFp4Scheme(INCScheme):
    """AutoRound NVFP4 W4A4, with optional CUDA input Hadamard rotation."""

    @staticmethod
    def can_handle(layer_config: "INCLayerConfig") -> bool:
        return (
            layer_config.data_type == "nv_fp"
            and layer_config.bits == 4
            and layer_config.group_size == 16
            and layer_config.sym
            and layer_config.quantized
        )

    def get_linear_method(
        self,
        config: "INCConfig",
        layer: torch.nn.Module,
        prefix: str,
        layer_config: "INCLayerConfig",
    ) -> "LinearMethodBase":
        if not current_platform.is_cuda():
            raise NotImplementedError("AutoRound NVFP4 requires a CUDA NVFP4 kernel")
        method = INCNvFp4LinearMethod(
            require_shared_scales=config.rotation_config is not None
        )
        if config.rotation_config is not None:
            from ..transform.linear import build_linear_transform_method

            return build_linear_transform_method(
                method, config.rotation_config["block_size"]
            )
        return method
