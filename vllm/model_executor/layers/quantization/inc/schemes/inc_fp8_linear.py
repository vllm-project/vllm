# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    get_marlin_input_dtype,
)

from .inc_scheme import INCLinearScheme


class INCFp8LinearScheme(INCLinearScheme):
    def __init__(self, prefix: str, weight_block_size: tuple[int, int]) -> None:
        self.quant_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=list(weight_block_size),
        )
        self.inner_method = Fp8LinearMethod(self.quant_config)
        self.inner_method.marlin_input_dtype = get_marlin_input_dtype(prefix)

    @classmethod
    def get_min_capability(cls) -> int:
        return Fp8Config.get_min_capability()

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
        self.inner_method.create_weights(
            layer=layer,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            input_size=input_size,
            output_size=output_size,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.inner_method.process_weights_after_loading(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.inner_method.apply(layer, x, bias)
