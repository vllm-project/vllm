# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization.utils.humming import (
    apply_humming_linear,
    convert_linear_layer_to_humming_standard,
    get_humming_linear_compute_config,
    humming_executed_input_quant_key,
    prepare_humming_linear_layer_config,
    quant_key_to_input_schema,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
)
from vllm.platforms import current_platform
from vllm.utils.humming import create_humming_locks
from vllm.utils.import_utils import has_humming

from .base import MxFp4LinearKernel, MxFp4LinearLayerConfig


class HummingMxFp4LinearKernel(MxFp4LinearKernel):
    """Humming GEMM Kernel for MXFP4."""

    weight_quant_config = {
        "quant_method": "humming",
        "dtype": "float4e2m1",
        "scale_dtype": "float8e8m0",
        "group_size": 32,
        "weight_scale_type": "group",
    }

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "Humming only supported on CUDA"

        if not has_humming():
            return False, "Humming is not installed"

        if not current_platform.has_device_capability(75):
            return False, "Humming only supported on SM75+"

        return True, None

    @classmethod
    def _supports_quant_scheme(
        cls,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        if weight_key != kMxfp4Static:
            return False
        try:
            quant_key_to_input_schema(activation_key)
        except ValueError:
            return False
        return True

    @staticmethod
    def _allows_input_schema_fallback(activation_key: QuantKey | None) -> bool:
        # Keep checkpoint-declared FP8 activations instead of falling back to
        # A16 for group-scaled weights.
        return (
            activation_key is None
            or activation_key.dtype != current_platform.fp8_dtype()
        )

    @classmethod
    def executed_activation_key(
        cls, activation_quant_key: QuantKey | None
    ) -> QuantKey | None:
        return humming_executed_input_quant_key(
            cls.weight_quant_config,
            activation_quant_key,
            allow_fallback=cls._allows_input_schema_fallback(activation_quant_key),
        )

    @classmethod
    def can_implement(cls, config: MxFp4LinearLayerConfig) -> tuple[bool, str | None]:
        try:
            quant_key_to_input_schema(config.activation_quant_key)
        except ValueError as error:
            return False, str(error)
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight_scale.data = layer.weight_scale.data.view(torch.float8_e8m0fnu)
        name_map = {"weight": "weight", "weight_scale": "weight_scale"}

        convert_linear_layer_to_humming_standard(layer=layer, name_map=name_map)
        act_key = self.config.activation_quant_key
        input_schema = quant_key_to_input_schema(act_key)
        self.layer_config = prepare_humming_linear_layer_config(
            layer,
            self.weight_quant_config,
            input_schema=input_schema,
            allow_input_schema_fallback=self._allows_input_schema_fallback(act_key),
        )
        self.compute_config = get_humming_linear_compute_config()
        self.locks = create_humming_locks(layer.weight.device)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return apply_humming_linear(
            layer,
            x,
            skip_bias_add=bias is None,
            layer_config=self.layer_config,
            compute_config=self.compute_config,
            locks=self.locks,
        )
