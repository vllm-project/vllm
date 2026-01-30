# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
from torch.nn.parameter import Parameter

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.fp8 import (
    Fp8Config,
    Fp8KVCacheMethod,
    Fp8LinearMethod,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm import (
    init_fp8_linear_kernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
    kFp8DynamicTokenSym,
)
from vllm.platforms import current_platform

ACTIVATION_SCHEMES = ["static", "dynamic"]

logger = init_logger(__name__)


class PTPCFp8Config(Fp8Config):
    """Config class for Per-Token-Per-Channel Dynamic Quantization Fp8."""

    def __init__(
        self,
        activation_scheme: str = "dynamic",
        ignored_layers: list[str] | None = None,
    ) -> None:
        if not current_platform.is_rocm():
            raise ValueError("ptpc_fp8 quantization is supported only on ROCm.")

        if not current_platform.has_device_capability(94):
            raise ValueError(
                "ptpc_fp8 quantization is supported only on AMD Instinct MI300 GPUs and newer."  # noqa: E501
            )
        if activation_scheme == "static":
            raise ValueError("ptpc_fp8 as of now only support dynamic quantization.")

        super().__init__(
            is_checkpoint_fp8_serialized=False,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
        )

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "ptpc_fp8"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "PTPCFp8Config":
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        return cls(activation_scheme=activation_scheme, ignored_layers=ignored_layers)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix, self.ignored_layers):
                return UnquantizedLinearMethod()
            return PTPCFp8LinearMethod(self)
        elif isinstance(layer, Attention):
            return Fp8KVCacheMethod(self)
        return None


class PTPCFp8LinearMethod(Fp8LinearMethod):
    """Linear method for Per-Token and Per-Channel FP8 Quantization.
    Only supports loading quantized BF16 model checkpoints with dynamic
    activation scaling. To load FP16 model checkpoints, user must specify
    to convert the FP16 model weight loading into BF16.
    The weight scaling factor will be initialized after
    the model weights are loaded.

    Limitations:
    1. Only support float8_e4m3fnuz data type due to the limitation of
       torch._scaled_mm (https://github.com/ROCm/pytorch/blob/8c0504d7f3fb0ee4c278c096a5c3caedb01129fa/aten/src/ATen/native/cuda/Blas.cpp#L1041)

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: PTPCFp8Config):
        assert current_platform.is_rocm(), (
            "PTPCFp8LinearMethod is only supported on ROCm."
        )
        super().__init__(quant_config=quant_config)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        # Call parent class create_weights to set up the base FP8 weights
        super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )

        # Override the fp8_linear kernel with PTPC-specific configuration
        self.fp8_linear = init_fp8_linear_kernel(
            activation_quant_key=kFp8DynamicTokenSym,
            weight_quant_key=kFp8DynamicTokenSym,
            out_dtype=torch.get_default_dtype(),
            N=output_size_per_partition,
            K=input_size_per_partition,
            module_name=self.__class__.__name__,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        assert layer.weight.data.dtype not in (torch.float16, torch.float32), (
            "Currently torch._scaled_mm (hipBLASLt) rowwise gemm only support "
            f"output dtype of bfloat16. {layer.weight.data.dtype} is specified."
        )

        if layer.weight.data.dtype == torch.bfloat16:
            # Quantize the weights.
            qweight, weight_scale = ops.scaled_fp8_quant(
                layer.weight, scale=None, use_per_token_if_dynamic=True
            )

            # Update the layer with the new values.
            layer.weight = Parameter(
                qweight.t(), requires_grad=False
            )  # Pretranspose the weight
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)
        else:
            assert layer.weight.data.dtype == current_platform.fp8_dtype()
            assert getattr(layer, "weight_scale", None) is not None
        layer.input_scale = None

        self.fp8_linear.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.fp8_linear.apply_weights(layer, x, bias)
