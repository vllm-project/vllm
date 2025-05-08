# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_fp8_marlin_linear, prepare_fp8_layer_for_marlin)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    Fp8LinearOp, maybe_create_device_identity, normalize_e4m3fn_to_e4m3fnuz)
from vllm.model_executor.parameter import (ChannelQuantScaleParameter,
                                           ModelWeightParameter)
from vllm.platforms import current_platform

logger = init_logger(__name__)


class FBGEMMFp8Config(QuantizationConfig):
    """Config class for FBGEMM Fp8."""

    def __init__(self, ignore_list: List[str], input_scale_ub: float):
        super().__init__()
        self.ignore_list = ignore_list if ignore_list else []
        self.input_scale_ub = input_scale_ub

        # For GPUs that lack FP8 hardware support, we can leverage the Marlin
        # kernel for fast weight-only FP8 quantization
        self.use_marlin = not current_platform.has_device_capability(89)
        self.fp8_linear = Fp8LinearOp()

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "fbgemm_fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FBGEMMFp8Config":
        ignore_list = cls.get_from_keys(config, ["modules_to_not_convert"])
        input_scale_ub = cls.get_from_keys(config, ["activation_scale_ub"])
        return cls(ignore_list=ignore_list, input_scale_ub=input_scale_ub)

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix, self.ignore_list):
                return UnquantizedLinearMethod()
            return FBGEMMFp8LinearMethod(self)
        return None


class FBGEMMFp8LinearMethod(LinearMethodBase):

    def __init__(self, quant_config: FBGEMMFp8Config):
        self.quant_config = quant_config
        self.fp8_linear = Fp8LinearOp(use_per_token_if_dynamic=True)
        self.out_dtype = torch.get_default_dtype()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        maybe_create_device_identity()
        weight_loader = extra_weight_attrs.get("weight_loader")
        del input_size, output_size
        output_size_per_partition = sum(output_partition_sizes)

        layer.logical_widths = output_partition_sizes

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        # WEIGHT
        weight = ModelWeightParameter(data=torch.empty(
            output_size_per_partition,
            input_size_per_partition,
            dtype=torch.float8_e4m3fn),
                                      input_dim=1,
                                      output_dim=0,
                                      weight_loader=weight_loader)
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        weight_scale = ChannelQuantScaleParameter(data=torch.empty(
            (sum(output_partition_sizes), 1), dtype=torch.float32),
                                                  output_dim=0,
                                                  weight_loader=weight_loader)
        weight_scale[:] = torch.finfo(torch.float32).min
        layer.register_parameter("weight_scale", weight_scale)

        # INPUT SCALE UPPER BOUND
        input_scale_ub = torch.nn.Parameter(torch.tensor(
            (self.quant_config.input_scale_ub), dtype=torch.float32),
                                            requires_grad=False)
        layer.input_scale_ub = input_scale_ub

    def process_weights_after_loading(self, layer: Module) -> None:
        # required by torch.compile
        layer.weight_scale = Parameter(layer.weight_scale.data,
                                       requires_grad=False)
        layer.weight = Parameter(layer.weight.data, requires_grad=False)

        weight = layer.weight

        if current_platform.is_fp8_fnuz():
            weight, weight_scale, input_scale = \
                normalize_e4m3fn_to_e4m3fnuz(
                    weight=weight,
                    weight_scale=layer.weight_scale,
                    input_scale=None)
            if input_scale is not None:
                layer.input_scale = Parameter(input_scale, requires_grad=False)
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)

        layer.weight = Parameter(weight.t(), requires_grad=False)
        if self.quant_config.use_marlin:
            prepare_fp8_layer_for_marlin(layer)
            # Activations not quantized for marlin.
            del layer.input_scale_ub

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        if self.quant_config.use_marlin:
            return apply_fp8_marlin_linear(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                workspace=layer.workspace,
                size_n=layer.output_size_per_partition,
                size_k=layer.input_size_per_partition,
                bias=bias)

        return self.fp8_linear.apply(input=x,
                                     weight=layer.weight,
                                     weight_scale=layer.weight_scale,
                                     out_dtype=self.out_dtype,
                                     input_scale=None,
                                     input_scale_ub=layer.input_scale_ub,
                                     bias=bias)
