from typing import Any, Dict, List, Optional

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase, UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    apply_fp8_linear, cutlass_fp8_supported, requantize_with_max_scale)
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger(__name__)

ACTIVATION_SCHEMES = ["static"]


class ModelOptFp8Config(QuantizationConfig):
    """Config class for ModelOpt FP8."""

    def __init__(self,
                 is_checkpoint_fp8_serialized: bool = False,
                 activation_scheme: str = "static",
                 ) -> None:
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
        if is_checkpoint_fp8_serialized:
            logger.warning("Detected fp8 checkpoint. Please note that the "
                           "format is experimental and subject to change.")
        if activation_scheme not in ACTIVATION_SCHEMES:
            raise ValueError(
                f"Unsupported activation scheme {activation_scheme}")
        self.activation_scheme = activation_scheme
       

    @classmethod
    def get_name(cls) -> str:
        return "modelopt"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["hf_quant_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ModelOptFp8Config":
        quant_config = cls.get_from_keys(config, ["quantization"])
        quant_method = quant_config["quant_algo"]
        is_checkpoint_fp8_serialized = ("FP8" in quant_method)
        activation_scheme = "static"
        return cls(is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
                   activation_scheme=activation_scheme)

    def get_quant_method(
            self, layer: torch.nn.Module, prefix: str) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention  # Avoid circular import
        from vllm.model_executor.layers.quantization.fp8 import Fp8KVCacheMethod # Avoid circular import
        if isinstance(layer, LinearBase):
            return ModelOptFp8LinearMethod(self)
        elif isinstance(layer, Attention):
            return Fp8KVCacheMethod(self)
        return None
            
    def get_scaled_act_names(self) -> List[str]:
        return []


class ModelOptQuantizer(torch.nn.Module):
    """Class to load amax values for Model Opt checkpoints."""

    def __init__(self, _amax, **extra_weight_attrs):
        super().__init__()
        self._amax = _amax
        set_weight_attrs(
            _amax,
            {
              "needs_scalar_to_array": True,
              **extra_weight_attrs
            },
        )
        return

    def forward(self, x):
        return x

class ModelOptFp8LinearMethod(LinearMethodBase):
    """Linear method for Model Optimizer static quantization.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Limitations[Same as Fp8LinearMethod]:
    1. Only support per-tensor quantization due to torch._scaled_mm support.
    2. Only support float8_e4m3fn data type due to the limitation of
       torch._scaled_mm (https://github.com/pytorch/pytorch/blob/2e48b39603411a41c5025efbe52f89560b827825/aten/src/ATen/native/cuda/Blas.cpp#L854-L856)

    Args:
        quant_config: The quantization config.
    """

    def __init__(self,
                 quant_config: ModelOptFp8Config):
        self.quant_config = quant_config
        self.cutlass_fp8_supported = cutlass_fp8_supported()

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
        del input_size, output_size
        output_size_per_partition = sum(output_partition_sizes)

        layer.process_after_load = True
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype 
        # Model Opt weights are not converted to FP8 when stored in
        # the checkpoint, so we use the original datatype. May change
        # in the future if the format of Model Opt checkpoint changes.
        weight_dtype = (torch.int8
                        if self.quant_config.is_checkpoint_fp8_serialized else
                        params_dtype)
        weight = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("weight", weight)
        set_weight_attrs(
            weight,
            {
                **extra_weight_attrs,
                "input_dim": 1,
                "output_dim": 0,
            },
        )

        # If checkpoint is serialized fp8, load them.
        # Otherwise, wait until process_weights_after_loading.
        if self.quant_config.is_checkpoint_fp8_serialized:
            # WEIGHT SCALE
            weight_amax = Parameter(
                torch.empty(len(output_partition_sizes), dtype=params_dtype),
                requires_grad=False,
            )
            weight_amax[:] = torch.finfo(params_dtype).min
            layer.add_module(
                "weight_quantizer",
                ModelOptQuantizer(weight_amax, **extra_weight_attrs),
            )
            

            # INPUT ACTIVATION SCALE
            if self.quant_config.activation_scheme == "static":
                input_amax = Parameter(
                    torch.empty(len(output_partition_sizes),
                                dtype=params_dtype),
                    requires_grad=False,
                )
                input_amax[:] = torch.finfo(params_dtype).min
                layer.add_module(
                    "input_quantizer",
                    ModelOptQuantizer(input_amax, **extra_weight_attrs),
                )

    def process_weights_after_loading(self, layer: Module) -> None:
        if (not hasattr(layer, "process_after_load")
                or not layer.process_after_load):
            return
        # If checkpoint is fp/bf16 and not serialized fp8, quantize the weights.
        if not self.quant_config.is_checkpoint_fp8_serialized:
            qweight, weight_scale = ops.scaled_fp8_quant(layer.weight,
                                                         scale=None)
            layer.weight = Parameter(qweight.t(), requires_grad=False)
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)
            layer.logical_widths = None
            layer.input_scale = None
            return

        else:
            weight_scale = (layer.weight_quantizer._amax.to(torch.float32)) / 448.0
            weight = layer.weight.view(torch.float8_e4m3fn)
            max_w_scale = weight_scale.max()
            # max_w_scale, weight = requantize_with_max_scale(
            #     weight=restored_fp8_tensor,
            #     weight_scale=weight_scale,
            #     logical_widths=layer.logical_widths,
            # )
            layer.weight = Parameter(weight.t(), requires_grad=False)
            layer.weight_scale = Parameter(max_w_scale, requires_grad=False)
            layer.input_scale = Parameter(
                (layer.input_quantizer._amax.max().to(torch.float32))/448.0, requires_grad=False) 
            
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not torch.cuda.is_current_stream_capturing(): 
            logger.info(f"fp8_linear: \n cutlass: {self.cutlass_fp8_supported} "
                     f"\nweight: {layer.weight}, "
                     f"\nx: {x},"
                     f"\nweight.scale={layer.weight_scale}, "
                     f"\ninput_scale: {layer.input_scale}")
        return apply_fp8_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            input_scale=layer.input_scale,
            bias=bias,
            cutlass_fp8_supported=self.cutlass_fp8_supported)
        
