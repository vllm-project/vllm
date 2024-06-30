from typing import List, Optional, Tuple, Union

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization.fp8 import (Fp8Config,
                                                         cutlass_fp8_supported,
                                                         per_tensor_quantize, 
                                                         apply_quantize)
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger(__name__)


class ModelOptQuantizer(torch.nn.Module):
    """Class to load amax values for Model Opt checkpoints."""

    def __init__(self, _amax, **extra_weight_attrs):
        super().__init__()
        self._amax = _amax
        set_weight_attrs(
            _amax,
            {
                **extra_weight_attrs,
                "needs_scalar_to_array": True,
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

    def __init__(self, quant_config: Fp8Config):
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
        # Model Opt weights are not converted to FP8 when stored in
        # the checkpoint, so we use the original datatype. May change
        # in the future if the format of Model Opt checkpoint changes.
        weight = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
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
                torch.empty(len(output_partition_sizes), dtype=torch.float32),
                requires_grad=False,
            )

            layer.add_module(
                "weight_quantizer",
                ModelOptQuantizer(weight_amax, **extra_weight_attrs),
            )

            # INPUT ACTIVATION SCALE
            if self.quant_config.activation_scheme == "static":
                input_amax = Parameter(
                    torch.empty(len(output_partition_sizes),
                                dtype=torch.float32),
                    requires_grad=False,
                )
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
            # WEIGHT_SCALE / WEIGHT
            # Convert the given weight to fp8 because model opt generates
            # quantization scales, but doesn't convert then to fp8.
            layer.weight_scale = layer.weight_quantizer._amax / 448
            max_w_scale = layer.weight_scale.max()

            weight = torch.empty_like(layer.weight, dtype=torch.float8_e4m3fn)
            start = 0
            for idx, logical_width in enumerate(layer.logical_widths):
                end = start + logical_width
                weight_dq = layer.weight[start:end, :]
                weight[start:end, :] = per_tensor_quantize(
                    weight_dq, layer.weight_scale.max())
                start = end

            layer.weight.data = weight
            layer.weight_scale = Parameter(max_w_scale, requires_grad=False)

            # WEIGHT
            #   Transpose weight for passing to torch._scaled_mm
            layer.weight = Parameter(weight.t(), requires_grad=False)
            
            if self.quant_config.activation_scheme == "static":
                layer.input_scale = Parameter(layer.input_quantizer._amax.max(),
                                              requires_grad=False)
            else:
                raise ValueError(
                    f"Unknown scheme {self.quant_config.activation_scheme}")

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

      return apply_quantize(layer, x, self.cutlass_fp8_supported, bias) 

