from typing import Optional

import torch
from torch.nn.parameter import Parameter

from vllm import quantization_ops
from vllm.model_executor.layers.linear import LinearMethodBase


class AWQLinearMethod(LinearMethodBase):

    def __init__(self, quant_config):
        self.quant_config = quant_config

    def create_weights(self, module: torch.nn.Module, input_size: int,
                       output_size: int, params_dtype: torch.dtype) -> None:
        if input_size % self.quant_config.group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")
        if output_size % self.quant_config.pack_factor != 0:
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")

        qweight = Parameter(
            torch.empty(
                input_size,
                output_size // self.quant_config.pack_factor,
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        qzeros = Parameter(
            torch.empty(
                input_size // self.quant_config.group_size,
                output_size // self.quant_config.pack_factor,
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        scales = Parameter(
            torch.empty(
                input_size // self.quant_config.group_size,
                output_size,
                device="cuda",
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        module.register_parameter("qweight", qweight)
        module.register_parameter("qzeros", qzeros)
        module.register_parameter("scales", scales)

    def apply_weights(self,
                      module: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        pack_factor = self.quant_config.pack_factor
        out_shape = (x.shape[:-1] + (module.qweight.shape[-1] * pack_factor, ))
        reshaped_x = x.reshape(-1, x.shape[-1])
        out = quantization_ops.awq_gemm(reshaped_x, module.qweight,
                                        module.scales, module.qzeros,
                                        pack_factor)
        if bias is not None:
            out = out + bias
        return out.reshape(out_shape)
