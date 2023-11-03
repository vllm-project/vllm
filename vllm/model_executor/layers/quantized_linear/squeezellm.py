from typing import Optional

import torch
from torch.nn.parameter import Parameter

from vllm import quantization_ops
from vllm.model_executor.layers.linear import LinearMethodBase


class SqueezeLLMLinearMethod(LinearMethodBase):

    def __init__(self, quant_config):
        self.quant_config = quant_config

    def create_weights(self, module: torch.nn.Module, input_size: int,
                       output_size: int, params_dtype: torch.dtype) -> None:
        if input_size % self.quant_config.group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")
        qweight = Parameter(
            torch.empty(
                input_size // self.quant_config.pack_factor,
                output_size,
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        lookup_table = Parameter(
            torch.empty(
                output_size,
                self.quant_config.weight_bits**2,
                device="cuda",
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        module.register_parameter("qweight", qweight)
        module.register_parameter("lookup_table", lookup_table)

    def apply_weights(self,
                      module: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        out_shape = x.shape[:-1] + (module.qweight.shape[-1], )
        reshaped_x = x.reshape(-1, x.shape[-1])
        # NOTE: The output tensor should be zero-initialized.
        out = torch.zeros(out_shape, device="cuda", dtype=torch.float16)
        quantization_ops.squeezellm_gemm(reshaped_x, module.qweight, out,
                                         module.lookup_table)

        if bias is not None:
            out = out + bias
        return out.reshape(out_shape)
