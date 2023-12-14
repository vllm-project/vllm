# Adapted from AutoGPTQ: https://github.com/PanQiWei/AutoGPTQ
from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from vllm import quantization_ops
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               set_weight_attrs)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

use_base_kernel = False
if not use_base_kernel:
    from auto_gptq.nn_modules.qlinear.qlinear_exllama import ext_q4_matmul


# Adapted from vllm minimal gptq branch: https://github.com/vllm-project/vllm/tree/minimal-gptq
def _gptq_matmul(
        x: torch.Tensor,
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        g_idx: torch.Tensor,
        shifter: torch.Tensor,
) -> torch.Tensor:
    """Matrix multiplication with GPTQ weights."""
    # qw: [input_size, output_size]
    qw = (qweight.unsqueeze(1) >> shifter.view(1, -1, 1)) & 0xf
    qw = qw.flatten(start_dim=0, end_dim=1)

    # qz: [input_size, output_size]
    qz = (qzeros[g_idx].unsqueeze(2) >> shifter.view(1, 1, -1)) & 0xf
    qz = qz + 1
    qz = qz.flatten(start_dim=1, end_dim=2)

    # qs: [input_size, output_size]
    qs = scales[g_idx]
    # w: [input_size, output_size]
    w = qs * (qw - qz).to(qs.dtype)

    # out: [batch_size, output_size]
    out = torch.matmul(x, w)
    return out


class GPTQConfig(QuantizationConfig):
    """Config class for GPTQ.
    """

    def __init__(
            self,
            weight_bits: int,
            group_size: int,
            zero_point: bool,
    ) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.zero_point = zero_point

        if self.weight_bits != 4:
            raise ValueError(
                "Currently, only 4-bit weight quantization is supported for "
                f"GPTQ, but got {self.weight_bits} bits.")
        self.pack_factor = 32 // self.weight_bits

    def __repr__(self) -> str:
        return (f"GPTQConfig(weight_bits={self.weight_bits}, "
                f"group_size={self.group_size}, "
                f"zero_point={self.zero_point})")

    def get_name(self) -> str:
        return "gptq"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.half]

    def get_min_capability(self) -> int:
        # The GPTQ kernel only supports Turing or newer GPUs.
        return 75

    @staticmethod
    def get_config_filenames() -> List[str]:
        return [
            "quant_config.json",  # E.g., casperhansen/vicuna-7b-v1.5-gptq
            "quantize_config.json",  # E.g., abhinavkulkarni/mosaicml-mpt-7b-instruct-w4-g128-gptq
        ]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GPTQConfig":
        weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
        group_size = cls.get_from_keys(config, ["q_group_size", "group_size"])
        zero_point = True
        return cls(weight_bits, group_size, zero_point)

    def get_linear_method(self) -> "GPTQLinearMethod":
        return GPTQLinearMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        return ["gelu", "gelu_fast", "gelu_new", "gelu_pytorch_tanh"]


class GPTQLinearMethod(LinearMethodBase):
    """Linear method for GPTQ.

    Args:
        quant_config: The GPTQ quantization config.
    """

    def __init__(self, quant_config: GPTQConfig):
        self.quant_config = quant_config

    def create_weights(self, input_size: int, output_size: int,
                       params_dtype: torch.dtype) -> Dict[str, torch.Tensor]:

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
                input_size // self.quant_config.pack_factor,
                output_size,
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight, {
                "input_dim": 0,
                "output_dim": 1,
                "packed_dim": 0,
                "pack_factor": self.quant_config.pack_factor,
            })
        qzeros = Parameter(
            torch.empty(
                input_size // self.quant_config.group_size,
                output_size // self.quant_config.pack_factor,
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qzeros, {
                "input_dim": 0,
                "output_dim": 1,
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor,
            })
        scales = Parameter(
            torch.empty(
                input_size // self.quant_config.group_size,
                output_size,
                device="cuda",
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(scales, {
            "input_dim": 0,
            "output_dim": 1,
        })
        g_idx = Parameter(torch.tensor(
            [i // self.quant_config.group_size for i in range(input_size)],
            device="cuda",
            dtype=torch.int32,
        ),
            requires_grad=False,
        )

        self.shifter = torch.tensor(
            [0, 4, 8, 12, 16, 20, 24, 28],
            device="cuda",
            dtype=torch.int32,
        )
        return {
            "qweight": qweight,
            "qzeros": qzeros,
            "scales": scales,
            "g_idx": g_idx
        }
    def apply_weights(self,
                      weights: Dict[str, torch.Tensor],
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        if use_base_kernel:
            qweight = weights["qweight"]
            qzeros = weights["qzeros"]
            scales = weights["scales"]
            out_shape = x.shape[:-1] + (qweight.shape[-1],)
            reshaped_x = x.reshape(-1, x.shape[-1])
            g_idx = weights["g_idx"]
            num_tokens = x.shape[:-1].numel()
            if num_tokens <= 32:
                output = torch.zeros(out_shape, dtype=x.dtype, device=x.device)

                quantization_ops.gptq_descact_matmul(reshaped_x, qweight,
                                                     output, scales,
                                                     qzeros, g_idx)
            else:
                output = _gptq_matmul(reshaped_x, qweight, qzeros,
                                      scales, g_idx, self.shifter)
        else:

            qweight_shape = weights["qweight_shape"]
            out_shape = x.shape[:-1] + (qweight_shape,)
            num_tokens = x.shape[:-1].numel()
            if num_tokens <= 32:
                q4 = weights["q4"]
                width = weights["width"]
                output = ext_q4_matmul(x, q4, width)
            else:
                qweight = weights["qweight"]
                qzeros = weights["qzeros"]
                scales = weights["scales"]
                g_idx = weights["g_idx"]
                reshaped_x = x.reshape(-1, x.shape[-1])
                output = _gptq_matmul(reshaped_x, qweight, qzeros,
                                      scales, g_idx, self.shifter)
                output = output

        if bias is not None:
            output = output + bias
        return output.reshape(out_shape)
