from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from vllm import quantization_ops
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               set_weight_attrs)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size)


class GPTQConfig(QuantizationConfig):
    """Config class for GPTQ.

    Reference: https://arxiv.org/abs/2306.00978
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        desc_act: bool,
    ) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.desc_act = desc_act
        self.pack_factor = 32 // self.weight_bits
        # exllama kernel v1 only supports 4 bit
        if self.weight_bits != 4:
            raise ValueError(
                "Currently, only 4-bit weight quantization is supported for "
                f"GPTQ, but got {self.weight_bits} bits.")

    def __repr__(self) -> str:
        return (f"GPTQConfig(weight_bits={self.weight_bits}, "
                f"group_size={self.group_size}, "
                f"desc_act={self.desc_act})")

    @classmethod
    def get_name(cls) -> str:
        return "gptq"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    # Need to figure it out
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return [
            "quantize_config.json",
        ]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GPTQConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        desc_act = cls.get_from_keys(config, ["desc_act"])
        return cls(weight_bits, group_size, desc_act)

    def get_linear_method(self) -> "GPTQLinearMethod":
        return GPTQLinearMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        return []


class GPTQLinearMethod(LinearMethodBase):
    """Linear method for GPTQ.

    Args:
        quant_config: The GPTQ quantization config.
    """

    def __init__(self, quant_config: GPTQConfig):
        self.quant_config = quant_config
        self.use_exllama = True

    def create_weights(self, input_size: int, output_size: int,
                       params_dtype: torch.dtype,
                       parallel_type: str = "none") -> Dict[str, torch.Tensor]:
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
        g_idx = Parameter(
            torch.tensor(
                [i // self.quant_config.group_size for i in range(input_size)],
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(g_idx, {"input_dim": 0})
        tp_size = get_tensor_model_parallel_world_size()
        if parallel_type == "row" and tp_size > 1 and (self.quant_config.desc_act
                and self.quant_config.group_size != -1):
            input_size = input_size * tp_size
            use_exllama = Parameter(torch.tensor(False, dtype=torch.bool, device="cuda"), requires_grad=False)
        else:
            use_exllama = Parameter(torch.tensor(True, dtype=torch.bool, device="cuda"), requires_grad=False)
        if self.quant_config.desc_act or self.quant_config.group_size == -1:
            input_dim = None
        else:
            input_dim = 0
        group_size = self.quant_config.group_size if self.quant_config.group_size != -1 else input_size
        qzeros = Parameter(
            torch.empty(
                input_size // group_size,
                output_size // self.quant_config.pack_factor,
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qzeros, {
                "input_dim": input_dim,
                "output_dim": 1,
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor,
            })
        scales = Parameter(
            torch.empty(
                input_size // group_size,
                output_size,
                device="cuda",
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(scales, {
            "input_dim": input_dim,
            "output_dim": 1,
        })
        return {
            "qweight": qweight,
            "g_idx": g_idx,
            "qzeros": qzeros,
            "scales": scales,
            "use_exllama": use_exllama,
        }

    def apply_weights(self,
                      weights: Dict[str, torch.Tensor],
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        #q4 = weights["q4"]
        qweight = weights["qweight"]
        out_shape = x.shape[:-1] + (qweight.shape[-1], )
        reshaped_x = x.reshape(-1, x.shape[-1])
        if weights["use_exllama"]:
            output = torch.empty((reshaped_x.shape[0], qweight.shape[-1]),
                                 dtype=torch.float16,
                                 device=x.device)
            quantization_ops.gemm_half_q_half(reshaped_x, weights["q4"], output,
                                              False)
        else:
            output = torch.zeros((reshaped_x.shape[0], qweight.shape[-1]),
                                 dtype=torch.float32,
                                 device=x.device)
            quantization_ops.gptq_descact_matmul(reshaped_x.float(),
                                                 weights["qweight"], output,
                                                 weights["scales"].float(),
                                                 weights["qzeros"], weights["g_idx"])
            output = output.half()
        if bias is not None:
            output = output + bias
        return output.reshape(out_shape)

    def temp_dq_size(self, input_size, output_size):
        return input_size * output_size * 2 + 128

    def temp_fwd_size(self, output_size, max_tokens):
        return output_size * max_tokens * 4 + 128

    def scratch_space_fixed(self, input_size, output_size, max_tokens):
        return self.temp_dq_size(input_size, output_size) + self.temp_fwd_size(
            output_size, max_tokens)

    def post_init(self, linear_weights, temp_dq):
        if not linear_weights["use_exllama"]:
            return
        none_tensor = torch.empty((1, 1), device="meta")
        height, width = linear_weights["qweight"].shape
        temp_dq = temp_dq.get_scratch_slice(
            self.temp_dq_size(height * self.quant_config.pack_factor, width))
        if not self.quant_config.desc_act:
            linear_weights["q4"] = quantization_ops.make_q_matrix(
                linear_weights["qweight"],
                none_tensor,
                none_tensor,
                linear_weights["qzeros"],
                linear_weights["scales"],
                none_tensor,
                temp_dq,
            )
        else:
            linear_weights["q_perm"] = torch.empty(
                (height * self.quant_config.pack_factor, ),
                dtype=torch.short,
                device=linear_weights["qweight"].device)
            linear_weights["q_invperm"] = torch.empty_like(linear_weights["q_perm"])
            linear_weights["q4"] = quantization_ops.make_q_matrix(
                linear_weights["qweight"],
                linear_weights["q_perm"],
                linear_weights["q_invperm"],
                linear_weights["qzeros"],
                linear_weights["scales"],
                linear_weights["g_idx"].cpu(),
                temp_dq,
            )
