import enum
from enum import Enum
from fractions import Fraction
from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from vllm._C import ops
from vllm.model_executor.layers.fused_moe import (fused_moe, fused_topk,
                                                  moe_align_block_size)
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               set_weight_attrs)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)


class GPTQConfig(QuantizationConfig):
    """Config class for GPTQ.

    Reference: https://arxiv.org/abs/2210.17323
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
        self.pack_factor = Fraction(32, self.weight_bits)
        if self.weight_bits not in [2, 3, 4, 8]:
            raise ValueError(
                "Currently, only 2/3/4/8-bit weight quantization is "
                f"supported for GPTQ, but got {self.weight_bits} bits.")

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
        return ["quantize_config.json"]

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


class ExllamaState(Enum):

    UNUSED = enum.auto()
    UNINITIALIZED = enum.auto()
    READY = enum.auto()


class GPTQLinearMethod(LinearMethodBase):
    """Linear method for GPTQ.

    Args:
        quant_config: The GPTQ quantization config.
    """

    def __init__(self, quant_config: GPTQConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        input_size_per_partition: int,
        output_size_per_partition: int,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        del output_size  # Unused.
        if input_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")
        if (output_size_per_partition % self.quant_config.pack_factor.numerator
                != 0):
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")

        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = input_size
        exllama_state = ExllamaState.UNINITIALIZED
        scale_and_zero_size = input_size // group_size
        scale_and_zero_input_dim = None
        if (input_size != input_size_per_partition
                and self.quant_config.group_size != -1):
            # For act-order models, we cannot use Exllama for row parallel layer
            if self.quant_config.desc_act:
                exllama_state = ExllamaState.UNUSED
            else:
                # we need to partition qzeros and scales for exllama kernel
                scale_and_zero_size = input_size_per_partition // group_size
                scale_and_zero_input_dim = 0

        qweight = Parameter(
            torch.empty(
                input_size_per_partition // self.quant_config.pack_factor,
                output_size_per_partition,
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
                [
                    i // self.quant_config.group_size
                    for i in range(input_size_per_partition)
                ],
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        # Ignore warning from fused linear layers such as QKVParallelLinear.
        set_weight_attrs(g_idx, {"input_dim": 0, "ignore_warning": True})
        qzeros = Parameter(
            torch.empty(
                scale_and_zero_size,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qzeros, {
                "input_dim": scale_and_zero_input_dim,
                "output_dim": 1,
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor,
            })
        scales = Parameter(
            torch.empty(
                scale_and_zero_size,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(scales, {
            "input_dim": scale_and_zero_input_dim,
            "output_dim": 1,
        })
        return {
            "qweight": qweight,
            "g_idx": g_idx,
            "qzeros": qzeros,
            "scales": scales,
            "exllama_state": exllama_state,
        }

    def apply_weights(self,
                      weights: Dict[str, Any],
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        qweight = weights["qweight"]
        out_shape = x.shape[:-1] + (qweight.shape[-1], )
        reshaped_x = x.reshape(-1, x.shape[-1])
        # exllama needs to shuffle the weight after the weight is loaded
        # here we do the shuffle on first forward pass
        if weights["exllama_state"] == ExllamaState.UNINITIALIZED:
            if self.quant_config.desc_act:
                weights["g_idx"] = torch.argsort(weights["g_idx"]).to(
                    torch.int)
            else:
                weights["g_idx"] = torch.empty((1, 1), device="meta")
            weights["exllama_state"] = ExllamaState.READY
            ops.gptq_shuffle(weights["qweight"], weights["g_idx"],
                             self.quant_config.weight_bits)
        output = ops.gptq_gemm(reshaped_x, weights["qweight"],
                               weights["qzeros"], weights["scales"],
                               weights["g_idx"],
                               weights["exllama_state"] == ExllamaState.READY,
                               self.quant_config.weight_bits)
        if bias is not None:
            output = output + bias
        return output.reshape(out_shape)

    def apply_moe_weights(self, w1: Dict[str,
                                         torch.Tensor], w2: Dict[str,
                                                                 torch.Tensor],
                          x: torch.Tensor, gating_output: torch.Tensor,
                          topk: int, renormalize: bool) -> torch.Tensor:
        # shuffle weights for exllama
        for w in [w1, w2]:
            if w["exllama_state"] == ExllamaState.UNINITIALIZED:
                if self.quant_config.desc_act:
                    w["g_idx"] = torch.argsort(w["g_idx"],
                                               dim=-1).to(torch.int)
                else:
                    w["g_idx"] = torch.empty((w["g_idx"].shape[0], 1),
                                             device="meta")
                w["exllama_state"] = ExllamaState.READY
                ops.gptq_shuffle(w["qweight"], w["g_idx"],
                                 self.quant_config.weight_bits)

        # Fused moe only supports 4-bit
        if self.quant_config.weight_bits != 4:
            return super().apply_moe_weights(w1, w2, x, gating_output, topk,
                                             renormalize)

        if x.shape[0] >= 128:
            dequant_w1 = ops.dequant_gptq(
                w1["qweight"], w1["qzeros"], w1["scales"], w1["g_idx"],
                self.quant_config.weight_bits,
                w1["exllama_state"] == ExllamaState.READY).permute(0, 2, 1)
            dequant_w2 = ops.dequant_gptq(
                w2["qweight"], w2["qzeros"], w2["scales"], w2["g_idx"],
                self.quant_config.weight_bits,
                w2["exllama_state"] == ExllamaState.READY).permute(0, 2, 1)
            return fused_moe(x, dequant_w1, dequant_w2, gating_output, topk,
                             renormalize)

        topk_weights, topk_ids = fused_topk(gating_output, topk, renormalize)
        (sorted_token_ids, expert_ids,
         num_tokens_post_padded) = moe_align_block_size(
             topk_ids, 8, w1["qweight"].shape[0])

        x = x.view(x.shape[0], 1, *x.shape[1:])
        gate_up = ops.group_gptq_gemm(
            x, w1["qweight"], w1["qzeros"], w1["scales"], w1["g_idx"],
            topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
            False, w1["exllama_state"] == ExllamaState.READY)

        out = torch.empty((gate_up.shape[:-1] + (gate_up.shape[-1] // 2, )),
                          dtype=x.dtype,
                          device=x.device)
        ops.silu_and_mul(out, gate_up)

        out = ops.group_gptq_gemm(out, w2["qweight"], w2["qzeros"],
                                  w2["scales"], w2["g_idx"], topk_weights,
                                  sorted_token_ids, expert_ids,
                                  num_tokens_post_padded, True,
                                  w2["exllama_state"] == ExllamaState.READY)

        return torch.sum(out, dim=1)
