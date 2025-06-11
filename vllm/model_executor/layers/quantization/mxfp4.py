# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.fused_moe import FusedMoE, FusedMoEMethodBase
from vllm.model_executor.layers.fused_moe.triton_kernels_moe import (
    triton_kernel_moe_forward)
from vllm.model_executor.layers.linear import (LinearBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped)
from vllm.model_executor.layers.utils import shuffle_weight
from vllm.model_executor.utils import set_weight_attrs


class Mxfp4Config(QuantizationConfig):

    def __init__(self):
        super().__init__()

    @classmethod
    def from_config(cls, config):
        return cls()

    @classmethod
    def get_min_capability(cls) -> int:
        return 90

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "mxfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float8_e4m3fn]

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention  # Avoid circular import

        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix=prefix,
                                ignored_layers=self.ignored_layers,
                                fused_mapping=self.packed_modules_mapping):
                return UnquantizedLinearMethod()
            raise NotImplementedError("Mxfp4 linear layer is not implemented")
        elif isinstance(layer, FusedMoE):
            return Mxfp4MoEMethod()
        elif isinstance(layer, Attention):
            return NotImplementedError(
                "Mxfp4 attention layer is not implemented")
        return None


class Mxfp4MoEMethod(FusedMoEMethodBase):

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        # TODO: we only register parameter here
        # and do not pre-allocate tensor
        # since they need to be transformed when loading
        # allocating will cause pytorch OOM
        # these dummy weight will be
        # replace in func FusedMoE::_load_weights_oai_mlp

        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w13_bias = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        layer.register_parameter("w13_bias", w13_bias)
        set_weight_attrs(w13_bias, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w2_bias = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        layer.register_parameter("w2_bias", w2_bias)
        set_weight_attrs(w2_bias, extra_weight_attrs)

        self.w13_precision_config = None
        self.w2_precision_config = None

    def process_weights_after_loading(self, layer):

        w13_bias = shuffle_weight(layer.w13_bias)
        w13_bias = w13_bias.to(torch.float32)
        w13_bias = F.pad(w13_bias, (0, layer.w13_right_pad, 0, 0),
                         mode="constant",
                         value=0)

        w2_bias = layer.w2_bias.to(torch.float32)
        w2_bias = F.pad(w2_bias, (0, layer.w2_right_pad, 0, 0),
                        mode="constant",
                        value=0)

        layer.w13_bias = Parameter(w13_bias, requires_grad=False)
        layer.w2_bias = Parameter(w2_bias, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
    ) -> torch.Tensor:

        return triton_kernel_moe_forward(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            gating_output=router_logits,
            topk=top_k,
            renormalize=renormalize,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            w1_bias=layer.w13_bias,
            w2_bias=layer.w2_bias,
            w1_precision=self.w13_precision_config,
            w2_precision=self.w2_precision_config,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
