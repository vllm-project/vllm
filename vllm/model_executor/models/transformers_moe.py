# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2024 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Wrapper around `transformers` MoE models."""
from typing import Any

import torch
import torch.nn as nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config.utils import getattr_iter
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.platforms import current_platform
from vllm.utils import direct_register_custom_op

from .transformers import (TransformersBase, TransformersForCausalLM,
                           TransformersForMultimodalLM,
                           can_enable_torch_compile, log_replacement)
from .utils import maybe_prefix


@CustomOp.register("transformers_fused_moe")
class TransformersFusedMoE(FusedMoE):
    """Custom FusedMoE for the Transformers backend."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._top_k_index: torch.Tensor = None

        def custom_routing_function(hidden_states, gating_output, topk,
                                    renormalize):
            """Return `top_k_weights` from `gating_output` and the
            `top_k_index` we stored in the layer earlier."""
            return gating_output, self._top_k_index

        self.custom_routing_function = custom_routing_function

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """In Transformers `experts.forward` will have this signature.

        We discard any extra kwargs because we cannot use them here."""
        return torch.ops.vllm.transformers_moe_forward(hidden_states,
                                                       top_k_index,
                                                       top_k_weights,
                                                       self.layer_name)


def transformers_moe_forward(hidden_states: torch.Tensor,
                             top_k_index: torch.Tensor,
                             top_k_weights: torch.Tensor,
                             layer_name: str) -> torch.Tensor:
    """Store the `top_k_index` in the layer and call the actual forward."""
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self._top_k_index = top_k_index
    # Clone hidden_states because it will be mutated in-place in FusedMoE
    return self.forward_impl(hidden_states.clone(), top_k_weights)


def transformers_moe_forward_fake(hidden_states: torch.Tensor,
                                  top_k_index: torch.Tensor,
                                  top_k_weights: torch.Tensor,
                                  layer_name: str) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="transformers_moe_forward",
    op_func=transformers_moe_forward,
    mutates_args=["hidden_states"],
    fake_impl=transformers_moe_forward_fake,
    dispatch_key=current_platform.dispatch_key,
    tags=(torch.Tag.needs_fixed_stride_order, ),
)


class TransformersMoEBase(TransformersBase):

    def __init__(self, *, vllm_config, prefix=""):
        self.check_version("4.57.0.dev0", "MoE models support")
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        if self.parallel_config.enable_expert_parallel:
            raise NotImplementedError(
                "Transformers backend does not support expert parallel yet.")
        if self.parallel_config.enable_eplb:
            raise NotImplementedError(
                "Transformers backend does not support expert parallel load "
                "balancing yet.")

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        """
        Params for weights, fp8 weight scales, fp8 activation scales
        (param_name, weight_name, expert_id, shard_id)
        """
        ckpt_names = [
            # (ckpt_gate_proj_name, ckpt_down_proj_name, ckpt_up_proj_name)
            ("gate_proj", "down_proj", "up_proj"),  # Most common MoE style
            ("w1", "w2", "w3"),  # Granite, Mixtral, Phi MoE style
            ("linear", "linear_1", "linear_v"),  # Grok1 style
        ]
        expert_mapping = []
        for gate_proj, down_proj, up_proj in ckpt_names:
            expert_mapping.extend(
                FusedMoE.make_expert_params_mapping(
                    ckpt_gate_proj_name=gate_proj,
                    ckpt_down_proj_name=down_proj,
                    ckpt_up_proj_name=up_proj,
                    num_experts=self.model_config.get_num_experts(),
                    num_redundant_experts=0,  # TODO: enable EPLB
                ))
        return expert_mapping

    def recursive_replace(self):
        """Initialize the MoE layers."""
        text_config = self.text_config

        # Positional arguments
        num_experts = self.model_config.get_num_experts()
        top_k = getattr_iter(text_config, ["num_experts_per_tok", "top_k"],
                             None)
        assert top_k is not None
        hidden_size = text_config.hidden_size
        intermediate_size = getattr_iter(
            text_config, ["moe_intermediate_size", "intermediate_size"], None)
        assert intermediate_size is not None

        # If there are shared experts, the results are
        # reduced after mlp.forward() not inside FusedMoE
        num_experts_shared = getattr_iter(text_config, [
            "num_experts_shared", "n_shared_experts", "moe_num_shared_experts"
        ], 0)
        reduce_results = num_experts_shared == 0

        def add_all_reduce(mlp: nn.Module):
            """Adds an all-reduce to the output of `mlp.forward()`."""

            class MLPWithAllReduce(mlp.__class__):

                def forward(self, *args, **kwargs):
                    output = super().forward(*args, **kwargs)
                    return self.experts.maybe_all_reduce_tensor_model_parallel(
                        output)

            mlp.__class__ = MLPWithAllReduce

        # Unused kwargs since we use custom_routing_function:
        # - `scoring_func` and `e_score_correction_bias` only used for grouped
        #    topk routing inside vLLM and are non-trivial to infer
        #    and hard code `use_grouped_topk=False`
        # - `renormalize` passed anyway because it's easy to infer
        # - `num_expert_group` and `topk_group` used for inferring expert
        #    placement strategy in FusedMoE
        # - `apply_router_weight_on_input` is already applied in Transformers
        renormalize = getattr(text_config, "norm_topk_prob", top_k > 1)
        num_expert_group = getattr(text_config, "n_group", None)
        topk_group = getattr(text_config, "topk_group", None)

        # MoE activation function
        activation = "silu"
        wrapped_arch = self.config.architectures[0].lower()
        if "gptoss" in wrapped_arch:
            activation = "swigluoai"
        elif "grok1" in wrapped_arch:
            activation = "gelu"

        # Expert mapping for `AutoWeightsLoader`
        expert_mapping = self.get_expert_mapping()

        # Configs
        parallel_config = self.parallel_config
        eplb_config = parallel_config.eplb_config

        # Expert parallel load balancing kwargs
        enable_eplb = parallel_config.enable_eplb
        num_redundant_experts = eplb_config.num_redundant_experts

        # Recursively fuse MoE layers
        def _recursive_replace(module: nn.Module, prefix: str):
            for child_name, child_module in module.named_children():
                qual_name = maybe_prefix(prefix, child_name)
                if (child_name == "experts"
                        and isinstance(child_module, nn.ModuleList)):
                    # Alias for readability
                    mlp = module
                    experts = child_module
                    # Do the experts have biases
                    has_bias = False
                    for experts_param_name, _ in experts.named_parameters():
                        if "bias" in experts_param_name:
                            has_bias = True
                            break
                    # Double check there are no shared experts
                    nonlocal reduce_results
                    if reduce_results:
                        for mlp_param_name, _ in mlp.named_parameters():
                            if "shared_expert" in mlp_param_name:
                                reduce_results = False
                                break
                    # Replace experts module with FusedMoE
                    fused_experts = TransformersFusedMoE(
                        num_experts=num_experts,
                        top_k=top_k,
                        hidden_size=hidden_size,
                        intermediate_size=intermediate_size,
                        reduce_results=reduce_results,
                        renormalize=renormalize,
                        # Hard coded because topk happens in Transformers
                        use_grouped_topk=False,
                        num_expert_group=num_expert_group,
                        topk_group=topk_group,
                        quant_config=self.quant_config,
                        prefix=qual_name,
                        activation=activation,
                        enable_eplb=enable_eplb,
                        num_redundant_experts=num_redundant_experts,
                        has_bias=has_bias,
                        expert_mapping=expert_mapping,
                    )
                    mlp.experts = fused_experts
                    log_replacement(qual_name, experts, fused_experts)
                    # If results are not all-reduced in FusedMoE, ensure they
                    # are all-reduced at the end of mlp.forward() if tensor
                    # parallel or expert parallel is enabled
                    if not reduce_results and (fused_experts.tp_size > 1
                                               or fused_experts.ep_size > 1):
                        add_all_reduce(mlp)
                else:
                    _recursive_replace(child_module, prefix=qual_name)

        _recursive_replace(self.model, prefix="model")
        # Continue with the replacement of layers in TransformersBase
        super().recursive_replace()


@support_torch_compile(enable_if=can_enable_torch_compile)
class TransformersMoEForCausalLM(TransformersMoEBase, TransformersForCausalLM):
    pass


@support_torch_compile(
    # set `positions` to last dim to support Qwen-mrope
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    },
    enable_if=can_enable_torch_compile)
class TransformersMoEForMultimodalLM(TransformersMoEForCausalLM,
                                     TransformersForMultimodalLM):
    pass
