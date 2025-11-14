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
"""Transformers modeling backend mixin for Mixture of Experts (MoE) models."""

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from vllm.config.utils import getattr_iter
from vllm.distributed import get_dp_group, get_ep_group
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.models.interfaces import MixtureOfExperts
from vllm.model_executor.models.utils import maybe_prefix
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from .utils import log_replacement

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@CustomOp.register("transformers_fused_moe")
class TransformersFusedMoE(FusedMoE):
    """Custom FusedMoE for the Transformers modeling backend."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._topk_ids: torch.Tensor = None

        def custom_routing_function(hidden_states, gating_output, topk, renormalize):
            """Return `topk_weights` from `gating_output` and the
            `topk_ids` we stored in the layer earlier."""
            topk_weights = gating_output
            topk_ids = self._topk_ids
            # Handle all gather in expert parallel
            if topk_ids.size(0) != hidden_states.size(0):
                dp_metadata = get_forward_context().dp_metadata
                sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
                is_sp = self.is_sequence_parallel
                dist_group = get_ep_group() if is_sp else get_dp_group()
                assert sizes[dist_group.rank_in_group] == topk_ids.shape[0]
                (topk_ids,) = dist_group.all_gatherv([topk_ids], 0, sizes)
            return topk_weights, topk_ids

        self.custom_routing_function = custom_routing_function

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """In Transformers `experts.forward` will have this signature.

        We discard any extra kwargs because we cannot use them here."""
        return torch.ops.vllm.transformers_moe_forward(
            hidden_states,
            topk_ids.to(torch.int32),
            topk_weights.to(torch.float32),
            self.layer_name,
        )


def transformers_moe_forward(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    """Store the `topk_ids` in the layer and call the actual forward."""
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self._topk_ids = topk_ids
    # Clone hidden_states because it will be mutated in-place in FusedMoE
    return self.forward_impl(hidden_states.clone(), topk_weights)


def transformers_moe_forward_fake(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="transformers_moe_forward",
    op_func=transformers_moe_forward,
    mutates_args=["hidden_states"],
    fake_impl=transformers_moe_forward_fake,
    dispatch_key=current_platform.dispatch_key,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


class MoEMixin(MixtureOfExperts):
    def __init__(self, *, vllm_config: "VllmConfig", prefix: str = ""):
        self.check_version("5.0.0.dev0", "MoE models support")
        # Skip MixtureOfExperts.__init__ and call the next class in MRO
        super(MixtureOfExperts, self).__init__(vllm_config=vllm_config, prefix=prefix)

    def set_eplb_state(
        self,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ):
        for moe_layer_idx, mlp_layer in enumerate(self.mlp_moe_layers):
            mlp_layer.experts.set_eplb_state(
                moe_layer_idx=moe_layer_idx,
                expert_load_view=expert_load_view,
                logical_to_physical_map=logical_to_physical_map,
                logical_replica_count=logical_replica_count,
            )

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ):
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for mlp in self.mlp_moe_layers:
            mlp.n_local_physical_experts = num_local_physical_experts
            mlp.n_physical_experts = num_physical_experts
            mlp.n_redundant_experts = self.num_redundant_experts
            mlp.experts.update_expert_map()

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
        num_experts = self.model_config.get_num_experts()
        num_redundant_experts = self.parallel_config.eplb_config.num_redundant_experts
        expert_mapping = []
        for gate_proj, down_proj, up_proj in ckpt_names:
            expert_mapping.extend(
                FusedMoE.make_expert_params_mapping(
                    ckpt_gate_proj_name=gate_proj,
                    ckpt_down_proj_name=down_proj,
                    ckpt_up_proj_name=up_proj,
                    num_experts=num_experts,
                    num_redundant_experts=num_redundant_experts,
                )
            )
        return expert_mapping

    def recursive_replace(self):
        """Initialize the MoE layers."""
        text_config = self.text_config

        # Positional arguments
        num_experts = self.model_config.get_num_experts()
        top_k = getattr_iter(text_config, ["num_experts_per_tok", "top_k"], None)
        assert top_k is not None
        hidden_size = text_config.hidden_size
        intermediate_size = getattr_iter(
            text_config, ["moe_intermediate_size", "intermediate_size"], None
        )
        assert intermediate_size is not None

        # If there are shared experts, the results are
        # reduced after mlp.forward() not inside FusedMoE
        num_shared_experts = getattr_iter(
            text_config,
            [
                "n_shared_experts",  # DeepSeek, Docs, GLM
                "moe_num_shared_experts",  # Aria, Ernie
            ],
            0,
        )
        reduce_results = num_shared_experts == 0

        def add_all_reduce(mlp: nn.Module):
            """Adds an all-reduce to the output of `mlp.forward()`."""

            class MLPWithAllReduce(mlp.__class__):
                def forward(self, *args, **kwargs):
                    output = super().forward(*args, **kwargs)
                    return self.experts.maybe_all_reduce_tensor_model_parallel(output)

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

        # Expert parallel load balancing kwargs
        enable_eplb = self.parallel_config.enable_eplb
        num_redundant_experts = self.parallel_config.eplb_config.num_redundant_experts

        # MixtureOfExperts mixin settings
        ep_size = get_ep_group().world_size

        self.mlp_moe_layers = []  # Used for MixtureOfExperts methods
        self.moe_layers = []
        self.expert_weights = []
        self.num_moe_layers = 0
        self.num_expert_groups = 1 if num_expert_group is None else num_expert_group
        self.num_logical_experts = num_experts
        self.num_physical_experts = num_experts + num_redundant_experts
        self.num_local_physical_experts = self.num_physical_experts // ep_size
        self.num_routed_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.num_redundant_experts = num_redundant_experts

        # Recursively fuse MoE layers
        def _recursive_replace(module: nn.Module, prefix: str):
            for child_name, child_module in module.named_children():
                qual_name = maybe_prefix(prefix, child_name)
                if child_name == "experts" and isinstance(child_module, nn.ModuleList):
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
                                # If the config does not specify num_shared_experts, but
                                # the model has shared experts, we assume there is one.
                                self.num_shared_experts = 1
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
                    # Update MixtureOfExperts mixin state
                    self.mlp_moe_layers.append(mlp)
                    self.moe_layers.append(fused_experts)
                    self.expert_weights.append(fused_experts.get_expert_weights())
                    self.num_moe_layers += 1
                    # If results are not all-reduced in FusedMoE, ensure they
                    # are all-reduced at the end of mlp.forward() if tensor
                    # parallel or expert parallel is enabled
                    if not reduce_results and (
                        fused_experts.tp_size > 1 or fused_experts.ep_size > 1
                    ):
                        add_all_reduce(mlp)
                else:
                    _recursive_replace(child_module, prefix=qual_name)

        _recursive_replace(self.model, prefix="model")
        # Continue with the replacement of layers in Base
        super().recursive_replace()
