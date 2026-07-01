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

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from vllm.config.utils import getattr_iter
from vllm.distributed import get_dp_group, get_ep_group
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.fused_moe import FusedMoE, MoERunner, RoutedExperts
from vllm.model_executor.models.interfaces import MixtureOfExperts
from vllm.model_executor.models.transformers.fusers.moe import MoEFuser
from vllm.model_executor.models.utils import maybe_prefix
from vllm.utils.torch_utils import direct_register_custom_op

from .utils import log_replacement

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


@dataclass
class TransformersMoEState:
    topk_ids: torch.Tensor | None = None
    is_sequence_parallel: bool = False


# --8<-- [start:transformers_fused_moe]
@PluggableLayer.register("transformers_fused_moe")
class TransformersMoERunner(MoERunner):
    """Custom FusedMoE for the Transformers modeling backend."""

    # --8<-- [end:transformers_fused_moe]
    def __init__(self, *args, moe_state: TransformersMoEState, **kwargs):
        super().__init__(*args, **kwargs)
        self._moe_state = moe_state
        self._moe_state.is_sequence_parallel = self.moe_config.is_sequence_parallel

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """In Transformers `experts.forward` will have this signature.

        We discard any extra kwargs because we cannot use them here."""
        # Note: we need to forward through a custom op so the topk_ids
        # can be transferred without interfering with cudagraphs.
        return torch.ops.vllm.transformers_moe_forward(
            hidden_states,
            topk_ids.to(torch.int32),
            topk_weights.to(torch.float32),
            self.layer_name,
        )

    def _forward_super(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        return super().forward(hidden_states, topk_weights)


def _transformers_moe_forward(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    """Store the `topk_ids` in the layer and call the actual forward."""
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self._moe_state.topk_ids = topk_ids
    return self._forward_super(hidden_states, topk_weights)


def _transformers_moe_forward_fake(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="transformers_moe_forward",
    op_func=_transformers_moe_forward,
    mutates_args=["hidden_states"],
    fake_impl=_transformers_moe_forward_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


class TransformersRoutedExperts(RoutedExperts):
    def get_expert_mapping(
        self, include_fused: bool = False
    ) -> list[tuple[str, str, int, str]]:
        common_names = ("gate_proj", "down_proj", "up_proj")
        common_map = super().get_expert_mapping(*common_names, include_fused)
        mixtral_map = super().get_expert_mapping("w1", "w2", "w3", include_fused)
        if not include_fused:
            return common_map + mixtral_map
        common_fused, common_unfused = common_map[:3], common_map[3:]
        mixtral_fused, mixtral_unfused = mixtral_map[:3], mixtral_map[3:]
        return common_fused + mixtral_fused + common_unfused + mixtral_unfused


class MoEMixin(MixtureOfExperts):
    def __init__(self, *, vllm_config: "VllmConfig", prefix: str = ""):
        self.check_version("5.0.0", "MoE models support")
        # Skip MixtureOfExperts.__init__ and call the next class in MRO
        super(MixtureOfExperts, self).__init__(vllm_config=vllm_config, prefix=prefix)

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ):
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for mlp in self.mlp_layers:
            mlp.n_local_physical_experts = num_local_physical_experts
            mlp.n_physical_experts = num_physical_experts
            mlp.n_redundant_experts = self.num_redundant_experts
            mlp.experts.update_expert_map()

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

        num_shared_experts = getattr_iter(
            text_config,
            [
                "n_shared_experts",  # DeepSeek, Docs, GLM
                "moe_num_shared_experts",  # Aria, Ernie
            ],
            0,
        )

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

        # Expert parallel load balancing kwargs
        enable_eplb = self.parallel_config.enable_eplb
        num_redundant_experts = self.parallel_config.eplb_config.num_redundant_experts

        # MixtureOfExperts mixin settings
        ep_size = get_ep_group().world_size

        self.mlp_layers = []  # Used for MixtureOfExperts methods
        self.moe_layers = []
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
                # Naive implementations will have experts as ModuleList
                is_modulelist = isinstance(child_module, nn.ModuleList)
                # Packed implementations will have experts as 3D tensors of shapes like:
                # gate_up_proj = (num_experts, 2 * intermediate_size, hidden_size)
                # down_proj = (num_experts, intermediate_size, hidden_size)
                params = list(child_module.parameters())
                is_3d = len(params) > 0 and all(p.ndim == 3 for p in params)
                if child_name == "experts" and (is_modulelist or is_3d):
                    # Alias for readability
                    mlp = module
                    experts = child_module
                    # Class of the fused block (parent of gate/experts/shared)
                    mlp_cls = type(mlp).__name__
                    experts_cls = type(experts).__name__
                    # Do the experts have biases
                    has_bias = False
                    for experts_param_name, _ in experts.named_parameters():
                        if "bias" in experts_param_name:
                            has_bias = True
                            break
                    # If the config does not specify num_shared_experts, but
                    # the model has shared experts, we assume there is one.
                    if self.num_shared_experts == 0:
                        for mlp_param_name, _ in mlp.named_parameters():
                            if "shared_expert" in mlp_param_name:
                                self.num_shared_experts = 1
                                break

                    kwargs: dict[str, Any] = dict(
                        num_experts=num_experts,
                        top_k=top_k,
                        hidden_size=hidden_size,
                        intermediate_size=intermediate_size,
                        renormalize=renormalize,
                        use_grouped_topk=False,
                        quant_config=self.quant_config,
                        prefix=qual_name,
                        activation=activation,
                        enable_eplb=enable_eplb,
                        num_redundant_experts=num_redundant_experts,
                        has_bias=has_bias,
                        routed_experts_cls=TransformersRoutedExperts,
                    )
                    if self.num_expert_groups <= 1 and (fuser := MoEFuser.match(mlp)):
                        # MLP forward is fully replaced.
                        # gate/router and shared expert (if any) runs in FusedMoE.
                        gate = fuser.build_gate(mlp, prefix)
                        shared_experts = fuser.build_shared_experts(mlp, prefix)
                        if shared_experts is not None:
                            self.hf_to_vllm_mapper.orig_to_new_stacked.update(
                                fuser.orig_to_new_stacked(prefix)
                            )
                        kwargs |= dict(
                            scoring_func=fuser.scoring_func,
                            is_sequence_parallel=(
                                self.parallel_config.use_sequence_parallel_moe
                            ),
                            gate=gate,
                            shared_experts=shared_experts,
                        )
                        fuser.rewrite_forward(mlp)
                        routed = "gate + experts"
                        if fuser.shared_name:
                            routed += " + shared experts"
                        logger.info_once(
                            "Fused: %s (%s) -> FusedMoE (internal routing)",
                            routed,
                            mlp_cls,
                        )
                    else:
                        # MLP forward is unmodified.
                        # gate/router and shared expert (if any) runs in Transformers.
                        # We then smuggle the topk_ids in using a custom op.
                        moe_state = TransformersMoEState()

                        def custom_routing_function(
                            hidden_states: torch.Tensor,
                            gating_output: torch.Tensor,
                            topk: int,
                            renormalize: bool,
                            moe_state: TransformersMoEState,
                        ):
                            """Return `topk_weights` from `gating_output` and the
                            `topk_ids` we stored in the layer earlier."""
                            topk_weights = gating_output
                            topk_ids = moe_state.topk_ids
                            assert topk_ids is not None
                            # Handle all gather in expert parallel
                            if topk_ids.size(0) != hidden_states.size(0):
                                dp_metadata = get_forward_context().dp_metadata
                                sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
                                is_sp = moe_state.is_sequence_parallel
                                group = get_ep_group() if is_sp else get_dp_group()
                                assert sizes[group.rank_in_group] == topk_ids.shape[0]
                                (topk_ids,) = group.all_gatherv([topk_ids], 0, sizes)
                            return topk_weights, topk_ids

                        kwargs |= dict(
                            num_expert_group=num_expert_group,
                            topk_group=topk_group,
                            custom_routing_function=partial(
                                custom_routing_function, moe_state=moe_state
                            ),
                            runner_cls=TransformersMoERunner,
                            runner_args={"moe_state": moe_state},
                        )
                        logger.info_once(
                            "Fused: experts (%s) -> FusedMoE (external routing)",
                            experts_cls,
                        )
                    fused_experts = FusedMoE(**kwargs)
                    mlp.experts = fused_experts
                    log_replacement(qual_name, experts, fused_experts)
                    # Update MixtureOfExperts mixin state
                    self.mlp_layers.append(mlp)
                    self.moe_layers.append(fused_experts)
                else:
                    _recursive_replace(child_module, prefix=qual_name)

        _recursive_replace(self.model, prefix="model")
        self.num_moe_layers = len(self.moe_layers)
        # Continue with the replacement of layers in Base
        super().recursive_replace()
