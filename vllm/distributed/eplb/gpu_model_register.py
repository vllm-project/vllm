# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import types
import typing
from typing import Callable

import torch

from vllm.model_executor.layers.fused_moe import SharedFusedMoE
from vllm.model_executor.models.utils import is_pp_missing_parameter


def set_eplb_state(
    self,
    expert_load_view: torch.Tensor,
    logical_to_physical_map: torch.Tensor,
    logical_replica_count: torch.Tensor,
) -> None:
    for layer_idx, layer in enumerate(self.moe_layers):
        # Register the expert weights.
        self.expert_weights.append(layer.get_expert_weights())
        layer.set_eplb_state(
            moe_layer_idx=layer_idx,
            expert_load_view=expert_load_view,
            logical_to_physical_map=logical_to_physical_map,
            logical_replica_count=logical_replica_count,
        )


def update_physical_experts_metadata(
    self,
    num_physical_experts: int,
    num_local_physical_experts: int,
) -> None:
    assert self.num_local_physical_experts == num_local_physical_experts
    self.num_physical_experts = num_physical_experts
    self.num_local_physical_experts = num_local_physical_experts
    self.num_redundant_experts = num_physical_experts - self.num_logical_experts
    for layer in self.model.layers:
        if isinstance(layer.mlp, self.example_moe):
            moe = layer.mlp
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_physical_experts = num_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()


def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
    # Params for weights, fp8 weight scales, fp8 activation scales
    # (param_name, weight_name, expert_id, shard_id)
    return SharedFusedMoE.make_expert_params_mapping(
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=self.config.n_routed_experts,
        num_redundant_experts=self.num_redundant_experts,
    )


def load_expert_weight(self, mapping, name, loaded_weight, params_dict):
    ignore_suffixes = (
        ".bias",
        "_bias",
        ".k_scale",
        "_k_scale",
        ".v_scale",
        "_v_scale",
        ".weight_scale",
        "_weight_scale",
        ".input_scale",
        "_input_scale",
    )
    expert_matched = False
    is_continue = False
    success = False
    name_mapped = ""
    param_name, weight_name, expert_id, shard_id = mapping
    if weight_name not in name:
        is_continue = True
        return expert_matched, is_continue, success, name_mapped

    # Anyway, this is an expert weight and should not be
    # attempted to load as other weights later
    expert_matched = True

    # Do not modify `name` since the loop may continue here
    # Instead, create a new variable
    name_mapped = name.replace(weight_name, param_name)

    if is_pp_missing_parameter(name_mapped, self):
        is_continue = True
        return expert_matched, is_continue, success, name_mapped

    # Skip loading extra parameters for GPTQ/modelopt models.
    if name_mapped.endswith(ignore_suffixes) and name_mapped not in params_dict:
        is_continue = True
        return expert_matched, is_continue, success, name_mapped

    param = params_dict[name_mapped]
    # We should ask the weight loader to return success or not
    # here since otherwise we may skip experts with other
    # available replicas.
    weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
    success = weight_loader(
        param,
        loaded_weight,
        name_mapped,
        shard_id=shard_id,
        expert_id=expert_id,
        return_success=True,
    )
    return expert_matched, is_continue, success, name_mapped


def model_register(model):
    """
    Registers custom methods related to Expert Parallel Load Balancing (EPLB)
    onto the vLLM model instance. It also determines the number of MoE layers
    based on the model configuration.

    Args:
        model: The vLLM model instance to which the methods will be added.
    """
    model.set_eplb_state = types.MethodType(set_eplb_state, model)
    model.load_expert_weight = types.MethodType(load_expert_weight, model)
    model.update_physical_experts_metadata = types.MethodType(
        update_physical_experts_metadata, model
    )
    model.model.get_expert_mapping = types.MethodType(get_expert_mapping, model.model)
    print("register complete")
