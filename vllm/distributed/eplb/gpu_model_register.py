# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import types

import torch


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
        moe = (
            getattr(layer, "mlp", None)
            or getattr(layer, "feed_forward", None)
            or getattr(layer, "block_sparse_moe", None)
        )
        if not isinstance(moe, self.example_moe):
            continue
        moe.n_local_physical_experts = num_local_physical_experts
        moe.n_physical_experts = num_physical_experts
        moe.n_redundant_experts = self.num_redundant_experts
        moe.experts.update_expert_map()


def model_register(model):
    """
    Registers custom methods related to Expert Parallel Load Balancing (EPLB)
    onto the vLLM model instance. It also determines the number of MoE layers
    based on the model configuration.

    Args:
        model: The vLLM model instance to which the methods will be added.
    """
    model.set_eplb_state = types.MethodType(set_eplb_state, model)
    model.update_physical_experts_metadata = types.MethodType(
        update_physical_experts_metadata, model
    )
    print("register complete")
