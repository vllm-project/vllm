# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import types

import torch


def get_expert_map(self, layer_id):
    """
    Retrieves the expert map for a specific MoE layer.
    This map typically indicates the mapping from logical expert IDs to physical expert IDs
    or some other internal representation of expert distribution within that layer.

    Args:
        self: The model instance (e.g., an instance of a vLLM model).
        layer_id: The index of the MoE layer.

    Returns:
        A torch.Tensor representing the expert map for the specified layer.
    """
    return self.model.layers[layer_id].mlp.experts.get_map()


def get_log2phy_map(self, layer_id):
    """
    Retrieves the logical-to-physical expert mapping for a specific MoE layer.
    This map determines which physical expert (identified by its global physical ID)
    a logical expert ID should map to for the current rank.

    Args:
        self: The model instance.
        layer_id: The index of the MoE layer.

    Returns:
        A torch.Tensor representing the logical-to-physical mapping for the specified layer.
    """
    return self.model.layers[layer_id].mlp.experts.get_log2phy_map()


def get_all_expert_map(self, num_moe_layers):
    """
    Aggregates the expert maps from all MoE layers into a single tensor.

    Args:
        self: The model instance.
        num_moe_layers: The total number of MoE layers in the model.

    Returns:
        A torch.Tensor of shape (num_moe_layers, num_experts_per_layer),
        where each row is the expert map for a corresponding MoE layer.
    """
    all_loads = []
    for layer_id in range(num_moe_layers):
        load_tensor = self.get_expert_map(layer_id)  # (num_experts_per_layer,)
        all_loads.append(load_tensor)

    return torch.stack(all_loads, dim=0)


def get_all_moe_loads(self):
    """
    Retrieves the current expert load (e.g., token counts routed to each expert)
    for all MoE layers. This typically reflects the load accumulated during the
    most recent forward pass.

    Args:
        self: The model instance.

    Returns:
        A torch.Tensor of shape (num_moe_layers, ...), where the dimensions
        after the first depend on how expert_load_view is structured (e.g.,
        (num_physical_experts) or (num_ranks, num_physical_experts)).
    """
    all_moe_loads = torch.stack(
        [self.model.layers[layer_id].mlp.experts.expert_load_view \
         for layer_id in range(self.num_moe_layers)],
        dim=0
    )
    return all_moe_loads


def clear_all_moe_loads(self):
    """
    Resets the expert load counters for all MoE layers.
    This is typically called after an aggregation step or at the beginning of a new
    load measurement period.

    Args:
        self: The model instance.
    """
    for layer_id in range(self.num_moe_layers):
        self.model.layers[layer_id].mlp.experts.clear_moe_load()


def model_register(model, model_config):
    """
    Registers custom methods related to Expert Parallel Load Balancing (EPLB)
    onto the vLLM model instance. It also determines the number of MoE layers
    based on the model configuration.

    Args:
        model: The vLLM model instance to which the methods will be added.
        model_config: The configuration object for the model, containing details
                      like model_type and layer counts.
    """
    model.get_expert_map = types.MethodType(get_expert_map, model)
    model.get_log2phy_map = types.MethodType(get_log2phy_map, model)
    model.get_all_expert_map = types.MethodType(get_all_expert_map, model)
    model.get_all_moe_loads = types.MethodType(get_all_moe_loads, model)
    model.clear_all_moe_loads = types.MethodType(clear_all_moe_loads, model)

    config = model_config.hf_config

    if config.model_type == "qwen3_moe":
        model.num_moe_layers = config.num_hidden_layers
    elif config.model_type == "deepseek_v2":
        num_dense_layers = config.first_k_dense_replace
        model.num_moe_layers = config.num_hidden_layers - num_dense_layers
    else:
        raise NotImplementedError("EPLB is not supported.")