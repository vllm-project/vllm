import types

import torch


def get_expert_map(self, layer_id):
    return self.model.layers[layer_id].mlp.experts.get_map()


def get_log2phy_map(self, layer_id):
    return self.model.layers[layer_id].mlp.experts.get_log2phy_map()


def get_all_expert_map(self, num_moe_layers):
    all_loads = []
    for layer_id in range(num_moe_layers):
        load_tensor = self.get_expert_map(layer_id)  # (num_experts_per_layer,)
        all_loads.append(load_tensor)

    return torch.stack(all_loads, dim=0)


def get_all_moe_loads(self):
    all_moe_loads = torch.stack(
        [self.model.layers[layer_id].mlp.experts.moe_load \
         for layer_id in range(self.num_moe_layers)],
        dim=0
    )
    return all_moe_loads


def clear_all_moe_loads(self):
    for layer_id in range(self.num_moe_layers):
        self.model.layers[layer_id].mlp.experts.clear_moe_load()


def model_register(model, model_config):
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