#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#
import json
from typing import Any

import torch
import torch.distributed as dist

from vllm.distributed.eplb.eplb_adaptor.abstract_adaptor import BaseAdaptor


class VllmEplbAdaptor(BaseAdaptor):

    def __init__(self, model, **args):
        super().__init__(**args)
        self.model = model
        self.rank_id = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.param_dict = dict(self.model.named_parameters())
        if self.model.config.model_type == "qwen3_moe":
            self.num_dense_layers = 0
            self.global_expert_num = self.model.config.num_experts
        else:
            self.num_dense_layers = self.model.config.first_k_dense_replace
            self.global_expert_num = self.model.config.n_routed_experts
        self.num_moe_layers = self.model.config.num_hidden_layers - self.num_dense_layers

        # TODO: init self.expert_weight_names depending on different model types, only deepseek v3 w8a8 and qwen3-moe is supported here
        if self.model.quant_config is not None:
            self.expert_weight_names = [
                "w13_weight", "w2_weight", "w13_weight_scale",
                "w13_weight_offset", "w2_weight_scale", "w2_weight_offset"
            ]
        else:
            self.expert_weight_names = ["w13_weight", "w2_weight"]

        self.expert_map_per_layer = dict(
        )  # reference to expert map on device for expert map update
        self.expert_map_per_layer_cpu = dict(
        )  # copy of expert map on CPU to avoid device synchronize frequently
        for layer_idx in range(self.num_moe_layers):
            self.expert_map_per_layer[self.num_dense_layers + layer_idx] = \
                self.model.get_expert_map(self.num_dense_layers + layer_idx)

        # TODO: here we set number of buffer tensor equal to number of expert in each laryer, which can be improved
        num_buffer_tensor = torch.where(
            self.expert_map_per_layer[self.num_dense_layers] != -1)[0].numel()
        self.buffer_tensor_list: list[list[Any]] = [
            [] for _ in range(num_buffer_tensor)
        ]
        self.init_buffer_tensor(num_buffer_tensor)

        self.expert_param_per_layer = dict()
        self.init_expert_param_per_layer()

        self.log2phy_map_per_layer = dict()
        for layer_idx in range(self.num_moe_layers):
            self.log2phy_map_per_layer[self.num_dense_layers + layer_idx] = \
                self.model.get_log2phy_map(self.num_dense_layers + layer_idx)

        self.all_topk_ids = []

    def init_buffer_tensor(self, num_buffer_tensor):
        for name in self.expert_weight_names:
            complete_name = "model.layers." + str(
                self.num_dense_layers) + ".mlp.experts." + name
            expert_tensor = self.param_dict[complete_name].data[
                            0:num_buffer_tensor]
            buffer_tensors = torch.empty_like(expert_tensor)
            for buffer_id in range(num_buffer_tensor):
                self.buffer_tensor_list[buffer_id].append(
                    buffer_tensors[buffer_id])

    def init_expert_param_per_layer(self):
        num_local_expert = self.param_dict["model.layers." + str(self.num_dense_layers) + \
                                           ".mlp.experts." + self.expert_weight_names[0]].data.shape[0]
        for moe_layer_id in range(self.num_moe_layers):
            layer_idx = self.num_dense_layers + moe_layer_id
            self.expert_param_per_layer[layer_idx] = list()
            for local_expert_id in range(num_local_expert):
                self.expert_param_per_layer[layer_idx].append([
                    self.param_dict["model.layers." + str(layer_idx) +
                                    ".mlp.experts." +
                                    name].data[local_expert_id]
                    for name in self.expert_weight_names
                ])

    def get_rank_expert_workload(self) -> torch.Tensor:
        self.moe_load = self.model.get_all_moe_loads()
        return self.moe_load

    def get_init_expert_map(self, num_moe_layers):
        expert_map = self.model.get_all_expert_map(num_moe_layers)
        if dist.is_initialized():
            world_size = dist.get_world_size()

        gathered = torch.empty(
            (world_size, *expert_map.shape),  # [W, L, E]
            dtype=expert_map.dtype,
            device=expert_map.device)

        dist.all_gather_into_tensor(gathered, expert_map)
        all_maps = gathered.permute(1, 0, 2)
        all_expert_maps = all_maps.cpu()

        for layer_idx in range(num_moe_layers):
            self.expert_map_per_layer_cpu[self.num_dense_layers + layer_idx] = \
                all_expert_maps[layer_idx][self.rank_id]

        return all_expert_maps

    def get_init_expert_map_from_file(self, num_moe_layers, expert_map_path):

        try:
            expert_map_tensor, layers_num, ranks_num = self._expert_file_to_tensor(
                expert_map_path)
            expert_map_all = self.local2global(expert_map_tensor)
        except (TypeError, FileNotFoundError, OSError):
            expert_map_all = self.determine_expert_map_all()

        for layer_idx in range(num_moe_layers):
            if self.model.config.model_type == "qwen3_moe":
                self.expert_map_per_layer_cpu[layer_idx] = \
                    expert_map_all[layer_idx][self.rank_id]
            else:
                self.expert_map_per_layer_cpu[layer_idx + 3] = \
                    expert_map_all[layer_idx][self.rank_id]
        return expert_map_all

    def _expert_file_to_tensor(self, expert_map_path: str):
        with open(expert_map_path, "r") as f:
            data = json.load(f)
            layers_num = data["moe_layer_count"]
            gpus_num = data["layer_list"][0]["device_count"]

            tensor_data = []
            for layer in data["layer_list"]:
                device_data = []
                for device in layer["device_list"]:
                    device_data.append(device["device_expert"])
                tensor_data.append(device_data)
            expert_map_tensor = torch.tensor(tensor_data, dtype=torch.int32)
            return expert_map_tensor, layers_num, gpus_num

    def do_update_expert_map(self, layer_id, updated_expert_map):
        self.expert_map_per_layer[layer_id].copy_(updated_expert_map)
        self.expert_map_per_layer_cpu[layer_id].copy_(updated_expert_map)

    def do_update_expert_weight(self, layer_id, local_expert_to_replace,
                                buffer_tensor_id):
        for expert_tensor, buffer_tensor in zip(
                self.expert_param_per_layer[layer_id][local_expert_to_replace],
                self.buffer_tensor_list[buffer_tensor_id]):
            expert_tensor.copy_(buffer_tensor)

    def do_update_log2phy_map(self, layer_id, updated_log2phy_map):
        if self.log2phy_map_per_layer[layer_id] is not None:
            self.log2phy_map_per_layer[layer_id].copy_(updated_log2phy_map)

    def local2global(self, placement_local: torch.Tensor) -> torch.Tensor:

        L, G, E_local = placement_local.shape
        device = placement_local.device

        max_id = torch.max(placement_local)
        E_global = (max_id + 1).item() if max_id >= 0 else 0

        if E_global == 0:
            return torch.empty((L, G, 0), dtype=torch.long, device=device)

        placement_global = torch.full((L, G, E_global),
                                      fill_value=-1,
                                      dtype=torch.long,
                                      device=device)

        valid = placement_local >= 0
        l_idx, g_idx, slot_idx = valid.nonzero(as_tuple=True)
        gid_idx = placement_local[l_idx, g_idx, slot_idx]

        placement_global[l_idx, g_idx, gid_idx] = slot_idx

        return placement_global

    def determine_expert_map_all(self):

        local_num_experts = self.global_expert_num // self.world_size

        expert_map_all = torch.full(
            (self.num_moe_layers, self.world_size, self.global_expert_num),
            -1,
            dtype=torch.int32)

        for r in range(self.world_size):
            if r < self.world_size - 1:
                start = r * local_num_experts
                end = (r + 1) * local_num_experts
                local_count = local_num_experts
            else:
                start = r * local_num_experts
                end = self.global_expert_num
                local_count = self.global_expert_num - r * local_num_experts

            local_ids = torch.arange(local_count, dtype=torch.int32)
            expert_map_all[:, r, start:end] = local_ids.unsqueeze(0).expand(
                self.num_moe_layers, -1)

        return expert_map_all
