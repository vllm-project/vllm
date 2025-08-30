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

from typing import Any

import torch

from vllm.distributed.eplb.eplb_adaptor.eplb_adaptor import EplbAdaptor


class QwenEplbAdaptor(EplbAdaptor):

    def __init__(self, model, **args):
        super().__init__(**args)
        self.num_dense_layers = 0
        self.global_expert_num = self.model.config.num_experts

        self.num_moe_layers = self.model.config.num_hidden_layers - self.num_dense_layers
        
        for layer_idx in range(self.num_moe_layers):
            self.expert_map_per_layer[self.num_dense_layers + layer_idx] = \
                self.model.get_expert_map(self.num_dense_layers + layer_idx)
        num_buffer_tensor = torch.where(
            self.expert_map_per_layer[self.num_dense_layers] != -1)[0].numel()
        self.buffer_tensor_list: list[list[Any]] = [
            [] for _ in range(num_buffer_tensor)
        ]
        self.init_buffer_tensor(num_buffer_tensor)
        self.init_expert_param_per_layer()
        for layer_idx in range(self.num_moe_layers):
            self.log2phy_map_per_layer[self.num_dense_layers + layer_idx] = \
                self.model.get_log2phy_map(self.num_dense_layers + layer_idx)

    def get_init_expert_map_from_file(self, num_moe_layers, expert_map_path):

        try:
            expert_map_tensor, layers_num, ranks_num = self._expert_file_to_tensor(
                expert_map_path)
            expert_map_all = self.local2global(expert_map_tensor)
        except (TypeError, FileNotFoundError, OSError):
            expert_map_all = self.determine_expert_map_all()

        for layer_idx in range(num_moe_layers):
            self.expert_map_per_layer_cpu[layer_idx] = \
                expert_map_all[layer_idx][self.rank_id]
        return expert_map_all