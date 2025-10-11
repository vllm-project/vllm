# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Expert Parallel Load Balancer (EPLB) Adaptor Implementation for vLLM.

This module implements distributed expert management for
Mixture-of-Experts (MoE) models in vLLM framework.
Key features include:

1. Expert Mapping Management
   - Maintains physical/logical expert mappings across devices
   - Handles expert placement updates during load balancing

2. Weight Synchronization
   - Manages expert weight buffers for dynamic parameter updates
   - Supports quantized expert weights (w8a8 format)

3. Distributed Coordination
   - Collects global expert workload metrics
   - Implements cross-rank expert map synchronization

Supported Model Architectures:
- DeepSeek V3 (with quantization support)
- Qwen3-MoE
- Kimi K2
- Standard MoE models with configurable expert layers

Note: Current implementation assumes homogeneous expert structure
across MoE layers.
"""

import json
from typing import Any

import torch
import torch.distributed as dist

from vllm.distributed.eplb.eplb_adaptor.abstract_adaptor import BaseAdaptor


class VllmEplbAdaptor(BaseAdaptor):
    """vLLM implementation of Expert Parallel Load Balancer (EPLB) adaptor.

    Handles expert mapping management, weight synchronization and distributed
    coordination for MoE models in vLLM framework.

    Attributes:
        model: vLLM model instance
        rank_id: Current process rank in distributed group
        world_size: Total number of processes in distributed group
        num_dense_layers: Number of dense layers before MoE layers
        global_expert_num: Total number of experts in the model
        num_moe_layers: Number of MoE layers in the model
        expert_weight_names: List of parameter names for expert weights
    """

    def __init__(self, model, **args):
        """Initialize adaptor with model configuration.

        Args:
            model: vLLM model instance containing MoE layers
            **args: Additional base class arguments
        """
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
        self.num_moe_layers = (
            self.model.config.num_hidden_layers - self.num_dense_layers
        )
        # TODO: init self.expert_weight_names depending on different model
        # types, only deepseek v3 w8a8 and qwen3-moe is supported here
        if self.model.quant_config is not None:
            self.expert_weight_names = [
                "w13_weight",
                "w2_weight",
                "w13_weight_scale",
                "w13_weight_offset",
                "w2_weight_scale",
                "w2_weight_offset",
            ]
        else:
            self.expert_weight_names = ["w13_weight", "w2_weight"]

        # reference to expert map on device for expert map update
        self.expert_map_per_layer = dict()

        # copy of expert map on CPU to avoid device synchronize frequently
        self.expert_map_per_layer_cpu = dict()

        for layer_idx in range(self.num_moe_layers):
            self.expert_map_per_layer[self.num_dense_layers + layer_idx] = (
                self.model.get_expert_map(self.num_dense_layers + layer_idx)
            )

        # TODO: here we set number of buffer tensor equal to number of expert
        # in each layer, which can be improved
        num_buffer_tensor = torch.where(
            self.expert_map_per_layer[self.num_dense_layers] != -1
        )[0].numel()
        self.buffer_tensor_list: list[list[Any]] = [
            [] for _ in range(num_buffer_tensor)
        ]
        self.init_buffer_tensor(num_buffer_tensor)

        self.expert_param_per_layer = dict()
        self.init_expert_param_per_layer()

        self.log2phy_map_per_layer = dict()
        for layer_idx in range(self.num_moe_layers):
            self.log2phy_map_per_layer[self.num_dense_layers + layer_idx] = (
                self.model.get_log2phy_map(self.num_dense_layers + layer_idx)
            )
        self.all_topk_ids = []

    def init_buffer_tensor(self, num_buffer_tensor):
        """Initialize buffer tensors for expert weight updates.

        Args:
            num_buffer_tensor: Number of buffer slots per expert parameter
        """
        for name in self.expert_weight_names:
            complete_name = (
                "model.layers." + str(self.num_dense_layers) + ".mlp.experts." + name
            )
            expert_tensor = self.param_dict[complete_name].data[0:num_buffer_tensor]
            buffer_tensors = torch.empty_like(expert_tensor)
            for buffer_id in range(num_buffer_tensor):
                self.buffer_tensor_list[buffer_id].append(buffer_tensors[buffer_id])

    def init_expert_param_per_layer(self):
        """Initialize expert parameter references for all MoE layers."""
        num_local_expert = self.param_dict[
            "model.layers."
            + str(self.num_dense_layers)
            + ".mlp.experts."
            + self.expert_weight_names[0]
        ].data.shape[0]
        for moe_layer_id in range(self.num_moe_layers):
            layer_idx = self.num_dense_layers + moe_layer_id
            self.expert_param_per_layer[layer_idx] = list()
            for local_expert_id in range(num_local_expert):
                self.expert_param_per_layer[layer_idx].append(
                    [
                        self.param_dict[
                            "model.layers." + str(layer_idx) + ".mlp.experts." + name
                        ].data[local_expert_id]
                        for name in self.expert_weight_names
                    ]
                )

    def get_rank_expert_workload(self) -> torch.Tensor:
        """Get current rank's expert workload statistics.

        Returns:
            torch.Tensor: Tensor containing MoE layer workload metrics
        """
        self.moe_load = self.model.get_all_moe_loads()
        return self.moe_load

    def get_init_expert_map(self, num_moe_layers):
        """Collect initial expert mappings across all ranks.

        Args:
            num_moe_layers: Number of MoE layers to process

        Returns:
            torch.Tensor: Global expert mapping tensor [layers, ranks, experts]
        """
        expert_map = self.model.get_all_expert_map(num_moe_layers)
        if dist.is_initialized():
            world_size = self.world_size

        gathered = torch.empty(
            (world_size, *expert_map.shape),  # [W, L, E]
            dtype=expert_map.dtype,
            device=expert_map.device,
        )

        dist.all_gather_into_tensor(gathered, expert_map)
        all_maps = gathered.permute(1, 0, 2)
        all_expert_maps = all_maps.cpu()

        for layer_idx in range(num_moe_layers):
            self.expert_map_per_layer_cpu[self.num_dense_layers + layer_idx] = (
                all_expert_maps[layer_idx][self.rank_id]
            )

        return all_expert_maps

    def get_init_expert_map_from_file(self, num_moe_layers, expert_map_path):
        """Retrieves initial expert mappings from a file.

        If file reading fails or the file does not exist, it falls back to
        the default expert map determination logic.

        Args:
            num_moe_layers: The number of MoE layers to process.
            expert_map_path: The path to the JSON file containing expert
                mapping information.

        Returns:
            torch.Tensor: A global expert mapping tensor of shape
                [layers, ranks, experts].
        """
        try:
            expert_map_tensor, _, _ = self._expert_file_to_tensor(expert_map_path)
            expert_map_all = self.local2global(expert_map_tensor)
        except (TypeError, FileNotFoundError, OSError, json.JSONDecodeError, KeyError):
            expert_map_all = self.determine_expert_map_all()

        for layer_idx in range(num_moe_layers):
            if self.model.config.model_type == "qwen3_moe":
                self.expert_map_per_layer_cpu[layer_idx] = expert_map_all[layer_idx][
                    self.rank_id
                ]
            else:  # adapt both dsv3 and kimik2
                self.expert_map_per_layer_cpu[layer_idx + self.num_dense_layers] = (
                    expert_map_all[layer_idx][self.rank_id]
                )
        return expert_map_all

    def _expert_file_to_tensor(self, expert_map_path: str):
        """Reads expert mappings from a JSON file and converts them to
        a PyTorch tensor.

        The file format is expected to contain 'moe_layer_count'
        and 'layer_list'. Each layer in 'layer_list' should contain a
        'device_list', and each device should contain 'device_expert'.

        Args:
            expert_map_path: The path to the JSON file containing
            expert mapping information.

        Returns:
            tuple: A tuple containing (expert map tensor,
                number of layers, number of GPUs).

        Raises:
            FileNotFoundError: If the file does not exist.
                json.JSONDecodeError: If the file content is not valid JSON.
            KeyError: If the JSON structure does not conform to expectations.
        """
        with open(expert_map_path) as f:
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
        """Performs an update of the expert map.

        Copies the updated expert map to both the on-device map
        and its CPU copy.

        Args:
            layer_id: The ID of the MoE layer to update.
            updated_expert_map: A PyTorch tensor containing
            the new expert map.
        """
        self.expert_map_per_layer[layer_id].copy_(updated_expert_map)
        self.expert_map_per_layer_cpu[layer_id].copy_(updated_expert_map)

    def do_update_expert_weight(
        self, layer_id, local_expert_to_replace, buffer_tensor_id
    ):
        """Performs an update of expert weights.

        Copies weights from a specified buffer tensor to
        the target local expert.

        Args:
            layer_id: The ID of the MoE layer containing
                the expert to update.
            local_expert_to_replace: The local ID of the expert
                whose weights are to be replaced.
            buffer_tensor_id: The ID of the buffer tensor containing
                the new weights.
        """
        for expert_tensor, buffer_tensor in zip(
            self.expert_param_per_layer[layer_id][local_expert_to_replace],
            self.buffer_tensor_list[buffer_tensor_id]
        ):
            expert_tensor.copy_(buffer_tensor)

    def do_update_log2phy_map(self, layer_id, updated_log2phy_map):
        """Performs an update of the logical-to-physical map.

        If a logical-to-physical map exists for the given layer,
        it is updated with the new values.

        Args:
            layer_id: The ID of the MoE layer to update.
            updated_log2phy_map: A PyTorch tensor containing
                the new logical-to-physical map.
        """
        if self.log2phy_map_per_layer[layer_id] is not None:
            self.log2phy_map_per_layer[layer_id].copy_(updated_log2phy_map)

    def local2global(self, placement_local: torch.Tensor) -> torch.Tensor:
        """Converts a local expert placement map to a global expert
        placement map.

        A local map typically only contains the IDs of local experts
        on each device. This function transforms it into a global view
        where each expert slot contains its global expert ID.

        For example, if the local placement is `[[0, 1], [2, 3]]` (meaning
        device 0 has experts 0,1 and device 1 has experts 2,3), the global
        placement would be `[[0, 1, -1, -1], [-1, -1, 0, 1]]` (meaning
        global expert 0 is in slot 0 on device 0, global expert 1 is in
        slot 1 on device 0, global expert 2 is in slot 0 on device 1,
        and global expert 3 is in slot 1 on device 1).

        Args:
            placement_local: A local expert placement tensor of shape
                [L, G, E_local], where L is the number of layers,
                G is the number of GPUs, and E_local is the number of
                local expert slots per GPU. Values represent local
                expert IDs.

        Returns:
            torch.Tensor: A global expert placement tensor of shape
                [L, G, E_global], where E_global is the total number of
                global experts. Values represent local slot IDs,
                with -1 indicating that this global expert is not
                on this device.
        """
        L, G, E_local = placement_local.shape
        device = placement_local.device

        max_id = torch.max(placement_local)
        E_global = (max_id + 1).item() if max_id >= 0 else 0

        if E_global == 0:
            return torch.empty((L, G, 0), dtype=torch.long, device=device)

        placement_global = torch.full(
            (L, G, E_global), fill_value=-1, dtype=torch.long, device=device
        )

        valid = placement_local >= 0
        l_idx, g_idx, slot_idx = valid.nonzero(as_tuple=True)
        gid_idx = placement_local[l_idx, g_idx, slot_idx]

        placement_global[l_idx, g_idx, gid_idx] = slot_idx

        return placement_global

    def determine_expert_map_all(self):
        """Determines the default expert mapping across all ranks.

        This method distributes experts evenly among each rank based on
        the total number of global experts and the world size. Each rank is
        responsible for a contiguous range of global experts.

        Returns:
            torch.Tensor: A global expert mapping tensor of shape
                [layers, ranks, global_expert_num]. The values in the tensor
                represent the local slot ID of that global expert on the
                corresponding rank.
        """
        local_num_experts = self.global_expert_num // self.world_size

        expert_map_all = torch.full(
            (self.num_moe_layers, self.world_size, self.global_expert_num),
            -1,
            dtype=torch.int32,
        )

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
                self.num_moe_layers, -1
            )

        return expert_map_all
