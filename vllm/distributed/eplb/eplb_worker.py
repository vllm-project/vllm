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

from multiprocessing import Process, Manager
from queue import Queue
from typing import Any

import networkx as nx  # type: ignore
import numpy as np
import torch
import torch.distributed as dist

from vllm.distributed import get_node_count, get_ep_group
from vllm.distributed.eplb.eplb_policy.policy_factory import PolicyFactory
from vllm.distributed.eplb.eplb_policy.abstract_policy import EplbPolicy
from vllm.logger import logger

from vllm.distributed.eplb.eplb_utils.eplb_utils import generate_log2phy_map


class EplbWorker:

    def __init__(self, shared_dict, policy_type, enable_d2d: bool = True):
        self.policy_type = policy_type
        self.policy = PolicyFactory.generate_policy(policy_type)
        self.shared_dict = shared_dict
        self.old_expert_maps = None
        self.enable_d2d = enable_d2d
        self.rank_id = dist.get_rank()

    def do_update(self):
        # put data in to queue
        # in process self.policy.generate_policy()
        # get epxert table && tensor

        # async stream
        # D2D
        # H2D

        # Get initial expert_map
        if self.old_expert_maps is None:
            self.old_expert_maps = self.get_init_expert_maps()
            if self.old_expert_maps is not None:
                self.num_local_experts = self.old_expert_maps.max() + 1
            else:
                raise ValueError("Failed to get expert_maps from shared_dict.")

        # Get MOE load information
        load_info = self.fetch_and_sum_load_info()
        if load_info is None:
            return

        # Get the updated expert table based on the workload information
        old_placement = self.global2local(self.old_expert_maps,
                                          self.num_local_experts)
        # 入参适配新的rebalance格式，
        new_placement = self.calculate_rebalance_experts(
            load_info, old_placement)

        if not torch.is_tensor(new_placement):
            new_placement = torch.tensor(new_placement)
        self.check_expert_placement(old_placement, new_placement)
        new_expert_maps = self.local2global(new_placement)
        self.update_expert_map(new_expert_maps)

        update_info = self.compose_expert_update_info_greedy(
            new_expert_maps, self.old_expert_maps)
        self.old_expert_maps = new_expert_maps
        logger.info("EPLB Process compute complete")

        packed_update_info = self.pack_update_info(update_info)

        return packed_update_info

    def check_expert_placement(self, old_placement, new_placement):
        num_layers = old_placement.shape[0]
        num_ranks = old_placement.shape[1]

        for layer_id in range(num_layers):
            # check if any logical expert is not placed on any rank
            if torch.unique(new_placement[layer_id]).numel() < torch.unique(
                    old_placement[layer_id]).numel():
                logger.error(
                    f"There exists expert not placed on any rank in layer {layer_id}"
                )
                new_placement[layer_id] = old_placement[layer_id]
                continue

            for rank_id in range(num_ranks):
                new_placement_check = new_placement[layer_id][rank_id]
                old_placement_check = old_placement[layer_id][rank_id]

                # check if same logical experts are placed on the same NPU
                if new_placement_check.numel() != torch.unique(
                        new_placement_check).numel():
                    logger.error(
                        f"Replicated experts are placed on the same NPU, expert placement on layer {layer_id}, rank {rank_id} is invalid"
                    )
                    new_placement[layer_id] = old_placement[layer_id]
                    break

                # check if there is any experts movement inside one NPU
                expert_not_move = torch.isin(new_placement_check,
                                             old_placement_check)
                if not torch.equal(new_placement_check[expert_not_move],
                                   old_placement_check[expert_not_move]):
                    logger.error(
                        f"There exists expert movement inside NPU, expert placement on layer {layer_id}, rank {rank_id} is invalid"
                    )
                    new_placement[layer_id] = old_placement[layer_id]
                    break

    def compose_expert_update_info_bipartite(self, updated_expert_maps_org,
                                             current_expert_maps_org):
        # transform numpy array to torch tensor
        updated_expert_maps = updated_expert_maps_org.clone()
        current_expert_maps = current_expert_maps_org.clone()
        updated_expert_maps = np.array(updated_expert_maps)
        current_expert_maps = np.array(current_expert_maps)

        num_layers = current_expert_maps.shape[0]

        for layer_id in range(num_layers):
            updated_expert_maps_this_layer = updated_expert_maps[layer_id]
            current_expert_maps_this_layer = current_expert_maps[layer_id]
            updated_expert_maps_this_layer_org = updated_expert_maps_org[
                layer_id]

            from typing import Any

            expert_send_info_this_layer: dict[Any, Any] = {}
            expert_recv_info_this_layer: dict[Any, Any] = {}

            # Guard Clause: if there is no expert weight update, avoid subsequent processing
            if (np.equal(updated_expert_maps_this_layer,
                         current_expert_maps_this_layer)).all():
                yield (expert_send_info_this_layer,
                       expert_recv_info_this_layer,
                       updated_expert_maps_this_layer_org, layer_id)

            # Parse expert_ids each rank needs to receive from other ranks
            dst_rank_indices, experts_to_recv = np.where(
                (current_expert_maps_this_layer == -1)
                & (updated_expert_maps_this_layer != -1))

            # record src ranks for potential transfer
            src_ranks_set = dict()
            for idx in range(len(dst_rank_indices)):
                expert_id = experts_to_recv[idx].item()
                if expert_id not in src_ranks_set:
                    src_ranks_set[expert_id] = np.where(
                        current_expert_maps_this_layer[:, expert_id] != -1)[0]

            # loop until all experts are scheduled
            while len(dst_rank_indices) > 0:
                # construct bipartite graph
                graph_expert_update: nx.Graph = nx.Graph()
                for idx in range(len(dst_rank_indices)):
                    dst_rank_id = dst_rank_indices[idx].item()
                    expert_id = experts_to_recv[idx].item()
                    # add src ranks
                    src_rank_ids = src_ranks_set[expert_id]
                    graph_expert_update.add_nodes_from(src_rank_ids,
                                                       bipartite=0)
                    # add dest rank
                    graph_expert_update.add_node(str(dst_rank_id), bipartite=1)
                    # add edges
                    for src_rank_id in src_rank_ids:
                        graph_expert_update.add_edge(src_rank_id,
                                                     str(dst_rank_id))

                # graph may not be connected
                connected_components = list(
                    nx.connected_components(graph_expert_update))
                all_matches = {}
                # matching in this loop
                for i, component in enumerate(connected_components):
                    subgraph = graph_expert_update.subgraph(component)
                    component_matching = nx.bipartite.maximum_matching(
                        subgraph)
                    all_matches.update(component_matching)

                for src_rank, dst_rank in all_matches.items():
                    dst_rank = int(dst_rank)
                    assert src_rank != dst_rank
                    if graph_expert_update.nodes[src_rank]['bipartite'] == 0:
                        # currently not scheduled experts in rank dst_rank
                        experts_v = experts_to_recv[np.where(
                            dst_rank_indices == dst_rank)]
                        # src: src_rank, dest: dst_rank, expert: expert_id
                        expert_id = np.intersect1d(
                            experts_v,
                            np.where(
                                current_expert_maps_this_layer[src_rank] != -1)
                        )[0]

                        # record send/rcv pairs
                        if src_rank not in expert_send_info_this_layer:
                            expert_send_info_this_layer[src_rank] = []
                        if dst_rank not in expert_recv_info_this_layer:
                            expert_recv_info_this_layer[dst_rank] = []
                        expert_send_info_this_layer[src_rank].append(
                            (dst_rank, expert_id))
                        expert_recv_info_this_layer[dst_rank].append(
                            (src_rank, expert_id))

                        remove_index = np.where(
                            np.logical_and(dst_rank_indices == dst_rank,
                                           experts_to_recv == expert_id))

                        # update
                        dst_rank_indices = np.delete(dst_rank_indices,
                                                     remove_index)
                        experts_to_recv = np.delete(experts_to_recv,
                                                    remove_index)

            yield (expert_send_info_this_layer, expert_recv_info_this_layer,
                   updated_expert_maps_this_layer_org, layer_id)

    # TODO: Here only expert weight exchange is considered, need to be extended to cover other weight update cases
    def compose_expert_update_info_greedy(self, updated_expert_maps,
                                          current_expert_maps):
        num_layers = current_expert_maps.shape[0]

        for layer_id in range(num_layers):
            updated_expert_maps_this_layer = updated_expert_maps[layer_id]
            current_expert_maps_this_layer = current_expert_maps[layer_id]

            expert_send_info_this_layer: dict[Any, Any] = {}
            expert_recv_info_this_layer: dict[Any, Any] = {}

            # Guard Clause: if there is no expert weight update, avoid subsequent processing
            if torch.equal(updated_expert_maps_this_layer,
                           current_expert_maps_this_layer):
                yield (expert_send_info_this_layer,
                       expert_recv_info_this_layer,
                       updated_expert_maps_this_layer, layer_id)

            # Parse expert_ids each rank needs to receive from other ranks
            dst_rank_indices, experts_to_recv = torch.where((current_expert_maps_this_layer == -1) \
                & (updated_expert_maps_this_layer != -1))

            # Parse expert_ids each rank needs to send to other ranks
            src_rank_indices, experts_to_send = torch.where((current_expert_maps_this_layer != -1) \
                & (updated_expert_maps_this_layer == -1))

            for idx in range(len(dst_rank_indices)):
                dst_rank_id = dst_rank_indices[idx].item()
                expert_id = experts_to_recv[idx].item()
                if dst_rank_id not in expert_recv_info_this_layer:
                    expert_recv_info_this_layer[dst_rank_id] = []

                if not torch.isin(torch.tensor(expert_id),
                                  experts_to_send).any():
                    # if expert_id are not sent out from any npu, it will be copied from one npu holding this expert
                    candidate_src_rank_indices = torch.where(
                        current_expert_maps_this_layer[:, expert_id] != -1)[0]
                else:
                    candidate_src_rank_indices = src_rank_indices[
                        experts_to_send == expert_id]

                # TODO: improve selection criterion of npu sending expert_id considering such as intra-node or inter-node...
                src_rank_id = candidate_src_rank_indices[0].item()
                if src_rank_id not in expert_send_info_this_layer:
                    expert_send_info_this_layer[src_rank_id] = []

                expert_send_info_this_layer[src_rank_id].append(
                    (dst_rank_id, expert_id))
                expert_recv_info_this_layer[dst_rank_id].append(
                    (src_rank_id, expert_id))

            yield (expert_send_info_this_layer, expert_recv_info_this_layer,
                   updated_expert_maps_this_layer, layer_id)

    def calculate_rebalance_experts(self, load_info, old_placement):
        """
        Compute `new_map` by calling the `rebalance_experts` method of the policy instance.
        """
        if self.old_expert_maps is None:
            return False, None, None

        # 返回值没有全部使用
        ep_group = get_ep_group().device_group
        num_replicas = model.num_physical_experts
        num_groups = model.num_expert_groups
        num_nodes = get_node_count()
        num_gpus = ep_group.size()
        old_global_expert_indices = EplbPolicy.convert_table(old_placement, num_layer=128)
        global_expert_load = EplbPolicy.convert_format(old_placement, load_info)

        changed, priority, new_map = self.policy.rebalance_experts(
            old_global_expert_indices,
            global_expert_load,
            num_replicas,
            num_groups,
            num_nodes,
            num_gpus
        )
        return self.policy.deployment

    def get_init_expert_maps(self):
        """
        Read the initial expert_map from shared_dict.
        """
        return self.shared_dict.get("expert_maps", None)

    def fetch_and_sum_load_info(self):
        """
        Each time the subprocess is awakened, read the latest moe_load
        (shape: [num_moe_layers, num_experts_per_layer]) from shared_dict.
        """
        return self.shared_dict.get("moe_load", None)

    def update_expert_map(self, expert_maps):

        self.shared_dict["expert_maps"] = expert_maps

    def global2local(self, placement: torch.Tensor,
                     E_local: int) -> tuple[torch.Tensor, torch.Tensor]:

        L, G, _ = placement.shape
        device = placement.device

        pt_local = torch.full((L, G, E_local),
                              fill_value=-1,
                              dtype=torch.long,
                              device=device)

        valid = placement >= 0
        l_idx, g_idx, k_idx = valid.nonzero(as_tuple=True)

        slot_idx = placement[l_idx, g_idx, k_idx]

        pt_local[l_idx, g_idx, slot_idx] = k_idx

        return pt_local

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

    def pack_update_info(self, update_info_generator):
        """
        Pack a list of update info tuples for efficient IPC.
        """
        send_all = []
        recv_all = []
        maps = []
        log2phy_all = []
        layer_ids = []

        for send_info, recv_info, new_expert_map, layer_id in update_info_generator:

            send_info_this_rank = send_info[
                self.rank_id] if self.rank_id in send_info else []
            recv_info_this_rank = recv_info[
                self.rank_id] if self.rank_id in recv_info else []
            send_all.append(send_info_this_rank)
            recv_all.append(recv_info_this_rank)

            maps.append(new_expert_map[self.rank_id].numpy().tolist())

            log2phy_map = generate_log2phy_map(new_expert_map)
            log2phy_all.append(log2phy_map[self.rank_id].numpy().tolist())

            layer_ids.append(layer_id)

        return list(zip(send_all, recv_all, maps, log2phy_all, layer_ids))


class EplbProcess:

    def __init__(self,
                 policy_type: int = 0,
                 enable_d2d: bool = True):
        """
        Args:
            shared_dict: Cross-process shared dict returned by Manager().dict()
            policy_type: Integer passed to PolicyFactory.generate_policy
            enable_d2d: Whether to enable D2D loading
        """
        self.manager = Manager()
        self.shared_dict = self.manager.dict({
            # 当前rank_id的专家表[num_layers,num_experts]
            "expert_map": None,
            # 热度负载信息 [num_layers, world_size, num_experts]
            "moe_load": None,
            # 所有的专家表[num_layers, world_size, num_experts]
            "expert_maps": None,
        })
        # self.shared_dict = shared_dict
        self.policy_type = policy_type
        self.enable_d2d = enable_d2d
        self.planner_q = Queue()
        self.block_update_q = Queue(maxsize=1)

        # Create EplbWorker instance
        self.worker = EplbWorker(self.shared_dict, self.policy_type,
                                 self.enable_d2d)

    def worker_process(self, planner_q, block_update_q):
        """
        Subprocess entry: bind to specified NPU, loop waiting for planner_q to wake up, call do_update, then notify main process update is complete.
        """
        while True:
            try:

                planner_q.get()

                packed_update_info = self.worker.do_update()

                while True:
                    if not block_update_q.empty():
                        continue
                    block_update_q.put(packed_update_info)
                    break

            except Exception as e:
                logger.warning(f"[EPLB subprocess Exiting due to error: {e}",
                               exc_info=True)
                break

    def _launch_process(self):
        """
        Use spawn method to launch subprocess and return (planner_q, block_update_q, proc).
        """
        proc = Process(target=self.worker_process,
                       args=(self.planner_q, self.block_update_q),
                       daemon=True)

        proc.start()
        return proc
