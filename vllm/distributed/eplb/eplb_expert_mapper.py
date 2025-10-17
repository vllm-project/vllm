import numpy as np
import torch
import networkx as nx

class ComposeExpertUpdate:
    def __init__(self, updated_expert_maps, current_expert_maps):
        self.updated_org = updated_expert_maps
        self.current_org = current_expert_maps
        # Internal views used by subclasses (torch or numpy)
        self.updated, self.current = self._prepare_internal(updated_expert_maps, current_expert_maps)
        self.num_layers = self.current.shape[0]
        self.num_ranks = self.current.shape[1]
        self.num_experts = self.current.shape[2]

    def _prepare_internal(self, updated, current):
        # Default: use torch tensors directly
        return updated, current

    def _is_equal(self, updated_layer, current_layer):
        raise NotImplementedError

    def _plan_transfers(self, layer_id, updated_layer, current_layer, send_dict, recv_dict):
        raise NotImplementedError

    def _map_to_yield(self, layer_id):
        return self.updated[layer_id]

    def generate(self):
        for layer_id in range(self.num_layers):
            updated_layer = self.updated[layer_id]
            current_layer = self.current[layer_id]

            expert_send_info_this_layer = {}
            expert_recv_info_this_layer = {}

            # Guard Clause: if there is no expert weight update, avoid subsequent processing
            if self._is_equal(updated_layer, current_layer):
                yield (
                    expert_send_info_this_layer,
                    expert_recv_info_this_layer,
                    self._map_to_yield(layer_id),
                    layer_id
                )

            # Main planning
            self._plan_transfers(
                layer_id, updated_layer, current_layer,
                expert_send_info_this_layer, expert_recv_info_this_layer
            )

            # Final yield
            yield (
                expert_send_info_this_layer,
                expert_recv_info_this_layer,
                self._map_to_yield(layer_id),
                layer_id
            )


class BipartiteExpertUpdate(ComposeExpertUpdate):
    """Networkx bipartite matching version."""

    def _prepare_internal(self, updated, current):
        updated_np = np.array(updated.clone())
        current_np = np.array(current.clone())
        return updated_np, current_np

    def _is_equal(self, updated_layer, current_layer):
        return np.equal(updated_layer, current_layer).all()

    def _map_to_yield(self, layer_id):
        return self.updated_org[layer_id]

    def _plan_transfers(self, layer_id, updated_layer, current_layer, send_dict, recv_dict):
        # Parse expert_ids each rank needs to receive from other ranks
        dst_rank_indices, experts_to_recv = np.where(
            (current_layer == -1) & (updated_layer != -1)
        )

        # record src ranks for potential transfer
        src_ranks_set = {}
        for idx in range(len(dst_rank_indices)):
            expert_id = experts_to_recv[idx].item()
            if expert_id not in src_ranks_set:
                src_ranks_set[expert_id] = np.where(current_layer[:, expert_id] != -1)[0]

        # Loop until all experts are scheduled
        while len(dst_rank_indices) > 0:
            # construct bipartite graph
            graph_expert_update = nx.Graph()
            for idx in range(len(dst_rank_indices)):
                dst_rank_id = dst_rank_indices[idx].item()
                expert_id = experts_to_recv[idx].item()
                # add src ranks
                src_rank_ids = src_ranks_set[expert_id]
                graph_expert_update.add_nodes_from(src_rank_ids, bipartite=0)
                # add dest rank
                graph_expert_update.add_node(str(dst_rank_id), bipartite=1)
                # add edges
                for src_rank_id in src_rank_ids:
                    graph_expert_update.add_edge(src_rank_id, str(dst_rank_id))

            # graph may not be connected
            connected_components = list(nx.connected_components(graph_expert_update))
            all_matches = {}
            # matching in this loop
            for i, component in enumerate(connected_components):
                subgraph = graph_expert_update.subgraph(component)
                component_matching = nx.bipartite.maximum_matching(subgraph)
                all_matches.update(component_matching)

            for src_rank, dst_rank in all_matches.items():
                dst_rank = int(dst_rank)
                assert src_rank != dst_rank
                if graph_expert_update.nodes[src_rank].get('bipartite') == 0:
                    # currently not scheduled experts in rank dst_rank
                    experts_v = experts_to_recv[np.where(dst_rank_indices == dst_rank)]
                    # src: src_rank, dest: dst_rank, expert: expert_id
                    expert_id = np.intersect1d(
                        experts_v, np.where(current_layer[src_rank] != -1)[0]
                    )[0]

                    # Ensure int() for numpy scalars
                    expert_id_int = int(expert_id)

                    # record send/rcv pairs
                    if src_rank not in send_dict:
                        send_dict[src_rank] = []
                    if dst_rank not in recv_dict:
                        recv_dict[dst_rank] = []
                    send_dict[src_rank].append((dst_rank, expert_id_int))
                    recv_dict[dst_rank].append((src_rank, expert_id_int))

                    remove_index = np.where(
                        np.logical_and(dst_rank_indices == dst_rank, experts_to_recv == expert_id)
                    )

                    # update
                    dst_rank_indices = np.delete(dst_rank_indices, remove_index)
                    experts_to_recv = np.delete(experts_to_recv, remove_index)


class GreedyExpertUpdate(ComposeExpertUpdate):
    """Greedy version."""
    def _prepare_internal(self, updated, current):
        # align devices
        if not torch.is_tensor(updated):
            updated = torch.as_tensor(updated)
        if not torch.is_tensor(current):
            current = torch.as_tensor(current)
        if updated.device != current.device:
            updated = updated.to(current.device)
        return updated, current

    def _is_equal(self, updated_layer, current_layer):
        return torch.equal(updated_layer, current_layer)

    def _plan_transfers(self, layer_id, updated_layer, current_layer, send_dict, recv_dict):
        # Parse expert_ids each rank needs to receive from other ranks
        dst_rank_indices, experts_to_recv = torch.where(
            (current_layer == -1) & (updated_layer != -1)
        )
        # Parse expert_ids each rank needs to send to other ranks
        src_rank_indices, experts_to_send = torch.where(
            (current_layer != -1) & (updated_layer == -1)
        )

        for idx in range(len(dst_rank_indices)):
            dst_rank_id = dst_rank_indices[idx].item()
            expert_id = experts_to_recv[idx].item()
            if dst_rank_id not in recv_dict:
                recv_dict[dst_rank_id] = []

            # if expert_id are not sent out from any npu, it will be copied from one npu holding this expert
            if not torch.isin(torch.tensor(expert_id), experts_to_send).any():
                candidate_src_rank_indices = torch.where(current_layer[:, expert_id] != -1)[0]
            else:
                candidate_src_rank_indices = src_rank_indices[experts_to_send == expert_id]

            src_rank_id = candidate_src_rank_indices[0].item()
            if src_rank_id not in send_dict:
                send_dict[src_rank_id] = []

            send_dict[src_rank_id].append((dst_rank_id, expert_id))
            recv_dict[dst_rank_id].append((src_rank_id, expert_id))

