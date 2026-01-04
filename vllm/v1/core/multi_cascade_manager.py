# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import List, Optional, Tuple

from vllm.config.attention import (MultiCascadeAllocateMethod,
                                   MultiCascadeConfig)
from vllm.v1.core.kv_cache_utils import (KVPrefixAlignedGroups, KVPrefixTrie,
                                         KVPrefixTrieNode)


def get_scheduled_requests(request_list: List[Tuple[str, bool]],
                           groups_list: List[Tuple[int, int, int]]) \
        -> KVPrefixAlignedGroups:

    # Post-process request and group list to remove
    # non-scheduled running requests
    actual_request_list = []
    actual_groups_list = []

    for group_num_common_blocks, start_id, end_id in groups_list:
        scheduled = [
            req_id
            for req_id, is_scheduled in request_list[start_id:end_id + 1]
            if is_scheduled
        ]

        actual_start_id = len(actual_request_list)
        actual_request_list.extend(scheduled)

        if scheduled:
            actual_groups_list.append(
                (group_num_common_blocks, actual_start_id,
                 len(actual_request_list)))

    return KVPrefixAlignedGroups(actual_request_list, actual_groups_list)


class MultiCascadeManager:
    """Manages grouping requests for multi-cascade attention."""

    def __init__(self, multi_cascade_config: MultiCascadeConfig):
        self.absorption_threshold = multi_cascade_config.absorption_threshold
        self.alloc_method = multi_cascade_config.allocate_method

    def alloc_groups(
            self,
            kv_prefix_trie: KVPrefixTrie) -> Optional[KVPrefixAlignedGroups]:
        match self.alloc_method:
            case MultiCascadeAllocateMethod.LEAF_PASS:
                return self.alloc_leaf_pass(kv_prefix_trie)
            case MultiCascadeAllocateMethod.FULL_PASS:
                return self.alloc_full_pass(kv_prefix_trie)
            case _:
                raise NotImplementedError("Must pass a valid grouping config"
                                          "for multi-cascade attention.")

    def alloc_leaf_pass(
            self,
            kv_prefix_trie: KVPrefixTrie) -> Optional[KVPrefixAlignedGroups]:
        """
        Allocates groups of requests based on the weight of each node.
        Traverses only trie leaves to find the best groupings.
        Returns:
            A tuple consisting of:
                request_list: List of requests which we group in model runner.
                groups_list: List of indices of the form
                (num_common_prefix_blocks, start, end)
                where all requests in request_list[start: end] form a group.
        """

        if not kv_prefix_trie:
            return None

        request_list: list[tuple[str, bool]] = []
        groups_list: list[tuple[int, int, int]] = []

        for block_id in kv_prefix_trie.block_id_to_req_id:
            groups_list.append(
                (kv_prefix_trie.block_id_to_leaf_node[block_id].depth,
                 len(request_list), len(request_list) +
                 len(kv_prefix_trie.block_id_to_req_id[block_id])))
            request_list.extend(
                list(
                    map(lambda k: (k, kv_prefix_trie.req_to_scheduled[k]),
                        kv_prefix_trie.block_id_to_req_id[block_id])))

        return get_scheduled_requests(request_list, groups_list)

    def alloc_full_pass(
            self,
            kv_prefix_trie: KVPrefixTrie) -> Optional[KVPrefixAlignedGroups]:
        """
        Allocates groups of requests based on the weight of each node.
        Traverses the entire trie to find best groupings.
        Returns:
            A tuple consisting of:
                request_list: List of requests which we group in model runner.
                groups_list: List of indices of the form
                (num_common_prefix_blocks, start, end)
                where all requests in request_list[start: end] form a group.
        """

        if not kv_prefix_trie:
            return None

        request_list: List[Tuple[str, bool]] = []
        groups_list: List[Tuple[int, int, int]] = []

        stack: List[Tuple[KVPrefixTrieNode,
                          bool]] = [(kv_prefix_trie.sentinel, False)]
        while stack:
            node, visited = stack.pop()
            if node is None:
                continue
            if visited:
                # Post-order logic
                node.min_accessible_leaf_id = min(node.min_accessible_leaf_id,
                                                  len(request_list))
                child_group_weights = sum(child.groups_weight
                                          for child in node.child_list)
                num_groups = sum(child.num_groups for child in node.child_list)
                start_direct_leaf = len(request_list)
                start_of_next_group = node.min_accessible_leaf_id + node.leaf_cnt
                if node.weight >= self.absorption_threshold * child_group_weights:
                    node.groups_weight = node.weight
                    del groups_list[-1 * num_groups:]
                    groups_list.append(
                        (node.depth, node.min_accessible_leaf_id,
                         start_of_next_group))
                    node.num_groups = 1
                else:
                    node.groups_weight = child_group_weights
                    groups_list.extend([
                        (node.depth, i, i + 1)
                        for i in range(start_direct_leaf, start_of_next_group)
                    ])
                    node.num_groups = num_groups + start_of_next_group - len(
                        request_list)
                request_list.extend(
                    list(
                        map(lambda k: (k, kv_prefix_trie.req_to_scheduled[k]),
                            node.node_req_ids)))
            else:
                # Push current node back as visited (for post-order)
                stack.append((node, True))
                for i in range(len(node.child_list) - 1, -1, -1):
                    child_node = node.child_list[i]
                    stack.append((child_node, False))

                # Pre-order logic
                node.min_accessible_leaf_id = float('inf')

        return get_scheduled_requests(request_list, groups_list)
