# Copyright (c) Microsoft Corporation.

from typing import List, Dict, Any

import numpy as np

SINGLE_REQUEST = -100

def prefix_cluster_preprocess(cluster, prompt_token_ids):
    for ind, prompt in enumerate(prompt_token_ids):
        cluster.add_prompt(prompt, ind)
    cluster.finalize()
    map_list, input_tokens = cluster.get_clustered_prompts(
        return_map_list=True)
    return map_list, input_tokens


def prefix_cluster_postprocess(map_list, output_list):
    assert len(map_list) == len(
        output_list), "# output < # input, please check"
    new_output_list = [(x, y) for x, y in zip(map_list, output_list)]
    sorted_new_output = sorted(new_output_list, key=lambda x: x[0])
    return [x[1] for x in sorted_new_output]


def print_batchllm_info(cluster):
    avg_sharing_degree, avg_prefix_len, avg_distinct_len, logical_tokens, \
        physical_tokens = cluster.summarize(
    )
    saving_ratio = (logical_tokens - physical_tokens) / logical_tokens * 100
    print(f"Avg. sharing degree: {avg_sharing_degree:.2f}, "
          f"avg. prefix length: {avg_prefix_len:.2f}, "
          f"avg. distinct length: {avg_distinct_len:.2f}, "
          f"logical tokens: {logical_tokens}, "
          f"physical tokens: {physical_tokens}, "
          f"saving ratio: {saving_ratio:.2f}%")


class PromptPrefixCluster:
    '''
    It clusters the prompts according to the common prefix. The prompts should
    be already tokenized, i.e., in token id format.
    '''

    class TokensNode:

        def __init__(self,
                     input_ids: List[int],
                     node_id: int = -100,
                     parent=None,
                     prompt_id=None):
            self.input_ids = input_ids
            self.node_id = node_id
            self.parent = parent
            self.prompt_id = prompt_id  # The prompt id if it is a leaf node.
            self.children: Dict[int, Any] = {}  # {node_id: TokensNode}
            
        def match_prefix(self, input_ids: List[int]) -> int:
            # Return the matched length of the input_ids.
            length = 0
            if input_ids is not []:
                for i, id in enumerate(input_ids):
                    if i >= len(self.input_ids) or id != self.input_ids[i]:
                        break
                    length += 1
            return length

        def add_child(self, node):
            self.children[node.node_id] = node

    def __init__(self, block_size: int = 1, multi_level_prefix: bool = False):
        self.node_number = 0
        self.root = self._create_node([], None)
        assert block_size == 1, "Blocked processing has not been implemented."
        self.block_size = block_size
        assert (
            not multi_level_prefix), "Multi-level prefix not supported yet."
        self.multi_level_prefix = multi_level_prefix
        self.finalized = False

    def reset(self):
        self.node_number = 0
        self.root = self._create_node([], None)
        self.finalized = False

    def _create_node(self, input_ids: List[int], parent=None, prompt_id=None):
        node = self.TokensNode(input_ids, self.node_number, parent, prompt_id)
        self.node_number += 1
        return node

    def _split_node(self, node: TokensNode, length: int):
        new_node = self._create_node(node.input_ids[length:], node)
        node.input_ids = node.input_ids[:length]
        new_node.children = node.children
        node.children = {new_node.node_id: new_node}
        for child in new_node.children.values():
            child.parent = new_node

        return node, new_node

    def add_prompt(self, input_ids: List[int], prompt_id):
        assert not self.finalized, "The prompt cluster has been finalized."
        assert len(input_ids) > 0, "The input_ids should not be empty."
        node = self.root
        while len(node.children) > 0:
            # Try to match the prefix of the children.
            matched = False
            for child in node.children.values():
                if child.prompt_id is not None:  # The leaf node.
                    continue
                matched_length = child.match_prefix(input_ids)
                if matched_length > 0:
                    if matched_length == len(child.input_ids):
                        node = child
                    else:
                        node, _ = self._split_node(child, matched_length)
                    input_ids = input_ids[matched_length:]
                    matched = True
                    break
            if not matched or len(input_ids) == 0:
                break
        if len(input_ids) != 0:
            new_node = self._create_node(input_ids, node)
            node.add_child(new_node)
            node = new_node
        leaf_node = self._create_node([], node, prompt_id)
        node.add_child(leaf_node)

    def _enlarge_1st_prefix_tokens_deprecated(self, node: TokensNode,
                                              level: int):
        '''
        This function tries to merge the level-1 node into the level-0 node if
        it helps to increase the shared tokens. It should not work for multi
        level prefix. It will also not deal with the nodes with a level larger
        than 2.
        '''
        # if level > 2:
        #     return
        changed = False
        children = list(node.children.values())
        depth = len(node.input_ids)
        for fork in children:
            if fork.input_ids is []:  # Leaf node.
                continue
            fork_width = len(fork.children)
            if fork_width == 1:
                continue
            fork_depth = len(fork.input_ids)
            # Increased sharing excluding the decreased sharing.
            gain = (fork_width - 1) * fork_depth - depth
            if (node.parent is not None) and (gain > 0):
                changed = True
                # Clone fork's parent (node) and concatenate with fork.
                fork.input_ids = node.input_ids + fork.input_ids
                fork.parent = node.parent
                node.parent.add_child(fork)
                node.children.pop(fork.node_id)
                # TODO: should update the depth here?
                self._enlarge_1st_prefix_tokens_deprecated(fork, level)

                if len(node.children) == 1:
                    # The node becomes a normal node.
                    # Merge the last child into it.
                    next_child = next(iter(node.children.values()))
                    # next_child_depth = len(next_child.input_ids)
                    node.input_ids = node.input_ids + next_child.input_ids
                    node.children = next_child.children
                    node.prompt_id = next_child.prompt_id
                    del next_child
                    self._enlarge_1st_prefix_tokens_deprecated(node, level)
                    break  # No child left.

                # # Revisit the node as it's child is changed.
                # self._enlarge_1st_prefix_tokens_deprecated(node, level)
                break
            else:
                changed |= self._enlarge_1st_prefix_tokens_deprecated(
                    fork, level + 1)
        return changed

    def _get_num_leaves(self, node: TokensNode):
        if len(node.children) == 0:
            return 1
        num_leaves = 0
        for child in node.children.values():
            num_leaves += self._get_num_leaves(child)
        return num_leaves

    def _enlarge_1st_prefix_tokens(self, node: TokensNode):
        if len(node.children) == 0:
            return
        children = list(node.children.values())
        for child in children:
            if child.input_ids is []:  # Leaf node.
                continue
            self._enlarge_1st_prefix_tokens(child)
            # Now the tree of `child` has the first prefix maximized, note this
            # tree excludes the `child` itself's input_ids but regard it as an
            # empty node.
            grandchildren = list(child.children.values())
            # Clone child and concatenate with grandchild if it enlarges the
            # shared prefix.
            child_depth = len(child.input_ids)
            for gchild in grandchildren:
                if gchild.input_ids is []:  # Leaf node.
                    continue
                gchild_width = self._get_num_leaves(gchild)
                gchild_depth = len(gchild.input_ids)
                # Increased sharing excluding the decreased sharing.
                gain = (gchild_width - 1) * gchild_depth - child_depth
                if gain > 0:
                    gchild.input_ids = child.input_ids + gchild.input_ids
                    gchild.parent = child.parent
                    child.parent.add_child(gchild)
                    child.children.pop(gchild.node_id)

                if len(child.children) == 1:
                    # The child becomes a normal node. Merge the last child into
                    # it.
                    next_child = next(iter(child.children.values()))
                    child.input_ids = child.input_ids + next_child.input_ids
                    child.children = next_child.children
                    child.prompt_id = next_child.prompt_id
                    del next_child
                    break

    def finalize(self):
        if not self.multi_level_prefix:
            if True:
                self._enlarge_1st_prefix_tokens(self.root)
            else:
                while self._enlarge_1st_prefix_tokens_deprecated(self.root, 0):
                    continue
        self.finalized = True

    def _expand_tokens(self, node: TokensNode):
        # Returns a list of paths, i.e., nested lists of token ids.
        prefix = node.input_ids
        if prefix is None:
            return [([], node.prompt_id)]

        branches = []
        for child in node.children.values():
            sub_branches = self._expand_tokens(child)
            branches.extend(sub_branches)

        assert len(branches) > 0

        results = []
        for branch in branches:
            input_ids, prompt_id = branch
            results.append((prefix + input_ids, prompt_id))

        return results

    def get_clustered_prompts(self, return_map_list=False):
        # For the single level prefix, only the first level fork will be
        # regarded as the shared one. It requires the finalize function to
        # be called before calling this function.
        assert self.finalized, "The prompt cluster has not been finalized."

        # Each item in `prompts` is a group of prompts with shared prefix. Note
        # such a group can contain only one prompt. The format of the group is:
        # [[shared_ids], [[distinct_ids_0], [distinct_ids_1], ...]]
        prompts = []
        map_list = []
        for node in self.root.children.values():
            prefix = node.input_ids

            distincts = []
            for child in node.children.values():
                distinct = self._expand_tokens(child)
                distincts.extend(distinct)
            if return_map_list:
                prompts.append([prefix, [x[0] for x in distincts]])
                map_list.extend([x[1] for x in distincts])
            else:
                prompts.append([prefix, distincts])
        if return_map_list:
            return map_list, prompts
        else:
            return prompts

    def verify(self, prompts: List[List[int]]):
        # Verify the correctness of the clustered prompts.
        clustered_prompts = self.get_clustered_prompts()
        for cluster in clustered_prompts:
            prefix = cluster[0]
            distincts = cluster[1]
            for (dist, id) in distincts:
                prompt = prefix + dist
                if prompt != prompts[id]:
                    print(f"Error: seq {id}: {prompt} != {prompts[id]}")
                    return False
        return True

    def summarize(self):
        group_sizes = []
        prefix_lengths = []
        distinct_lengths = []
        clustered_prompts = self.get_clustered_prompts()
        logical_tokens = 0
        physical_tokens = 0
        for cluster in clustered_prompts:
            prefix = cluster[0]
            distincts = cluster[1]
            group_size = len(distincts)
            group_sizes.append(group_size)
            prefix_tokens = len(prefix)
            prefix_lengths.append(prefix_tokens)
            cluster_distinct_lengths = [len(dist) for dist, _ in distincts]
            distinct_lengths.extend(cluster_distinct_lengths)
            cluster_distinct_tokens = sum(cluster_distinct_lengths)
            logical_tokens += (prefix_tokens * group_size +
                               cluster_distinct_tokens)
            physical_tokens += prefix_tokens + cluster_distinct_tokens

        return np.mean(group_sizes), np.mean(prefix_lengths), np.mean(
            distinct_lengths), logical_tokens, physical_tokens

    def _print_pretty(self, node: TokensNode, token_num: int, width: int = 4):
        if node.input_ids is not []:
            for input_id in node.input_ids:
                print(f'{input_id:<{width}}', end='')
        else:
            print(f"<prompt{node.prompt_id}>")
            return
        if node.input_ids is []:
            local_intent_num = 0
        else:
            local_intent_num = 2 + len(node.input_ids)
            if len(node.input_ids) > 0:
                local_intent_num += len(node.input_ids) - 1
        node_token_num = len(
            node.input_ids) if node.input_ids is not [] else 0
        new_token_num = token_num + node_token_num
        for id, child in enumerate(node.children.values()):
            if id != 0:
                intend = ' ' * new_token_num * width
                print(f"{intend}", end='')
            self._print_pretty(child, new_token_num)

    def print_pretty(self):
        self._print_pretty(self.root, 0)
