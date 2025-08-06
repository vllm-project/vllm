# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with TreeAttention."""

import ast
from dataclasses import dataclass


@dataclass
class TreeDrafterParams:
    tree_choices: list[tuple[int, ...]]
    attn_mask: list[list[bool]]
    # Cumulative number of drafts at each level.
    cu_drafts_per_level: list[int]
    # Number of child drafts that each token has at the given level.
    child_drafts_per_level: list[int]
    # Maps each draft token to its level in the tree.
    draft_levels: list[int]

    @staticmethod
    def from_spec_token_tree(spec_token_tree: str) -> "TreeDrafterParams":
        # Parse the speculative token tree.
        tree_choices: list[tuple[int, ...]] = ast.literal_eval(spec_token_tree)
        # Sort the tree breadth-first.
        tree_choices.sort(key=lambda t: (len(t), t))
        # Only trees with fixed branching factor per level are
        # currently supported for tree attention.
        _assert_fixed_branching_factor_per_level(tree_choices, spec_token_tree)

        tree_depth = len(tree_choices[-1]) + 1
        # Precompute per-level properties of the tree.
        num_nodes_per_level = [0] * tree_depth
        num_nodes_per_level[0] = 1
        for node in tree_choices:
            num_nodes_per_level[len(node)] += 1

        cu_drafts_per_level = [0]
        child_drafts_per_level = []
        draft_levels = []
        for level in range(1, tree_depth):
            cu_drafts_per_level.append(cu_drafts_per_level[-1] +
                                       num_nodes_per_level[level])
            child_drafts_per_level.append(num_nodes_per_level[level] //
                                          num_nodes_per_level[level - 1])
            draft_levels += [level - 1] * num_nodes_per_level[level]

        # Construct the tree attention bias.
        depth_counts = _get_depth_counts(tree_choices)
        attn_mask = _prepare_tree_attn_bias(
            tree_choices,
            depth_counts,
        )

        return TreeDrafterParams(
            tree_choices=tree_choices,
            attn_mask=attn_mask,
            cu_drafts_per_level=cu_drafts_per_level,
            child_drafts_per_level=child_drafts_per_level,
            draft_levels=draft_levels,
        )

def _has_fixed_branching_factor(tree_nodes, level):
    """
    Checks if all nodes at the given level have the same number of children.
    """
    next_level_nodes = [node for node in tree_nodes if len(node) == level + 1]
    if len(next_level_nodes) == 0:
        return True

    level_nodes = [node for node in tree_nodes if len(node) == level]
    child_counts = []
    for parent in level_nodes:
        child_counts.append(
            sum(1 for child in next_level_nodes if child[:-1] == parent)
        )
    return len(set(child_counts)) <= 1  # All counts are the same.

def _assert_fixed_branching_factor_per_level(
    tree_nodes: list[tuple[int, ...]],
    spec_token_tree: str) -> None:
    """
    Asserts that each level of the tree has a fixed branching factor. That is,
    the number of children per node is the same within a level, but can vary
    across levels.
    """
    tree_depth = len(tree_nodes[-1]) + 1
    for level in range(1, tree_depth):
        assert _has_fixed_branching_factor(tree_nodes, level), \
            f"The configured spec token tree '{spec_token_tree}' has variable " \
            f"branching at level {level}. Tree speculative decoding requires " \
            f"a uniform number of children per level."

def _get_depth_counts(sorted_tree_choices: list[tuple[int, ...]]) -> list[int]:
    """
    Counts the number of choices at each depth of the tree.
    """
    depth_counts = []
    prev_depth = 0
    for path in sorted_tree_choices:
        depth = len(path)
        if depth != prev_depth:
            depth_counts.append(0)
        depth_counts[depth - 1] += 1
        prev_depth = depth
    return depth_counts


def _prepare_tree_attn_bias(
    sorted_tree_choices: list[tuple[int, ...]],
    depth_counts: list[int],
) -> list[list[bool]]:
    # +1 comes from the additional root node.
    tree_len = len(sorted_tree_choices) + 1
    tree_attn_mask = [[False for _ in range(tree_len)]
                      for _ in range(tree_len)]

    mask_val = True
    for i in range(tree_len):
        # Set diagonal to all True. Each token should attend to itself.
        tree_attn_mask[i][i] = mask_val
        # Set root column to all True. All tokens attend to it.
        tree_attn_mask[i][0] = mask_val

    # Set all ancestors to True.
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_tree_choice = sorted_tree_choices[start + j]
            if len(cur_tree_choice) == 1:
                continue

            for c in range(len(cur_tree_choice) - 1):
                ancestor_idx = sorted_tree_choices.index(
                    cur_tree_choice[:c + 1]) + 1
                tree_attn_mask[j + start + 1][ancestor_idx] = mask_val
        start += depth_counts[i]
    return tree_attn_mask
