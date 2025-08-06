# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with TreeAttention."""

import ast
from dataclasses import dataclass
from typing import Optional


@dataclass
class TreeDrafterParams:
    tree_choices: list[tuple[int, ...]]
    attn_mask: list[list[bool]]
    cu_drafts_per_level: list[int]
    child_drafts_per_level: list[int]

    @staticmethod
    def from_spec_token_tree(spec_token_tree: str) -> "TreeDrafterParams":
        # Parse the speculative token tree.
        tree_choices: list[tuple[int, ...]] = ast.literal_eval(spec_token_tree)
        # Sort the tree breadth-first.
        tree_choices.sort(key=lambda t: (len(t), t))

        tree_depth = len(tree_choices[-1])
        # Precompute per-level properties of the tree.
        num_drafts_per_level = [0] * tree_depth
        for node in tree_choices:
            num_drafts_per_level[len(node) - 1] += 1
        cu_drafts_per_level = [num_drafts_per_level[0]]
        child_drafts_per_level = [num_drafts_per_level[0]]
        for level in range(1, tree_depth):
            cu_drafts_per_level.append(
                cu_drafts_per_level[-1] + num_drafts_per_level[level]
            )
            child_drafts_per_level.append(
                num_drafts_per_level[level] // num_drafts_per_level[level - 1]
            )

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
        )


def _get_depth_counts(sorted_tree_choices: list[tuple[int, ...]]) -> list[int]:
    # Count the number of choices at each depth of the tree.
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
    tree_attn_mask = [[False for _ in range(tree_len)] for _ in range(tree_len)]

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
                ancestor_idx = sorted_tree_choices.index(cur_tree_choice[: c + 1]) + 1
                tree_attn_mask[j + start + 1][ancestor_idx] = mask_val
        start += depth_counts[i]
    return tree_attn_mask
