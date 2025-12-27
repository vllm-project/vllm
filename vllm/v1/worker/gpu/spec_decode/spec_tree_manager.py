# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
SpecTreeManager - Port from TRT-LLM for tree-aware speculative decoding.

This module manages the tree structure for Eagle3 speculative decoding,
computing attention masks and position offsets for tree-based speculation.

Key differences from TRT-LLM:
- Uses float32 bias matrix instead of packed int32 for Triton attention
- Integrates with vLLM's TreeAttentionBackend
- Supports SM90 and SM120 architectures
"""
import math
from itertools import accumulate
from typing import List, Optional

import torch


class SpecTreeManager:
    """Manages speculative decoding tree structure and attention masks.

    Ported from TRT-LLM: tensorrt_llm/_torch/speculative/spec_tree_manager.py

    Attributes:
        use_dynamic_tree: Whether using dynamic tree structure per request
        max_total_draft_tokens: Max nodes in tree (excluding root)
        max_draft_len: Number of draft layers (tree depth - 1)
        eagle_choices: Static tree structure definition
        num_trees: 1 for static tree, max_num_requests for dynamic
    """

    def __init__(
        self,
        max_num_requests: int,
        use_dynamic_tree: bool,
        max_total_draft_tokens: int,
        max_draft_len: int,
        eagle_choices: List[List[int]],
        dynamic_tree_max_topK: int = 10,
        device: str = "cuda",
    ):
        self.use_dynamic_tree = use_dynamic_tree
        self.max_total_draft_tokens = max_total_draft_tokens
        self.max_draft_len = max_draft_len
        self.eagle_choices = eagle_choices
        self.num_trees = max_num_requests if use_dynamic_tree else 1
        self.dynamic_tree_max_topK = dynamic_tree_max_topK
        self.device = device
        self.cur_draft_layer_idx = 0
        self.top_k_list: List[torch.Tensor] = []

        # Internal mappings for static tree
        self.index_mapping_set: dict = {}
        self.nodes_list_per_layer: List[List[int]] = []

        # Initialize buffers
        # eagle_paths: [num_trees, max_tokens+1, max_draft_len+1]
        # Stores parent path for each node (ancestor indices)
        self.eagle_paths = (
            torch.ones(
                (
                    self.num_trees,
                    self.max_total_draft_tokens + 1,
                    self.max_draft_len + 1,
                ),
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            )
            * -1
        )

        # spec_dec_mask_matrix: [num_trees, max_tokens+1, max_tokens+1]
        # Binary attention mask - which tokens can attend to which
        self.spec_dec_mask_matrix = (
            torch.eye(
                self.max_total_draft_tokens + 1,
                dtype=torch.int32,
                device=device,
            )
            .unsqueeze(0)
            .repeat(self.num_trees, 1, 1)
        )

        # tree_attn_bias: [num_trees, max_tokens+1, max_tokens+1]
        # Float attention bias for Triton unified_attention (qq_bias parameter)
        # 0.0 for allowed attention, -inf for blocked
        self.tree_attn_bias = torch.zeros(
            (
                self.num_trees,
                self.max_total_draft_tokens + 1,
                self.max_total_draft_tokens + 1,
            ),
            dtype=torch.float32,
            device=device,
        )

        # spec_dec_position_offsets: [num_trees, max_tokens+1]
        # Position offset for each tree node (depth in tree)
        self.spec_dec_position_offsets = torch.zeros(
            (self.num_trees, self.max_total_draft_tokens + 1),
            dtype=torch.int32,
            device=device,
        )

        # Packed mask for XQA (SM90 only, not used on SM120)
        num_packed_blocks = math.ceil((self.max_total_draft_tokens + 1) / 32)
        self.spec_dec_packed_mask = torch.zeros(
            (self.num_trees, self.max_total_draft_tokens + 1, num_packed_blocks),
            dtype=torch.int32,
            device=device,
        )

        # Initialize tree structure
        if self.use_dynamic_tree:
            self._init_tree_info_for_dynamic_tree()
        else:
            self._init_tree_info_for_static_tree()

    def _init_tree_info_for_dynamic_tree(self):
        """Initialize buffers for dynamic tree structure."""
        self.top_k_list = [
            torch.ones(
                self.dynamic_tree_max_topK,
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            )
            * self.dynamic_tree_max_topK
        ]

    def _init_tree_info_for_static_tree(self):
        """Initialize static tree from eagle_choices.

        Parses eagle_choices (e.g., [(0,), (0,0), (0,1), (1,), (1,0)]) into:
        - index_mapping_set: maps choice tuples to node indices
        - eagle_paths: ancestor paths for each node
        - nodes_list_per_layer: nodes at each tree depth
        - spec_dec_mask_matrix: which nodes can attend to which
        - tree_attn_bias: float bias for Triton attention
        - spec_dec_position_offsets: tree depth for each node
        """
        self.index_mapping_set = {}
        self.nodes_list_per_layer = [[] for _ in range(self.max_draft_len + 1)]
        child_nodes_list = [[] for _ in range(self.max_total_draft_tokens + 1)]

        # 1) Map choice tuples to indices
        for i, choice in enumerate(self.eagle_choices):
            self.index_mapping_set[str(choice)] = i + 1

        # 2) Reconstruct eagle_paths (ancestor paths)
        self.eagle_paths.fill_(-1)
        self.eagle_paths[0][0][0] = 0  # root node
        for i, choice in enumerate(self.eagle_choices):
            self.eagle_paths[0][i + 1][0] = 0  # all paths start at root
            for j in range(len(choice)):
                ancestor_choice = choice[: j + 1]
                self.eagle_paths[0][i + 1][j + 1] = self.index_mapping_set[
                    str(ancestor_choice)
                ]

        # 3) Compute nodes per layer
        self.nodes_list_per_layer[0].append(0)  # root at layer 0
        for choice in self.eagle_choices:
            cur_layer = len(choice)
            self.nodes_list_per_layer[cur_layer].append(
                self.index_mapping_set[str(choice)]
            )

        # 4) Compute child_nodes_list
        for choice in self.eagle_choices:
            if len(choice) == 1:
                # Direct children of root
                child_nodes_list[0].append(self.index_mapping_set[str(choice)])
            else:
                parent_choice = choice[:-1]
                parent_idx = self.index_mapping_set[str(parent_choice)]
                child_nodes_list[parent_idx].append(self.index_mapping_set[str(choice)])

        # 5) Compute top_k_list per layer
        for i in range(self.max_draft_len):
            cur_layer_nodes = self.nodes_list_per_layer[i]
            tmp_top_k_list = [
                len(child_nodes_list[node])
                for node in cur_layer_nodes
                if len(child_nodes_list[node]) > 0
            ]
            if tmp_top_k_list:
                self.top_k_list.append(
                    torch.tensor(
                        tmp_top_k_list, dtype=torch.int32, device="cpu", pin_memory=True
                    )
                )
            else:
                self.top_k_list.append(
                    torch.zeros(0, dtype=torch.int32, device="cpu", pin_memory=True)
                )

        # 6) Compute spec_dec_mask_matrix from eagle_paths
        for i, path in enumerate(self.eagle_paths[0]):
            indices = path[path > -1]
            self.spec_dec_mask_matrix[0][i, indices] = 1

        # 7) Convert mask_matrix to tree_attn_bias for Triton attention
        # 0 in mask -> -inf in bias (blocked), 1 in mask -> 0 in bias (allowed)
        self.tree_attn_bias.fill_(float("-inf"))
        for tree_idx in range(self.num_trees):
            self.tree_attn_bias[tree_idx][
                self.spec_dec_mask_matrix[tree_idx] == 1
            ] = 0.0

        # 8) Compute position offsets (tree depth)
        start_idx = 0
        for i in range(self.max_draft_len + 1):
            num_nodes_this_layer = len(self.nodes_list_per_layer[i])
            self.spec_dec_position_offsets[
                :, start_idx : start_idx + num_nodes_this_layer
            ] = i
            start_idx += num_nodes_this_layer

        # 9) Compute packed mask for XQA (SM90 only)
        self._compute_spec_dec_packed_mask(
            self.spec_dec_mask_matrix, self.spec_dec_packed_mask
        )

        # 10) Copy top_k_list to CUDA
        self.top_k_list_cuda = [
            tk.to(device=self.device, dtype=torch.int32) for tk in self.top_k_list
        ]

        # Compute max top_k across all layers
        self.max_top_k = max(
            (tk.max().item() for tk in self.top_k_list_cuda if tk.numel() > 0),
            default=1,
        )

    def _compute_spec_dec_packed_mask(
        self,
        mask_matrix: torch.Tensor,
        packed_mask: torch.Tensor,
    ):
        """Pack binary mask matrix into int32 bitfield for XQA.

        Each row of packed_mask contains ceil(max_tokens/32) int32 values,
        where each bit represents one element of the mask.

        Note: This is only used for XQA on SM90. Triton attention uses
        tree_attn_bias (float) instead.
        """
        num_trees = mask_matrix.shape[0]
        num_blocks = math.ceil((self.max_total_draft_tokens + 1) / 32)

        int_tensor = mask_matrix.reshape(-1, self.max_total_draft_tokens + 1)
        packed_flat = packed_mask.reshape(-1, num_blocks)

        for block_idx in range(num_blocks):
            start_idx = block_idx * 32
            end_idx = min(start_idx + 32, self.max_total_draft_tokens + 1)
            if end_idx <= start_idx:
                break

            block_bits = int_tensor[:, start_idx:end_idx]
            weights = torch.pow(
                2,
                torch.arange(
                    end_idx - start_idx, dtype=torch.int32, device=int_tensor.device
                ),
            )
            block_value = torch.sum(block_bits * weights, dim=-1)
            packed_flat[:, block_idx] = block_value

    def get_tree_attn_bias(self, tree_idx: int = 0) -> torch.Tensor:
        """Get attention bias for Triton unified_attention.

        Returns:
            Float tensor [max_tokens+1, max_tokens+1] usable as qq_bias parameter.
        """
        return self.tree_attn_bias[tree_idx]

    def get_eagle_paths(self, tree_idx: int = 0) -> torch.Tensor:
        """Get ancestor paths for each node in tree."""
        return self.eagle_paths[tree_idx]

    def get_position_offsets(self, tree_idx: int = 0) -> torch.Tensor:
        """Get position offsets (tree depth) for each node."""
        return self.spec_dec_position_offsets[tree_idx]

    def get_top_k_list(self, draft_layer_id: int) -> torch.Tensor:
        """Get top_k values for specified draft layer."""
        assert 0 <= draft_layer_id < len(self.top_k_list)
        return self.top_k_list[draft_layer_id]

    def dump_tree_info(self):
        """Debug: print tree structure info."""
        print(f"use_dynamic_tree: {self.use_dynamic_tree}")
        print(f"max_total_draft_tokens: {self.max_total_draft_tokens}")
        print(f"max_draft_len: {self.max_draft_len}")
        print(f"num_trees: {self.num_trees}")
        print(f"top_k_list: {self.top_k_list}")
        if not self.use_dynamic_tree:
            print(f"index_mapping_set: {self.index_mapping_set}")
            print(f"nodes_list_per_layer: {self.nodes_list_per_layer}")
            print(f"eagle_paths[0]:\n{self.eagle_paths[0]}")
            print(f"spec_dec_mask_matrix[0]:\n{self.spec_dec_mask_matrix[0]}")
            print(f"tree_attn_bias[0]:\n{self.tree_attn_bias[0]}")
            print(f"spec_dec_position_offsets[0]: {self.spec_dec_position_offsets[0]}")

    def get_bias_for_builder(self, tree_idx: int = 0) -> torch.Tensor:
        """Get attention bias in format for TreeAttentionMetadataBuilder.

        Returns 2D tensor [tree_len, tree_len] compatible with the existing
        _prepare_tree_attn_bias() format used by TreeAttentionMetadataBuilder.

        This enables dynamic tree support where the bias can be swapped
        per-request instead of being static for the entire run.

        Usage:
            builder.set_tree_attn_bias(tree_manager.get_bias_for_builder())

        Args:
            tree_idx: Tree index (for dynamic trees, each request has own tree)

        Returns:
            Float tensor [max_tokens+1, max_tokens+1] ready for qq_bias parameter
        """
        return self.tree_attn_bias[tree_idx].to(torch.float32)

    def get_sliced_bias_for_drafting(
        self,
        draft_index: int,
        max_query_len: int,
        tree_idx: int = 0,
    ) -> torch.Tensor:
        """Get sliced attention bias for a specific drafting level.

        Mirrors the slicing logic in TreeAttentionMetadataBuilder.build_for_drafting().

        Args:
            draft_index: Draft level (0 = root, 1+ = subsequent levels)
            max_query_len: Query length for the current level
            tree_idx: Tree index

        Returns:
            Sliced bias tensor for the draft level
        """
        if draft_index == 0:
            # Root level uses empty bias (prefill)
            return torch.empty(0, device=self.device)

        # Slice bias excluding root level
        bias = self.tree_attn_bias[tree_idx]
        start = 1
        end = 1 + max_query_len
        return bias[start:end, start:end].contiguous()


def create_spec_tree_manager_from_choices(
    eagle_choices: List[tuple],
    max_num_requests: int = 1,
    device: str = "cuda",
) -> SpecTreeManager:
    """Create SpecTreeManager from eagle_choices list.

    Args:
        eagle_choices: List of tuples representing tree structure.
                       Example: [(0,), (0,0), (0,1), (1,), (1,0)]
        max_num_requests: Max concurrent requests (for dynamic tree)
        device: CUDA device

    Returns:
        Initialized SpecTreeManager
    """
    # Convert tuples to lists if needed
    choices = [list(c) for c in eagle_choices]

    # Compute max_draft_len (max depth)
    max_draft_len = max(len(c) for c in choices) if choices else 1

    # Compute max_total_draft_tokens
    max_total_draft_tokens = len(choices)

    return SpecTreeManager(
        max_num_requests=max_num_requests,
        use_dynamic_tree=False,  # Static tree from choices
        max_total_draft_tokens=max_total_draft_tokens,
        max_draft_len=max_draft_len,
        eagle_choices=choices,
        device=device,
    )
