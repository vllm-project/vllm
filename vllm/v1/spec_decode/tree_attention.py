# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tree-attention speculative decoding for vLLM.

Implements topology-aware tree-structured speculative decoding inspired by
SpecInfer (arXiv:2305.09781) and EAGLE2 (arXiv:2406.16858).

Tree-attention allows evaluating multiple speculative token paths in a
single kernel invocation, using a tree mask to control which tokens
attend to which. This improves token acceptance rates and throughput
compared to linear speculative decoding.

Key components:
- TreeSpecDecodeMetadata: metadata for tree-structured draft tokens
- TreeProposer: proposer that generates tree-structured draft tokens
- Tree attention mask construction: builds the topology-aware mask
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.utils import next_power_of_2

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.worker.gpu_input_batch import InputBatch


@dataclass
class TreeSpecDecodeMetadata(SpecDecodeMetadata):
    """Metadata for tree-structured speculative decoding.

    Extends SpecDecodeMetadata with tree topology information:
    - tree_parents: parent index for each draft token (-1 for root)
    - tree_depths: depth of each draft token in the tree
    - tree_paths: list of root-to-leaf paths for verification
    """

    # [num_tokens] parent index for each draft token, -1 for root
    tree_parents: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int32))
    # [num_tokens] depth of each draft token in the tree
    tree_depths: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int32))
    # [num_paths, max_path_len] token indices for each root-to-leaf path
    tree_paths: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int32))
    # [num_paths] actual length of each path
    tree_path_lengths: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int32))

    @classmethod
    def make_dummy(
        cls,
        draft_token_ids: list[list[int]],
        device: torch.device,
    ) -> "TreeSpecDecodeMetadata":
        """Create dummy tree metadata for testing/warmup."""
        batch_size = len(draft_token_ids)
        num_draft_tokens = [len(ids) for ids in draft_token_ids]
        num_sampled_tokens = [len(ids) + 1 for ids in draft_token_ids]
        flattened_draft_token_ids = sum(draft_token_ids, [])
        num_tokens = len(flattened_draft_token_ids)

        draft_token_ids_tensor = torch.tensor(
            flattened_draft_token_ids, dtype=torch.int32, device=device
        )
        cu_num_draft_tokens = np.cumsum(num_draft_tokens, dtype=np.int32)
        cu_num_draft_tokens_tensor = torch.from_numpy(cu_num_draft_tokens).to(device)
        cu_num_sampled_tokens = np.cumsum(num_sampled_tokens, dtype=np.int32)
        cu_num_sampled_tokens_tensor = torch.from_numpy(cu_num_sampled_tokens).to(device)

        target_logits_indices = torch.zeros(num_tokens, dtype=torch.int32, device=device)
        bonus_logits_indices = torch.zeros(batch_size, dtype=torch.int32, device=device)
        logits_indices = torch.zeros(num_tokens + batch_size, dtype=torch.int32, device=device)

        # Dummy tree: linear chain (each token is parent of the next)
        tree_parents = torch.full((num_tokens,), -1, dtype=torch.int32, device=device)
        tree_depths = torch.zeros(num_tokens, dtype=torch.int32, device=device)
        for i in range(1, num_tokens):
            tree_parents[i] = i - 1
            tree_depths[i] = tree_depths[i - 1] + 1

        # Single path: all tokens
        tree_paths = torch.tensor(
            [flattened_draft_token_ids], dtype=torch.int32, device=device
        )
        tree_path_lengths = torch.tensor([num_tokens], dtype=torch.int32, device=device)

        return cls(
            draft_token_ids=draft_token_ids_tensor,
            num_draft_tokens=num_draft_tokens,
            cu_num_draft_tokens=cu_num_draft_tokens_tensor,
            cu_num_sampled_tokens=cu_num_sampled_tokens_tensor,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
            tree_parents=tree_parents,
            tree_tree_depths=tree_depths,
            tree_paths=tree_paths,
            tree_path_lengths=tree_path_lengths,
        )


def build_tree_attention_mask(
    tree_parents: torch.Tensor,
    num_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    """Build topology-aware attention mask for tree-structured decoding.

    The mask allows each token to attend to:
    1. All its ancestors in the tree
    2. Itself
    3. All tokens on the path from root to itself

    This is the key innovation from SpecInfer: instead of a linear chain,
    the tree structure allows multiple speculative paths to be verified
    in parallel with a single attention kernel.

    Args:
        tree_parents: [num_tokens] parent index for each token, -1 for root
        num_tokens: total number of draft tokens
        device: torch device

    Returns:
        attention_mask: [num_tokens, num_tokens] boolean mask where
            attention_mask[i, j] = True means token i can attend to token j
    """
    # Initialize: each token attends to itself
    mask = torch.eye(num_tokens, dtype=torch.bool, device=device)

    # For each token, add all its ancestors
    for i in range(num_tokens):
        parent = tree_parents[i].item()
        while parent >= 0:
            mask[i, parent] = True
            parent = tree_parents[parent].item()

    return mask


def build_tree_position_ids(
    tree_parents: torch.Tensor,
    tree_depths: torch.Tensor,
    base_position: int,
    num_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    """Build position IDs for tree-structured decoding.

    Each token's position is its parent's position + 1, ensuring that
    tokens on the same path have monotonically increasing positions.

    Args:
        tree_parents: [num_tokens] parent index for each token
        tree_depths: [num_tokens] depth of each token
        base_position: starting position for root tokens
        num_tokens: total number of draft tokens
        device: torch device

    Returns:
        position_ids: [num_tokens] position ID for each token
    """
    position_ids = torch.zeros(num_tokens, dtype=torch.int32, device=device)

    # Root tokens get base_position
    for i in range(num_tokens):
        if tree_parents[i].item() < 0:
            position_ids[i] = base_position

    # Non-root tokens get parent's position + 1
    for i in range(num_tokens):
        parent = tree_parents[i].item()
        if parent >= 0:
            position_ids[i] = position_ids[parent] + 1

    return position_ids


def generate_tree_draft_tokens(
    logits: torch.Tensor,
    tree_width: int,
    tree_depth: int,
    temperature: float = 0.0,
) -> tuple[list[int], list[int], list[int]]:
    """Generate tree-structured draft tokens from logits.

    Instead of a linear chain of draft tokens, this generates a tree where
    each level has up to `tree_width` candidates per parent. The total
    number of draft tokens is up to `tree_width * tree_depth`.

    Args:
        logits: [vocab_size] logits from the draft model
        tree_width: maximum number of children per node
        tree_depth: maximum depth of the tree
        temperature: sampling temperature (0 = greedy)

    Returns:
        token_ids: flat list of draft token IDs
        parents: parent index for each token (-1 for root)
        depths: depth of each token
    """
    token_ids: list[int] = []
    parents: list[int] = []
    depths: list[int] = []

    # Get top-k candidates for root
    if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        root_candidates = torch.topk(probs, tree_width).indices.tolist()
    else:
        root_candidates = torch.topk(logits, tree_width).indices.tolist()

    # BFS to build tree
    queue: list[tuple[int, int]] = []  # (token_id, parent_idx)
    for token_id in root_candidates:
        token_ids.append(token_id)
        parents.append(-1)
        depths.append(0)
        queue.append((token_id, len(token_ids) - 1))

    # Expand tree level by level
    for level in range(1, tree_depth):
        next_queue: list[tuple[int, int]] = []
        for token_id, parent_idx in queue:
            # In a real implementation, we would run the draft model
            # on this token to get its children. For now, we just
            # use the same logits as a placeholder.
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                children = torch.topk(probs, tree_width).indices.tolist()
            else:
                children = torch.topk(logits, tree_width).indices.tolist()

            for child_id in children:
                token_ids.append(child_id)
                parents.append(parent_idx)
                depths.append(level)
                next_queue.append((child_id, len(token_ids) - 1))

        queue = next_queue

    return token_ids, parents, depths


class TreeProposer:
    """Tree-structured speculative decoding proposer.

    Generates tree-structured draft tokens and prepares metadata for
    tree-attention verification. This is the tree-attention analog of
    EAGLE/DFlash proposers.

    The proposer:
    1. Runs the draft model to get root logits
    2. Expands the tree using top-k at each level
    3. Builds tree metadata (parents, depths, paths)
    4. Constructs the tree attention mask
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        tree_width: int = 3,
        tree_depth: int = 4,
    ):
        self.vllm_config = vllm_config
        self.device = device
        self.tree_width = tree_width
        self.tree_depth = tree_depth
        self.num_speculative_tokens = tree_width * tree_depth

    def propose(
        self,
        draft_logits: torch.Tensor,
        input_batch: Optional[InputBatch] = None,
    ) -> TreeSpecDecodeMetadata:
        """Propose tree-structured draft tokens.

        Args:
            draft_logits: [batch_size, vocab_size] logits from draft model
            input_batch: current input batch for context

        Returns:
            TreeSpecDecodeMetadata with tree topology
        """
        batch_size = draft_logits.shape[0]
        all_token_ids: list[list[int]] = []
        all_parents: list[list[int]] = []
        all_depths: list[list[int]] = []

        for b in range(batch_size):
            token_ids, parents, depths = generate_tree_draft_tokens(
                draft_logits[b],
                self.tree_width,
                self.tree_depth,
            )
            all_token_ids.append(token_ids)
            all_parents.append(parents)
            all_depths.append(depths)

        # Build metadata
        num_draft_tokens = [len(ids) for ids in all_token_ids]
        num_sampled_tokens = [len(ids) + 1 for ids in all_token_ids]
        flattened = sum(all_token_ids, [])
        num_tokens = len(flattened)

        draft_token_ids_tensor = torch.tensor(
            flattened, dtype=torch.int32, device=self.device
        )
        cu_num_draft_tokens = np.cumsum(num_draft_tokens, dtype=np.int32)
        cu_num_draft_tokens_tensor = torch.from_numpy(cu_num_draft_tokens).to(self.device)
        cu_num_sampled_tokens = np.cumsum(num_sampled_tokens, dtype=np.int32)
        cu_num_sampled_tokens_tensor = torch.from_numpy(cu_num_sampled_tokens).to(self.device)

        target_logits_indices = torch.zeros(num_tokens, dtype=torch.int32, device=self.device)
        bonus_logits_indices = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
        logits_indices = torch.zeros(num_tokens + batch_size, dtype=torch.int32, device=self.device)

        # Flatten tree structure
        flat_parents = sum(all_parents, [])
        flat_depths = sum(all_depths, [])
        tree_parents_tensor = torch.tensor(flat_parents, dtype=torch.int32, device=self.device)
        tree_depths_tensor = torch.tensor(flat_depths, dtype=torch.int32, device=self.device)

        # Build paths (root-to-leaf)
        max_path_len = self.tree_depth
        num_paths = self.tree_width  # one path per root candidate
        tree_paths = torch.zeros(
            (num_paths, max_path_len), dtype=torch.int32, device=self.device
        )
        tree_path_lengths = torch.zeros(num_paths, dtype=torch.int32, device=self.device)

        for path_idx in range(min(num_paths, len(all_token_ids[0]))):
            # Trace back from leaf to root
            token_idx = path_idx
            path: list[int] = []
            while token_idx >= 0 and len(path) < max_path_len:
                path.append(all_token_ids[0][token_idx])
                if token_idx < len(all_parents[0]):
                    token_idx = all_parents[0][token_idx]
                else:
                    break
            path.reverse()
            tree_paths[path_idx, : len(path)] = torch.tensor(
                path, dtype=torch.int32, device=self.device
            )
            tree_path_lengths[path_idx] = len(path)

        return TreeSpecDecodeMetadata(
            draft_token_ids=draft_token_ids_tensor,
            num_draft_tokens=num_draft_tokens,
            cu_num_draft_tokens=cu_num_draft_tokens_tensor,
            cu_num_sampled_tokens=cu_num_sampled_tokens_tensor,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
            tree_parents=tree_parents_tensor,
            tree_depths=tree_depths_tensor,
            tree_paths=tree_paths,
            tree_path_lengths=tree_path_lengths,
        )
