# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import torch

from vllm.logger import init_logger
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.sample.rejection_sampler import (PLACEHOLDER_TOKEN_ID,
                                              RejectionSampler)
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.tree_spec_decode.tree_drafter_params import TreeDrafterParams

logger = init_logger(__name__)


class TreeRejectionSampler(RejectionSampler):

    def __init__(
        self,
        tree_drafter_params: TreeDrafterParams,
        max_batch_size: int,
        main_sampler: Sampler,
        device: Optional[torch.device],
    ):
        super().__init__()
        tree_mask = torch.tensor(tree_drafter_params.attn_mask,
                                 device=device)[:, 1:].contiguous()
        self.expanded_tree_mask = tree_mask.expand((max_batch_size, -1, -1))
        self.batch_indices = torch.arange(max_batch_size, device=device)
        # Cumulative # of tokens per level, including the root token.
        self.cu_tokens_per_level = [
            num_drafts + 1
            for num_drafts in tree_drafter_params.cu_drafts_per_level
        ]
        # Num children at every level.
        self.main_sampler = main_sampler

        # Get tree depth (# levels) and width (# drafts at last level).
        self.tree_depth = len(self.cu_tokens_per_level)
        self.tree_width = self.cu_tokens_per_level[
            -1] - self.cu_tokens_per_level[-2]
        self.tree_size = self.cu_tokens_per_level[-1]

        # Get per-level slices of draft indices, and per-level indices
        # for their corresponding parents.
        num_children_per_level = tree_drafter_params.child_drafts_per_level
        self.draft_slices = [(0, 0)]
        self.parent_indices: list[list[int]] = [[]]
        parents_end = 0
        for level in range(1, self.tree_depth):
            # Add slice of draft indices for this level.
            self.draft_slices.append((self.cu_tokens_per_level[level - 1],
                                      self.cu_tokens_per_level[level]))
            # Add indices for this level's parents.
            parents_start = parents_end
            parents_end = self.cu_tokens_per_level[level - 1]
            num_children = num_children_per_level[level - 1]
            indices = []
            for parent_idx in range(parents_start, parents_end):
                indices += [parent_idx] * num_children
            self.parent_indices.append(indices)

        # Precompute indices for logits corresponding to tree-internal
        # tokens across batches.
        num_tree_internal_tokens = self.cu_tokens_per_level[-2]
        self.tree_internal_index_offsets = torch.arange(
            num_tree_internal_tokens, device=device)

    def forward(
        self,
        metadata: SpecDecodeMetadata,
        # [num_tokens, vocab_size]
        draft_probs: Optional[torch.Tensor],
        # [num_tokens, vocab_size]
        target_logits: torch.Tensor,
        bonus_token_ids: Optional[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        """
        Args:
            metadata:
                Metadata for spec decoding.
            draft_probs (Optional[torch.Tensor]):
                Probability distribution for the draft tokens. Shape is
                [num_tokens, vocab_size]. Can be None if probabilities are
                not provided, which is the case for ngram spec decode.
            target_logits (torch.Tensor):
                Target model's logits probability distribution.
                Shape is [num_tokens, vocab_size]. Here, probabilities from
                different requests are flattened into a single tensor because
                this is the shape of the output logits.
            bonus_token_ids_tensor (Optional[torch.Tensor]):
                Not used, and expected to be None. This method will generate
                a bonus token for each request depending on which branch is
                accepted.
            sampling_metadata (vllm.v1.sample.metadata.SamplingMetadata):
                Additional metadata needed for sampling, such as temperature,
                top-k/top-p parameters, or other relevant information.
        Returns:
            output_token_ids (torch.Tensor):
                A tensor containing the final output token IDs.
        """
        assert bonus_token_ids is None

        # Example tree structure (2 levels of drafts):
        #         0 (root)
        #        / \
        #       1   2   level 1
        #     / |   | \
        #    3  4   5  6    level 2

        device = target_logits.device
        num_reqs = len(metadata.num_draft_tokens)
        # [8, 8, 0, 0, 0, 0, 0, 0]
        num_draft_tokens = torch.tensor(metadata.num_draft_tokens,
                                        device=device)
        draft_tree_size = self.tree_size - 1
        # [1, 1, 0, 0, 0, 0, 0, 0]
        tree_decode_mask = num_draft_tokens == draft_tree_size
        # [0, 1, 2, 3, 4, 5, 6, 7]
        start_indices = torch.arange(num_reqs, device=device)
        # [0, 9, 18, 19, 20, 21, 22, 23]
        start_indices[1:] += metadata.cu_num_draft_tokens[:-1]

        # Compute target probabilities for all logits corresponding to internal
        # nodes in the tree.
        vocab_size = target_logits.shape[-1]
        # [0, 9]
        tree_decode_start_indices = start_indices[tree_decode_mask]
        # [[0, 1, 2],
        #  [9, 10, 11]]
        tree_internal_indices = tree_decode_start_indices.unsqueeze(
            1) + self.tree_internal_index_offsets
        num_tree_decodes, num_logits_per_batch = tree_internal_indices.shape
        tree_internal_logits = target_logits[tree_internal_indices.flatten()]
        target_probs = self.compute_probs(
            tree_internal_logits,
            num_logits_per_batch,
            sampling_metadata,
        ).view(num_tree_decodes, -1, vocab_size)

        # Sample tokens from the target probabilities.
        # TODO(TheEpicDolphin): Add support for probabilistic-style rejection
        # sampling, as used in EAGLE.
        target_token_ids = target_probs.argmax(dim=-1).cpu()

        # Reshape the draft token ids to [num_tree_decodes, draft_tree_size].
        draft_token_ids = metadata.draft_token_ids.view(num_tree_decodes, -1)

        # Move sampled target and draft token tensors to CPU.
        # [[311, 6435, 96618],
        #  [279, 11, 15861]]
        target_token_ids_cpu = target_token_ids.cpu()
        # [[311, 8844, 2349, 387, 4732, 96618, 311, 334],
        #  [3634, 279, 323, 11, 438, 15861, 3634, 7016]]
        draft_token_ids_cpu = draft_token_ids.cpu()

        # For each batch, find longest path from the root node.
        path_lengths = torch.zeros(
            # +1 for the root token.
            (num_tree_decodes, draft_tree_size + 1),
            dtype=torch.int32)
        path_lengths[:, 0] = 1
        for level in range(1, self.tree_depth):
            # level 2:
            # (3, 9)
            start, end = self.draft_slices[level]
            # [1, 1, 1, 2, 2, 2]
            parent_indices = self.parent_indices[level]
            # [[0, 0, 0, 0, 0, 0],
            #  [0, 1, 0, 1, 0, 0]]
            sample_match = draft_token_ids_cpu[:, start - 1:end -
                                               1] == target_token_ids_cpu[:,
                                                                          parent_indices]
            nonzero_length = path_lengths[:, parent_indices] > 0
            # [[1, 2, 0, 0, 0, 0, 0, 0, 0],  ->  [[1, 2, 0, 0, 0, 0, 0, 0, 0],
            #  [1, 0, 2, 0, 0, 0, 0, 0, 0]]       [1, 0, 2, 0, 0, 0, 3, 0, 0]]
            path_lengths[:,
                         start:end].masked_fill_(sample_match & nonzero_length,
                                                 level + 1)
        # [1, 6]
        accepted_token_index_offsets = path_lengths.argmax(dim=-1).to(device)

        # Get boolean masks for the paths to the accepted tokens.
        # [0, 1]
        tree_batch_indices = self.batch_indices[:num_tree_decodes]
        # [[[1, 0, 0, 0, 0, 0, 0],  <- batch 0
        #   [1, 1, 0, 0, 0, 0, 0],
        #   [1, 0, 1, 0, 0, 0, 0],
        #   [1, 1, 0, 1, 0, 0, 0],
        #   [1, 1, 0, 0, 1, 0, 0],
        #   [1, 0, 1, 0, 0, 1, 0],
        #   [1, 0, 1, 0, 0, 0, 1]],
        #  [[1, 0, 0, 0, 0, 0, 0],  <- batch 1
        #   [1, 1, 0, 0, 0, 0, 0],
        #   [1, 0, 1, 0, 0, 0, 0],
        #   [1, 1, 0, 1, 0, 0, 0],
        #   [1, 1, 0, 0, 1, 0, 0],
        #   [1, 0, 1, 0, 0, 1, 0],
        #   [1, 0, 1, 0, 0, 0, 1]]]
        tree_mask = self.expanded_tree_mask[:num_tree_decodes]
        # [1, 6] => [[1, 0, 0, 0, 0, 0], <- batch 0
        #            [0, 1, 0, 0, 0, 1]]  <- batch 1
        path_masks = tree_mask[tree_batch_indices,
                               accepted_token_index_offsets]

        # Create output buffer.
        output_token_ids = torch.empty(
            # +1 for the bonus token.
            (num_reqs, draft_tree_size + 1),
            dtype=torch.
            int32,  # Consistent with SamplerOutput.sampled_token_ids.
            device=device,
        )
        output_token_ids.fill_(PLACEHOLDER_TOKEN_ID)

        # Set accepted draft tokens.
        accepted_draft_tokens = draft_token_ids[path_masks]
        scatter_mask = torch.zeros_like(output_token_ids, dtype=torch.bool)
        scatter_mask[tree_decode_mask, :-1] = path_masks
        output_token_ids.masked_scatter_(scatter_mask, accepted_draft_tokens)

        # Sample and add a bonus token to the accepted paths.
        bonus_token_indices = start_indices
        bonus_token_indices[tree_decode_mask] += accepted_token_index_offsets
        bonus_sampler_output = self.main_sampler(
            logits=target_logits[bonus_token_indices],
            sampling_metadata=sampling_metadata,
        )
        output_token_ids[:,
                         -1] = bonus_sampler_output.sampled_token_ids.view(-1)
        return output_token_ids

    def compute_probs(self, logits: torch.Tensor, logits_per_batch: int,
                      sampling_metadata: SamplingMetadata):
        if sampling_metadata.all_greedy:
            return logits

        assert sampling_metadata.temperature is not None
        temperature = sampling_metadata.temperature.repeat_interleave(
            logits_per_batch)
        logits.div_(temperature.view(-1, 1))

        top_k = None
        if sampling_metadata.top_k is not None:
            top_k = sampling_metadata.top_k.repeat_interleave(logits_per_batch)
        top_p = None
        if sampling_metadata.top_p is not None:
            top_p = sampling_metadata.top_p.repeat_interleave(logits_per_batch)
        logits = apply_top_k_top_p(logits, top_k, top_p)
        output_probs = logits.softmax(dim=-1, dtype=torch.float32)
        return output_probs
