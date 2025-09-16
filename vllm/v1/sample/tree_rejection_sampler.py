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
        self.tree_internal_size = self.cu_tokens_per_level[-2]
        self.tree_index_offsets = torch.arange(self.tree_size, device=device)

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
        draft_tree_size = self.tree_size - 1
        tree_internal_index_offsets = (
            self.tree_index_offsets[:self.tree_internal_size])
        draft_index_offsets = self.tree_index_offsets[:draft_tree_size]

        num_reqs = len(metadata.num_draft_tokens)
        # [1, 8, 8, 0, 0, 0, 0, 0]
        num_draft_tokens = torch.tensor(metadata.num_draft_tokens,
                                        device=device)
        # [0, 1, 1, 0, 0, 0, 0, 0]
        is_tree_decode = num_draft_tokens == draft_tree_size
        # [0, 1, 2, 3, 4, 5, 6, 7]
        start_indices = torch.arange(num_reqs, device=device)
        # [0, 2, 11, 20, 21, 22, 23, 24]
        start_indices[1:] += metadata.cu_num_draft_tokens[:-1]

        # Create output token ids buffer.
        output_token_ids = torch.empty(
            # +1 for the bonus token.
            (num_reqs, draft_tree_size + 1),
            dtype=torch.
            int32,  # Consistent with SamplerOutput.sampled_token_ids.
            device=device,
        )
        output_token_ids.fill_(PLACEHOLDER_TOKEN_ID)

        # [0, 0, 0, 0, 0, 0, 0, 0]
        accepted_index_offsets = torch.zeros_like(is_tree_decode,
                                                  dtype=torch.int32)

        num_tree_decodes = is_tree_decode.sum()
        if num_tree_decodes > 0:
            # Compute target probabilities for all logits corresponding to
            # internal nodes in the tree.
            vocab_size = target_logits.shape[-1]
            # [0, 9]
            tree_decode_start_indices = start_indices[is_tree_decode]
            # [[0, 1, 2],
            #  [9, 10, 11]]
            tree_internal_indices = (tree_decode_start_indices.unsqueeze(1) +
                                     tree_internal_index_offsets)
            tree_internal_logits = target_logits[
                tree_internal_indices.flatten()]
            target_probs = self.compute_tree_target_probs(
                tree_internal_logits,
                is_tree_decode,
                num_tree_decodes,
                sampling_metadata,
            ).view(num_tree_decodes, -1, vocab_size)

            # Sample tokens from the target probabilities.
            # TODO(TheEpicDolphin): Add support for probabilistic-style
            # rejection sampling, as used in EAGLE.
            target_token_ids = target_probs.argmax(dim=-1)

            # Get the draft token ids for batches with full draft trees.
            # [0, 0]
            draft_start_indices = torch.zeros(num_tree_decodes,
                                              device=device,
                                              dtype=torch.int32)
            # [0, 8]
            draft_start_indices[1:] = (
                metadata.cu_num_draft_tokens[is_tree_decode][:-1])
            # [[0, 1, 2, ... , 7]
            #  [8, 9, 10, ... , 15]]
            tree_draft_indices = (draft_start_indices.unsqueeze(1) +
                                  draft_index_offsets)
            draft_token_ids = metadata.draft_token_ids[tree_draft_indices]

            # Move sampled target and draft token tensors to CPU.
            # [[311, 6435, 96618],
            #  [279, 11, 15861]]
            target_token_ids_cpu = target_token_ids.cpu()
            # [[311, 8844, 2349, 387, 4732, 96618, 311, 334],
            #  [3634, 279, 323, 11, 438, 15861, 3634, 7016]]
            draft_token_ids_cpu = draft_token_ids.cpu()

            # For each tree decode batch, find longest path from the root node.
            path_lengths_cpu = torch.zeros(
                # +1 for the root token.
                (num_tree_decodes, draft_tree_size + 1),
                dtype=torch.int32,
                device="cpu")
            path_lengths_cpu[:, 0] = 1
            for level in range(1, self.tree_depth):
                # level 2:
                # (3, 9)
                start, end = self.draft_slices[level]
                # [1, 1, 1, 2, 2, 2]
                parent_indices = self.parent_indices[level]
                # [[0, 0, 0, 0, 0, 0],
                #  [0, 1, 0, 1, 0, 0]]
                sample_match = (draft_token_ids_cpu[:, start - 1:end - 1] ==
                                target_token_ids_cpu[:, parent_indices])
                nonzero_length = path_lengths_cpu[:, parent_indices] > 0
                # [[1, 2, 0, 0, 0, 0, 0, 0, 0],-> [[1, 2, 0, 0, 0, 0, 0, 0, 0],
                #  [1, 0, 2, 0, 0, 0, 0, 0, 0]]    [1, 0, 2, 0, 0, 0, 3, 0, 0]]
                path_lengths_cpu[:, start:end].masked_fill_(
                    sample_match & nonzero_length, level + 1)
            # [1, 6, 0, 0, 0, 0, 0, 0]
            path_lengths = path_lengths_cpu.argmax(dim=-1).to(
                device, dtype=torch.int32)
            accepted_index_offsets[is_tree_decode] = path_lengths

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
            path_masks = tree_mask[tree_batch_indices, path_lengths]

            # Set accepted draft tokens.
            accepted_draft_tokens = draft_token_ids[path_masks]
            scatter_mask = torch.zeros_like(output_token_ids, dtype=torch.bool)
            scatter_mask[is_tree_decode, :-1] = path_masks
            output_token_ids.masked_scatter_(scatter_mask,
                                             accepted_draft_tokens)

        # Sample and add a bonus token to the accepted paths.
        # [0, 2 + 1, 11 + 6, 20, 21, 22, 23, 24]
        bonus_token_indices = start_indices + accepted_index_offsets
        bonus_sampler_output = self.main_sampler(
            logits=target_logits[bonus_token_indices],
            sampling_metadata=sampling_metadata,
        )
        output_token_ids[:,
                         -1] = bonus_sampler_output.sampled_token_ids.view(-1)
        return output_token_ids

    def compute_tree_target_probs(self, logits: torch.Tensor,
                                  is_tree_decode: torch.Tensor,
                                  num_tree_decodes: int,
                                  sampling_metadata: SamplingMetadata):
        if sampling_metadata.all_greedy:
            return logits

        # How many times to repeat the temperature, top-k, and top-p
        # for each tree-decode batch.
        num_repeats = logits.shape[0] // num_tree_decodes

        assert sampling_metadata.temperature is not None
        temperature = sampling_metadata.temperature[is_tree_decode]
        temperature = temperature.repeat_interleave(num_repeats)
        logits.div_(temperature.view(-1, 1))

        top_k = None
        if sampling_metadata.top_k is not None:
            top_k = sampling_metadata.top_k[is_tree_decode]
            top_k = top_k.repeat_interleave(num_repeats)
        top_p = None
        if sampling_metadata.top_p is not None:
            top_p = sampling_metadata.top_p[is_tree_decode]
            top_p = top_p.repeat_interleave(num_repeats)
        logits = apply_top_k_top_p(logits, top_k, top_p)
        output_probs = logits.softmax(dim=-1, dtype=torch.float32)
        return output_probs
