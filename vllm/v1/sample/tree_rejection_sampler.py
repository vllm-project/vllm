# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import torch

from vllm.logger import init_logger
from vllm.tree_drafter_params import TreeDrafterParams
from vllm.triton_utils import tl
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import RejectionSampler, compute_probs
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

logger = init_logger(__name__)

PLACEHOLDER_TOKEN_ID: tl.constexpr = -1
EPS: torch.float = 1e-10


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
                                 device=device)[1:, 1:]
        self.expanded_tree_mask = tree_mask.expand((max_batch_size, -1, -1))
        self.batch_indices = torch.arange(max_batch_size, device=device)
        self.cu_num_tokens_per_tree_level = torch.tensor(
            # Prepend with 0 for the root level, which has 0 draft tokens.
            [0] + tree_drafter_params.cu_drafts_per_level,
            device=device,
        )
        self.num_children_per_tree_level = (
            tree_drafter_params.child_drafts_per_level)
        self.main_sampler = main_sampler

    """
    The implementation strictly follows the algorithm described in
        https://arxiv.org/abs/2211.17192.
    However, we want to clarify the terminology used in the implementation:
    accepted tokens: tokens that are accepted based on the relationship
            between the "raw" draft and target probabilities.
    recovered tokens: tokens that are sampled based on the adjusted probability
        distribution, which is derived from both the draft and target
        probabilities.
    bonus tokens:
        If all proposed tokens are accepted, the bonus token is added to the
        end of the sequence. The bonus token is only sampled from the target
        probabilities. We pass in the bonus tokens instead of sampling them
        in the rejection sampler to allow for more flexibility in the
        sampling process. For example, we can use top_p, top_k sampling for
        bonus tokens, while spec decode does not support these sampling
        strategies.
    output tokens:
        Tokens are finally generated with the rejection sampler.
        output tokens = accepted tokens + recovered tokens + bonus tokens
    """

    def forward(
        self,
        metadata: SpecDecodeMetadata,
        # [num_tokens, 1]
        draft_probs: Optional[torch.Tensor],
        # [num_tokens, vocab_size]
        target_logits: torch.Tensor,
        # [batch_size, 1]
        bonus_token_ids: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        """
        Args:
            metadata:
                Metadata for spec decoding.
            draft_probs (Optional[torch.Tensor]):
                Probability distribution for the draft tokens. Shape is
                [num_tokens, 1]. Can be None if probabilities are
                not provided, which is the case for ngram spec decode.
            target_logits (torch.Tensor):
                Target model's logits probability distribution.
                Shape is [num_tokens, vocab_size]. Here, probabilities from
                different requests are flattened into a single tensor because
                this is the shape of the output logits.
                NOTE: `target_logits` can be updated in place to save memory.
            bonus_token_ids_tensor (torch.Tensor):
                A tensor containing bonus tokens. Shape is [batch_size, 1].
                Bonus tokens are added to the end of the sequence if all
                proposed tokens are accepted. We generate the bonus tokens
                outside of the rejection sampler with the default sampling
                strategy. It allows for more flexibility in the sampling
                process such as top_p, top_k sampling.
            sampling_metadata (vllm.v1.sample.metadata.SamplingMetadata):
                Additional metadata needed for sampling, such as temperature,
                top-k/top-p parameters, or other relevant information.
        Returns:
            output_token_ids (torch.Tensor):
                A tensor containing the final output token IDs.
        """
        assert draft_probs is not None

        batch_size = len(metadata.num_draft_tokens)
        batch_indices = self.batch_indices[:batch_size]
        expanded_tree_mask = self.expanded_tree_mask[:batch_size]
        tree_size = self.cu_num_tokens_per_tree_level[-1]
        tree_depth = len(self.num_children_per_tree_level)
        assert metadata.num_draft_tokens[
            0] == tree_size, f"Expected each batch to have {tree_size} draft tokens, but got: {metadata.num_draft_tokens}"

        # Compute the target probabilities for the draft tokens.
        indices = torch.arange(metadata.cu_num_draft_tokens[-1],
                               device=target_logits.device)
        target_probs = compute_probs(
            target_logits,
            metadata.cu_num_draft_tokens,
            sampling_metadata,
        )[indices, metadata.draft_token_ids]

        # Reshape the target probs, draft probs, and draft tokens to
        # [batch_size, tree_size].
        target_probs = target_probs.view(batch_size, -1)
        draft_probs = draft_probs.view(batch_size, -1)
        draft_token_ids = metadata.draft_token_ids.view(batch_size, -1)

        # Generate random samples from uniform distribution.
        random_samples = torch.rand_like(draft_probs)
        # Perform one-shot specualtive sampling.
        accepted_tokens = random_samples < (target_probs / (draft_probs + EPS))

        # Compute the paths of accepted tokens in the tree, starting from the
        # root node and ending at the first rejected token.
        first_rejected_level = torch.ones(
            (batch_size, 1),
            device=self.expanded_tree_mask.device,
            dtype=torch.int32)
        for level in range(tree_depth):
            # Broadcast the first rejected level for each child.
            num_children = self.num_children_per_tree_level[level]
            first_rejected_level = first_rejected_level.repeat_interleave(
                num_children, dim=1)

            # Get start and end token indices for the current tree level.
            start = self.cu_num_tokens_per_tree_level[level]
            end = self.cu_num_tokens_per_tree_level[level + 1]

            # Increase the first rejected level for each accepted token whose
            # ancestors were all accepted.
            acceptance_mask = accepted_tokens[:, start:end] & (first_rejected_level == level + 1)
            first_rejected_level += acceptance_mask

        # For each batch, get the longest accepted path in the tree.
        rejected_level, rejected_level_child_index = first_rejected_level.max(
            dim=1)

        # Get the offsets into the tree level.
        rejected_level_offset = self.cu_num_tokens_per_tree_level[
            rejected_level - 1]
        rejected_position = rejected_level_offset + rejected_level_child_index

        # Sample from the target distribution an extra token for each
        # accepted path length < tree_depth.
        resampled = rejected_level < (tree_depth + 1)
        resampled_batches = batch_indices[resampled]
        resampled_positions = rejected_position[resampled]
        target_logits_to_resample = target_logits.view(batch_size, tree_size,
                                                       -1)[resampled_batches,
                                                           resampled_positions]
        sampler_output = self.main_sampler(
            logits=target_logits_to_resample,
            sampling_metadata=sampling_metadata,
        )
        draft_token_ids[
            resampled_batches,
            resampled_positions] = sampler_output.sampled_token_ids.view(-1)

        # Insert placeholder tokens for rejected tokens.
        path_masks = expanded_tree_mask[batch_indices, rejected_position]
        output_token_ids = torch.where(path_masks, draft_token_ids,
                                       PLACEHOLDER_TOKEN_ID)
        return output_token_ids.view(-1)
