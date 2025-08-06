# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.tree_drafter_params import TreeDrafterParams
from vllm.triton_utils import tl
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

logger = init_logger(__name__)

PLACEHOLDER_TOKEN_ID: tl.constexpr = -1
EPS: torch.float = 1e-10


class TreeRejectionSampler(nn.Module):

    def __init__(
        self,
        tree_drafter_params: TreeDrafterParams,
        max_batch_size: int,
        main_sampler: Sampler,
        device: Optional[torch.device],
    ):
        super().__init__()
        tree_mask = torch.tensor(tree_drafter_params.attn_mask,
                                 device=device)[:, 1:]
        self.expanded_tree_mask = tree_mask.expand((max_batch_size, -1, -1))
        self.batch_indices = torch.arange(max_batch_size, device=device)
        # Cumulative # of draft tokens per level.
        self.cu_drafts_per_level = tree_drafter_params.cu_drafts_per_level
        # Cumulative # of tokens per level, including the root token.
        self.cu_tokens_per_level = [
            num_drafts + 1 for num_drafts in self.cu_drafts_per_level
        ]
        self.main_sampler = main_sampler

        # Get tree depth (# levels) and width (# drafts at last level).
        self.tree_depth = len(self.cu_drafts_per_level)
        self.tree_width = self.cu_drafts_per_level[
            -1] - self.cu_drafts_per_level[-2]

        # Used for getting the flattened tree position for any draft token,
        # indexed by it's level and position in the level.
        tree_draft_positions = torch.zeros(
            (self.tree_depth, self.tree_width),
            device=device,
            dtype=torch.int32,
        )
        for level in range(1, self.tree_depth):
            start = self.cu_tokens_per_level[level - 1]
            end = self.cu_tokens_per_level[level]
            level_num_drafts = end - start
            level_draft_positions = torch.arange(start, end, device=device)
            tree_draft_positions[
                level] = level_draft_positions.repeat_interleave(
                    self.tree_width // level_num_drafts)
        self.expanded_tree_draft_positions = tree_draft_positions.expand(
            (max_batch_size, -1, -1))

    def forward(
        self,
        metadata: SpecDecodeMetadata,
        # [num_tokens, vocab_size]
        draft_probs: Optional[torch.Tensor],
        # [num_tokens, vocab_size]
        target_logits: torch.Tensor,
        bonus_token_ids: Optional[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
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
            accepted_token_indices (list[torch.Tensor]):
                Contains the accepted token indices for each tree drafting
                request.
        """
        assert bonus_token_ids is None

        draft_tree_size = self.cu_drafts_per_level[-1]
        total_num_draft_tokens = sum(metadata.num_draft_tokens)
        num_tree_decodes = total_num_draft_tokens // draft_tree_size
        batch_indices = self.batch_indices[:num_tree_decodes]
        tree_mask = self.expanded_tree_mask[:num_tree_decodes]
        tree_draft_positions = self.expanded_tree_draft_positions[:num_tree_decodes]

        # Get only the logits associated with tree-drafted requests.
        num_tree_logits = num_tree_decodes * (1 + draft_tree_size)
        tree_target_logits = target_logits[:num_tree_logits]

        # Compute target probabilities for all logits corresponding to internal
        # nodes in the tree.
        num_internal_tokens = self.cu_tokens_per_level[-2]
        vocab_size = tree_target_logits.shape[-1]
        tree_target_logits = tree_target_logits.view(num_tree_decodes, -1, vocab_size)
        tree_internal_target_logits = tree_target_logits[:, :num_internal_tokens].clone()
        target_probs = self.compute_probs(
            tree_internal_target_logits,
            sampling_metadata,
        )

        # Reshape the draft token ids to [num_tree_decodes, draft_tree_size].
        draft_token_ids = metadata.draft_token_ids.view(num_tree_decodes, -1)

        # Below tensor will hold 1 for a token if accepted, and 0 if rejected.
        tree_acceptances = torch.zeros(
            (num_tree_decodes, self.tree_depth, self.tree_width),
            device=tree_mask.device,
            dtype=torch.int32)
        parents_end = 0
        for level in range(self.tree_depth - 1):
            # Get target and draft start and end token indices for the current
            # tree level.
            parents_start = parents_end
            parents_end = self.cu_tokens_per_level[level]
            drafts_start = self.cu_drafts_per_level[level]
            drafts_end = self.cu_drafts_per_level[level + 1]

            # Get the target probabilities and drafted token ids for the
            # current level.
            level_target_probs = target_probs[:, parents_start:parents_end]
            level_draft_token_ids = draft_token_ids[:, drafts_start:drafts_end]
            # Accept/reject tokens at the current level.
            level_acceptances = self.rejection_sample(level_target_probs,
                                                      level_draft_token_ids)

            # Broadcast the acceptances to the width of the tree.
            num_level_drafts = drafts_end - drafts_start
            tree_acceptances[:,
                             level, :] = (level_acceptances.repeat_interleave(
                                 self.tree_width // num_level_drafts, dim=1))

        # Get the boolean mask for the maximum length path of accepted tokens.
        path_lengths = tree_acceptances.argmin(dim=1)
        accepted_path_levels, accepted_path_indices = path_lengths.max(dim=1)
        accepted_paths = tree_draft_positions[batch_indices,
                                              accepted_path_levels,
                                              accepted_path_indices]
        path_masks = tree_mask[batch_indices, accepted_paths]

        # Create output buffer.
        num_req = len(metadata.num_draft_tokens)
        output_token_ids = torch.empty(
            # +1 for the bonus token.
            (num_req, draft_tree_size + 1),
            dtype=torch.
            int32,  # Consistent with SamplerOutput.sampled_token_ids.
            device=draft_token_ids.device,
        )
        output_token_ids.fill_(PLACEHOLDER_TOKEN_ID)

        # Set accepted draft tokens.
        output_token_ids[:num_tree_decodes, :draft_tree_size][path_masks] = draft_token_ids[path_masks]

        # Sample and add a bonus token to the accepted paths.
        bonus_target_logits = tree_target_logits[batch_indices, accepted_paths]
        bonus_sampler_output = self.main_sampler(
            logits=bonus_target_logits,
            sampling_metadata=sampling_metadata,
        )
        output_token_ids[:num_tree_decodes, -1] = bonus_sampler_output.sampled_token_ids.view(
                -1)

        if num_req > num_tree_decodes:
            # In some cases, we may have leftover requests with 0 draft tokens.
            # Sample a bonus token for each.
            bonus_sampler_output = self.main_sampler(
                logits=target_logits[num_tree_logits:],
                sampling_metadata=sampling_metadata,
            )
            output_token_ids[num_tree_decodes:,
                             -1] = bonus_sampler_output.sampled_token_ids.view(
                                 -1)

        accepted_token_indices = [torch.where(row)[0] for row in path_masks]
        return output_token_ids, accepted_token_indices

    def compute_probs(self, logits: torch.Tensor,
                      sampling_metadata: SamplingMetadata):
        if sampling_metadata.all_greedy:
            return logits

        assert sampling_metadata.temperature is not None
        batch_size, tokens_per_batch, vocab_size = logits.shape
        logits.div_(sampling_metadata.temperature.view(-1, 1, 1))

        top_k = None
        if sampling_metadata.top_k is not None:
            top_k = sampling_metadata.top_k.repeat_interleave(tokens_per_batch)
        top_p = None
        if sampling_metadata.top_p is not None:
            top_p = sampling_metadata.top_p.repeat_interleave(tokens_per_batch)
        logits = apply_top_k_top_p(logits.view(-1, vocab_size), top_k,
                                   top_p)
        output_probs = logits.softmax(dim=-1, dtype=torch.float32)
        return output_probs.view(batch_size, -1, vocab_size)

    def rejection_sample(self, target_probs: torch.Tensor,
                         draft_tokens: torch.Tensor):
        # TODO(TheEpicDolphin): Add support for probabilistic-style rejection
        # sampling, as used in EAGLE.
        target_argmax = target_probs.argmax(dim=-1)
        target_sampled_tokens = target_argmax.repeat_interleave(
            draft_tokens.shape[-1] // target_argmax.shape[-1],
            dim=1,
        )
        return target_sampled_tokens == draft_tokens

    @staticmethod
    def parse_output(
        output_token_ids: torch.Tensor,
        vocab_size: int,
    ) -> list[list[int]]:
        """Parse the output of the rejection sampler.

        Args:
            output_token_ids: The sampled token IDs in shape
                [batch_size, max_spec_len + 1]. The rejected tokens are
                replaced with `PLACEHOLDER_TOKEN_ID` by the rejection sampler
                and will be filtered out in this function.
            vocab_size: The size of the vocabulary.

        Returns:
            A list of lists of token IDs.
        """
        output_token_ids_np = output_token_ids.cpu().numpy()
        # Create mask for valid tokens.
        valid_mask = ((output_token_ids_np != PLACEHOLDER_TOKEN_ID) &
                      (output_token_ids_np < vocab_size))
        outputs = [
            row[valid_mask[i]].tolist()
            for i, row in enumerate(output_token_ids_np)
        ]
        return outputs
