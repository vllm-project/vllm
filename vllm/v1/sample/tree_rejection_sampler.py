# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.sample.rejection_sampler import (PLACEHOLDER_TOKEN_ID,
                                              RejectionSampler)
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.tree_spec_decode.tree_drafter_params import TreeDrafterParams

logger = init_logger(__name__)

BLOCK_SIZE: int = 64


class TreeRejectionSampler(RejectionSampler):

    def __init__(
        self,
        tree_drafter_params: TreeDrafterParams,
        max_batch_size: int,
        main_sampler: Sampler,
        device: Optional[torch.device],
    ):
        super().__init__(main_sampler, device)
        self.tree_mask = torch.tensor(tree_drafter_params.attn_mask,
                                      device=device)[:, 1:].contiguous()
        # Cumulative # of tokens per level, including the root token.
        self.cu_tokens_per_level = [
            num_drafts + 1
            for num_drafts in tree_drafter_params.cu_drafts_per_level
        ]

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
        self.tree_index_offsets = np.arange(self.tree_size)

    def forward(
        self,
        metadata: SpecDecodeMetadata,
        # [num_tokens, vocab_size]
        draft_probs: Optional[torch.Tensor],
        # [num_tokens, vocab_size]
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        """
        Args:
            metadata:
                Metadata for spec decoding.
            draft_probs (Optional[torch.Tensor]):
                Probability distribution for the draft tokens. Shape is
                [num_tokens, vocab_size]. Can be None if probabilities are
                not provided, which is the case for ngram spec decode.
            logits (torch.Tensor):
                Target model's logits probability distribution.
                Shape is [num_tokens, vocab_size]. Here, probabilities from
                different requests are flattened into a single tensor because
                this is the shape of the output logits.
            sampling_metadata (vllm.v1.sample.metadata.SamplingMetadata):
                Additional metadata needed for sampling, such as temperature,
                top-k/top-p parameters, or other relevant information.
        Returns:
            output_token_ids (torch.Tensor):
                A tensor containing the final output token IDs.
        """

        # Example tree structure (2 levels of drafts):
        #         0 (root)
        #        / \
        #       1   2   level 1
        #     / |   | \
        #    3  4   5  6    level 2

        draft_tree_size = self.tree_size - 1
        tree_internal_index_offsets = (
            self.tree_index_offsets[:self.tree_internal_size])
        draft_index_offsets = self.tree_index_offsets[:draft_tree_size]
        draft_token_ids = metadata.draft_token_ids
        draft_token_ids_cpu = draft_token_ids.cpu()

        # 8
        num_reqs = len(metadata.num_draft_tokens)
        # [1, 8, 8, 0, 0, 0, 0, 0]
        num_draft_tokens = np.array(metadata.num_draft_tokens)
        # [2, 9, 9, 1, 1, 1, 1, 1]
        num_tokens = num_draft_tokens + 1
        # [0, 1, 1, 0, 0, 0, 0, 0]
        is_tree_decode = num_draft_tokens == draft_tree_size

        # [0, 0, 0, 0, 0, 0, 0, 0]
        start_indices = np.zeros(num_reqs, dtype=np.int32)
        # [0, 2, 11, 20, 21, 22, 23, 24]
        np.cumsum(num_tokens[:-1], out=start_indices[1:])

        # Create output token ids buffer.
        output_token_ids = torch.empty(
            # +1 for the bonus token.
            (num_reqs, draft_tree_size + 1),
            dtype=torch.
            int32,  # Consistent with SamplerOutput.sampled_token_ids.
            device=self.device,
        )

        # [0, 0, 0, 0, 0, 0, 0, 0]
        accepted_index_offsets = np.zeros(num_reqs, dtype=np.int32)

        num_tree_decodes = is_tree_decode.sum()
        if num_tree_decodes > 0:
            # Compute target probabilities for all logits corresponding to
            # internal nodes in the tree.
            vocab_size = logits.shape[-1]
            # [2, 11]
            tree_decode_start_indices = start_indices[is_tree_decode]
            # [[2, 3, 4],
            #  [11, 12, 13]]
            tree_internal_indices = (tree_decode_start_indices[:, None] +
                                     tree_internal_index_offsets)
            # [2, 3, 4, 11, 12, 13]
            tree_internal_indices_tensor = torch.from_numpy(
                tree_internal_indices.flatten()).to(self.device)
            target_probs = self.compute_tree_target_probs(
                logits[tree_internal_indices_tensor],
                is_tree_decode,
                num_tree_decodes,
                sampling_metadata,
            ).view(num_tree_decodes, -1, vocab_size)
            # Sample target token ids from the target probabilities.
            # TODO(TheEpicDolphin): Add support for probabilistic-style
            # rejection sampling, as used in EAGLE.
            target_token_ids = target_probs.argmax(dim=-1)

            # Get the draft token ids for batches with full draft trees.
            # [0, 0]
            draft_start_indices = np.zeros(num_tree_decodes, dtype=np.int32)
            # [1, 9]
            np.cumsum(num_draft_tokens[is_tree_decode][:-1],
                      out=draft_start_indices[1:])
            # [[1, 2, 3, ... , 8]
            #  [9, 10, 11, ... , 16]]
            tree_draft_indices = (draft_start_indices[:, None] +
                                  draft_index_offsets)
            # [[311, 8844, 2349, 387, 4732, 96618, 311, 334],
            #  [3634, 279, 323, 11, 438, 15861, 3634, 7016]]
            draft_token_ids_np = draft_token_ids_cpu[tree_draft_indices].numpy(
            )

            # Move sampled target token ids to CPU.
            # [[311, 6435, 96618],
            #  [279, 11, 15861]]
            target_token_ids_np = target_token_ids.cpu().numpy()

            # For each tree decode batch, find longest path from the root node.
            path_lengths = np.zeros(
                # +1 for the root token.
                (num_tree_decodes, draft_tree_size + 1),
                dtype=np.int32)
            path_lengths[:, 0] = 1
            for level in range(1, self.tree_depth):
                # level 2:
                # (3, 9)
                start, end = self.draft_slices[level]
                # [1, 1, 1, 2, 2, 2]
                parent_indices = self.parent_indices[level]
                # [[0, 0, 0, 0, 0, 0],
                #  [0, 1, 0, 1, 0, 0]]
                sample_match = (draft_token_ids_np[:, start - 1:end - 1] ==
                                target_token_ids_np[:, parent_indices])
                nonzero_length = path_lengths[:, parent_indices] > 0
                combined_mask = sample_match & nonzero_length
                # [[1, 2, 0, 0, 0, 0, 0, 0, 0],-> [[1, 2, 0, 0, 0, 0, 0, 0, 0],
                #  [1, 0, 2, 0, 0, 0, 0, 0, 0]]    [1, 0, 2, 0, 0, 0, 3, 0, 0]]
                path_lengths[:, start:end][combined_mask] = level + 1
            # [1, 6, 0, 0, 0, 0, 0, 0]
            accepted_index_offsets[is_tree_decode] = path_lengths.argmax(
                axis=-1)

        # Calculate grid dimensions.
        grid_dim_0 = num_reqs
        grid_dim_1 = triton.cdiv(draft_tree_size, BLOCK_SIZE)
        grid = (grid_dim_0, grid_dim_1)

        # Launch kernel to set accepted draft token ids in output buffer.
        accepted_index_offsets_tensor = torch.from_numpy(
            accepted_index_offsets).to(self.device, non_blocking=True)
        _scatter_accepted_tokens_kernel[grid](
            draft_token_ids_ptr=draft_token_ids,
            output_token_ids_ptr=output_token_ids,
            accepted_offsets_ptr=accepted_index_offsets_tensor,
            tree_mask_ptr=self.tree_mask,
            draft_tree_size=draft_tree_size,
            placeholder_token_id=PLACEHOLDER_TOKEN_ID,
            block_size=BLOCK_SIZE,
        )

        # Sample and add a bonus token to the accepted paths.
        # [0, 2 + 1, 11 + 6, 20, 21, 22, 23, 24]
        bonus_token_indices = start_indices + accepted_index_offsets
        bonus_token_indices_tensor = torch.from_numpy(bonus_token_indices).to(
            self.device, non_blocking=True)
        bonus_sampler_output = self.main_sampler(
            logits=logits[bonus_token_indices_tensor],
            sampling_metadata=sampling_metadata,
        )
        output_token_ids[:,
                         -1] = bonus_sampler_output.sampled_token_ids.view(-1)
        return SamplerOutput(
            sampled_token_ids=output_token_ids,
            logprobs_tensors=bonus_sampler_output.logprobs_tensors,
        )

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


@triton.jit
def _scatter_accepted_tokens_kernel(
    draft_token_ids_ptr,
    output_token_ids_ptr,
    accepted_offsets_ptr,
    tree_mask_ptr,
    draft_tree_size,
    placeholder_token_id: tl.constexpr,
    block_size: tl.constexpr,
):
    """
    For batches that correspond to tree decodes, accepted token ids from
    draft_token_ids are scattered to the corresponding indices in
    output_token_ids. All other indices are set to placeholder_token_id.

    Whether a token from draft_token_ids is accepted or not is determined by
    indexing into tree_mask via accepted_offsets.

    Args:
        draft_token_ids_ptr: [num_reqs, draft_tree_size] Draft token ids.
        output_token_ids_ptr: [num_reqs, draft_tree_size + 1] Output buffer.
        accepted_offsets_ptr: [num_reqs] - Indices into tree_mask rows.
        tree_mask_ptr: [draft_tree_size + 1, draft_tree_size] - Boolean masks
                       for paths to each node in the tree.
        draft_tree_size: Size of draft tree.
        placeholder_token_id: Placeholder token id for rejected tokens.
        block_size: Block size

    Grid: (num_reqs, ceil(draft_tree_size / BLOCK_SIZE))
    """

    req_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    # Get the accepted token index offset for this request.
    accepted_offset = tl.load(accepted_offsets_ptr + req_idx)

    # Calculate which tokens this block processes.
    block_start = block_idx * block_size
    token_offsets = block_start + tl.arange(0, block_size)
    token_mask = token_offsets < draft_tree_size

    # Get accepted path mask. Index as tree_mask[accepted_offset, :].
    path_mask_base = accepted_offset * draft_tree_size
    accepted_path_mask = tl.load(tree_mask_ptr + path_mask_base +
                                 token_offsets,
                                 mask=token_mask,
                                 other=0)

    # Load draft tokens for this request.
    draft_base = req_idx * draft_tree_size
    draft_tokens = tl.load(draft_token_ids_ptr + draft_base + token_offsets,
                           mask=token_mask,
                           other=placeholder_token_id)

    # Select draft tokens based on the accepted path mask.
    output_tokens = tl.where(accepted_path_mask, draft_tokens,
                             placeholder_token_id)

    # Store to output at the same positions.
    output_width = draft_tree_size + 1
    output_base = req_idx * output_width
    tl.store(output_token_ids_ptr + output_base + token_offsets,
             output_tokens,
             mask=token_mask)
