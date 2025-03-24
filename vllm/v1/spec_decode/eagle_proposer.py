# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from typing import List, Tuple

import torch
from torch import Tensor, nn

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata


class EagleProposer:

    def __init__(self, vllm_config: VllmConfig, model: nn.Module,
                 sampling_metadata: SamplingMetadata):
        self._partial_prefill_last_token_hidden_states = defaultdict(
            lambda: torch.zeros(256, dtype=torch.float32))
        self._vllm_config = vllm_config
        self._model = model
        self._sampling_metadata = sampling_metadata

    def propose(
            self, req_ids: List, target_model_input_ids: Tensor,
            target_model_positions: Tensor, target_model_hidden_states: Tensor,
            target_model_cumulative_seq_lens: Tensor,
            accepted_token_ids: Tensor, prefill_mask: Tensor,
            num_lookahead_slots: int,
            attention_metadata: FlashAttentionMetadata
    ) -> Tuple[Tensor, Tensor]:
        target_model_seq_lens = torch.diff(
            target_model_cumulative_seq_lens,
            prepend=torch.tensor(
                [0], device=target_model_cumulative_seq_lens.device))
        target_model_start_locs = torch.cat([
            torch.tensor([0], device=target_model_cumulative_seq_lens.device),
            target_model_cumulative_seq_lens[:-1]
        ])
        accepted_token_lengths = (accepted_token_ids != -1).sum(dim=1)
        # Verify that the accepted_token_lengths for prefill sequences is <= 1.
        # It can be zero for partial prefills.
        assert torch.all(accepted_token_lengths[prefill_mask] <= 0), \
            "All accepted_token_lengths where prefill_mask is True must be <= 0"

        (completed_prefill_mask, partial_prefill_mask, non_prefill_mask,
         zero_position_mask) = self._construct_masks(prefill_mask,
                                                     target_model_positions,
                                                     target_model_start_locs,
                                                     accepted_token_lengths)

        (eagle_seq_lens, eagle_start_locs, eagle_num_tokens) =\
            self._construct_eagle_seq_metadata(
                target_model_seq_lens, accepted_token_lengths,
                prefill_mask, zero_position_mask)

        eagle_previous_hidden_state = self._get_prev_hidden_states_for_proposer(
            eagle_num_tokens, target_model_hidden_states,
            target_model_start_locs, target_model_seq_lens,
            accepted_token_lengths, eagle_start_locs, eagle_seq_lens,
            non_prefill_mask, partial_prefill_mask, completed_prefill_mask,
            zero_position_mask, req_ids)

        eagle_positions = self._get_positions_for_proposer(
            target_model_positions, target_model_start_locs, eagle_start_locs,
            eagle_seq_lens, eagle_num_tokens, completed_prefill_mask,
            partial_prefill_mask)

        eagle_input_ids = self._get_input_ids_for_proposer(
            target_model_input_ids, target_model_start_locs,
            target_model_seq_lens, eagle_start_locs, eagle_seq_lens,
            eagle_num_tokens, completed_prefill_mask, partial_prefill_mask,
            zero_position_mask, accepted_token_ids)

        eagle_prefill_attn_metadata = self._construct_eagle_prefill_attn_metadata(
            attention_metadata, eagle_num_tokens, target_model_start_locs,
            target_model_seq_lens, eagle_start_locs, eagle_seq_lens,
            partial_prefill_mask, completed_prefill_mask, zero_position_mask)

        # We will sample everything except partial prefills
        eagle_hidden_states = None
        seq_to_sample_mask = ~partial_prefill_mask
        with set_forward_context(eagle_prefill_attn_metadata,
                                 self._vllm_config):
            eagle_hidden_states = self._model(
                input_ids=eagle_input_ids,
                positions=eagle_positions,
                previous_hidden_states=eagle_previous_hidden_state,
            )
            logits_indices = \
                eagle_start_locs[seq_to_sample_mask] + eagle_seq_lens[seq_to_sample_mask] - 1
            sample_hidden_states = eagle_hidden_states[logits_indices]
            logits = self._model.compute_logits(sample_hidden_states, None)
            sampler_output = self._model.sample(
                logits=logits,
                sampling_metadata=self._sampling_metadata,
            )

        eagle_speculation_attn_metadata = self._construct_eagle_speculate_attn_metadata(
            eagle_prefill_attn_metadata, seq_to_sample_mask, eagle_start_locs,
            eagle_seq_lens)
        eagle_input_ids = sampler_output.sampled_token_ids.squeeze(1)
        eagle_positions = torch.arange(0, seq_to_sample_mask.sum() + 1)
        eagle_previous_hidden_state = eagle_hidden_states
        with set_forward_context(eagle_speculation_attn_metadata,
                                 self._vllm_config):

            for _ in range(0, num_lookahead_slots):
                eagle_hidden_states = self._model(
                    input_ids=eagle_input_ids,
                    positions=eagle_positions,
                    previous_hidden_states=eagle_previous_hidden_state,
                )
                eagle_previous_hidden_state = eagle_hidden_states
                logits = self._model.compute_logits(eagle_hidden_states, None)
                sampler_output = self._model.sample(
                    logits=logits,
                    sampling_metadata=self._sampling_metadata,
                )
                eagle_input_ids = sampler_output.sampled_token_ids.squeeze(1)
                eagle_speculation_attn_metadata = \
                    self._update_eagle_speculate_attn_metadata(
                        eagle_speculation_attn_metadata)

        print('eagle_input_ids ' + str(eagle_input_ids))

        return target_model_input_ids, target_model_input_ids

    def _construct_masks(
        self, prefill_mask: Tensor, target_model_positions: Tensor,
        target_model_start_locs: Tensor, accepted_token_lengths: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Constructs various boolean masks based on input conditions.

        Args:
            prefill_mask: A boolean tensor indicating which sequences are in the prefill phase.
            target_model_positions: Token positions as passed to the target model.
            target_model_start_locs: Indices indicating the starting positions of sequences.
            accepted_token_lengths: A tensor representing the number of accepted tokens per sequence.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: 
                - completed_prefill_mask: Mask for prefill sequences with accepted tokens.
                - partial_prefill_mask: Mask for prefill sequences with no accepted tokens.
                - non_prefill_mask: Mask for non-prefill sequences.
                - zero_position_mask: Mask indicating sequences that start at position 0.
        """
        completed_prefill_mask = prefill_mask.bool() & (accepted_token_lengths
                                                        > 0)
        partial_prefill_mask = prefill_mask.bool() & (accepted_token_lengths
                                                      == 0)
        non_prefill_mask = ~prefill_mask
        first_positions = target_model_positions[target_model_start_locs]
        zero_position_mask = first_positions == 0
        return (completed_prefill_mask, partial_prefill_mask, non_prefill_mask,
                zero_position_mask)

    def _construct_eagle_seq_metadata(
        self,
        target_model_seq_lens: Tensor,
        accepted_token_lengths: Tensor,
        prefill_mask: Tensor,
        zero_position_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, int]:
        """
        Constructs sequence metadata for the Eagle model, including sequence lengths, 
        start locations, and total token count.

        Args:
            target_model_seq_lens: Sequence lengths from the target model.
            accepted_token_lengths: Number of accepted tokens per sequence.
            prefill_mask: Mask indicating prefill sequences.
            zero_position_mask: Mask indicating sequences whose first positon is 0.

        Returns:
            Tuple[Tensor, Tensor, int]: 
                - eagle_seq_lens: Sequence lengths for the Eagle model.
                - eagle_start_locs: Start indices for each sequence.
                - eagle_num_tokens: Total number of tokens in the Eagle model.
        """
        # Initialize Eagle sequence lengths with zeros
        eagle_seq_lens = torch.zeros_like(target_model_seq_lens)

        # Compute sequence lengths for prefill sequences
        eagle_seq_lens[prefill_mask & zero_position_mask] = (
            target_model_seq_lens[prefill_mask & zero_position_mask] +
            accepted_token_lengths[prefill_mask & zero_position_mask] - 1)
        eagle_seq_lens[prefill_mask & ~zero_position_mask] = (
            target_model_seq_lens[prefill_mask & ~zero_position_mask] +
            accepted_token_lengths[prefill_mask & ~zero_position_mask])

        # Compute sequence lengths for non-prefill sequences
        non_prefill_mask = ~prefill_mask
        eagle_seq_lens[non_prefill_mask] = accepted_token_lengths[
            non_prefill_mask]

        # Compute start locations for each sequence
        eagle_start_locs = torch.cat([
            torch.zeros(1, dtype=torch.long),
            torch.cumsum(eagle_seq_lens, dim=0)
        ])[:-1]

        # Compute total number of tokens
        eagle_num_tokens = int(eagle_seq_lens.sum().item())

        return eagle_seq_lens, eagle_start_locs, eagle_num_tokens

    def _update_eagle_speculate_attn_metadata(
        self,
        eagle_prefill_attention_metadata: FlashAttentionMetadata,
    ) -> FlashAttentionMetadata:
        eagle_prefill_attention_metadata.seq_lens = \
            eagle_prefill_attention_metadata.seq_lens + 1
        eagle_prefill_attention_metadata.max_seq_len = \
            eagle_prefill_attention_metadata.seq_lens.max().item()
        eagle_prefill_attention_metadata.slot_mapping =\
            eagle_prefill_attention_metadata.slot_mapping + 1
        return eagle_prefill_attention_metadata

    def _construct_eagle_speculate_attn_metadata(
        self,
        eagle_prefill_attention_metadata: FlashAttentionMetadata,
        sampled_mask: Tensor,
        eagle_start_locs: Tensor,
        eagle_seq_lens: Tensor,
    ) -> FlashAttentionMetadata:
        eagle_speculate_attn_metadata = eagle_prefill_attention_metadata
        eagle_speculate_attn_metadata.num_actual_tokens = sampled_mask.sum(
        ).item()
        eagle_speculate_attn_metadata.max_query_len = 1
        eagle_speculate_attn_metadata.query_start_loc = torch.arange(
            0, eagle_speculate_attn_metadata.num_actual_tokens + 1)
        eagle_speculate_attn_metadata.seq_lens = \
            eagle_speculate_attn_metadata.seq_lens[sampled_mask] + 1
        eagle_speculate_attn_metadata.max_seq_len = \
            eagle_speculate_attn_metadata.seq_lens.max().item()
        last_slot_indices = eagle_start_locs[sampled_mask] + eagle_seq_lens[
            sampled_mask] - 1
        eagle_speculate_attn_metadata.slot_mapping =\
            eagle_speculate_attn_metadata.slot_mapping[last_slot_indices] + 1
        return eagle_speculate_attn_metadata

    def _construct_eagle_prefill_attn_metadata(
            self, attention_metadata: FlashAttentionMetadata,
            num_eagle_tokens: int, target_model_start_locs: Tensor,
            target_model_seq_lens: Tensor, eagle_start_locs: Tensor,
            eagle_seq_lens: Tensor, partial_prefill_mask: Tensor,
            completed_prefill_mask: Tensor,
            zero_position_mask: Tensor) -> FlashAttentionMetadata:
        eagle_seq_lens = attention_metadata.seq_lens - \
            target_model_seq_lens + eagle_seq_lens
        eagle_slot_mapping = self._update_attention_metadata_slot_mappings(
            attention_metadata.slot_mapping, num_eagle_tokens,
            target_model_start_locs, eagle_start_locs, eagle_seq_lens,
            partial_prefill_mask, completed_prefill_mask, zero_position_mask)
        eagle_attention_metadata = FlashAttentionMetadata(
            num_actual_tokens=num_eagle_tokens,
            max_query_len=int(eagle_seq_lens.max().item()),
            query_start_loc=torch.cat(
                [eagle_start_locs,
                 torch.tensor([num_eagle_tokens])]),
            max_seq_len=int(eagle_seq_lens.max().item()),
            seq_lens=eagle_seq_lens,
            block_table=attention_metadata.block_table,
            slot_mapping=eagle_slot_mapping,
            use_cascade=False,
            common_prefix_len=0,
            cu_prefix_query_lens=None,
            prefix_kv_lens=None,
            suffix_kv_lens=None,
            num_input_tokens=1,
        )
        return eagle_attention_metadata

    def _update_attention_metadata_slot_mappings(
            self, target_model_slot_mapping: Tensor, num_eagle_tokens: int,
            target_model_start_locs: Tensor, eagle_start_locs: Tensor,
            eagle_seq_lens: Tensor, partial_prefill_mask: Tensor,
            completed_prefill_mask: Tensor,
            zero_position_mask: Tensor) -> Tensor:
        eagle_slot_mapping = torch.zeros((num_eagle_tokens),
                                         dtype=target_model_slot_mapping.dtype)
        prefill_mask = partial_prefill_mask | completed_prefill_mask
        # Handle prefill and zero positions first.
        non_prefill_zero_start_mask = ~prefill_mask & zero_position_mask
        start_indices = eagle_start_locs[non_prefill_zero_start_mask]
        end_indices = start_indices + eagle_seq_lens[
            non_prefill_zero_start_mask]
        range_lengths = end_indices - start_indices
        start_pos = target_model_slot_mapping[
            target_model_start_locs[non_prefill_zero_start_mask]]
        eagle_slot_mapping = self._populate_positions(eagle_slot_mapping,
                                                      start_indices,
                                                      range_lengths, start_pos)

        # Prefill with non zero mask
        non_zero_prefill_mask = ~zero_position_mask & prefill_mask
        start_indices = eagle_start_locs[non_zero_prefill_mask]
        end_indices = start_indices + eagle_seq_lens[non_zero_prefill_mask]
        range_lengths = end_indices - start_indices
        start_pos = target_model_slot_mapping[
            target_model_start_locs[non_zero_prefill_mask]]
        eagle_slot_mapping = self._populate_positions(eagle_slot_mapping,
                                                      start_indices,
                                                      range_lengths, start_pos)

        return eagle_slot_mapping

    def _expand_indices(self, starts: Tensor, ends: Tensor) -> Tensor:
        """
        Expands start and end indices into a contiguous range of indices.

        Args:
            starts: A tensor containing the start indices for each range.
            ends: A tensor containing the end indices for each range.

        Returns:
            Tensor: A flattened tensor containing all indices from the specified ranges.
        """
        indices = torch.cat([
            torch.arange(start, end, dtype=torch.long)
            for start, end in zip(starts, ends)
        ])
        return indices

    def _get_prev_hidden_states_for_proposer(
            self, eagle_num_tokens: int,
            target_model_hidden_state: torch.Tensor,
            target_model_start_locs: torch.Tensor,
            target_model_seq_lens: torch.Tensor,
            accepted_token_lengths: torch.Tensor,
            eagle_start_locs: torch.Tensor, eagle_seq_lens: torch.Tensor,
            non_prefill_mask: torch.Tensor, partial_prefill_mask: torch.Tensor,
            completed_prefill_mask: torch.Tensor,
            zero_position_mask: torch.Tensor, req_ids: list) -> torch.Tensor:
        """
        Constructs the previous hidden states tensor for the Eagle model. 
        Copies hidden states for both prefill and non-prefill sequences, maintaining 
        last token hidden states for partial prefills to be used in subsequent forward passes.
        
        Args:
            eagle_num_tokens: Number of tokens in this forward pass.
            target_model_hidden_state: Hidden states from the target model.
            target_model_start_locs: Start locations of sequences in the target model.
            target_model_seq_lens: Sequence lengths in the target model.
            accepted_token_lengths: Number of accepted tokens per sequence.
            eagle_start_locs: Start locations in the Eagle model.
            eagle_seq_lens: Sequence lengths in the Eagle model.
            non_prefill_mask: Mask for non-prefill sequences.
            partial_prefill_mask: Mask for partial prefill sequences.
            completed_prefill_mask: Mask for completed prefill sequences.
            zero_position_mask: Mask for sequences starting at position 0.
            req_ids: Request IDs corresponding to sequences.

        Returns:
            Tensor: The previous hidden states for the Eagle model.
        """
        eagle_prev_hidden_states = torch.zeros(
            (eagle_num_tokens, target_model_hidden_state.shape[1]),
            dtype=target_model_hidden_state.dtype)

        # Non-Prefill: Copy hidden states corresponding to accepted tokens.
        target_model_starts = target_model_start_locs[non_prefill_mask]
        target_model_ends = target_model_starts + accepted_token_lengths[
            non_prefill_mask]
        eagle_starts = eagle_start_locs[non_prefill_mask]
        eagle_ends = eagle_starts + accepted_token_lengths[non_prefill_mask]

        eagle_prev_hidden_states[self._expand_indices(eagle_starts, eagle_ends)] = \
            target_model_hidden_state[self._expand_indices(target_model_starts, target_model_ends)]

        # Prefill: Copy hidden states for both partial and completed prefill sequences.
        prefill_mask = partial_prefill_mask | completed_prefill_mask

        target_model_starts = target_model_start_locs[prefill_mask]
        target_model_ends_all = target_model_start_locs + target_model_seq_lens
        target_model_ends = torch.where(
            completed_prefill_mask,
            target_model_ends_all,
            target_model_ends_all -
            1  # Exclude last token for partial prefills.
        )[prefill_mask]

        # For non-first prefill sequences (non-zero start position),
        # the first hidden state comes from a previous iteration.
        eagle_starts_all = torch.where(zero_position_mask, eagle_start_locs,
                                       eagle_start_locs + 1)
        eagle_starts = eagle_starts_all[prefill_mask]
        eagle_ends = eagle_start_locs[prefill_mask] + eagle_seq_lens[
            prefill_mask]

        eagle_prev_hidden_states[self._expand_indices(eagle_starts, eagle_ends)] = \
            target_model_hidden_state[self._expand_indices(target_model_starts, target_model_ends)]

        # Append last hidden state for non-first prefill sequences.
        non_zero_prefill_mask = prefill_mask & ~zero_position_mask
        indices = non_zero_prefill_mask.nonzero(as_tuple=True)[0].tolist()
        req_ids_from_prev_step = [req_ids[i] for i in indices]
        target_indices = eagle_start_locs[non_zero_prefill_mask]

        for req_id, target_index in zip(req_ids_from_prev_step,
                                        target_indices):
            eagle_prev_hidden_states[
                target_index] = self._partial_prefill_last_token_hidden_states[
                    req_id]

        # Store last token hidden states for partial prefills.
        indices = partial_prefill_mask.nonzero(as_tuple=True)[0].tolist()
        req_ids_from_prev_step = [req_ids[i] for i in indices]
        target_indices = target_model_start_locs[
            partial_prefill_mask] + target_model_seq_lens[partial_prefill_mask]

        for req_id, target_index in zip(req_ids_from_prev_step,
                                        target_indices):
            self._partial_prefill_last_token_hidden_states[
                req_id] = target_model_hidden_state[target_index]

        # Remove stored hidden states for completed prefill sequences.
        indices = completed_prefill_mask.nonzero(as_tuple=True)[0].tolist()
        req_ids_for_completed_prefill = [req_ids[i] for i in indices]

        for req_id in req_ids_for_completed_prefill:
            self._partial_prefill_last_token_hidden_states.pop(req_id, None)

        return eagle_prev_hidden_states

    def _populate_positions(self, position_tensor: Tensor,
                            start_indices: Tensor, lengths: Tensor,
                            start_positions: Tensor) -> Tensor:
        """
        Populates the position tensor by assigning position values at the corresponding indices.

        Args:
            position_tensor: Tensor that will be populated with positions.
            start_indices: Start indices for the ranges.
            lengths: Length of each range to be populated.
            start_positions: Starting positions for each range.

        Returns:
            Tensor: The updated position tensor with the assigned position values.
        """
        position_values = torch.cat([
            torch.arange(start_position,
                         start_position + length,
                         dtype=torch.long,
                         device=position_tensor.device)
            for start_position, length in zip(start_positions, lengths)
        ])

        start_indices_expanded = start_indices.repeat_interleave(lengths)
        index_increments = torch.cat([
            torch.arange(0,
                         length,
                         dtype=torch.long,
                         device=position_tensor.device) for length in lengths
        ])

        position_tensor[start_indices_expanded +
                        index_increments] = position_values
        return position_tensor

    def _get_positions_for_proposer(
        self,
        target_model_positions: Tensor,
        target_model_start_locs: Tensor,
        eagle_start_locs: Tensor,
        eagle_seq_lens: Tensor,
        eagle_num_tokens: int,
        completed_prefill_mask: Tensor,
        partial_prefill_mask: Tensor,
    ) -> Tensor:
        """
        Generates position tensor for the Eagle model based on the target model positions.

        Args:
            target_model_positions: Positions from the target model.
            target_model_start_locs: Start locations in the target model.
            eagle_start_locs: Start locations in the Eagle model.
            eagle_seq_lens: Sequence lengths in the Eagle model.
            eagle_num_tokens: Total number of tokens for the Eagle model.
            completed_prefill_mask: Mask for completed prefill sequences.
            partial_prefill_mask: Mask for partial prefill sequences.

        Returns:
            Tensor: The position tensor for the Eagle model.
        """
        eagle_positions = torch.zeros(eagle_num_tokens,
                                      dtype=target_model_positions.dtype)
        prefill_mask = partial_prefill_mask | completed_prefill_mask

        # Positions are offset by 1 in the Eagle model
        # Handle prefill sequences
        start_indices = eagle_start_locs[prefill_mask]
        range_lengths = eagle_seq_lens[prefill_mask]
        start_pos = torch.maximum(
            target_model_positions[target_model_start_locs[prefill_mask]] - 1,
            torch.tensor(0, device=target_model_positions.device))

        eagle_positions = self._populate_positions(eagle_positions,
                                                   start_indices,
                                                   range_lengths, start_pos)

        # Handle non-prefill sequences.
        # For non-prefill sequences, we start from the first position because
        # this corresponds to the next token to be processed in the Eagle model
        non_prefill_mask = ~prefill_mask
        start_indices = eagle_start_locs[non_prefill_mask]
        range_lengths = eagle_seq_lens[non_prefill_mask]
        start_pos = target_model_positions[
            target_model_start_locs[non_prefill_mask]]

        eagle_positions = self._populate_positions(eagle_positions,
                                                   start_indices,
                                                   range_lengths, start_pos)

        return eagle_positions

    def _get_input_ids_for_proposer(
        self,
        target_model_input_ids: Tensor,
        target_model_start_locs: Tensor,
        target_model_seq_lens: Tensor,
        eagle_start_locs: Tensor,
        eagle_seq_lens: Tensor,
        eagle_num_tokens: int,
        completed_prefill_mask: Tensor,
        partial_prefill_mask: Tensor,
        zero_position_mask: Tensor,
        accepted_token_ids: Tensor,
    ) -> Tensor:
        """
        Constructs input tokens for the Eagle model by copying input tokens for both 
        prefill and non-prefill sequences. Additionally, appends accepted tokens 
        for completed prefills.

        Args:
            target_model_input_ids: Input token IDs used by the target model.
            target_model_start_locs: Start positions in the target model.
            target_model_seq_lens: Sequence lengths in the target model.
            eagle_start_locs: Start positions in the Eagle model.
            eagle_seq_lens: Sequence lengths in the Eagle model.
            eagle_num_tokens: Total number of tokens to be passed to Eagle for
                this forward pass.
            completed_prefill_mask: Mask indicating completed prefill sequences.
            partial_prefill_mask: Mask indicating partial prefill sequences.
            zero_position_mask: Mask for sequences starting at position 0.
            accepted_token_ids: The accepted token IDs.

        Returns:
            Tensor: The input IDs for the Eagle model.
        """
        eagle_input_ids = torch.zeros(eagle_num_tokens,
                                      dtype=target_model_input_ids.dtype)

        # Handle Prefill Sequences (Completed + Partial)
        prefill_mask = partial_prefill_mask | completed_prefill_mask
        # For prefills starting at zero positon drop the first token.
        target_model_starts = torch.where(
            zero_position_mask, target_model_start_locs + 1,
            target_model_start_locs)[prefill_mask]
        target_model_ends = (target_model_start_locs +
                             target_model_seq_lens)[prefill_mask]

        eagle_starts = eagle_start_locs[prefill_mask]
        eagle_ends = torch.where(
            completed_prefill_mask, eagle_start_locs + eagle_seq_lens - 1,
            eagle_start_locs + eagle_seq_lens)[prefill_mask]

        # Copy input IDs from target model to Eagle model
        eagle_input_ids[self._expand_indices(eagle_starts, eagle_ends)] = \
            target_model_input_ids[self._expand_indices(target_model_starts, target_model_ends)]

        # Append accepted tokens for completed prefills
        completed_start_locs = eagle_start_locs[completed_prefill_mask]
        completed_seq_lens = eagle_seq_lens[completed_prefill_mask]
        # For completed prefills there is only 1 accepted token.
        accepted_tokens = accepted_token_ids[completed_prefill_mask, 0]

        eagle_input_ids[completed_start_locs + completed_seq_lens -
                        1] = accepted_tokens

        # Handle Non-Prefill Sequences
        non_prefill_mask = ~prefill_mask
        non_prefill_starts = eagle_start_locs[non_prefill_mask]
        non_prefill_lens = eagle_seq_lens[non_prefill_mask]
        non_prefill_accepted_tokens = accepted_token_ids[non_prefill_mask]

        for start_idx, seq_len, accepted_token in zip(
                non_prefill_starts, non_prefill_lens,
                non_prefill_accepted_tokens):
            eagle_input_ids[start_idx:start_idx +
                            seq_len] = accepted_token[:seq_len]

        return eagle_input_ids
