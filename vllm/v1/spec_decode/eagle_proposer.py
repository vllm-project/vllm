from typing import Optional
import torch
from torch import Tensor
import numpy as np
from typing import List
from collections import defaultdict
from typing import Tuple
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata


class EagleProposer:

    def __init__(self):
        self.partial_prefill_last_token_hidden_states = defaultdict(
            lambda: torch.zeros(256, dtype=torch.float32)
        )
        #self._model = model

    def propose(
        self,
        req_ids: List,
        target_model_input_ids: Tensor,
        target_model_positions: Tensor,
        target_model_hidden_states: Tensor,
        target_model_cumulative_seq_lens: Tensor,
        accepted_token_ids: Tensor,
        is_prefill: Tensor,
        num_lookahead_slots: int,
        attention_metadata: FlashAttentionMetadata
    ) ->  Tuple[Tensor, Tensor] :
        target_model_seq_lens = torch.diff(
            target_model_cumulative_seq_lens, prepend=torch.tensor([0],
            device=target_model_cumulative_seq_lens.device))
        target_model_start_locs = torch.cat(
            [torch.tensor([0], device=target_model_cumulative_seq_lens.device),
            target_model_cumulative_seq_lens[:-1]])
        accepted_token_lengths = (accepted_token_ids != -1).sum(dim=1)
        completed_prefill_mask = is_prefill.bool() & (accepted_token_lengths > 0)
        partial_prefill_mask = is_prefill.bool() & (accepted_token_lengths == 0)
        non_prefill_mask = ~is_prefill
        first_positions = target_model_positions[target_model_start_locs]
        zero_position_mask = first_positions == 0
        eagle_seq_lens = torch.zeros_like(target_model_seq_lens)
        print('eagle_sequence_lengths ' + str(eagle_seq_lens))
        print('is_prefill ' + str(is_prefill))
        print('zero_position_mask ' + str(zero_position_mask))

        eagle_seq_lens[is_prefill & zero_position_mask] = (
            target_model_seq_lens[is_prefill & zero_position_mask] \
                + accepted_token_lengths[is_prefill & zero_position_mask] - 1
        )
        eagle_seq_lens[is_prefill & ~zero_position_mask] = (
            target_model_seq_lens[is_prefill & ~zero_position_mask] + \
                accepted_token_lengths[is_prefill & ~zero_position_mask]
        )
        eagle_seq_lens[non_prefill_mask] = accepted_token_lengths[non_prefill_mask]
        eagle_start_locs = torch.cat([
            torch.zeros(1, dtype=torch.long),
            torch.cumsum(eagle_seq_lens, dim=0)])[:-1]
        
        eagle_num_tokens = int(eagle_seq_lens.sum().long().item())
        print('eagle_sequence_lengths ' + str(eagle_seq_lens))
        eagle_previous_hidden_state = self._get_prev_hidden_states_for_proposer(
            req_ids, target_model_hidden_states,
            target_model_start_locs, target_model_seq_lens,
            eagle_start_locs,
            eagle_seq_lens, eagle_num_tokens, completed_prefill_mask,
            partial_prefill_mask, non_prefill_mask, zero_position_mask, 
            accepted_token_lengths
        )
        print('target_model_hidden_states ' + str(target_model_hidden_states))
        print('eagle_previous_hidden_state ' + str(eagle_previous_hidden_state))
        #eagle_input_ids = self._get_input_ids_for_proposer(target_model_input_ids, )

        eagle_positions = self._get_positions_for_proposer(
            target_model_positions, target_model_start_locs,
            eagle_start_locs, eagle_seq_lens, eagle_num_tokens,
            completed_prefill_mask, partial_prefill_mask)
        print('eagle_positions ' + str(eagle_positions))

        eagle_input_ids = self._get_input_ids_for_proposer(
            target_model_input_ids, target_model_start_locs,
            target_model_seq_lens, eagle_start_locs,
            eagle_seq_lens, eagle_num_tokens,
            completed_prefill_mask, partial_prefill_mask,
            zero_position_mask, accepted_token_ids)
        print('eagle_input_ids ' + str(eagle_input_ids))

        return target_model_input_ids, target_model_input_ids
    
    def _updateAttentionMetadata(
        self,
        attention_metadata: FlashAttentionMetadata,
        num_eagle_tokens: int,
        target_model_seq_lens: Tensor,
        eagle_start_locs: Tensor,
        eagle_seq_lens: Tensor,
        partial_prefill_mask: Tensor,
        completed_prefill_mask: Tensor) -> FlashAttentionMetadata:
        eagle_seq_lens = attention_metadata.seq_lens - \
            target_model_seq_lens + eagle_seq_lens
        eagle_slot_mapping = torch.zeros(
            (num_eagle_tokens), dtype=attention_metadata.slot_mapping.dtype)
        prefill_mask = partial_prefill_mask | completed_prefill_mask
        start_indices =  eagle_start_locs[prefill_mask]
        end_indices = start_indices + eagle_seq_lens[prefill_mask]
        range_lengths = end_indices - start_indices
        start_pos = attention_metadata.slot_mapping[start_indices]
        eagle_slot_mapping = self._populate_positions(
            eagle_slot_mapping, start_indices, range_lengths, start_pos)
        start_indices =  eagle_start_locs[~prefill_mask]
        end_indices = start_indices + eagle_seq_lens[~prefill_mask]
        range_lengths = end_indices - start_indices
        start_pos = attention_metadata.slot_mapping[~start_indices] + 1
        eagle_slot_mapping = self._populate_positions(
            eagle_slot_mapping, start_indices, range_lengths, start_pos)
        eagle_attention_metadata = FlashAttentionMetadata(
            num_actual_tokens=num_eagle_tokens,
            max_query_len=int(eagle_seq_lens.max().item()),
            query_start_loc=torch.cat(
                [eagle_start_locs, torch.tensor([num_eagle_tokens])]),
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


    
    def _expand_indices(self, starts: Tensor, ends: Tensor):
        indices = torch.cat([
            torch.arange(start, end, dtype=torch.long) 
            for start, end in zip(starts, ends)
        ])
        print('indices ' + str(indices))
        return indices

    def _get_prev_hidden_states_for_proposer(
        self,
        req_ids: List,
        target_model_hidden_state: Tensor,
        target_model_start_locs: Tensor,
        target_model_seq_lens: Tensor,
        eagle_start_locs: Tensor,
        eagle_seq_lens: Tensor,
        eagle_num_tokens: int,
        completed_prefill_mask: Tensor,
        partial_prefill_mask: Tensor,
        non_prefill_mask:  Tensor,
        zero_position_mask: Tensor,
        accepted_token_lengths: Tensor
    ) ->  Tensor :
        eagle_prev_hidden_states = torch.zeros(
            (eagle_num_tokens, target_model_hidden_state.shape[1]),
            dtype=target_model_hidden_state.dtype)
        print('eagle_prev_hidden_states ' + str(eagle_prev_hidden_states))
        print('target_model_hidden_state ' + str(target_model_hidden_state))
        # First find the start and end locations for non prefills and copy the hidden states
        # Target model start and end indices
        print('target_model_start_locs ' + str(target_model_start_locs))
        print('non_prefill_mask ' + str(non_prefill_mask))
        target_model_starts = target_model_start_locs[non_prefill_mask]
        target_model_ends = \
            target_model_starts + accepted_token_lengths[non_prefill_mask]
        # Eagle start and end indices  
        eagle_starts = eagle_start_locs[non_prefill_mask]
        eagle_ends = \
            eagle_starts + accepted_token_lengths[non_prefill_mask]
        # Copy the non prefill hidden states.
        print('non_prefill_target_start_indices ' + str(target_model_starts))
        print('non_prefill_target_model_end_indices ' + str(target_model_ends))
        print('non_prefill_eagle_start_indices ' + str(eagle_starts))
        print('non_prefill_eagle_end_indices ' + str(eagle_ends))

        eagle_prev_hidden_states[self._expand_indices(
            eagle_starts, eagle_ends)] = \
                target_model_hidden_state[self._expand_indices(
                    target_model_starts, target_model_ends)]
        
        # Now find the start and end locations for prefills and copy the hidden states
        prefill_mask = partial_prefill_mask | completed_prefill_mask
        # Target model start and end indices
        target_model_starts = \
            target_model_start_locs[prefill_mask]
        target_model_ends = torch.where(
            completed_prefill_mask,
            target_model_start_locs + target_model_seq_lens,
            torch.where(
                partial_prefill_mask,
                target_model_start_locs + target_model_seq_lens - 1,
                target_model_start_locs  # Fallback for non-prefill sequences. Will be removed
            )
        )[prefill_mask]
        # Eagle start and end indices
        eagle_starts = torch.where(
            zero_position_mask,
            eagle_start_locs,
            eagle_start_locs + 1
        )[prefill_mask] 
        eagle_ends = eagle_start_locs[prefill_mask] + eagle_seq_lens[prefill_mask]
        # Copy the prefill hidden states.
        print('eagle_starts ' + str(eagle_starts))
        print('eagle_ends ' + str(eagle_ends))
        print('target_model_starts ' + str(target_model_starts))
        print('target_model_ends ' + str(target_model_ends))
        eagle_prev_hidden_states[self._expand_indices(
            eagle_starts, eagle_ends)] = \
                target_model_hidden_state[self._expand_indices(
                    target_model_starts, target_model_ends)]
        
        # For the prefills which have non zero first positions we need to copy
        # over the hidden state from the
        # partial_prefill_last_token_hidden_states map.
        indices = (prefill_mask & ~zero_position_mask).nonzero(as_tuple=True)[0].tolist()
        req_ids_from_prev_step = [req_ids[i] for i in indices]
        target_indices = eagle_start_locs[prefill_mask & ~zero_position_mask]
        for req_id, target_index in zip(req_ids_from_prev_step, target_indices) :
            eagle_prev_hidden_states[target_index] = \
                self.partial_prefill_last_token_hidden_states[req_id]

        # For the partial prefills we need to copy over the last token hidden
        # states into partial_prefill_last_token_hidden_states.
        indices = (partial_prefill_mask).nonzero(as_tuple=True)[0].tolist()
        req_ids_from_prev_step = [req_ids[i] for i in indices]
        target_indices = \
            target_model_start_locs[partial_prefill_mask] + \
            target_model_seq_lens[partial_prefill_mask]
        for req_id, target_index in zip(req_ids_from_prev_step, target_indices) :
            self.partial_prefill_last_token_hidden_states[req_id] = \
                target_model_hidden_state[target_index]
        
        # Remove the completed prefill req_ids from 
        # partial_prefill_last_token_hidden_states
        indices = (completed_prefill_mask).nonzero(as_tuple=True)[0].tolist()
        req_ids_for_completed_prefill = [req_ids[i] for i in indices]
        for req_id in req_ids_for_completed_prefill:
            self.partial_prefill_last_token_hidden_states.pop(req_id, None)

        return eagle_prev_hidden_states

    def _populate_positions(
        self,
        position_tensor: Tensor,
        start_indices: Tensor,
        lengths: Tensor,
        start_positions: Tensor
    )-> Tensor:
        position_values = torch.cat([torch.arange(
            start_position, start_position + length,
            dtype=torch.long, device=position_tensor.device) \
                for start_position, length in zip(start_positions, lengths)])
        start_indices_expanded = start_indices.repeat_interleave(lengths)
        index_increments = torch.cat([torch.arange(
            0, length, dtype=torch.long,
            device=position_tensor.device) for length in lengths])
        position_tensor[start_indices_expanded + index_increments] = position_values
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
        eagle_positions = torch.zeros(
            (eagle_num_tokens), dtype=target_model_positions.dtype)
        prefill_mask = partial_prefill_mask | completed_prefill_mask
        # First handle prefill starting at zero positions.
        start_indices =  eagle_start_locs[prefill_mask]
        end_indices = start_indices + eagle_seq_lens[prefill_mask]
        range_lengths = end_indices - start_indices
        start_pos = torch.maximum(
            target_model_positions[target_model_start_locs[prefill_mask]] - 1,
            torch.tensor(0, device=target_model_positions.device)
        )
        print('start_indices ' + str(start_indices))
        print('range_lengths ' + str(range_lengths))
        print('start_pos ' + str(start_pos))
        eagle_positions = self._populate_positions(
            eagle_positions, start_indices, range_lengths, start_pos)

        # Handle non prefills.
        start_indices =  eagle_start_locs[~prefill_mask]
        end_indices = start_indices + eagle_seq_lens[~prefill_mask]
        start_pos = target_model_positions[target_model_start_locs[~prefill_mask]]
        range_lengths = end_indices - start_indices
        eagle_positions = self._populate_positions(
            eagle_positions, start_indices, range_lengths, start_pos)
        
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
        eagle_input_ids = torch.zeros((eagle_num_tokens), dtype=target_model_input_ids.dtype)
        # First handle prefills.
        prefill_mask = partial_prefill_mask | completed_prefill_mask
        target_model_starts = torch.where(
            zero_position_mask, 
            target_model_start_locs + 1,
            target_model_start_locs
        )[prefill_mask]
        target_model_ends = target_model_start_locs[prefill_mask] + \
            target_model_seq_lens[prefill_mask]
        print('target_model_starts  ' + str(target_model_starts))
        print('target_model_ends  ' + str(target_model_ends))
        eagle_starts = eagle_start_locs[prefill_mask]
        eagle_ends = torch.where(
            completed_prefill_mask, 
            eagle_start_locs + eagle_seq_lens - 1,
            eagle_start_locs + eagle_seq_lens
        )[prefill_mask]
        print('eagle_starts  ' + str(eagle_starts))
        print('eagle_ends  ' + str(eagle_ends))

        # Now copy
        eagle_input_ids[self._expand_indices(
            eagle_starts, eagle_ends)] = \
                target_model_input_ids[self._expand_indices(
                    target_model_starts, target_model_ends)]
        
        # Copy the accepted tokens for the completed prefills.
        for start_idx, seq_len, accepted_token in zip(
                eagle_start_locs[completed_prefill_mask],
                eagle_seq_lens[completed_prefill_mask],
                accepted_token_ids[completed_prefill_mask]):
            print('start_idx + seq_len ' + str(start_idx + seq_len))
            eagle_input_ids[start_idx + seq_len - 1] = accepted_token[0]

        
        # Handling of non-prefill sequences.
        for start_idx, len, accepted_token in zip(
                eagle_start_locs[~prefill_mask],
                eagle_seq_lens[~prefill_mask],
                accepted_token_ids[~prefill_mask]):
            print('start_idx  ' + str(start_idx ))
            print('start_idx + seq_len ' + str(start_idx + len))
            eagle_input_ids[start_idx : start_idx + len] = accepted_token[:len]
        
        return eagle_input_ids





        

        
