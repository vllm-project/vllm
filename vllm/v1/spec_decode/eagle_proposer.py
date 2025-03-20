from typing import Optional
import torch
from torch import Tensor
import numpy as np
from typing import List
from collections import defaultdict
from typing import Tuple
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata


class EagleProposer:

    def __init__(self, model):
        self.partial_prefill_last_token_hidden_states = defaultdict(
            lambda: torch.zeros(256, dtype=torch.float32)
        )
        self._model = model

    def propose(
        self,
        req_ids: List,
        input_ids_of_original_model: Tensor,
        position_of_original_model: Tensor,
        hidden_state: Tensor,
        cumulative_seq_lens: Tensor,
        accepted_token_ids: Tensor,
        accepted_token_id_positions: Tensor,
        is_prefill: Tensor,
        num_lookahead_slots_for_seq: Tensor,
        attention_metadata: FlashAttentionMetadata
    ) ->  Tuple[Tensor, Tensor] :
        
        sequence_lengths = cumulative_seq_lens[1:] - cumulative_seq_lens[:-1]

        accepted_token_lengths = (accepted_token_ids != -1).sum(dim=1)

        new_hidden_state_lengths = torch.where(
            is_prefill,
            sequence_lengths + accepted_token_lengths,
            accepted_token_lengths
        )
        total_tokens = new_hidden_state_lengths.sum()

        new_hidden_states = torch.zeros((total_tokens, hidden_state.shape[1]), dtype=hidden_state.dtype)

        proposer_previous_hidden_state = self._get_prev_hidden_states_for_proposer(

        )
        proposer_positions = self._get_positions_for_proposer(

        )
        proposer_input_ids = self._get_input_ids_for_proposer(

        )

        completed_prefill_mask = is_prefill.bool() & (accepted_token_lengths == 0)
        partial_prefill_mask = (accepted_token_lengths == 0) & is_prefill.bool()
        non_prefill_mask = ~is_prefill
        hidden_state_src_starts = cumulative_seq_lens[:-1]
        hidden_state_src_end = torch.where(
            non_prefill_mask, hidden_state_src_starts + accepted_token_lengths - 1,
            torch.where(
                completed_prefill_mask,
                hidden_state_src_starts + sequence_lengths,
                hidden_state_src_starts + sequence_lengths - 1)
        )
        hidden_state_tgt_starts = torch.where(
            is_prefill,
            cumulative_seq_lens[:-1] + 1,
            cumulative_seq_lens[:-1]
        )

        hidden_state_tgt_ends = hidden_state_tgt_starts + accepted_token_lengths
        src_indices = torch.cat([
            torch.arange(start, end + 1) 
            for start, end in zip(hidden_state_src_starts, hidden_state_src_end)
        ])

        tgt_indices = torch.cat([
            torch.arange(start, start + (end - start) + 1)
            for start, end in zip(hidden_state_tgt_starts, hidden_state_tgt_ends)
        ])
        new_hidden_states[tgt_indices] = hidden_state[src_indices]
        prefill_indices = cumulative_seq_lens[:-1][is_prefill.bool()]
        prefill_hidden_states = torch.tensor([
            self.partial_prefill_last_token_hidden_states[req_id].clone()
            for req_id in np.array(req_ids)[partial_prefill_mask.cpu().numpy()]
        ], dtype=new_hidden_states.dtype, device=new_hidden_states.device)
        new_hidden_states[prefill_indices] = prefill_hidden_states
        for idx, req_id in enumerate(np.array(req_ids)[partial_prefill_mask.cpu().numpy()]):
            self.partial_prefill_last_token_hidden_states[req_id] = \
                hidden_state[hidden_state_src_end[partial_prefill_mask][idx]]
        for req_id in np.array(req_ids)[(completed_prefill_mask | non_prefill_mask).cpu().numpy()]:
            if req_id in self.partial_prefill_last_token_hidden_states:
                del self.partial_prefill_last_token_hidden_states[req_id]
        self._model()
        return new_hidden_states  # Modify return value accordingly
    


    def _get_prev_hidden_states_for_proposer(
        self,
        req_ids: List,
        hidden_state: Tensor,
        original_seq_start_locs: Tensor,
        proposer_seq_start_locs: Tensor,
        proposer_seq_lens: Tensor,
        hidden_state_from_prev_prefill: Tensor,
        is_prefill: Tensor,
        accepted_token_lengths: Tensor
    ) ->  Tensor :
        completed_prefill_mask = is_prefill.bool() & (accepted_token_lengths > 0)
        non_prefill_mask = ~is_prefill

        hidden_state_src_starts = original_seq_start_locs
        hidden_state_src_ends = torch.where(
            non_prefill_mask, hidden_state_src_starts + accepted_token_lengths - 1,
            torch.where(
                completed_prefill_mask,
                hidden_state_src_starts + proposer_seq_lens,
                hidden_state_src_starts + proposer_seq_lens - 1)
        )
        hidden_state_tgt_starts = torch.where(
            hidden_state_from_prev_prefill,
            proposer_seq_start_locs + 1,
            proposer_seq_start_locs
        )
        hidden_state_tgt_ends = torch.where(
            non_prefill_mask, hidden_state_tgt_starts + accepted_token_lengths - 1,
            torch.where(
                completed_prefill_mask,
                hidden_state_tgt_starts + proposer_seq_lens,
                hidden_state_src_starts + proposer_seq_lens - 1)
        )
        total_tokens = proposer_seq_lens.sum()
        new_hidden_states = torch.zeros((total_tokens, hidden_state.shape[1]), dtype=hidden_state.dtype)
        src_indices = torch.cat([
            torch.arange(start, end + 1, dtype=torch.long) 
            for start, end in zip(hidden_state_src_starts, hidden_state_src_ends)
        ])
        tgt_indices = torch.cat([
            torch.arange(start, start + (end - start) + 1)
            for start, end in zip(hidden_state_tgt_starts, hidden_state_tgt_ends)
        ])
        print('hidden_state_src_starts ' + str(hidden_state_src_starts))
        print('hidden_state_src_ends ' + str(hidden_state_src_ends))
       
        print('hidden_state_tgt_starts ' + str(hidden_state_tgt_starts))
        print('hidden_state_tgt_ends ' + str(hidden_state_tgt_ends))

        print('tgt_indices ' + str(tgt_indices))
        print('src_indices ' + str(src_indices))
        new_hidden_states[tgt_indices] = hidden_state[src_indices]

        hidden_states_to_concat = []
        for req_id in req_ids:
            hidden_states_to_concat.append(self.partial_prefill_last_token_hidden_states[req_id])
        if hidden_states_to_concat:
            hidden_states_from_prev_step: Tensor = torch.cat(hidden_states_to_concat).view(-1)
            new_hidden_states[proposer_seq_start_locs[hidden_state_from_prev_prefill]] =  hidden_states_from_prev_step

        return new_hidden_states

    def _get_positions_for_proposer(
        self,
        req_ids: List,
        input_ids_of_original_model: Tensor,
        position_of_original_model: Tensor,
        original_seq_start_locs: Tensor,
        original_seq_lens: Tensor,
        proposer_seq_start_locs: Tensor,
        proposer_seq_lens: Tensor,
        is_prefill: Tensor,
        accepted_token_lengths: Tensor
    ):
        completed_prefill_mask = is_prefill.bool() & (accepted_token_lengths > 0)
        non_prefill_mask = ~is_prefill

        position_src_starts = original_seq_start_locs
        position_src_starts[non_prefill_mask] += 1
        position_src_ends = torch.where(
            non_prefill_mask,
            proposer_seq_start_locs + accepted_token_lengths - 1,
            proposer_seq_start_locs + original_seq_lens
        )
        position_tgt_starts = proposer_seq_start_locs
        position_tgt_ends = torch.where(
            non_prefill_mask, position_tgt_starts + accepted_token_lengths - 1,
            torch.where(
                completed_prefill_mask,
                position_tgt_starts + proposer_seq_lens - 1,
                position_tgt_starts + proposer_seq_lens)
        )
        total_tokens = proposer_seq_lens.sum()
        proposer_positions = torch.zeros((total_tokens,), dtype=hidden_state.dtype)
        src_indices = torch.cat([
            torch.arange(start, end + 1, dtype=torch.long) 
            for start, end in zip(position_src_starts, position_src_ends)
        ])
        tgt_indices = torch.cat([
            torch.arange(start, start + (end - start) + 1)
            for start, end in zip(position_tgt_starts, position_tgt_ends)
        ])
        proposer_positions[tgt_indices] = position_of_original_model[src_indices]
        last_index_of_completed_prefill_seq = \
            proposer_seq_start_locs[completed_prefill_mask] + \
                 proposer_seq_lens[completed_prefill_mask]
        last_index_of_non_prefill_seq  = \
            proposer_seq_start_locs[non_prefill_mask] + \
                 proposer_seq_lens[non_prefill_mask] + accepted_token_lengths[non_prefill_mask]
        proposer_positions[last_index_of_completed_prefill_seq] = \
            proposer_positions[last_index_of_completed_prefill_seq - 1] + 1
        proposer_positions[last_index_of_non_prefill_seq] = \
            proposer_positions[last_index_of_non_prefill_seq - 1] + 1

    def _get_positions_for_proposer(
        self,
        req_ids: List,
        input_ids_of_original_model: Tensor,
        position_of_original_model: Tensor,
        original_seq_start_locs: Tensor,
        original_seq_lens: Tensor,
        proposer_seq_start_locs: Tensor,
        proposer_seq_lens: Tensor,
        is_prefill: Tensor,
        accepted_token_lengths: Tensor
    ):
        # Identify masks
        completed_prefill_mask = is_prefill & (accepted_token_lengths > 0)
        non_prefill_mask = ~is_prefill

        # Compute start and end positions
        position_src_starts = original_seq_start_locs.clone()
        position_src_starts[non_prefill_mask] += 1

        # Compute source and target ranges more efficiently
        src_range_lengths = torch.where(
            non_prefill_mask,
            accepted_token_lengths,
            original_seq_lens
        )
        tgt_range_lengths = torch.where(
            non_prefill_mask,
            accepted_token_lengths,
            proposer_seq_lens
        )

        # Allocate memory
        total_tokens = proposer_seq_lens.sum()
        proposer_positions = torch.zeros(total_tokens, dtype=hidden_state.dtype)

        # Efficiently generate src/tgt indices
        src_indices = torch.repeat_interleave(position_src_starts, src_range_lengths)
        tgt_indices = torch.repeat_interleave(proposer_seq_start_locs, tgt_range_lengths)

        # Assign positions
        proposer_positions[tgt_indices] = position_of_original_model[src_indices]

        # Efficiently update final index values
        last_indices = torch.cat([
            proposer_seq_start_locs[completed_prefill_mask] + proposer_seq_lens[completed_prefill_mask],
            proposer_seq_start_locs[non_prefill_mask] + proposer_seq_lens[non_prefill_mask] +
            accepted_token_lengths[non_prefill_mask]
        ])
        proposer_positions[last_indices] = proposer_positions[last_indices - 1] + 1

        return proposer_positions

    def _get_input_ids_for_proposer(
        self,
        req_ids: List,
        input_ids_of_original_model: Tensor,
        position_of_original_model: Tensor,
        original_seq_start_locs: Tensor,
        original_seq_lens: Tensor, 
        proposer_seq_start_locs: Tensor,
        proposer_seq_lens: Tensor,
        is_prefill: Tensor,
        accepted_token_lengths: Tensor,
        accepted_token_ids: Tensor,
    ):
        completed_prefill_mask = is_prefill.bool() & (accepted_token_lengths > 0)
        non_prefill_mask = ~is_prefill
        input_ids_prefill_src_starts = original_seq_start_locs[is_prefill]
        zero_position_mask = position_of_original_model[input_ids_prefill_src_starts] == 0
        input_ids_prefill_src_starts[zero_position_mask] += 1
        input_ids_prefill_src_ends = \
            original_seq_start_locs[is_prefill] + original_seq_lens[is_prefill]
        input_ids_prefill_tgt_starts = proposer_seq_start_locs[is_prefill]
        input_ids_prefill_tgt_ends = torch.where(
            zero_position_mask, 
            input_ids_prefill_tgt_starts + original_seq_lens[is_prefill] - 1,
            input_ids_prefill_tgt_starts + original_seq_lens[is_prefill]
        )
        non_prefill_tgt_starts = proposer_seq_start_locs[non_prefill_mask]
        non_prefill_tgt_ends = non_prefill_tgt_starts + accepted_token_lengths[non_prefill_mask] - 1

        total_tokens = proposer_seq_lens.sum()
        input_ids_proposer = torch.zeros((total_tokens,), dtype=input_ids_of_original_model.dtype)
        for idx, (start, end, tgt_start) in enumerate(zip(input_ids_prefill_src_starts, input_ids_prefill_src_ends, input_ids_prefill_tgt_starts)):
            input_ids_proposer[tgt_start: tgt_start + (end - start)] = input_ids_of_original_model[start:end]

        non_prefill_tgt_starts = proposer_seq_start_locs[non_prefill_mask]
        for idx, (start, length) in enumerate(zip(non_prefill_tgt_starts, accepted_token_lengths[non_prefill_mask])):
            input_ids_proposer[start : start + length] = accepted_token_ids[idx, :length]
        for idx, (tgt_loc, src_loc) in enumerate(zip(completed_prefill_tgt_locs, completed_prefill_src_locs)):
            input_ids_proposer[tgt_loc] = accepted_token_ids[idx, 0]

        return input_ids_proposer






        

        
