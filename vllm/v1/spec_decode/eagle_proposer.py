# SPDX-License-Identifier: Apache-2.0
import itertools

import torch
from torch import Tensor, nn

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata


class EagleProposer:

    def __init__(self, vllm_config: VllmConfig, model: nn.Module,
                 sampling_metadata: SamplingMetadata):
        """
        Initialize EagleProposer with necessary configuration and model.
        """
        self._vllm_config = vllm_config
        self._model = model
        self._sampling_metadata = sampling_metadata

    def generate_draft_token_ids(
            self, *, target_model_input_ids: Tensor,
            target_model_positions: Tensor, target_model_hidden_states: Tensor,
            target_model_seq_lens: list[int],
            sampled_token_ids: list[list[int]],
            next_prompt_token_ids: list[list[int]], is_prefill: list[bool],
            num_draft_tokens_to_propose: int,
            attention_metadata: FlashAttentionMetadata) -> list[SamplerOutput]:
        """
        Generates speculative draft token IDs using the Eagle model.

        This function aligns the Eagle model's KV cache with the target
        model’s output before generating speculative tokens. It first
        performs a prefill pass, followed by iterative speculation passes
        to generate additional draft tokens.

        Example:
        Consider three sequences:
        - S1: A completed prefill sequence starting at position 0.
        - S2: A partial prefill sequence starting at a nonzero position.
        - S3: A decoding sequence.

        Target Model:
        Sequences: [S1, S2, S3]
        Tokens: [T11, T12, T13, T14, T21, T22, T23, T31, T32, T33]
        Positions: [0, 1, 2, 3, 9, 10, 11, 44, 45, 46]
        Hidden States: [H11, H12, H13, H14, H21, H22, H23, H31, H32, H33]
        Sampled Tokens: [[T15], [], [T32]]
        Next Prompt Tokens: [[], [T24], []]

        The first forward pass in Eagle aligns its KV cache with that
        of the target model and generated the first proposal. The
        input to the eagle model for the first forward pass will be
        the following.

        Eagle Prefill Forward Pass:
        Sequences: [S1, S2, S3]
        Tokens: [T12, T13, T14, T15, T22, T23, T24, T32]
        Positions: [0, 1, 2, 3, 9, 10, 11, 44]
        Previous Hidden States: [H11, H12, H13, H14, H21, H22, H23, H31]
        Sampled Tokens: [[T16], [T25], [T33']]

        Note that for S1, we drop T11 (position 0). For S2 and S3, 
        T21 and T31 are skipped since they were processed earlier. 
        Eagle positions are always one less than the target model 
        due to dropping the first token.        

        Subsequent Eagle speculative passes generate additional tokens
        per sequence as needed.

        Args:
            target_model_input_ids: Input token IDs used by the target model.
            target_model_positions: The token positions used by the target
                model.
            target_model_hidden_states: Hidden states from the target model.
            target_model_seq_lens: Sequence lengths in the target model.
            sampled_token_ids: Previously sampled/accepted tokens from the
                target model.
            next_prompt_token_ids: The next prompt token for a sequence if it
                is a partial prefill sequence and empty otherwise.
            is_prefill: Boolean flags indicating prefill sequences.
            num_draft_tokens_to_propose: Number of speculative tokens to
                generate per sequence.
            attention_metadata: Attention metadata from the target model's
                forward pass.

        Returns:
            list[SamplerOutput]: A list of sampled token outputs from Eagle,
            with length `num_draft_tokens_to_propose`. Each SamplerOutput 
            includes sampled tokens for all sequences, including 
            partial prefill ones.
        """
        # Compute start positions for each sequence in the target model.
        target_model_start_locs = [0] + list(
            itertools.accumulate(target_model_seq_lens))[:-1]

        # Determine expected sequence lengths in the Eagle model:
        # - For prefill sequences, lengths remain unchanged.
        # - For decoding sequences, lengths match the number of
        #   accepted tokens.
        eagle_seq_lens = [
            target_model_seq_lens[i]
            if is_prefill[i] else len(sampled_token_ids[i])
            for i in range(len(target_model_seq_lens))
        ]
        eagle_num_tokens = sum(eagle_seq_lens)
        num_seqs = len(eagle_seq_lens)
        eagle_start_locs = [0] + list(
            itertools.accumulate(eagle_seq_lens))[:-1]

        # Convert lists to tensors for computation
        sampled_token_ids_tensors = [
            torch.tensor(tokens,
                         dtype=torch.int,
                         device=target_model_positions.device)
            for tokens in sampled_token_ids
        ]
        target_model_seq_lens_tensor = torch.tensor(
            target_model_seq_lens,
            dtype=torch.int,
            device=target_model_positions.device)
        eagle_start_locs_tensor = torch.tensor(
            eagle_start_locs,
            dtype=torch.int,
            device=target_model_positions.device)
        eagle_seq_lens_tensor = torch.tensor(
            eagle_seq_lens,
            dtype=torch.int,
            device=target_model_positions.device)

        # Initialize Eagle model input tensors
        eagle_input_ids = torch.zeros(eagle_num_tokens,
                                      dtype=target_model_positions.dtype,
                                      device=target_model_input_ids.device)
        eagle_positions = torch.zeros(eagle_num_tokens,
                                      dtype=target_model_positions.dtype,
                                      device=target_model_positions.device)
        eagle_prev_hidden_states = torch.zeros(
            (eagle_num_tokens, target_model_hidden_states.shape[1]),
            dtype=target_model_hidden_states.dtype,
            device=target_model_hidden_states.device)
        eagle_slot_mappings = torch.zeros(eagle_num_tokens,
                                          dtype=target_model_positions.dtype,
                                          device=target_model_positions.device)

        # Populate Eagle model inputs for the first forward pass.
        for req_idx in range(num_seqs):
            eagle_start_loc = eagle_start_locs[req_idx]
            eagle_seq_len = eagle_seq_lens[req_idx]
            target_model_start_loc = target_model_start_locs[req_idx]
            target_model_start_position = target_model_positions[
                target_model_start_loc]

            # Assign positions: Start positions match the target model.
            eagle_positions[eagle_start_loc:eagle_start_loc + eagle_seq_len] = \
                torch.arange(
                    target_model_start_position,
                    target_model_start_position + eagle_seq_len)

            # Assign previous hidden states from the target model.
            eagle_prev_hidden_states[
                eagle_start_loc:eagle_start_loc + eagle_seq_len] = \
                target_model_hidden_states[
                    target_model_start_loc:\
                        target_model_start_loc + eagle_seq_len]

            # Assign slot mappings for attention.
            target_model_start_slot_position = attention_metadata.slot_mapping[
                target_model_start_loc]
            eagle_slot_mappings[
                eagle_start_loc:eagle_start_loc + eagle_seq_len] = \
                    torch.arange(
                        target_model_start_slot_position,
                        target_model_start_slot_position + eagle_seq_len)

            # Populate input IDs based on prefill or decoding.
            if is_prefill[req_idx]:
                # Drop the first token and use either next prompt
                # or sampled token.
                eagle_input_ids[
                    eagle_start_loc:eagle_start_loc + eagle_seq_len - 1] = \
                        target_model_input_ids[target_model_start_loc + 1: \
                            target_model_start_loc + eagle_seq_len]
                eagle_input_ids[eagle_start_loc + eagle_seq_len - 1] = \
                    next_prompt_token_ids[req_idx][0] \
                        if next_prompt_token_ids[req_idx] \
                            else sampled_token_ids_tensors[req_idx][0]
            else:
                # Use sampled tokens for decoding sequences.
                eagle_input_ids[eagle_start_loc:\
                    eagle_start_loc + eagle_seq_len] = \
                        sampled_token_ids_tensors[req_idx][:eagle_seq_len]

        # Construct the attention metadata to use in the Eagle model
        # Adjust attention metadata for Eagle model
        eagle_attention_metadata = attention_metadata
        eagle_attention_metadata.num_actual_tokens = eagle_num_tokens
        eagle_attention_metadata.seq_lens = \
            eagle_attention_metadata.seq_lens - \
                target_model_seq_lens_tensor + eagle_seq_lens_tensor
        eagle_attention_metadata.max_seq_len = \
            eagle_attention_metadata.seq_lens.max().item()
        eagle_attention_metadata.query_start_loc = torch.cat(
            [eagle_start_locs_tensor,
             torch.tensor([eagle_num_tokens])])
        eagle_attention_metadata.max_query_len = \
            eagle_seq_lens_tensor.max().item()
        eagle_attention_metadata.slot_mapping = eagle_slot_mappings

        # Compute the logit indices to use for sampling. These are the
        # last indices for each sequence.
        logits_indices = eagle_start_locs_tensor + eagle_seq_lens_tensor - 1

        result: list[SamplerOutput] = []

        with set_forward_context(eagle_attention_metadata, self._vllm_config):
            eagle_hidden_states = self._model(
                input_ids=eagle_input_ids,
                positions=eagle_positions,
                previous_hidden_states=eagle_prev_hidden_states,
            )
            sampled_hidden_states = eagle_hidden_states[logits_indices]
            logits = self._model.compute_logits(sampled_hidden_states, None)
            sampler_output = self._model.sample(
                logits=logits, sampling_metadata=self._sampling_metadata)
            result.extend(sampler_output)

        if num_draft_tokens_to_propose == 1:
            return result

        # Prepare for more speculative decode passes.
        eagle_positions = eagle_positions[logits_indices] + 1
        eagle_input_ids = sampler_output[0].sampled_token_ids.squeeze(-1)
        eagle_prev_hidden_states = sampled_hidden_states

        eagle_attention_metadata.num_actual_tokens = num_seqs
        eagle_attention_metadata.seq_lens = \
            eagle_attention_metadata.seq_lens[logits_indices] + 1
        eagle_attention_metadata.max_seq_len =\
            eagle_attention_metadata.seq_lens.max().item()
        eagle_attention_metadata.query_start_loc = torch.arange(
            0, num_seqs + 1, device=eagle_positions.device)
        eagle_attention_metadata.max_query_len = 1
        eagle_attention_metadata.slot_mapping =\
            eagle_attention_metadata.slot_mapping[logits_indices] + 1

        for _ in range(num_draft_tokens_to_propose - 1):
            with set_forward_context(eagle_attention_metadata,
                                     self._vllm_config):
                eagle_hidden_states = self._model(
                    input_ids=eagle_input_ids,
                    positions=eagle_positions,
                    previous_hidden_states=eagle_prev_hidden_states,
                )
                logits = self._model.compute_logits(eagle_hidden_states, None)
                sampler_output = self._model.sample(
                    logits=logits, sampling_metadata=self._sampling_metadata)
                result.extend(sampler_output)

                eagle_input_ids = sampler_output[0].sampled_token_ids.squeeze(
                    -1)
                eagle_positions += 1
                eagle_prev_hidden_states = eagle_hidden_states
                eagle_attention_metadata.slot_mapping += 1
                eagle_attention_metadata.seq_lens += 1
                eagle_attention_metadata.max_seq_len =\
                    eagle_attention_metadata.seq_lens.max().item()

        return result
