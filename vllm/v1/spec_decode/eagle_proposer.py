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
            self, target_model_input_ids: Tensor,
            target_model_positions: Tensor, target_model_hidden_states: Tensor,
            target_model_seq_lens: list[int],
            sampled_token_ids: list[list[int]],
            next_prompt_token_ids: list[list[int]], is_prefill: list[bool],
            num_draft_tokens_to_propose: int,
            attention_metadata: FlashAttentionMetadata) -> list[SamplerOutput]:
        """
        Generate draft token IDs based on model input and attention metadata.
        """
        # Calculate start locations for sequences and their respective lengths
        target_model_start_locs = [0] + list(
            itertools.accumulate(target_model_seq_lens))[:-1]
        # Calculate the expected eagle sequence lengths. For prefills
        # the sequence lengths will be the same as the
        eagle_seq_lens = [
            target_model_seq_lens[i]
            if is_prefill[i] else len(sampled_token_ids[i])
            for i in range(len(target_model_seq_lens))
        ]
        eagle_num_tokens = sum(eagle_seq_lens)
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

        # Initialize tensors for eagle input data
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

        # Populate eagle tensors based on prefill or non-prefill sequences
        for req_idx in range(len(eagle_seq_lens)):
            eagle_start_loc = eagle_start_locs[req_idx]
            eagle_seq_len = eagle_seq_lens[req_idx]
            target_model_start_loc = target_model_start_locs[req_idx]
            target_model_start_position = target_model_positions[
                target_model_start_loc]

            # Populate Eagle positions using the positions passed to the
            # target model.
            eagle_positions[eagle_start_loc:eagle_start_loc +
                            eagle_seq_len] = torch.arange(
                                target_model_start_position,
                                target_model_start_position + eagle_seq_len)

            # Populate Eagle previous hidden states using the hidden states
            # returned by the target model.
            eagle_prev_hidden_states[
                eagle_start_loc:eagle_start_loc +
                eagle_seq_len] = target_model_hidden_states[
                    target_model_start_loc:target_model_start_loc +
                    eagle_seq_len]

            # Populate Eagle slot mappings
            target_model_start_slot_position = attention_metadata.slot_mapping[
                target_model_start_loc]
            eagle_slot_mappings[eagle_start_loc:eagle_start_loc +
                                eagle_seq_len] = torch.arange(
                                    target_model_start_slot_position,
                                    target_model_start_slot_position +
                                    eagle_seq_len)

            if is_prefill[req_idx]:
                # Handle prefill: Drop the first input token and
                # copy the last one from next prompt or sampled token. We use
                # the next prompt token for partial prefills or the sampled token
                # in case of completed prefills.
                eagle_input_ids[eagle_start_loc : \
                    eagle_start_loc + eagle_seq_len -1] = \
                        target_model_input_ids[
                            target_model_start_loc + 1 : \
                                target_model_start_loc + eagle_seq_len]
                eagle_input_ids[
                    eagle_start_loc +
                    eagle_seq_len] = next_prompt_token_ids[req_idx][0] if len(
                        next_prompt_token_ids[req_idx]
                    ) == 1 else sampled_token_ids_tensors[req_idx][0]
            else:
                # Handle non-prefill: Copy sampled and accepted token IDs
                eagle_input_ids[eagle_start_loc : eagle_start_loc + eagle_seq_len] = \
                    sampled_token_ids_tensors[req_idx][:eagle_seq_len]

        # Update Attention Metadata for Eagle
        eagle_attention_metadata = attention_metadata
        eagle_attention_metadata.num_actual_tokens = eagle_num_tokens
        # Calculate the sequence lengths for the Eagle metadata. To do
        # that we subtract the sequence lengths of the target model for
        # current forward pass and increment it by the sequence lengths
        # of the eagle model for the current forward pass.
        eagle_attention_metadata.seq_lens = (
            eagle_attention_metadata.seq_lens - target_model_seq_lens_tensor +
            eagle_seq_lens_tensor)
        eagle_attention_metadata.max_seq_len = \
            eagle_attention_metadata.seq_lens.max().item()
        # Replace the existing query start locations with those of the
        # Eagle model.
        eagle_attention_metadata.query_start_loc = torch.cat(
            [eagle_start_locs_tensor,
             torch.tensor([eagle_num_tokens])])
        eagle_attention_metadata.max_query_len = \
            eagle_seq_lens_tensor.max().item()
        eagle_attention_metadata.slot_mapping = eagle_slot_mappings

        # Create partial prefill mask for sequences with next prompt tokens
        partial_prefill_mask = torch.tensor(
            [len(tokens) > 0 for tokens in next_prompt_token_ids],
            dtype=torch.bool,
            device=target_model_positions.device)

        # Compute logits indices for sequences. We will only sample and propose
        # for the non partial prefills.
        logits_indices = (eagle_start_locs_tensor + eagle_seq_lens_tensor -
                          1)[~partial_prefill_mask]

        # Initialize result container
        result: list[SamplerOutput] = []

        # Perform forward pass through the model
        with set_forward_context(eagle_attention_metadata, self._vllm_config):
            eagle_hidden_states = self._model(
                input_ids=eagle_input_ids,
                positions=eagle_positions,
                previous_hidden_states=eagle_prev_hidden_states,
            )

            # Select hidden states for sampling
            sampled_hidden_states = eagle_hidden_states[logits_indices]

            # Compute logits and sample next tokens
            logits = self._model.compute_logits(sampled_hidden_states, None)
            sampler_output = self._model.sample(
                logits=logits,
                sampling_metadata=self._sampling_metadata,
            )

            result.extend(sampler_output)

        # Update eagle-related states with new sampled information
        eagle_positions = eagle_positions[logits_indices]
        eagle_input_ids = torch.stack([
            s.sampled_token_ids for s in sampler_output
        ])  # Ensure tensor consistency
        eagle_prev_hidden_states = eagle_hidden_states

        # Update attention metadata for the next iteration
        eagle_attention_metadata.num_actual_tokens = len(logits_indices)
        eagle_attention_metadata.seq_lens = \
            eagle_attention_metadata.seq_lens[~partial_prefill_mask] + 1
        eagle_attention_metadata.max_seq_len = \
            eagle_attention_metadata.seq_lens.max().item()
        eagle_attention_metadata.query_start_loc = torch.arange(
            0,
            len(logits_indices) + 1,
            device=eagle_attention_metadata.seq_lens.device)
        eagle_attention_metadata.max_query_len = 1
        eagle_attention_metadata.slot_mapping = \
            eagle_slot_mappings[logits_indices] + 1

        # Generate additional draft tokens if requested
        for _ in range(num_draft_tokens_to_propose):
            with set_forward_context(eagle_attention_metadata,
                                     self._vllm_config):
                eagle_hidden_states = self._model(
                    input_ids=eagle_input_ids,
                    positions=eagle_positions,
                    previous_hidden_states=eagle_prev_hidden_states,
                )
                logits = self._model.compute_logits(eagle_hidden_states, None)
                sampler_output = self._model.sample(
                    logits=logits,
                    sampling_metadata=self._sampling_metadata,
                )
                result.extend(sampler_output)
                eagle_input_ids = sampler_output[0].sampled_token_ids
                eagle_positions = eagle_positions + 1
                eagle_prev_hidden_states = eagle_hidden_states
                eagle_attention_metadata.slot_mapping = eagle_attention_metadata.slot_mapping + 1

        return result
