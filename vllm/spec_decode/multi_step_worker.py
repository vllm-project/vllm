# SPDX-License-Identifier: Apache-2.0

import copy
import weakref
from typing import Dict, List, Set, Tuple

import torch

from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.platforms import current_platform
from vllm.sequence import (ExecuteModelRequest, HiddenStates, SequenceData,
                           SequenceGroupMetadata)

if current_platform.is_cuda_alike():
    from vllm.spec_decode.draft_model_runner import TP1DraftModelRunner

from vllm.spec_decode.interfaces import (SpeculativeProposals,
                                         SpeculativeProposer)
from vllm.spec_decode.proposer_worker_base import ProposerWorkerBase
from vllm.spec_decode.top1_proposer import Top1Proposer
from vllm.worker.worker_base import DelegateWorkerBase


class MultiStepWorker(ProposerWorkerBase, DelegateWorkerBase):
    """The MultiStepWorker is equivalent to a Worker except that it allows
    multiple forward passes in a single call, assuming the scheduler has
    allocated enough space to store the additional KV. This reduces overhead
    by invoking the scheduler less.

    The MultiStepWorker does not support cache swap operations, or beam search.
    Cache swap operations do not require large modifications. On the other hand,
    beam search requires memory allocations during sequence forks and thus
    requires more thought for MultiStepWorker support.
    """

    def __init__(self, *args, **kwargs):
        DelegateWorkerBase.__init__(self, *args, **kwargs)
        # Lazy initialization list.
        self._proposer: SpeculativeProposer

    def init_device(self) -> None:
        self.worker.init_device()
        self._proposer = Top1Proposer(
            weakref.proxy(self),  # type: ignore[arg-type]
            self.device,
            self.vocab_size,
            max_proposal_len=self.max_model_len,
        )

    def set_include_gpu_probs_tensor(self) -> None:
        # Need include_gpu_probs_tensor for MultiStepWorker
        self.model_runner.sampler.include_gpu_probs_tensor = True
        if hasattr(self.model_runner.model, "sampler"):
            (self.model_runner.model.sampler.include_gpu_probs_tensor) = True

    def set_should_modify_greedy_probs_inplace(self) -> None:
        self.model_runner.sampler.should_modify_greedy_probs_inplace = True
        if hasattr(self.model_runner.model, "sampler"):
            (self.model_runner.model.sampler.should_modify_greedy_probs_inplace
             ) = True

    @torch.inference_mode()
    def sampler_output(
        self,
        execute_model_req: ExecuteModelRequest,
        sample_len: int,
        seq_ids_with_bonus_token_in_last_step: Set[int],
    ) -> Tuple[List[SamplerOutput], bool]:
        """Run the model forward pass sample_len times. Returns the list of
        sampler output, one per model forward pass, along with indicator of
        whether torch tensor in sampler output need to be transposed in latter
        sampler_output_to_torch logic.

        For multi step worker, this indicator shall be True.
        """
        self._raise_if_unsupported(execute_model_req)
        # Expand the batch for sequences with a bonus token.
        # Perform a forward pass on the expanded batch and filter the
        # response to retain only the original sequences' responses.
        expanded_request, indices_of_seq_with_bonus_tokens =\
            self._expand_execute_model_request(
                execute_model_req, seq_ids_with_bonus_token_in_last_step)

        # Run model sample_len times.
        model_outputs: List[SamplerOutput] = []
        if current_platform.is_cuda_alike() and isinstance(
                self.model_runner, TP1DraftModelRunner
        ) and self.model_runner.supports_gpu_multi_step(expanded_request):
            # Here we run the draft_model_runner with multi-step prepare
            # on the GPU directly
            expanded_request.num_steps = sample_len
            self.model_runner.set_indices_of_seq_with_bonus_tokens(
                indices_of_seq_with_bonus_tokens)
            model_outputs = self.execute_model(
                execute_model_req=expanded_request)
        else:
            # Here we run multi-step directly, with every step prepared
            # on the CPU.
            # TODO: Remove this branch once DraftModelRunner supports TP>1
            # and other restrictions that are part of DraftModelRunner's
            # supports_gpu_multi_step(..)
            if expanded_request.previous_hidden_states is not None:
                self.worker.model_runner.return_hidden_states = True
            for _ in range(sample_len):
                model_output: List[SamplerOutput] = self.worker.execute_model(
                    execute_model_req=expanded_request)
                assert (len(model_output) == 1
                        ), "composing multistep workers not supported"
                model_output = model_output[0]
                self._maybe_update_previous_hidden_states(
                    model_output, expanded_request)

                self._append_new_tokens(
                    model_output, expanded_request.seq_group_metadata_list,
                    indices_of_seq_with_bonus_tokens)
                model_outputs.append(model_output)

        # move indices to device to avoid stream sync
        indices_of_seq_with_bonus_tokens = torch.tensor(
            indices_of_seq_with_bonus_tokens, device=self.device)
        filtered_model_outputs = self._filter_model_output(
            model_outputs, indices_of_seq_with_bonus_tokens)
        return filtered_model_outputs, True

    @staticmethod
    def _maybe_update_previous_hidden_states(
            model_output: SamplerOutput,
            expanded_request: ExecuteModelRequest) -> None:
        """
        Updates the previous hidden states in an expanded request
        in-place with the hidden states from the model output. 
        """
        if expanded_request.previous_hidden_states is not None:
            expanded_request.previous_hidden_states = HiddenStates(
                model_output.hidden_states,
                expanded_request.seq_group_metadata_list)

    @staticmethod
    def _expand_execute_model_request(
        execute_model_req: ExecuteModelRequest,
        seq_with_bonus_token_in_last_step: set,
    ) -> Tuple[ExecuteModelRequest, List[int]]:
        """
        Expands the execute model request based on sequences with bonus
        tokens.

        For each sequence with a bonus token, this method creates a new
        sequence without the bonus token and adds it to the execute model
        request. The original sequence groups are also retained. The indices
        of the original sequence groups are returned for further processing.

        Args:
            execute_model_req (ExecuteModelRequest): The original execute
            model request.
            seq_with_bonus_token_in_last_step (set): Set of sequence IDs that 
            contain bonus tokens.

        Returns:
            Tuple[ExecuteModelRequest, List[int]]: The updated execute model
            request with expanded sequences and a list of indices corresponding
            to the original sequence groups.
        """
        updated_seq_group_metadata_list: List[SequenceGroupMetadata] = []
        updated_execute_model_req = execute_model_req.clone(
            updated_seq_group_metadata_list)
        indices_of_original_sequence_groups = []
        for seq_group in execute_model_req.seq_group_metadata_list:
            seq_group_has_bonus_tokens = False
            for seq_id, _ in seq_group.seq_data.items():
                # Identify sequences with bonus tokens in the sequence group.
                if seq_id in seq_with_bonus_token_in_last_step:
                    seq_group_has_bonus_tokens = True
                    break
            if seq_group_has_bonus_tokens:
                #Create new sequences without the last bonus token. These new
                # sequence have the same sequence id as the original sequence.
                # We create a new sequence group and add them there.
                updated_seq_group_without_bonus_token  = \
                    MultiStepWorker._copy_seq_metadata_excluding_last_token(
                        seq_group, seq_with_bonus_token_in_last_step)
                updated_seq_group_metadata_list.append(
                    updated_seq_group_without_bonus_token)
            # Add the original sequence group.
            updated_seq_group_metadata_list.append(
                MultiStepWorker._shallow_copy_seq_group_metadata(seq_group))
            # Record the index of the original sequence group.
            indices_of_original_sequence_groups.append(
                len(updated_seq_group_metadata_list) - 1)

        updated_execute_model_req.seq_group_metadata_list =\
            updated_seq_group_metadata_list

        if isinstance(updated_execute_model_req.previous_hidden_states,
                      HiddenStates):
            updated_execute_model_req.previous_hidden_states\
                .expand_with_bonus_tokens(seq_with_bonus_token_in_last_step)

        return updated_execute_model_req, indices_of_original_sequence_groups

    @staticmethod
    def _filter_model_output(
            expanded_batch_outputs: List[SamplerOutput],
            output_indices_to_retain: torch.Tensor) -> List[SamplerOutput]:
        """
        Filters the model output to include only the specified sequence
        outputs. This method contracts the expanded batch output from the
        model to retain the outputs of only those sequences indicated by the
        provided indices.

        Args:
            expanded_batch_output (List[SamplerOutput]): The expanded output
                batch from the model.
            output_indices_to_retain (torch.Tensor): Indices of the model
                outputs to retain.

        Returns:
            List[SamplerOutput]: A list containing the filtered model 
            outputs for the specified indices.
        """
        return [
            SamplerOutput(
                outputs=[
                    expanded_batch_output.outputs[i]
                    for i in output_indices_to_retain
                ] if len(expanded_batch_output.outputs) > 0 else [],
                sampled_token_probs=(
                    expanded_batch_output.
                    sampled_token_probs[output_indices_to_retain]
                    if expanded_batch_output.sampled_token_probs is not None
                    else None),
                logprobs=(
                    expanded_batch_output.logprobs[output_indices_to_retain]
                    if expanded_batch_output.logprobs is not None else None),
                sampled_token_ids=(expanded_batch_output.
                                   sampled_token_ids[output_indices_to_retain]
                                   if expanded_batch_output.sampled_token_ids
                                   is not None else None))
            for expanded_batch_output in expanded_batch_outputs
        ]

    def get_spec_proposals(
        self,
        execute_model_req: ExecuteModelRequest,
        seq_ids_with_bonus_token_in_last_step: set,
    ) -> SpeculativeProposals:
        """Produce speculations given an input batch of sequences. The number of
        speculative tokens per sequence is determined by max_proposal_len.
        """
        return self._proposer.get_spec_proposals(
            execute_model_req, seq_ids_with_bonus_token_in_last_step)

    @staticmethod
    def _append_new_tokens(
            model_output: List[SamplerOutput],
            seq_group_metadata_list: List[SequenceGroupMetadata],
            indices_of_seq_with_bonus_tokens: List[int]) -> None:
        """Given model output from a single run, append the tokens to the
        sequences. This is normally done outside of the worker, but it is
        required if the worker is to perform multiple forward passes.
        """
        count = 0
        for index, (seq_group_metadata, sequence_group_outputs) in enumerate(
                zip(seq_group_metadata_list, model_output)):
            seq_group_metadata.is_prompt = False

            for seq_output in sequence_group_outputs.samples:
                # NOTE: Beam search is not supported, so we can assume that
                # parent_seq_id == seq_id.
                seq = seq_group_metadata.seq_data[seq_output.parent_seq_id]

                token_id = seq_output.output_token
                token_logprob = seq_output.logprobs[token_id]
                # Determine the actual token ID to be generated,
                # considering bonus tokens
                if index != indices_of_seq_with_bonus_tokens[count]:
                    bonus_seq_metadata = seq_group_metadata_list[
                        indices_of_seq_with_bonus_tokens[count]]
                    _, bonus_token_seq_data = next(
                        iter(bonus_seq_metadata.seq_data.items()))
                    token_id = bonus_token_seq_data.output_token_ids[-1]
                else:
                    count += 1

                seq.append_token_id(token_id, token_logprob.logprob,
                                    seq_output.output_embed)
                seq.update_num_computed_tokens(1)

    @staticmethod
    def _shallow_copy_seq_group_metadata(
        seq_group_metadata: SequenceGroupMetadata, ) -> SequenceGroupMetadata:
        """Copy input data structures to remove side-effects when input data
        structures are shared with other modules.

        Helpful when the vLLM scheduler runs in the same process as the worker.
        The alternative is deep-copying (or other form of deep copy); this has
        performance downsides.
        """
        # Shallow-copy the SequenceGroupMetadata. This allows us to
        # append tokens and change is_prompt without external side-effects.
        # We must shallow-copy seq_group_metadata as is_prompt could change.
        new_seq_group_metadata = copy.copy(seq_group_metadata)

        # We must shallow-copy seq_data as we will append token ids
        new_seq_data: Dict[int, SequenceData] = {}
        for seq_id, old_seq_data in seq_group_metadata.seq_data.items():
            new_seq_data[seq_id] = copy.copy(old_seq_data)
            new_seq_data[seq_id].output_token_ids =\
                old_seq_data.output_token_ids[:]

        new_seq_group_metadata.seq_data = new_seq_data
        return new_seq_group_metadata

    @staticmethod
    def _copy_seq_metadata_excluding_last_token(
        seq_group_metadata: SequenceGroupMetadata,
        seq_ids_to_copy: Set[int],
    ) -> SequenceGroupMetadata:
        """
        Creates a shallow copy of the given SequenceGroupMetadata, retaining
        only the sequence IDs specified in seq_ids_to_copy. For each of these
        sequence IDs, all output_token_ids except the last one are copied.
        Sequence IDs not in seq_ids_to_copy are excluded from the copy.
        
        Parameters:
        seq_group_metadata (SequenceGroupMetadata): The original sequence
            group metadata.
        seq_ids_to_copy (Set[int]): The set of sequence IDs to include in the
            copy.
        
        Returns:
        SequenceGroupMetadata: A shallow copy of the sequence group metadata
            with the specified modifications.
        """
        # Shallow-copy the SequenceGroupMetadata.
        new_seq_group_metadata = copy.copy(seq_group_metadata)
        # Shallow-copy seq_data and modify the output_token_ids.
        new_seq_data: Dict[int, SequenceData] = {}
        for seq_id, old_seq_data in seq_group_metadata.seq_data.items():
            if (seq_id in seq_ids_to_copy):
                new_seq_data[seq_id] = copy.copy(old_seq_data)
                # Copy all the output token ids except the last.
                # Also reduce num_computed_tokens by 1 since we are not
                # including the last output token.
                # NOTE: num_computed_tokens is not directly used by the
                # speculative decoding workers, as it is only relevant for
                # chunked prefill, which is disabled for speculative decoding.
                # However, to maintain consistency in num_computed_tokens,
                # we update it here.
                new_seq_data[seq_id].output_token_ids =\
                    old_seq_data.output_token_ids[:-1]
                new_seq_data[seq_id].update_num_computed_tokens(-1)
        new_seq_group_metadata.seq_data = new_seq_data
        return new_seq_group_metadata

    def _assert_enough_kv_space(
            self, seq_group_metadata_list: List[SequenceGroupMetadata],
            num_steps: int) -> None:
        """Assert there are enough physical blocks per sequence to store the
        current KV plus additional KV from num_steps tokens.
        """
        assert self.model_runner.block_size is not None
        for seq_group_metadata in seq_group_metadata_list:
            # Only one seq_id is guaranteed because there is no beam search.
            seq_id = list(seq_group_metadata.seq_data.keys())[0]
            seq = seq_group_metadata.seq_data[seq_id]

            # After num_steps, the seq len will be the current seq len
            # plus one token per step.
            final_seq_len = seq.get_len() + num_steps

            # We will have final_seq_len - 1 KV because vLLM saves KV for a
            # token in the iteration after the token was generated.
            required_num_kv_slots = final_seq_len - 1

            # The allocated number of kv slots is the number of allocated blocks
            # times the number of slots of block.
            number_physical_blocks = len(
                seq_group_metadata.block_tables[seq_id])
            allocated_kv_slots = (number_physical_blocks *
                                  self.model_runner.block_size)

            if required_num_kv_slots > allocated_kv_slots:
                request_id = seq_group_metadata.request_id
                raise ValueError(
                    "The worker attempted to run "
                    f"{num_steps} times but found insufficient KV space for "
                    f"{request_id=} {seq_id=}. ({allocated_kv_slots=} "
                    f"{required_num_kv_slots=}).")

    def _raise_if_unsupported(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> None:
        """MultiStepWorker does not yet implement support for cache swap
        operations or beam search.
        """
        if any([
                execute_model_req.blocks_to_swap_in,
                execute_model_req.blocks_to_swap_out,
                execute_model_req.blocks_to_copy
        ]):
            raise NotImplementedError(
                "MultiStepWorker does not support cache operations")

        if any(
                len(seq_group_metadata.seq_data.keys()) != 1
                for seq_group_metadata in
                execute_model_req.seq_group_metadata_list):
            raise NotImplementedError(
                "MultiStepWorker does not support beam search.")

    def maybe_load_lm_head_weight(
        self,
        lm_head_weight: torch.Tensor,
    ) -> None:
        weight_loader = getattr(
            self.worker.model_runner.model_runner.model.lm_head.weight,
            "weight_loader", default_weight_loader)
        weight_loader(
            self.worker.model_runner.model_runner.model.lm_head.weight,
            lm_head_weight)
