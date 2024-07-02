import copy
import weakref
from typing import Dict, List, Tuple

import torch

from vllm.sequence import (ExecuteModelRequest, SamplerOutput, SequenceData,
                           SequenceGroupMetadata)
from vllm.spec_decode.draft_model_runner import TP1DraftModelRunner
from vllm.spec_decode.interfaces import (SpeculativeProposals,
                                         SpeculativeProposer)
from vllm.spec_decode.proposer_worker_base import ProposerWorkerBase
from vllm.spec_decode.top1_proposer import Top1Proposer
from vllm.worker.worker import Worker


class MultiStepWorker(Worker, ProposerWorkerBase):
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
        super().__init__(*args, **kwargs)

        # Lazy initialization list.
        self._proposer: SpeculativeProposer

    def init_device(self) -> None:
        super().init_device()

        self._proposer = Top1Proposer(
            weakref.proxy(self),  # type: ignore[arg-type]
            self.device,
            self.vocab_size,
            max_proposal_len=self.max_model_len,
        )

    def set_include_gpu_probs_tensor(self) -> None:
        # Need include_gpu_probs_tensor for multi_step_worker
        self.model_runner.model.sampler.include_gpu_probs_tensor = True

    @torch.inference_mode()
    def sampler_output(
        self,
        execute_model_req: ExecuteModelRequest,
        sample_len: int,
        seq_ids_with_bonus_token_in_last_step: set,
    ) -> Tuple[List[SamplerOutput], bool]:
        """Run the model forward pass sample_len times. Returns the list of
        sampler output, one per model forward pass, along with indicator of
        whether torch tensor in sampler output need to be transposed in latter
        sampler_output_to_torch logic.

        For multi step worker, this indicator shall be True.
        """
        self._raise_if_unsupported(execute_model_req)

        # Shallow copy input data so modifications (such as appending tokens)
        # do not cause side-effects.
        copied_seq_group_metadata_list = self._shallow_copy_inputs(
            execute_model_req.seq_group_metadata_list)
        copied_execute_model_req = execute_model_req.clone(
            copied_seq_group_metadata_list)

        # Assert enough KV space for sample_len tokens per sequence.
        self._assert_enough_kv_space(execute_model_req.seq_group_metadata_list,
                                     sample_len)
        # Run model sample_len times.
        model_outputs = []
        # Step 0: Expand the batch for sequences with a bonus token.
        # Perform a forward pass on the expanded batch and filter the 
        # response to retain only the original sequences' responses.
        expanded_request, indices_of_seq_with_bonus_tokens =\
            self._expand_execute_model_request(
                copied_execute_model_req, seq_ids_with_bonus_token_in_last_step)
        # Run model sample_len times.
        model_outputs: List[SamplerOutput] = []
        if isinstance(self.model_runner, TP1DraftModelRunner):
            copied_execute_model_req.num_steps = sample_len
            model_outputs = self.execute_model(
                execute_model_req=expanded_request)
        else:
            # TODO: Remove this branch once DraftModelRunner supports TP>1.
            for _ in range(sample_len):
                model_output: List[SamplerOutput] = super().execute_model(
                    execute_model_req=expanded_request)
                assert (len(model_output) == 1
                        ), "composing multistep workers not supported"
                model_output = model_output[0]

                self._append_new_tokens(model_output,
                                        copied_seq_group_metadata_list)
                model_outputs.append(model_output)
        return model_outputs, True

    @staticmethod
    def _expand_execute_model_request(
        execute_model_req: ExecuteModelRequest,
        seq_with_bonus_token_in_last_step: set,
    ) -> Tuple[ExecuteModelRequest, List[int]]:
        """
        Expands the execute model request based on sequences with bonus tokens.

        For each sequence with a bonus token, this method creates a new sequence
        without the bonus token and adds it to the execute model request. The original
        sequence groups are also retained. The indices of the original sequence groups
        are returned for further processing.

        Args:
            execute_model_req (ExecuteModelRequest): The original execute model request.
            seq_with_bonus_token_in_last_step (set): Set of sequence IDs that contain bonus tokens.

        Returns:
            Tuple[ExecuteModelRequest, List[int]]: The updated execute model request with expanded 
            sequences and a list of indices corresponding to the original sequence groups.
        """
        updated_seq_group_metadata_list = []
        updated_execute_model_req = execute_model_req.clone(updated_seq_group_metadata_list)
        indices_of_original_sequence_groups = []
        for seq_group in execute_model_req.seq_group_metadata_list:
            seq_ids_with_bonus_tokens = []
            for seq_id, seq_data in seq_group.seq_data.items():
                # Identify sequences with bonus tokens in the sequence group.
                if seq_id in seq_with_bonus_token_in_last_step:
                    seq_ids_with_bonus_tokens.append(seq_id)
            if seq_ids_with_bonus_tokens:
                #Create new sequences without the last bonus token. These new
                # sequence have the same sequence id as the original sequence. 
                # We create a new sequence group and add them there.
                updated_seq_group_without_bonus_token = \
                    MultiStepWorker._shallow_copy_sequence_group_metadata(
                        seq_group)
                seq_group_without_bonus_token_data = {
                    seq_id: SequenceData(
                        prompt_token_ids=seq_group.seq_data[seq_id].prompt_token_ids,
                        output_token_ids=seq_group.seq_data[seq_id].output_token_ids[:-1]
                    )
                    for seq_id in seq_ids_with_bonus_tokens
                }
                # Update the number of computed tokens for the new sequences.
                for seq_data in seq_group_without_bonus_token_data.values():
                    seq_data.update_num_computed_tokens(len(seq_data.output_token_ids) - 1)
                # Add the new sequence groups (without bonus tokens) first. We add these
                # first because these are the ones without the bonus tokens and hence
                # should be processed before the ones with the bonus tokens.
                updated_seq_group_without_bonus_token.seq_data = seq_group_without_bonus_token_data
                updated_seq_group_metadata_list.append(updated_seq_group_without_bonus_token)
            # Add the original sequence group.
            updated_seq_group_metadata_list.append(seq_group)
            # Record the index of the original sequence group.
            indices_of_original_sequence_groups.append(len(updated_seq_group_metadata_list) - 1)

        updated_execute_model_req.seq_group_metadata_list = updated_seq_group_metadata_list
        return updated_execute_model_req, indices_of_original_sequence_groups

    @staticmethod
    def _filter_model_output(
        expanded_batch_output: SamplerOutput,
        output_indices_to_retain: List[int]
    ) -> List[SamplerOutput]:
        """
        Filters the model output to include only the specified sequence outputs.

        This method contracts the expanded batch output from the model to retain 
        the outputs of only those sequences indicated by the provided indices.

        Args:
            expanded_batch_output (SamplerOutput): The expanded output batch 
                from the model.
            output_indices_to_retain (List[int]): Indices of the model outputs to
                retain.

        Returns:
            List[SamplerOutput]: A list containing the filtered model 
            outputs for the specified indices.
        """
        return [
            SamplerOutput(
                outputs=[expanded_batch_output.outputs[i] for i in output_indices_to_retain],
                sampled_token_probs=(
                    expanded_batch_output.sampled_token_probs[output_indices_to_retain]
                    if expanded_batch_output.sampled_token_probs is not None else None
                ),
                logprobs=(
                    expanded_batch_output.logprobs[output_indices_to_retain]
                    if expanded_batch_output.logprobs is not None else None
                ),
                sampled_token_ids=(
                    expanded_batch_output.sampled_token_ids[output_indices_to_retain]
                    if expanded_batch_output.sampled_token_ids is not None else None
                )
            )      
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
            seq_group_metadata_list: List[SequenceGroupMetadata]) -> None:
        """Given model output from a single run, append the tokens to the
        sequences. This is normally done outside of the worker, but it is
        required if the worker is to perform multiple forward passes.
        """
        for seq_group_metadata, sequence_group_outputs in zip(
                seq_group_metadata_list, model_output):
            seq_group_metadata.is_prompt = False

            for seq_output in sequence_group_outputs.samples:
                # NOTE: Beam search is not supported, so we can assume that
                # parent_seq_id == seq_id.
                seq = seq_group_metadata.seq_data[seq_output.parent_seq_id]

                token_id = seq_output.output_token
                token_logprob = seq_output.logprobs[token_id]

                seq.append_token_id(token_id, token_logprob.logprob)
                seq.update_num_computed_tokens(1)

    @staticmethod
    def _shallow_copy_inputs(
        seq_group_metadata_list: List[SequenceGroupMetadata]
    ) -> List[SequenceGroupMetadata]:
        """Copy input data structures to remove side-effects when input data
        structures are shared with other modules.

        Helpful when the vLLM scheduler runs in the same process as the worker.
        The alternative is deep-copying (or other form of deep copy); this has
        performance downsides.
        """

        # Shallow-copy the list of SequenceGroupMetadata. This allows us to
        # append tokens and change is_prompt without external side-effects.
        new_seq_group_metadata_list: List[SequenceGroupMetadata] = []

        for old_seq_group_metadata in seq_group_metadata_list:
            # We must shallow-copy seq_group_metadata as is_prompt could change.
            seq_group_metadata = copy.copy(old_seq_group_metadata)
            new_seq_group_metadata_list.append(seq_group_metadata)

            # We must shallow-copy seq_data as we will append token ids
            new_seq_data: Dict[int, SequenceData] = {}
            for seq_id, old_seq_data in seq_group_metadata.seq_data.items():
                new_seq_data[seq_id] = copy.copy(old_seq_data)
                new_seq_data[
                    seq_id].output_token_ids = old_seq_data.output_token_ids[:]

            seq_group_metadata.seq_data = new_seq_data

        return new_seq_group_metadata_list

    @staticmethod
    def _shallow_copy_sequence_group_metadata(
        seq_group_metadata: SequenceGroupMetadata) -> SequenceGroupMetadata:
        return SequenceGroupMetadata(
            seq_group_metadata.request_id,
            seq_group_metadata.is_prompt,
            seq_group_metadata.seq_data,
            seq_group_metadata.sampling_params,
            seq_group_metadata.block_tables,
            seq_group_metadata.do_sample,
            seq_group_metadata.pooling_params,
            seq_group_metadata.token_chunk_size,
            seq_group_metadata.lora_request,
            seq_group_metadata.computed_block_nums,
            seq_group_metadata.state,
            seq_group_metadata.multi_modal_data,
            seq_group_metadata.encoder_seq_data,
            seq_group_metadata.cross_block_table,            
        )


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
