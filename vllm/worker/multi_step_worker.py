from typing import List

import torch

from vllm.sequence import SamplerOutput, SequenceGroupMetadata, ExecuteModelData
from vllm.worker.worker import Worker
from vllm.worker.base_worker import LoraNotSupportedWorker
from vllm.anyscale.profiler_utils import nvtx_range
from vllm.model_executor.layers.sampler import RawSamplerOutput, pythonize_sampler_output
from vllm.model_executor.input_metadata import MultiStepInputMetadata, InputMetadata


class MultiStepWorker(Worker, LoraNotSupportedWorker):
    """The MultiStepWorker is equivalent to a Worker except that it allows
    multiple forward passes in a single call, assuming the scheduler has
    allocated enough space to store the additional KV. This reduces overhead
    by invoking the scheduler less and also by requiring less interprocess
    communication.

    Currently, the MultiStepWorker does not support cache swap operations, delta
    sequence group metadata updates, or beeam search. The first two can be
    added, however adding beam search is more complicated as it requires memory
    allocations during forks.
    """

    @staticmethod
    def _get_num_steps_from_num_preallocated_slots(
            num_preallocated_slots: int) -> int:
        """Determine the number of steps the MultiStepWorker should run given
        the number of slots preallocated by the scheduler.

        This is num_preallocated_slots plus one because the last generated token
        will not have its KV generated yet.
        """
        return num_preallocated_slots + 1

    @torch.inference_mode()
    @nvtx_range("multi_step_worker.execute_model")
    def execute_model(
            self,
            execute_model_data: ExecuteModelData,
            *,
            return_python_output: bool = True) -> List[SamplerOutput]:
        """Run the model forward pass num_steps times. Returns the list of
        sampler output, one per model forward pass.
        """
        (seq_group_metadata_list, _, _, _, _,
         return_logits) = (execute_model_data.seq_group_metadata_list,
                           execute_model_data.finished_request_ids_list,
                           execute_model_data.blocks_to_swap_in,
                           execute_model_data.blocks_to_swap_out,
                           execute_model_data.blocks_to_copy,
                           execute_model_data.return_logits)

        # Return if there are no input sequences.
        # We can do nothing here since input metadata deltas and
        # cache events are not supported.
        if not seq_group_metadata_list:
            return [SamplerOutput([])]

        num_steps = self._get_num_steps_from_num_preallocated_slots(
            execute_model_data.num_preallocated_slots)

        # Set num_preallocated_slots to zero; the single step worker does not
        # need to know about any other slots.
        old_num_preallocated_slots = execute_model_data.num_preallocated_slots
        execute_model_data.num_preallocated_slots = 0

        # Assert enough KV space for num_steps tokens per sequence.
        self._assert_enough_kv_space(
            execute_model_data.seq_group_metadata_list, num_steps)

        # Prepare input tensors.
        (
            input_tokens,
            input_positions,
            multi_step_input_metadata,
            _,
            _,
        ) = self._prepare_inputs(seq_group_metadata_list,
                                 return_logits=return_logits,
                                 num_steps=num_steps)

        # Run model num_steps times.
        model_outputs = []
        prev_parameters_tensors = (None, None)
        for _ in range(num_steps):
            # TODO(cade,antoni) This code breaks abstractions to improve
            # latency. We should refactor this so that `advance_step` can be
            # performed without blocking the GPU. Then this can become cuda
            # graphable, and simpler!
            with nvtx_range("multi_step_worker.run_single_step"):
                if model_outputs:
                    (input_tokens, input_positions,
                     input_metadata) = (self._advance_step(
                         model_outputs[-1],
                         input_metadata.selected_token_indices, input_tokens,
                         input_positions, multi_step_input_metadata))
                else:
                    input_metadata = multi_step_input_metadata.get_next_step()

                output = self.captured_model.execute_if_capturable(
                    input_ids=input_tokens,
                    positions=input_positions,
                    input_metadata=input_metadata,
                    cache_events=None,
                    sampling_parameters_tensors=prev_parameters_tensors[0],
                    sampling_token_tensors=prev_parameters_tensors[1],
                )
                prev_parameters_tensors = (output.sampling_parameters_tensors,
                                           output.sampling_token_tensors)

            model_outputs.append(output)

        execute_model_data.num_preallocated_slots = old_num_preallocated_slots

        if return_python_output:
            model_outputs = [
                pythonize_sampler_output(o, input_metadata)
                for o in model_outputs
            ]

        return model_outputs

    def _advance_step(
            self, last_output: RawSamplerOutput,
            last_selected_token_indices: torch.Tensor,
            input_tokens: torch.Tensor, input_positions: torch.Tensor,
            multi_step_input_metadata: MultiStepInputMetadata
    ) -> InputMetadata:
        sampled_tokens = last_output.sampled_tokens.flatten()
        # Sampled tokens from last step become new input tokens.
        input_tokens[:last_selected_token_indices.shape[0]] = sampled_tokens
        input_tokens[last_selected_token_indices.shape[0]:] = 0
        new_input_positions = input_positions[
            last_selected_token_indices].add_(1)
        input_positions[:last_selected_token_indices.
                        shape[0]] = new_input_positions
        input_positions[last_selected_token_indices.shape[0]:] = 0
        input_metadata = multi_step_input_metadata.get_next_step()
        return input_tokens, input_positions, input_metadata

    def _assert_enough_kv_space(
            self, seq_group_metadata_list: List[SequenceGroupMetadata],
            num_steps: int) -> None:
        """Assert there are enough physical blocks per sequence to store the
        current KV plus additional KV from num_steps tokens.
        """
        assert self.block_size is not None
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
            allocated_kv_slots = number_physical_blocks * self.block_size

            if required_num_kv_slots > allocated_kv_slots:
                request_id = seq_group_metadata.request_id
                raise ValueError(
                    "The worker attempted to run "
                    f"{num_steps} times but found insufficient KV space for "
                    f"{request_id=} {seq_id=}. ({allocated_kv_slots=} "
                    f"{required_num_kv_slots=}).")

    def _raise_if_unsupported(self,
                              execute_model_data: ExecuteModelData) -> None:
        """MultiStepWorker does not yet implement support for cache swap
        operations, incremental seq group metadata, or beam search.
        """
        (seq_group_metadata_list, _, blocks_to_swap_in, blocks_to_swap_out,
         blocks_to_copy) = (execute_model_data.seq_group_metadata_list,
                            execute_model_data.finished_request_ids_list,
                            execute_model_data.blocks_to_swap_in,
                            execute_model_data.blocks_to_swap_out,
                            execute_model_data.blocks_to_copy)

        if any([blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy]):
            raise NotImplementedError(
                "MultiStepWorker does not support cache operations")

        if any(not isinstance(seq_group_metadata, SequenceGroupMetadata)
               for seq_group_metadata in seq_group_metadata_list):
            raise NotImplementedError(
                "MultiStepWorker only supports SequenceGroupMetadata input "
                "(not deltas).")

        if any(
                len(seq_group_metadata.seq_data.keys()) != 1
                for seq_group_metadata in seq_group_metadata_list):
            raise NotImplementedError(
                "MultiStepWorker does not support beam search.")
