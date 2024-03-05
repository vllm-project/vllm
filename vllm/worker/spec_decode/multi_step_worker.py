from typing import List, Dict
import copy

import torch

from vllm.sequence import SamplerOutput, SequenceGroupMetadata
from vllm.worker.worker import Worker
from vllm.worker.spec_decode.util import SpeculativeProposals, sampler_output_to_torch


class MultiStepWorker(Worker):
    """The MultiStepWorker is equivalent to a Worker except that it allows
    multiple forward passes in a single call, assuming the scheduler has
    allocated enough space to store the additional KV. This reduces overhead
    by invoking the scheduler less.

    The MultiStepWorker does not support cache swap operations, or beam search.
    Cache swap operations do not require large modifications. On the other hand,
    beam search requires memory allocations during sequence forks and thus
    requires more thought for MultiStepWorker support.
    """

    @torch.inference_mode()
    def execute_model_multi_step(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
        num_steps: int,
    ) -> List[SamplerOutput]:
        """Run the model forward pass num_steps times. Returns the list of
        sampler output, one per model forward pass.
        """
        self._raise_if_unsupported(seq_group_metadata_list, blocks_to_swap_in,
                                   blocks_to_swap_out, blocks_to_copy)

        # Shallow copy input data so modifications (such as appending tokens)
        # do not cause side-effects.
        copied_seq_group_metadata_list = self._shallow_copy_inputs(
            seq_group_metadata_list)

        # Assert enough KV space for num_steps tokens per sequence.
        self._assert_enough_kv_space(seq_group_metadata_list, num_steps)

        # Run model num_steps times.
        model_outputs = []
        for _ in range(num_steps):
            model_output = super().execute_model(
                seq_group_metadata_list=copied_seq_group_metadata_list,
                blocks_to_swap_in=blocks_to_swap_in,
                blocks_to_swap_out=blocks_to_swap_out,
                blocks_to_copy=blocks_to_copy,
            )

            self._append_new_tokens(model_output,
                                    copied_seq_group_metadata_list)
            model_outputs.append(model_output)

        return model_outputs

    def get_spec_proposals(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
        max_proposal_len: int,
    ) -> SpeculativeProposals:
        """
        - create proposal lens tensor
            - determine which seqs are over len
            - set k = 0
        - create new batch that ignores k=0
        - do normal fwd pass
        - construct output
            - inject empty rows for token ids/probs tensor
        """
        
        # TODO
        max_model_len = 2048

        proposal_lens: List[int] = []
        nonzero_proposal_len_seqs: List[SequenceGroupMetadata] = []
        #nonzero_proposal_len_indices: List[int] = []
        #zero_proposal_len_indices: List[int] = []
        for i, seq_group_metadata in enumerate(seq_group_metadata_list):
            seq_data = next(iter(seq_group_metadata.seq_data.values()))
            seq_len = seq_data.get_len()

            if seq_len + max_proposal_len < max_model_len:
                proposal_lens.append(max_proposal_len)
                nonzero_proposal_len_seqs.append(seq_group_metadata)
                #nonzero_proposal_len_indices.append(i)
            else:
                proposal_lens.append(0)
                #zero_proposal_len_indices.append(i)
        
        # create new batch which ignores k=0
        # problem -> prepare inputs encodes some things based off of indices
        #       -> we may violate some assumption (num_prompt, etc)
        # going to continue for now..
        
        # run fwd pass

        sampler_output = self.execute_model_multi_step(
            seq_group_metadata_list=nonzero_proposal_len_seqs,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            num_steps=max_proposal_len,
        )

        # Now, reformat the output GPU tensors such that each sequence has
        # a proposal. the proposal can be empty, e.g. [-1, -1, -1]

        proposal_tokens, proposal_probs = sampler_output_to_torch(sampler_output)

        proposals = SpeculativeProposals(
            # TODO remove unused.
            spec_seqs=[],
            non_spec_seqs=[],
            all_seqs=[],
            original_indices=None,
            proposal_token_ids=proposal_tokens,
            proposal_probs=proposal_probs,
        )


    def _append_new_tokens(
            self, model_output: SamplerOutput,
            seq_group_metadata_list: SequenceGroupMetadata) -> None:
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

                seq.append_token_id(token_id, token_logprob)

    def _shallow_copy_inputs(
        self, seq_group_metadata_list: List[SequenceGroupMetadata]
    ) -> List[SequenceGroupMetadata]:
        """Copy input data structures to remove side-effects when input data
        structures are shared with other modules.

        The multi-step worker must be able to append tokens to sequences after
        a forward pass. This necessitates modification of the data structures
        used by the worker. Since these data structures are shared with other
        parts of vLLM, like the scheduler, we must take care not to introduce
        unexpected side-effects.

        When Ray is used to orchestrate worker processes (such as when the
        tensor-parallel degree is >1), this is not a problem because the input
        datastructures will be serialized and created anew in the worker
        process.

        However, when Ray is not used to orchestrate the worker processes (such
        as when the tensor-parallel degree is 1), this is a problem. We avoid
        the problem by shallow-copying the input datastructures (specifically,
        the parts that will change in multiple steps).
        """

        # Shallow-copy the list of SequenceGroupMetadata. This allows us to
        # append tokens and change is_prompt without external side-effects.
        new_seq_group_metadata_list = []

        for old_seq_group_metadata in seq_group_metadata_list:
            # We must shallow-copy seq_group_metadata as is_prompt could change.
            seq_group_metadata = copy.copy(old_seq_group_metadata)
            new_seq_group_metadata_list.append(seq_group_metadata)

            # We must shallow-copy seq_data as we will append token ids
            new_seq_data = {}
            for seq_id, old_seq_data in seq_group_metadata.seq_data.items():
                new_seq_data[seq_id] = copy.copy(old_seq_data)
                new_seq_data[
                    seq_id].output_token_ids = old_seq_data.output_token_ids[:]

            seq_group_metadata.seq_data = new_seq_data

        return new_seq_group_metadata_list

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
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        """MultiStepWorker does not yet implement support for cache swap
        operations or beam search.
        """
        if any([blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy]):
            raise NotImplementedError(
                "MultiStepWorker does not support cache operations")

        if any(
                len(seq_group_metadata.seq_data.keys()) != 1
                for seq_group_metadata in seq_group_metadata_list):
            raise NotImplementedError(
                "MultiStepWorker does not support beam search.")
