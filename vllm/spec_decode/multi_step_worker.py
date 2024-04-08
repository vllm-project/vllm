import copy
from typing import Dict, List, Optional, Tuple

import torch

from vllm.sequence import SamplerOutput, SequenceGroupMetadata
from vllm.spec_decode.interfaces import (SpeculativeProposals,
                                         SpeculativeProposer)
from vllm.spec_decode.util import sampler_output_to_torch
from vllm.worker.worker import Worker


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._proposer: Optional[DraftModelTop1Proposer] = None

    def init_device(self):
        super().init_device()

        self._proposer = DraftModelTop1Proposer(
            self,
            self.device,
            self.max_model_len,
            self.vocab_size,
        )

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
        """Produce speculations given an input batch of sequences. The number of
        speculative tokens per sequence is determined by max_proposal_len.
        """

        return self._proposer.get_proposals(
            seq_group_metadata_list,
            blocks_to_swap_in,
            blocks_to_swap_out,
            blocks_to_copy,
            max_proposal_len,
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

                seq.append_token_id(token_id, token_logprob.logprob)

    def _shallow_copy_inputs(
        self, seq_group_metadata_list: List[SequenceGroupMetadata]
    ) -> List[SequenceGroupMetadata]:
        """Copy input data structures to remove side-effects when input data
        structures are shared with other modules.

        Helpful when the vLLM scheduler runs in the same process as the worker.
        The alternative is deep-copying (or other form of deep copy); this has
        performance downsides.
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


class DraftModelTop1Proposer(SpeculativeProposer):
    """Helper class which separates out sequences which would exceed the max
    model length when speculated upon.

    This allows combinations of models such as JackFram/llama-68m draft with
    meta-llama/Llama2-13b-chat-hf, as llama-68m has max_position_embeddings of
    2048 while Llama2-13b has max_position_embeddings of 4096.

    We treat the sequences which exceed the proposal draft model length as
    "non-spec sequences". Essentially they skip the draft model and go through
    normal decoding in the target model.

    Currently, only proposal_lens of 0 and k are supported, where k is a global
    batch proposal length. In the future vLLM should support per-sequence
    proposal lengths.
    """

    def __init__(
        self,
        draft_worker: MultiStepWorker,
        device: str,
        max_model_len: int,
        vocab_size: int,
    ):
        self._draft_worker = draft_worker
        self._device = device
        self._max_model_len = max_model_len
        self._vocab_size = vocab_size

    def get_proposals(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
        max_proposal_len: int,
    ) -> SpeculativeProposals:
        """Get speculative proposals given the input batch.

        Sequences which would exceed the max model length are skipped during
        speculation.
        """

        # Split speculative- and non-speculative- sequences.
        (proposal_lens, nonzero_proposal_len_seqs,
         nonzero_proposal_len_indices) = self._split_by_max_model_len(
             seq_group_metadata_list, max_proposal_len)

        if nonzero_proposal_len_seqs:
            # Speculate tokens using the draft worker for the speculative
            # sequences.
            maybe_sampler_output = self._draft_worker.execute_model_multi_step(
                seq_group_metadata_list=nonzero_proposal_len_seqs,
                blocks_to_swap_in=blocks_to_swap_in,
                blocks_to_swap_out=blocks_to_swap_out,
                blocks_to_copy=blocks_to_copy,
                num_steps=max_proposal_len,
            )
        else:
            # If no sequences can be speculated, set sampler output to None.
            maybe_sampler_output = None

        # Combine speculative- and non-speculative sequences into the same
        # representation.
        proposal_tokens, proposal_probs, proposal_lens = self._merge_outputs(
            batch_size=len(seq_group_metadata_list),
            max_proposal_len=max_proposal_len,
            maybe_sampler_output=maybe_sampler_output,
            proposal_lens=proposal_lens,
            nonzero_proposal_len_indices=nonzero_proposal_len_indices,
        )

        proposals = SpeculativeProposals(
            proposal_token_ids=proposal_tokens,
            proposal_probs=proposal_probs,
            proposal_lens=proposal_lens,
        )

        return proposals

    def _split_by_max_model_len(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        max_proposal_len: int,
    ) -> Tuple[List[int], List[SequenceGroupMetadata], List[int]]:
        """Determine which sequences would exceed the max model length.
        """

        proposal_lens: List[int] = []
        nonzero_proposal_len_seqs: List[SequenceGroupMetadata] = []
        nonzero_proposal_len_indices: List[int] = []
        for i, seq_group_metadata in enumerate(seq_group_metadata_list):
            seq_data = next(iter(seq_group_metadata.seq_data.values()))
            seq_len = seq_data.get_len()

            # Currently only proposal lens of 0 or the global batch proposal len
            # are supported.
            if seq_len + max_proposal_len < self._max_model_len:
                proposal_lens.append(max_proposal_len)
                nonzero_proposal_len_seqs.append(seq_group_metadata)
                nonzero_proposal_len_indices.append(i)
            else:
                proposal_lens.append(0)

        return (proposal_lens, nonzero_proposal_len_seqs,
                nonzero_proposal_len_indices)

    def _merge_outputs(
        self,
        batch_size: int,
        max_proposal_len: int,
        maybe_sampler_output: Optional[SamplerOutput],
        proposal_lens: List[int],
        nonzero_proposal_len_indices: List[int],
    ) -> Tuple[torch.Tensor, torch.tensor, torch.Tensor]:
        """After speculations are produced, merge the speculation results with
        the skipped sequences.
        """
        if maybe_sampler_output is None:
            # If no speculative tokens, the sampler output will be None.
            # In this case we return empty tensors.
            proposal_tokens = torch.zeros(0,
                                          max_proposal_len,
                                          dtype=torch.long,
                                          device=self._device)
            proposal_probs = torch.zeros(0,
                                         max_proposal_len,
                                         self._vocab_size,
                                         dtype=torch.float32,
                                         device=self._device)
            proposal_lens = torch.zeros(len(proposal_lens),
                                        dtype=torch.long,
                                        device=self._device)
            return proposal_tokens, proposal_probs, proposal_lens

        sampler_output = maybe_sampler_output

        proposal_tokens, proposal_probs = sampler_output_to_torch(
            sampler_output)

        # Now, reformat the output GPU tensors such that each sequence has
        # a proposal. the proposal can be empty, e.g. [-1, -1, -1]

        entire_proposal_tokens = torch.full(size=(batch_size,
                                                  *proposal_tokens.shape[1:]),
                                            fill_value=-1,
                                            dtype=torch.long,
                                            device=self._device)
        entire_proposal_tokens[nonzero_proposal_len_indices] = proposal_tokens
        entire_proposal_probs = torch.zeros(batch_size,
                                            *proposal_probs.shape[1:],
                                            dtype=torch.float32,
                                            device=self._device)
        entire_proposal_probs[nonzero_proposal_len_indices] = proposal_probs

        proposal_tokens, proposal_probs = (entire_proposal_tokens,
                                           entire_proposal_probs)

        proposal_lens = torch.zeros(batch_size,
                                    dtype=torch.long,
                                    device=self._device)
        proposal_lens[nonzero_proposal_len_indices] = max_proposal_len

        return proposal_tokens, proposal_probs, proposal_lens
