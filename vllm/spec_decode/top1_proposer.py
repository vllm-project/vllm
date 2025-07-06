# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import List, Optional, Set, Tuple

import torch

from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest, SequenceGroupMetadata
from vllm.spec_decode.interfaces import (SpeculativeProposals,
                                         SpeculativeProposer)
from vllm.spec_decode.proposer_worker_base import ProposerWorkerBase
from vllm.spec_decode.util import sampler_output_to_torch


class Top1Proposer(SpeculativeProposer):
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
        worker: ProposerWorkerBase,
        device: str,
        vocab_size: int,
        max_proposal_len: Optional[int] = None,
    ):
        self._worker = worker
        self._device = device
        self.max_proposal_len = max_proposal_len
        self._vocab_size = vocab_size

    def get_spec_proposals(
        self,
        execute_model_req: ExecuteModelRequest,
        seq_ids_with_bonus_token_in_last_step: Set[int],
    ) -> SpeculativeProposals:
        """Get speculative proposals given the input batch.

        Sequences which would exceed the max model length are skipped during
        speculation.
        """
        proposal_len = execute_model_req.num_lookahead_slots
        seq_group_metadata_list = execute_model_req.seq_group_metadata_list

        # Split speculative- and non-speculative- sequences.
        (
            proposal_lens,
            nonzero_proposal_len_seqs,
            nonzero_proposal_len_indices,
        ) = self._split_by_proposal_len(seq_group_metadata_list, proposal_len)

        if nonzero_proposal_len_seqs:
            # Speculate tokens using the draft worker for the speculative
            # sequences.
            # If sampler_transposed is true, then maybe_sampler_output's
            # token_ids is like [batch] format in proposal_len size list,
            # while if it is false, the format would be [proposal_len]
            # in batch size list
            hidden_states = execute_model_req.previous_hidden_states
            if hidden_states is not None:
                hidden_states.prune(nonzero_proposal_len_seqs)
            nonzero_execute_model_req = ExecuteModelRequest(
                seq_group_metadata_list=nonzero_proposal_len_seqs,
                num_lookahead_slots=proposal_len,
                previous_hidden_states=hidden_states,
            )
            maybe_sampler_output, transposed = self._worker.sampler_output(
                execute_model_req=nonzero_execute_model_req,
                sample_len=proposal_len,
                seq_ids_with_bonus_token_in_last_step=\
                    seq_ids_with_bonus_token_in_last_step,
            )
            (
                proposal_lens,
                maybe_sampler_output,
                nonzero_proposal_len_indices,
            ) = self._remove_no_proposal_seqs(proposal_lens,
                                              maybe_sampler_output,
                                              nonzero_proposal_len_indices,
                                              transposed)
        else:
            # If no sequences can be speculated, set sampler output to None.
            maybe_sampler_output = None
            transposed = False

        # Combine speculative- and non-speculative sequences into the same
        # representation.
        proposal_tokens, proposal_probs, proposal_lens = self._merge_outputs(
            batch_size=len(seq_group_metadata_list),
            proposal_len=proposal_len,
            maybe_sampler_output=maybe_sampler_output,
            proposal_lens=proposal_lens,
            nonzero_proposal_len_indices=nonzero_proposal_len_indices,
            sampler_transposed=transposed,
        )

        proposals = SpeculativeProposals(proposal_token_ids=proposal_tokens,
                                         proposal_probs=proposal_probs,
                                         proposal_lens=proposal_lens,
                                         no_proposals=maybe_sampler_output
                                         is None)
        return proposals

    def _split_by_proposal_len(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        proposal_len: int,
    ) -> Tuple[List[int], List[SequenceGroupMetadata], List[int]]:
        """Split sequences by two groups:
        1. Sequences with non-zero proposal length.
        2. Sequences with zero proposal length (due to disabled speculation
        or exceed the maximum model length).
        """

        proposal_lens: List[int] = []
        nonzero_proposal_len_seqs: List[SequenceGroupMetadata] = []
        nonzero_proposal_len_indices: List[int] = []
        for i, seq_group_metadata in enumerate(seq_group_metadata_list):
            # The speculative decoding for this request has either been disabled
            # (e.g. due to high traffic) or this is a prompt request.
            if (seq_group_metadata.is_prompt
                    or seq_group_metadata.num_speculative_tokens == 0):
                proposal_lens.append(0)
                continue

            seq_data = next(iter(seq_group_metadata.seq_data.values()))
            seq_len = seq_data.get_len()

            # Currently only proposal lens of 0 or the global batch proposal len
            # are supported.
            # If max_proposal_len is defined, then we shall not exceed this
            # quota for nonzero_proposal
            new_k = 0
            if (self.max_proposal_len is None
                    or seq_len + proposal_len < self.max_proposal_len):
                new_k = proposal_len
                nonzero_proposal_len_seqs.append(seq_group_metadata)
                nonzero_proposal_len_indices.append(i)
            proposal_lens.append(new_k)
            seq_group_metadata.num_speculative_tokens = new_k

        return (
            proposal_lens,
            nonzero_proposal_len_seqs,
            nonzero_proposal_len_indices,
        )

    @staticmethod
    def _remove_no_proposal_seqs(proposal_lens, maybe_sampler_output,
                                 nonzero_proposal_len_indices, transposed):
        """Remove sequences from nonzero_proposal_len_indices and reset
        their proposal_len to 0 the draft worker does not provide a proposal
        (maybe_sampler_output=None). This can avoid scoring overheads.
        """

        # If maybe_sampler_output is None, then the draft worker did not
        # provide a proposal for any sequence and thus no action needed.
        # Also we do not support transposed maybe_sampler_output for now
        # because it seems not straightforward for draft workers outputting
        # transposed sampler outputs to handle the case of no proposal.
        if maybe_sampler_output is None or transposed:
            return (proposal_lens, maybe_sampler_output,
                    nonzero_proposal_len_indices)

        new_proposal_lens: List[int] = []
        new_nonzero_proposal_len_indices: List[int] = []
        new_maybe_sampler_output: List[SamplerOutput] = []
        nonzero_proposal_len_idx_ptr = 0
        seq_idx = 0
        while seq_idx < len(
                proposal_lens) and nonzero_proposal_len_idx_ptr < len(
                    nonzero_proposal_len_indices):
            if seq_idx < nonzero_proposal_len_indices[
                    nonzero_proposal_len_idx_ptr]:
                # Sequence is not in the original nonzero_proposal_len_indices,
                # meaning that it has a proposal length of 0 before sending to
                # the draft worker.
                assert proposal_lens[seq_idx] == 0
                new_proposal_lens.append(0)
            else:
                # Sequence is in the original nonzero_proposal_len_indices
                if maybe_sampler_output[nonzero_proposal_len_idx_ptr] is None:
                    # but does not have a proposal from the draft worker.
                    new_proposal_lens.append(0)
                else:
                    # and has a proposal from the draft worker. Add it to the
                    # new nonzero proposal list and keep the sampler output.
                    new_proposal_lens.append(proposal_lens[seq_idx])
                    new_nonzero_proposal_len_indices.append(seq_idx)
                    new_maybe_sampler_output.append(
                        maybe_sampler_output[nonzero_proposal_len_idx_ptr])
                nonzero_proposal_len_idx_ptr += 1
            seq_idx += 1

        # The remaining sequences should have proposal length of 0.
        new_proposal_lens.extend(proposal_lens[seq_idx:])

        # We assume sampler_output will not be a list of all Nones.
        # In this case this function should not be called.
        assert new_maybe_sampler_output
        return (new_proposal_lens, new_maybe_sampler_output,
                new_nonzero_proposal_len_indices)

    def _merge_outputs(
        self,
        batch_size: int,
        proposal_len: int,
        maybe_sampler_output: Optional[List[SamplerOutput]],
        proposal_lens: List[int],
        nonzero_proposal_len_indices: List[int],
        sampler_transposed: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """After speculations are produced, merge the speculation results with
        the skipped sequences.
        """
        if maybe_sampler_output is None:
            # If no speculative tokens, the sampler output will be None.
            # In this case we return empty proposals.
            proposal_tokens = torch.tensor(-1,
                                           dtype=torch.long,
                                           device=self._device).expand(
                                               batch_size, proposal_len)
            proposal_probs = torch.tensor(0,
                                          dtype=torch.float32,
                                          device=self._device).expand(
                                              batch_size, proposal_len,
                                              self._vocab_size)
            proposal_lens_tensor = torch.tensor(0,
                                                dtype=torch.long,
                                                device=self._device).expand(
                                                    len(proposal_lens))
            return proposal_tokens, proposal_probs, proposal_lens_tensor

        sampler_output = maybe_sampler_output
        proposal_tokens, proposal_probs, *_ = sampler_output_to_torch(
            sampler_output, sampler_transposed)

        # Now, reformat the output GPU tensors such that each sequence has
        # a proposal. the proposal can be empty, e.g. [-1, -1, -1]

        entire_proposal_tokens = proposal_tokens.new_full(
            size=(batch_size, *proposal_tokens.shape[1:]),
            fill_value=-1,
        )
        entire_proposal_tokens[nonzero_proposal_len_indices] = proposal_tokens
        entire_proposal_probs = proposal_probs.new_zeros(
            batch_size,
            *proposal_probs.shape[1:],
        )
        entire_proposal_probs[nonzero_proposal_len_indices] = proposal_probs

        proposal_tokens, proposal_probs = (
            entire_proposal_tokens,
            entire_proposal_probs,
        )

        proposal_lens_tensor = torch.zeros(batch_size,
                                           dtype=torch.long,
                                           device=self._device)
        proposal_lens_tensor[nonzero_proposal_len_indices] = proposal_len

        return proposal_tokens, proposal_probs, proposal_lens_tensor
