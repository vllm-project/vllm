from typing import List, Optional, Tuple

import torch

from vllm.sequence import (ExecuteModelRequest, SamplerOutput,
                           SequenceGroupMetadata)
from vllm.spec_decode.interfaces import (SpeculativeProposals,
                                         SpeculativeProposer)
from vllm.spec_decode.util import sampler_output_to_torch
from vllm.worker.worker_base import WorkerBase


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
        worker: WorkerBase,
        device: str,
        vocab_size: int,
        max_proposal_len: Optional[int] = None,
    ):
        self._worker = worker
        self._device = device
        self.max_proposal_len = max_proposal_len
        self._vocab_size = vocab_size

    def get_proposals(
        self,
        execute_model_req: ExecuteModelRequest,
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
        ) = self._split_by_max_model_len(seq_group_metadata_list, proposal_len)

        if nonzero_proposal_len_seqs:
            # Speculate tokens using the draft worker for the speculative
            # sequences.
            # If sampler_transposed is true, then maybe_sampler_output's
            # token_ids is like [batch] format in proposal_len size list,
            # while if it is false, the format would be [proposal_len]
            # in batch size list
            nonzero_execute_model_req = ExecuteModelRequest(
                seq_group_metadata_list=nonzero_proposal_len_seqs,
                num_lookahead_slots=proposal_len,
            )
            maybe_sampler_output, transposed = self._worker.sampler_output(
                execute_model_req=nonzero_execute_model_req,
                sample_len=proposal_len,
            )
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

        proposals = SpeculativeProposals(
            proposal_token_ids=proposal_tokens,
            proposal_probs=proposal_probs,
            proposal_lens=proposal_lens,
        )

        return proposals

    def _split_by_max_model_len(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        proposal_len: int,
    ) -> Tuple[List[int], List[SequenceGroupMetadata], List[int]]:
        """Determine which sequences would exceed the max model length."""

        proposal_lens: List[int] = []
        nonzero_proposal_len_seqs: List[SequenceGroupMetadata] = []
        nonzero_proposal_len_indices: List[int] = []
        for i, seq_group_metadata in enumerate(seq_group_metadata_list):
            seq_data = next(iter(seq_group_metadata.seq_data.values()))
            seq_len = seq_data.get_len()

            # Currently only proposal lens of 0 or the global batch proposal len
            # are supported.
            # If max_proposal_len is defined, then we shall no exccess this
            # quota for nonzero_proposal
            if (self.max_proposal_len is None
                    or seq_len + proposal_len < self.max_proposal_len):
                proposal_lens.append(proposal_len)
                nonzero_proposal_len_seqs.append(seq_group_metadata)
                nonzero_proposal_len_indices.append(i)
            else:
                proposal_lens.append(0)

        return (
            proposal_lens,
            nonzero_proposal_len_seqs,
            nonzero_proposal_len_indices,
        )

    def _merge_outputs(
        self,
        batch_size: int,
        proposal_len: int,
        maybe_sampler_output: Optional[SamplerOutput],
        proposal_lens: List[int],
        nonzero_proposal_len_indices: List[int],
        sampler_transposed: bool,
    ) -> Tuple[torch.Tensor, torch.tensor, torch.Tensor]:
        """After speculations are produced, merge the speculation results with
        the skipped sequences.
        """
        if maybe_sampler_output is None:
            # If no speculative tokens, the sampler output will be None.
            # In this case we return empty proposals.
            proposal_tokens = torch.full(
                size=(
                    batch_size,
                    proposal_len,
                ),
                fill_value=-1,
                dtype=torch.long,
                device=self._device,
            )
            proposal_probs = torch.zeros(
                batch_size,
                proposal_len,
                self._vocab_size,
                dtype=torch.float32,
                device=self._device,
            )
            proposal_lens_tensor = torch.zeros(len(proposal_lens),
                                               dtype=torch.long,
                                               device=self._device)
            return proposal_tokens, proposal_probs, proposal_lens_tensor

        sampler_output = maybe_sampler_output
        proposal_tokens, proposal_probs, _ = sampler_output_to_torch(
            sampler_output, sampler_transposed)

        # Now, reformat the output GPU tensors such that each sequence has
        # a proposal. the proposal can be empty, e.g. [-1, -1, -1]

        entire_proposal_tokens = torch.full(
            size=(batch_size, *proposal_tokens.shape[1:]),
            fill_value=-1,
            dtype=torch.long,
            device=self._device,
        )
        entire_proposal_tokens[nonzero_proposal_len_indices] = proposal_tokens
        entire_proposal_probs = torch.zeros(
            batch_size,
            *proposal_probs.shape[1:],
            dtype=torch.float32,
            device=self._device,
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
