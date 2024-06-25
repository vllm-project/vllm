from itertools import chain, count
from typing import Iterator, List, Tuple

import torch

from vllm.sequence import (ExecuteModelRequest, SamplerOutput, SequenceData,
                           SequenceGroupMetadata, get_all_seq_ids)
from vllm.spec_decode.interfaces import (SpeculativeProposals,
                                         SpeculativeScorer, SpeculativeScores)
from vllm.spec_decode.util import (nvtx_range, sampler_output_to_torch,
                                   split_batch_by_proposal_len)
from vllm.worker.worker_base import WorkerBase

SeqId = int
TargetSeqId = int
TokenId = int


class BatchExpansionTop1Scorer(SpeculativeScorer):
    """Implements a speculative scorer that uses batch expansion to get
    probabilities of speculative tokens according to the scoring model.

    Batch expansion converts a list of sequences and multiple query positions
    to a new batch of sequences, each with a single query position. This allows
    for MQA-like scoring in speculative decoding without requiring an MQA
    kernel.

    It is strictly less efficient than MQA scoring.

    It only supports scoring the top1 proposal tokens of the proposer, instead
    of topk/tree.
    """

    def __init__(self, scorer_worker: WorkerBase, device: str,
                 vocab_size: int):
        self._scorer_worker = scorer_worker
        self._device = device
        self._vocab_size = vocab_size

    @nvtx_range("BatchExpansionTop1Scorer.score_proposals")
    def score_proposals(
        self,
        execute_model_req: ExecuteModelRequest,
        proposals: SpeculativeProposals,
    ) -> SpeculativeScores:
        """Score the proposed tokens via the scorer model.

        This converts each input sequence to a set of k+1 target sequences. The
        target sequences have the unique continuations to be scored and a
        unique sequence ID that is different from all input sequence ids.

        If a speculative sequence length would exceed the max model length, then
        no speculation is produced for that sequence.

        Args:
            execute_model_req: The execution request.
            proposals: The speculative proposals to score.
        Returns:
            SpeculativeScores: The scores of each speculative token, along with
                which sequences were ignored during scoring.
        """

        # TODO(cade) perform this on GPU to remove blocking call.
        proposal_lens_list = proposals.proposal_lens.tolist()
        proposal_token_ids_list = proposals.proposal_token_ids.tolist()

        # Filter the list to ignore -1 proposals.
        proposal_token_ids_list_without_skips = [
            proposals for proposals in proposal_token_ids_list
            if -1 not in proposals
        ]

        (spec_indices, non_spec_indices, target_seq_group_metadata_list,
         num_scoring_tokens) = self._expand_batch(
             seq_group_metadata_list=execute_model_req.seq_group_metadata_list,
             proposal_token_ids_list=proposal_token_ids_list_without_skips,
             proposal_lens_list=proposal_lens_list,
         )

        target_sampler_output = self._scorer_worker.execute_model(
            execute_model_req=execute_model_req.clone(
                seq_group_metadata_list=target_seq_group_metadata_list))
        assert len(target_sampler_output) == 1, "expected single-step output"
        target_sampler_output = target_sampler_output[0]

        all_tokens, all_probs, spec_logprobs = self._contract_batch(
            contracted_bs=len(execute_model_req.seq_group_metadata_list),
            target_sampler_output=target_sampler_output,
            proposals=proposals,
            num_scoring_tokens=num_scoring_tokens,
            non_spec_indices=non_spec_indices,
            spec_indices=spec_indices,
            k=execute_model_req.num_lookahead_slots,
        )

        return SpeculativeScores(
            probs=all_probs,
            token_ids=all_tokens,
            logprobs=spec_logprobs,
            hidden_states=target_sampler_output.hidden_states,
        )

    def _expand_batch(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        proposal_token_ids_list: List[List[TokenId]],
        proposal_lens_list: List[int],
    ) -> Tuple[List[int], List[int], List[SequenceGroupMetadata], int]:
        """Given the input sequences and potentially multiple corresponding
        proposal tokens, create a new batch where each sequence has a single
        query token.
        """

        # vLLM currently only supports proposal lens equal to zero or the batch
        # proposal len. This adds some complexity (splitting the batch into spec
        # and non spec sequences) and should be removed in the future. It can be
        # done by supporting per-sequence proposal lens.
        spec_seqs, spec_indices = split_batch_by_proposal_len(
            seq_group_metadata_list,
            proposal_lens_list,
            select_proposal_len_zero=False)
        non_spec_seqs, non_spec_indices = split_batch_by_proposal_len(
            seq_group_metadata_list,
            proposal_lens_list,
            select_proposal_len_zero=True)

        target_seq_group_metadata_list = self._create_scoring_model_input(
            seq_group_metadata_list=spec_seqs,
            proposal_token_ids=proposal_token_ids_list,
            # NOTE: We determine the seq ids in the expanded batch using the
            # full seq_group_metadata_list, instead of only spec_seqs.
            target_seq_ids_iter=self._create_target_seq_id_iterator(
                seq_ids=get_all_seq_ids(seq_group_metadata_list)),
        )

        num_scoring_tokens = len(target_seq_group_metadata_list)
        target_seq_group_metadata_list.extend(non_spec_seqs)

        return (spec_indices, non_spec_indices, target_seq_group_metadata_list,
                num_scoring_tokens)

    def _contract_batch(
            self, contracted_bs: int, target_sampler_output: SamplerOutput,
            proposals: SpeculativeProposals, num_scoring_tokens: int,
            non_spec_indices: List[int], spec_indices: List[int],
            k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Contract the expanded batch back into its original size.
        This maps the scores of speculative tokens back to their original
        sequences.

        contracted_bs is the original batch size, and the batch size that the
        target_sampler_output will be contracted to.
        """
        (target_token_ids, target_probs, target_logprobs,
         non_spec_target_token_ids, non_spec_target_probs,
         non_spec_target_logprobs) = self._split_scoring_output(
             target_sampler_output, num_scoring_tokens)

        # Map distinct sequences used to score each token
        # of shape [batch_size * k + 1] back to [batch_size, k + 1].
        expanded_batch_size, k = proposals.proposal_token_ids.shape

        # The number of tokens in the expanded batch used for speculation is
        # equal to the total expanded batch size minus the number of samples for
        # non-speculative sequences.
        non_spec_expanded_bs, _ = non_spec_target_token_ids.shape
        spec_expanded_bs = expanded_batch_size - non_spec_expanded_bs

        target_token_ids = target_token_ids.reshape(spec_expanded_bs, k + 1)
        target_probs = target_probs.reshape(*target_token_ids.shape,
                                            self._vocab_size)
        target_logprobs = target_logprobs.reshape(target_probs.shape)

        all_tokens = target_token_ids.new_full(size=(contracted_bs, k + 1),
                                               fill_value=-1)
        all_probs = target_probs.new_zeros(*all_tokens.shape, self._vocab_size)
        all_logprobs = target_logprobs.new_full(size=all_probs.shape,
                                                fill_value=-float("inf"))

        if non_spec_indices:
            all_tokens[non_spec_indices, :1] = non_spec_target_token_ids
            all_probs[non_spec_indices, :1, :] = non_spec_target_probs
            all_logprobs[non_spec_indices, :1, :] = non_spec_target_logprobs

        if spec_indices:
            all_tokens[spec_indices] = target_token_ids
            all_probs[spec_indices] = target_probs
            all_logprobs[spec_indices] = target_logprobs

        return all_tokens, all_probs, all_logprobs

    def _create_scoring_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        proposal_token_ids: List[List[TokenId]],  # shape: [batch_size, k]
        target_seq_ids_iter: Iterator[TargetSeqId],
    ) -> List[SequenceGroupMetadata]:
        """Given the original input sequences and proposed tokens from the draft
        model, create a list of target sequences that can be used for scoring.

        target_seq_ids_iter provides sequence ids for the expanded batch,
        fulfilling the requirement that no seq id in the expanded batch is equal
        to the seq id in the original batch.
        """

        if not seq_group_metadata_list:
            return []

        target_seq_group_metadata = list(
            chain.from_iterable(
                self._create_target_seq_group_metadata(
                    seq_group_metadata,
                    proposal_token_ids,
                    i,
                    target_seq_ids_iter,
                ) for i, seq_group_metadata in enumerate(
                    seq_group_metadata_list)))

        return target_seq_group_metadata

    def _create_target_seq_group_metadata(
        self,
        input_seq_group_metadata: SequenceGroupMetadata,
        proposal_token_ids: List[List[TokenId]],  # shape: [batch_size, k]
        batch_index: int,
        target_seq_ids_iter: Iterator[TargetSeqId],
    ) -> List[SequenceGroupMetadata]:
        """Given an input sequence group metadata and a list of draft tokens,
        create a list of target SequenceGroupMetadata, one for each
        token id that needs to be scored.

        Naive speculative decoding requires K target model scores, one for each
        draft model token. However one can add a bonus token such that if each
        token is accepted, then a final token may be sampled from the model.
        This function creates K+1 target SequenceGroupMetadata to take
        advantage of the bonus token.
        """
        assert not input_seq_group_metadata.is_prompt, (
            "Speculating on "
            "prompts not yet supported")
        assert len(input_seq_group_metadata.seq_data) == 1, (
            "Beam search "
            "not supported in speculative decoding")
        input_seq_id = next(iter(input_seq_group_metadata.seq_data.keys()))

        token_ids_to_score = self._get_token_ids_to_score(
            proposal_token_ids[batch_index])

        target_seq_group_metadata_list: List[SequenceGroupMetadata] = []
        for token_ids in token_ids_to_score:
            target_seq_group_metadata_list.append(
                self._create_single_target_seq_group_metadata(
                    input_seq_group_metadata,
                    input_seq_id,
                    next(target_seq_ids_iter),
                    token_ids,
                ))

        return target_seq_group_metadata_list

    def _create_single_target_seq_group_metadata(
        self,
        seq_group_metadata: SequenceGroupMetadata,
        seq_id: SeqId,
        target_seq_id: TargetSeqId,
        token_ids: List[TokenId],
    ) -> SequenceGroupMetadata:
        """Create a single target SequenceGroupMetadata.

        Args:
            seq_group_metadata: The metadata for the input sequence.
            seq_id: The input sequence ID.
            target_seq_id: The corresponding target sequence ID.
            token_ids: The list of token ids that are to be appended to the
                input sequence.
        """
        seq_data = seq_group_metadata.seq_data[seq_id]
        prompt_token_ids = seq_data.get_prompt_token_ids()
        new_output_token_ids = [*seq_data.get_output_token_ids(), *token_ids]

        new_seq_data_dict = {
            target_seq_id:
            SequenceData(
                prompt_token_ids=prompt_token_ids,
                output_token_ids=new_output_token_ids,
            ),
        }
        # This is a hack. Technically, spec decoding should compute
        # num_lookahead slots at one shot, but instead, it expands the batch
        # and evaluate one by one right now. context_len is seq_len - 1 because
        # the kv cache is filled by a previous batch in the batch expansion.
        for data in new_seq_data_dict.values():
            data.update_num_computed_tokens(data.get_len() - 1)

        return SequenceGroupMetadata(
            request_id=seq_group_metadata.request_id,
            is_prompt=seq_group_metadata.is_prompt,
            seq_data=new_seq_data_dict,
            sampling_params=seq_group_metadata.sampling_params,
            block_tables={
                target_seq_id: seq_group_metadata.block_tables[seq_id],
            },
            lora_request=None,
            token_chunk_size=1,
        )

    def _split_scoring_output(
        self, sampler_output: SamplerOutput, num_scoring_tokens: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor]:
        """Split the target model output into speculative and non-speculative
        output.
        """

        # vLLM currently only supports proposal lens equal to zero or the batch
        # proposal len. This adds some complexity (splitting the batch into spec
        # and non spec sequences) and should be removed in the future. It can be
        # done by supporting per-sequence proposal lens.
        #
        # First samples are from speculative scoring, latter samples are non-
        # speculative samples.
        split_sizes = [
            num_scoring_tokens,
            sampler_output.sampled_token_ids.numel() - num_scoring_tokens
        ]
        (spec_probs, non_spec_probs
         ) = sampler_output.sampled_token_probs.split(split_sizes)
        (spec_sampled_tokens, non_spec_sampled_tokens
         ) = sampler_output.sampled_token_ids.flatten().split(split_sizes)
        (
            spec_logprobs,
            non_spec_logprobs,
        ) = sampler_output.logprobs.split(split_sizes)

        # Convert scores to tensors.
        sampler_output.sampled_token_probs = spec_probs
        sampler_output.sampled_token_ids = spec_sampled_tokens
        sampler_output.logprobs = spec_logprobs
        (target_token_ids, target_probs,
         target_logprobs) = sampler_output_to_torch([sampler_output], True)

        # Convert non-speculative output tokens to tensors.
        sampler_output.sampled_token_probs = non_spec_probs
        sampler_output.sampled_token_ids = non_spec_sampled_tokens
        sampler_output.logprobs = non_spec_logprobs
        (non_spec_target_token_ids, non_spec_target_probs,
         non_spec_target_logprobs) = sampler_output_to_torch([sampler_output],
                                                             True)

        return (target_token_ids, target_probs, target_logprobs,
                non_spec_target_token_ids, non_spec_target_probs,
                non_spec_target_logprobs)

    def _create_target_seq_id_iterator(
            self, seq_ids: List[SeqId]) -> Iterator[TargetSeqId]:
        """Create an iterator for creating target sequence ids.
        Target sequence ids are distinct from sequence ids because we create a
        distinct target sequence id for each proposal token to be scored.

        This implementation increments a counter starting at 1 + max of all
        provided input sequence ids.
        """
        return count(start=max(seq_ids) + 1)

    def _get_token_ids_to_score(
        self,
        full_spec_token_ids: List[TokenId]  # shape: [k]
    ) -> List[List[TokenId]]:
        """Given an int tensor of proposal token ids, return a list of
        token ids that should be scored.

        Returns k+1 output lists. The additional one is used for generating the
        bonus token.

        Example:
            Input: [0, 1, 2, 3] (k=4)
            Output: (k+1 lists)
                []
                [0]
                [0, 1]
                [0, 1, 2]
                [0, 1, 2, 3]
        """
        empty_token_ids: List[TokenId] = []

        token_ids_to_score = [empty_token_ids]
        token_ids_to_score.extend([
            full_spec_token_ids[:i + 1]
            for i in range(len(full_spec_token_ids))
        ])
        return token_ids_to_score
