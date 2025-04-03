# SPDX-License-Identifier: Apache-2.0

from array import array
from itertools import chain, count
from typing import Iterator, List, Optional, Tuple

import torch

from vllm import SamplingParams
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import (VLLM_INVALID_TOKEN_ID, VLLM_TOKEN_ID_ARRAY_TYPE,
                           ExecuteModelRequest, SequenceData,
                           SequenceGroupMetadata, get_all_seq_ids)
from vllm.spec_decode.interfaces import (SpeculativeProposals,
                                         SpeculativeScorer, SpeculativeScores)
from vllm.spec_decode.util import nvtx_range, split_batch_by_proposal_len

SeqId = int
TargetSeqId = int
TokenId = int

DEFAULT_SIMPLE_SAMPLING_PARAMS = SamplingParams()


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

        # Filter the list to ignore invalid proposals.
        proposal_token_ids_list_without_skips = [
            proposals for proposals in proposal_token_ids_list
            if VLLM_INVALID_TOKEN_ID not in proposals
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

        if not non_spec_indices:
            # All sequence groups in batch have spec decoding enabled
            return self._contract_batch_all_spec(
                target_sampler_output=target_sampler_output,
                proposals=proposals,
            )
        else:
            # Batch has a mix of spec decode enabled and disabled seq groups
            return self._contract_batch(
                execute_model_req.seq_group_metadata_list,
                target_sampler_output=target_sampler_output,
                proposals=proposals,
                num_scoring_tokens=num_scoring_tokens,
                non_spec_indices=non_spec_indices,
                spec_indices=spec_indices,
                k=execute_model_req.num_lookahead_slots,
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
        (spec_seqs, spec_indices), (non_spec_seqs, non_spec_indices) = \
            split_batch_by_proposal_len(
                seq_group_metadata_list, proposal_lens_list)

        spec_expanded_seqs = self._create_scoring_model_input(
            seq_group_metadata_list=spec_seqs,
            proposal_token_ids=proposal_token_ids_list,
            # NOTE: We determine the seq ids in the expanded batch using the
            # full seq_group_metadata_list, instead of only spec_seqs.
            target_seq_ids_iter=self._create_target_seq_id_iterator(
                seq_ids=get_all_seq_ids(seq_group_metadata_list)),
        )

        num_scoring_tokens = len(spec_expanded_seqs)
        # Batch speculative and non-speculative (e.g. chunked prefill) requests
        # but make sure order is prefill|decode due to backend requirement.
        target_seq_group_metadata_list = non_spec_seqs + spec_expanded_seqs

        return (spec_indices, non_spec_indices, target_seq_group_metadata_list,
                num_scoring_tokens)

    def _contract_non_speculative(
            self, scores: SpeculativeScores,
            seq_group_metadata_list: List[SequenceGroupMetadata],
            non_spec_indices: List[int], non_spec_outputs: SpeculativeScores,
            has_prompt_log: bool) -> SpeculativeScores:
        """
            Augment input `scores` with non-speculative requests outputs. 
            This includes decode requests with speculation turned off, as well
            as prefill requests when `enable_chunked_prefill` is set.
            For the latter, prefills are further separated into terminal and 
            non-terminal chunks (from which no token is sampled).
        """
        if not non_spec_indices:
            return scores

        if has_prompt_log:
            # When prompt_logprobs is enabled, prefills yield output token
            # (and respective prob) in the last entry (prompt|out):
            # [.|.|.|prefill0_out|.|prefill1_out|decode0_out|..].
            # With chunked prefill, non-terminal chunks have -1 on each
            # position: they're still picked, but they're discarded later.
            seq_meta = seq_group_metadata_list
            nospec_sizes = torch.tensor([
                seq_meta[i].token_chunk_size if seq_meta[i].is_prompt else 1
                for i in non_spec_indices
            ])
            nospec_sampled_token_idxs = torch.cumsum(nospec_sizes, 0).add_(-1)
        else:
            # In this case only sampled tokens are returned, select all.
            nospec_sampled_token_idxs = list(
                range(len(non_spec_outputs.token_ids)))

        scores.token_ids[non_spec_indices, :1] = \
            non_spec_outputs.token_ids[nospec_sampled_token_idxs].unsqueeze(1)
        scores.probs[non_spec_indices, :1, :] = \
            non_spec_outputs.probs[nospec_sampled_token_idxs].unsqueeze(1)
        scores.logprobs[non_spec_indices, :1, :] = \
            non_spec_outputs.logprobs[nospec_sampled_token_idxs].unsqueeze(1)
        if scores.hidden_states is not None:
            assert non_spec_outputs.hidden_states is not None
            scores.hidden_states[non_spec_indices, :1, :] = \
                non_spec_outputs.hidden_states[nospec_sampled_token_idxs].unsqueeze(1)
        return scores

    def _contract_batch(
            self,
            contracted_seq_group_metadata_list: List[SequenceGroupMetadata],
            target_sampler_output: SamplerOutput,
            proposals: SpeculativeProposals, num_scoring_tokens: int,
            non_spec_indices: List[int], spec_indices: List[int],
            k: int) -> SpeculativeScores:
        """Contract the expanded batch back into its original size.
        This maps the scores of speculative tokens back to their original
        sequences.

        contracted_bs is the original batch size, and the batch size that the
        target_sampler_output will be contracted to.
        """
        contracted_bs = len(contracted_seq_group_metadata_list)
        (target_token_ids, target_probs, target_logprobs, target_hidden_states,
         non_spec_target_token_ids, non_spec_target_probs,
         non_spec_target_logprobs,
         non_spec_target_hidden_states) = self._split_scoring_output(
             target_sampler_output, num_scoring_tokens)

        # Map distinct sequences used to score each token
        # of shape [batch_size * k + 1] back to [batch_size, k + 1].
        expanded_batch_size, k = proposals.proposal_token_ids.shape

        # The number of tokens in the expanded batch used for speculation is
        # equal to the total expanded batch size minus the number of samples for
        # non-speculative sequences, prefill chunks with no out tokens included
        non_spec_expanded_bs = len(non_spec_indices)
        spec_expanded_bs = expanded_batch_size - non_spec_expanded_bs

        target_token_ids = target_token_ids.reshape(spec_expanded_bs, k + 1)
        target_probs = target_probs.reshape(*target_token_ids.shape,
                                            self._vocab_size)
        target_logprobs = target_logprobs.reshape(target_probs.shape)

        if target_hidden_states is not None:
            target_hidden_states = target_hidden_states.reshape(
                *target_token_ids.shape, target_hidden_states.shape[-1])

        all_tokens = target_token_ids.new_full(size=(contracted_bs, k + 1),
                                               fill_value=-1)
        all_probs = target_probs.new_zeros(*all_tokens.shape, self._vocab_size)
        all_logprobs = target_logprobs.new_full(size=all_probs.shape,
                                                fill_value=-float("inf"))

        if target_sampler_output.hidden_states is not None:
            all_hidden_states = target_hidden_states.new_zeros(
                size=(contracted_bs, k + 1, target_hidden_states.shape[-1]))
        else:
            all_hidden_states = None

        has_prompt_log = any((sg.sampling_params.prompt_logprobs
                              and sg.sampling_params.prompt_logprobs > 0)
                             for sg in contracted_seq_group_metadata_list)
        # When prompt logprobs is enabled, lens of returned tensors go from
        # n_sampled (requests with do_sample=True) to n_prompt+n_prefills.
        # We adjust stride accordingly to get the generated tokens and
        # their probs, but pass on prompt_logprobs as is.
        prompt_logprobs = None
        if (not self._scorer_worker.model_runner.disable_logprobs\
            and has_prompt_log):
            prompt_logprobs = [
                o.prompt_logprobs for o in target_sampler_output.outputs
            ]
        elif not has_prompt_log:
            # When prompt logprobs are not to be returned,
            # we can ignore non-terminal chunks (no out token).
            non_spec_indices = [
                idx for idx in non_spec_indices
                if contracted_seq_group_metadata_list[idx].do_sample
            ]

        # "Contract" speculative.
        if spec_indices:
            all_tokens[spec_indices] = target_token_ids
            all_probs[spec_indices] = target_probs
            all_logprobs[spec_indices] = target_logprobs
            if all_hidden_states is not None:
                all_hidden_states[spec_indices] = target_hidden_states

        spec_scores = SpeculativeScores(probs=all_probs,
                                        token_ids=all_tokens,
                                        logprobs=all_logprobs,
                                        hidden_states=all_hidden_states,
                                        prompt_logprobs=prompt_logprobs)

        non_spec_outputs = SpeculativeScores(
            probs=non_spec_target_probs,
            token_ids=non_spec_target_token_ids,
            logprobs=non_spec_target_logprobs,
            hidden_states=non_spec_target_hidden_states)
        # Contract remaining nonspec entries based on non_spec_indices, if any.
        return self._contract_non_speculative(
            spec_scores, contracted_seq_group_metadata_list, non_spec_indices,
            non_spec_outputs, has_prompt_log)

    def _contract_batch_all_spec(
        self,
        target_sampler_output: SamplerOutput,
        proposals: SpeculativeProposals,
    ) -> SpeculativeScores:
        """Contract the expanded batch back into its original size.
        This maps the scores of speculative tokens back to their original
        sequences.

        It assumes all sequences in the batch were previously expanded.
        """

        # Map distinct sequences used to score each token
        # of shape [batch_size * k + 1] back to [batch_size, k + 1].
        contracted_bs, k = proposals.proposal_token_ids.shape

        # Reshape tensors to original batch size
        target_token_ids = target_sampler_output.sampled_token_ids.reshape(
            contracted_bs, k + 1)
        target_probs = target_sampler_output.sampled_token_probs.reshape(
            *target_token_ids.shape, self._vocab_size)
        target_logprobs = target_sampler_output.logprobs.reshape(
            target_probs.shape)
        target_hidden_states = target_sampler_output.hidden_states
        if target_hidden_states is not None:
            target_hidden_states = target_hidden_states.reshape(
                *target_token_ids.shape, target_hidden_states.shape[-1])

        return SpeculativeScores(probs=target_probs,
                                 token_ids=target_token_ids,
                                 logprobs=target_logprobs,
                                 hidden_states=target_hidden_states,
                                 prompt_logprobs=None)

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
        assert len(input_seq_group_metadata.seq_data) == 1, (
            "Beam search "
            "not supported in speculative decoding")
        input_seq_id = next(iter(input_seq_group_metadata.seq_data.keys()))

        token_ids_to_score = self._get_token_ids_to_score(
            proposal_token_ids[batch_index])

        sampling_params = input_seq_group_metadata.sampling_params
        target_seq_group_metadata_list: List[SequenceGroupMetadata] = []
        for i, token_ids in enumerate(token_ids_to_score):
            target_seq_group_metadata_list.append(
                self._create_single_target_seq_group_metadata(
                    input_seq_group_metadata,
                    input_seq_id,
                    next(target_seq_ids_iter),
                    token_ids,
                    sampling_params=sampling_params,
                ))

        return target_seq_group_metadata_list

    @staticmethod
    def _create_single_target_seq_group_metadata(
        seq_group_metadata: SequenceGroupMetadata,
        seq_id: SeqId,
        target_seq_id: TargetSeqId,
        token_ids: List[TokenId],
        sampling_params: SamplingParams,
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
        prompt_token_ids = seq_data.prompt_token_ids_array
        new_output_token_ids = [*seq_data.get_output_token_ids(), *token_ids]
        mrope_position_delta = seq_data.mrope_position_delta

        new_seq_data_dict = {
            target_seq_id:
            SequenceData(
                prompt_token_ids,
                _output_token_ids=array(VLLM_TOKEN_ID_ARRAY_TYPE,
                                        new_output_token_ids),
            ),
        }
        # This is a hack. Technically, spec decoding should compute
        # num_lookahead slots at one shot, but instead, it expands the batch
        # and evaluate one by one right now. context_len is seq_len - 1 because
        # the kv cache is filled by a previous batch in the batch expansion.
        for data in new_seq_data_dict.values():
            data.update_num_computed_tokens(data.get_len() - 1)
            data.mrope_position_delta = mrope_position_delta

        return SequenceGroupMetadata(
            request_id=seq_group_metadata.request_id,
            is_prompt=seq_group_metadata.is_prompt,
            seq_data=new_seq_data_dict,
            sampling_params=sampling_params,
            block_tables={
                target_seq_id: seq_group_metadata.block_tables[seq_id],
            },
            lora_request=None,
            token_chunk_size=1,
        )

    @staticmethod
    def _split_scoring_output(
        sampler_output: SamplerOutput, num_scoring_tokens: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               Optional[torch.Tensor], torch.Tensor, torch.Tensor,
               torch.Tensor, Optional[torch.Tensor]]:
        """Split the target model output into speculative and non-speculative
        output.
        """

        # vLLM currently only supports proposal lens equal to zero or the batch
        # proposal len. This adds some complexity (splitting the batch into spec
        # and non spec sequences) and should be removed in the future. It can be
        # done by supporting per-sequence proposal lens.
        #
        # First samples are non-speculative, latter samples are from speculative
        # scoring (prefill|decode order).
        split_sizes = (sampler_output.sampled_token_ids.numel() -
                       num_scoring_tokens, num_scoring_tokens)
        (non_spec_probs,
         spec_probs) = sampler_output.sampled_token_probs.split(split_sizes)
        (non_spec_sampled_tokens, spec_sampled_tokens
         ) = sampler_output.sampled_token_ids.flatten().split(split_sizes)
        (non_spec_logprobs,
         spec_logprobs) = sampler_output.logprobs.split(split_sizes)

        if sampler_output.hidden_states is not None:
            (non_spec_hidden_states, spec_hidden_states
             ) = sampler_output.hidden_states.split(split_sizes)
        else:
            non_spec_hidden_states, spec_hidden_states = None, None

        return (spec_sampled_tokens, spec_probs, spec_logprobs,
                spec_hidden_states, non_spec_sampled_tokens, non_spec_probs,
                non_spec_logprobs, non_spec_hidden_states)

    @staticmethod
    def _create_target_seq_id_iterator(
            seq_ids: List[SeqId]) -> Iterator[TargetSeqId]:
        """Create an iterator for creating target sequence ids.
        Target sequence ids are distinct from sequence ids because we create a
        distinct target sequence id for each proposal token to be scored.

        This implementation increments a counter starting at 1 + max of all
        provided input sequence ids.
        """
        return count(start=max(seq_ids) + 1)

    @staticmethod
    def _get_token_ids_to_score(
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
        token_ids_to_score.extend(full_spec_token_ids[:i + 1]
                                  for i in range(len(full_spec_token_ids)))
        return token_ids_to_score
