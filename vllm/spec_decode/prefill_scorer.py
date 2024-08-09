from copy import deepcopy
from itertools import chain, count
from typing import Iterator, List, Tuple

import torch

from vllm.sequence import (
    ExecuteModelRequest,
    SamplerOutput,
    SequenceData,
    SequenceGroupMetadata,
    SequenceGroupState,
    SequenceStage,
    get_all_seq_ids,
)
from vllm.spec_decode.interfaces import (
    SpeculativeProposals,
    SpeculativeScorer,
    SpeculativeScores,
)
from vllm.spec_decode.util import (
    nvtx_range,
    sampler_output_to_torch,
    split_batch_by_proposal_len,
)
from vllm.worker.worker_base import WorkerBase

SeqId = int
TargetSeqId = int
TokenId = int


class PrefillTop1Scorer(SpeculativeScorer):
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

    def __init__(self, scorer_worker: WorkerBase, device: str, vocab_size: int):
        self._scorer_worker = scorer_worker
        self._device = device
        self._vocab_size = vocab_size

    @nvtx_range("PrefillTop1Scorer.score_proposals")
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
            proposals for proposals in proposal_token_ids_list if -1 not in proposals
        ]

        prefill_seq_group_metadata_list = []
        for i, seq_group_metadata in enumerate(
            execute_model_req.seq_group_metadata_list
        ):
            _seq_group_metadata = deepcopy(seq_group_metadata)
            _seq_data = {}
            for k, v in _seq_group_metadata.seq_data.items():
                prompt_token_ids = list(v.prompt_token_ids)

                prompt_token_ids.extend(v.output_token_ids)

                # we need to compute the last output token
                num_computed_tokens = len(prompt_token_ids) - 1

                prompt_token_ids.extend(proposal_token_ids_list_without_skips[i])
                v.prompt_token_ids = prompt_token_ids
                v.output_token_ids = []
                # v.stage = SequenceStage.PREFILL
                # num_computed_tokens = v.get_num_computed_tokens()
                v.reset_state_for_recompute()
                v.update_num_computed_tokens(num_computed_tokens)

                _seq_data[k] = v
                # _seq_data[k].update_num_computed_tokens(num_computed_tokens)

            _seq_group_metadata = SequenceGroupMetadata(
                request_id=seq_group_metadata.request_id,
                is_prompt=True,
                seq_data=_seq_data,
                sampling_params=seq_group_metadata.sampling_params,
                block_tables=seq_group_metadata.block_tables,
                do_sample=seq_group_metadata.do_sample,
                pooling_params=seq_group_metadata.pooling_params,
                token_chunk_size=len(prompt_token_ids) - num_computed_tokens,
            )

            _seq_group_metadata.sampling_params.prompt_logprobs = 1

            prefill_seq_group_metadata_list.append(_seq_group_metadata)

        _execute_model_req = execute_model_req.clone(
            seq_group_metadata_list=prefill_seq_group_metadata_list
        )

        target_sampler_output = self._scorer_worker.execute_model(
            execute_model_req=_execute_model_req
        )
        assert len(target_sampler_output) == 1, "expected single-step output"
        target_sampler_output = target_sampler_output[0]

        # TODO
        num_spec_tokens = len(proposal_token_ids_list_without_skips[0])

        all_probs = target_sampler_output.sampled_token_probs[
            -(1 + num_spec_tokens) :, :
        ]

        all_probs.unsqueeze_(0)
        all_tokens = all_probs.argmax(dim=-1)

        all_probs = torch.zeros_like(all_probs)
        # TODO refactor
        for i, t in enumerate(all_tokens[0]):
            all_probs[0, i, t] = 1.0

        # all_probs.max(dim=-1) must be 1.0 here

        spec_logprobs = torch.log(1e-6 + all_probs)

        return SpeculativeScores(
            probs=all_probs,
            token_ids=all_tokens,
            logprobs=spec_logprobs,
            hidden_states=target_sampler_output.hidden_states,
        )

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
                )
                for i, seq_group_metadata in enumerate(seq_group_metadata_list)
            )
        )

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
            "Speculating on " "prompts not yet supported"
        )
        assert len(input_seq_group_metadata.seq_data) == 1, (
            "Beam search " "not supported in speculative decoding"
        )
        input_seq_id = next(iter(input_seq_group_metadata.seq_data.keys()))

        token_ids_to_score = self._get_token_ids_to_score(
            proposal_token_ids[batch_index]
        )

        target_seq_group_metadata_list: List[SequenceGroupMetadata] = []
        for token_ids in token_ids_to_score:
            target_seq_group_metadata_list.append(
                self._create_single_target_seq_group_metadata(
                    input_seq_group_metadata,
                    input_seq_id,
                    next(target_seq_ids_iter),
                    token_ids,
                )
            )

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
            target_seq_id: SequenceData(
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

        if (
            seq_group_metadata.state is not None
            and seq_group_metadata.state.generator is not None
        ):
            generator = torch.Generator(
                device=seq_group_metadata.state.generator.device
            )
            generator.set_state(seq_group_metadata.state.generator.get_state())
            state = SequenceGroupState(generator=generator)
        else:
            state = None

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
            state=state,
        )

    def _split_scoring_output(
        self, sampler_output: SamplerOutput, num_scoring_tokens: int
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
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
            sampler_output.sampled_token_ids.numel() - num_scoring_tokens,
        ]
        (spec_probs, non_spec_probs) = sampler_output.sampled_token_probs.split(
            split_sizes
        )
        (spec_sampled_tokens, non_spec_sampled_tokens) = (
            sampler_output.sampled_token_ids.flatten().split(split_sizes)
        )
        (
            spec_logprobs,
            non_spec_logprobs,
        ) = sampler_output.logprobs.split(split_sizes)

        # Convert scores to tensors.
        sampler_output.sampled_token_probs = spec_probs
        sampler_output.sampled_token_ids = spec_sampled_tokens
        sampler_output.logprobs = spec_logprobs
        (target_token_ids, target_probs, target_logprobs) = sampler_output_to_torch(
            [sampler_output], True
        )

        # Convert non-speculative output tokens to tensors.
        sampler_output.sampled_token_probs = non_spec_probs
        sampler_output.sampled_token_ids = non_spec_sampled_tokens
        sampler_output.logprobs = non_spec_logprobs
        (non_spec_target_token_ids, non_spec_target_probs, non_spec_target_logprobs) = (
            sampler_output_to_torch([sampler_output], True)
        )

        return (
            target_token_ids,
            target_probs,
            target_logprobs,
            non_spec_target_token_ids,
            non_spec_target_probs,
            non_spec_target_logprobs,
        )

    def _create_target_seq_id_iterator(
        self, seq_ids: List[SeqId]
    ) -> Iterator[TargetSeqId]:
        """Create an iterator for creating target sequence ids.
        Target sequence ids are distinct from sequence ids because we create a
        distinct target sequence id for each proposal token to be scored.

        This implementation increments a counter starting at 1 + max of all
        provided input sequence ids.
        """
        return count(start=max(seq_ids) + 1)

    def _get_token_ids_to_score(
        self, full_spec_token_ids: List[TokenId]  # shape: [k]
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
        token_ids_to_score.extend(
            [full_spec_token_ids[: i + 1] for i in range(len(full_spec_token_ids))]
        )
        return token_ids_to_score
