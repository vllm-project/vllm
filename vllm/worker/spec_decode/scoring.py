from typing import Iterator, List, Tuple, Optional, Union, Dict
from itertools import chain, count
from functools import cached_property
import logging
import time
from dataclasses import dataclass

import torch
import traceback

from vllm.worker.spec_decode.metrics import SpecDecodeWorkerMetrics, AsyncMetricsCollector
from vllm.sequence import (SamplerOutput, SequenceGroupMetadata, SequenceData,
                           SequenceGroupOutput, SequenceOutput)
from vllm.worker.worker import Worker
from vllm.worker.spec_decode.multi_step_worker import MultiStepWorker
from vllm.model_executor.layers.rejection_sampler import RejectionSampler
from vllm.model_executor.parallel_utils.parallel_state import get_tensor_model_parallel_group
from vllm.config import CacheConfig
from vllm.utils import in_wsl
from vllm.worker.spec_decode.util import nvtx_range, sampler_output_to_torch, get_all_seq_ids
from vllm.worker.spec_decode.interfaces import SpeculativeScorer, SpeculativeProposals, SpeculativeScores

SeqId = int
TargetSeqId = int
TokenId = int

class BatchExpansionTop1Scorer(SpeculativeScorer):
    def __init__(self, scorer_worker: Worker, device: str, vocab_size: int):
        self._scorer_worker = scorer_worker
        self._device = device
        self._vocab_size = vocab_size
        

    @nvtx_range("BatchExpansionTop1Scorer.score_proposals")
    def score_proposals(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata], blocks_to_swap_in: Optional[Dict[int, int]],
        blocks_to_swap_out: Optional[Dict[int, int]],
        blocks_to_copy: Optional[Dict[int, List[int]]],
        k: int,
        proposals: SpeculativeProposals,
    ) -> SpeculativeScores:
        """Score the proposed tokens via the target model.

        This converts each input sequence to a set of k+1 target sequences. The
        target sequences have the unique continuations to be scored and a
        unique sequence ID that is different from all input sequence ids.

        This adds overhead and should be removed. It is done because the sampler
        currently operates on sequences instead of queries.
        """

        # TODO(cade) perform this on GPU to remove blocking call.
        proposal_lens_list = proposals.proposal_lens.tolist()
        proposal_token_ids_list = proposals.proposal_token_ids.tolist()

        spec_indices, non_spec_indices, target_seq_group_metadata_list, num_scoring_tokens = self._expand_batch(
            seq_group_metadata_list=seq_group_metadata_list,
            proposal_token_ids_list=proposal_token_ids_list,
            proposal_lens_list=proposal_lens_list,
        )

        target_sampler_output = self._scorer_worker.execute_model(
            seq_group_metadata_list=target_seq_group_metadata_list,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            return_python_output=False)
        
        all_tokens, all_probs = self._contract_batch(
            original_bs=len(seq_group_metadata_list),
            target_sampler_output=target_sampler_output, 
            proposals=proposals,
            num_scoring_tokens=num_scoring_tokens, non_spec_indices=non_spec_indices, spec_indices=spec_indices, k=k,
        )

        return SpeculativeScores(
            probs=all_probs,
            token_ids=all_tokens,
        )

    def _expand_batch(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        proposal_token_ids_list: List[int],
        proposal_lens_list: List[int],
    ) -> Tuple[List[int], List[int], List[SequenceGroupMetadata], int]:

        spec_seqs = [seq_group for seq_group, proposal_len in zip(seq_group_metadata_list, proposal_lens_list) if proposal_len != 0]
        spec_indices = [i for i, (_, proposal_len) in enumerate(zip(seq_group_metadata_list, proposal_lens_list)) if proposal_len != 0]

        non_spec_seqs = [seq_group for seq_group, proposal_len in zip(seq_group_metadata_list, proposal_lens_list) if proposal_len == 0]
        non_spec_indices = [i for i, (_, proposal_len) in enumerate(zip(seq_group_metadata_list, proposal_lens_list)) if proposal_len == 0]

        # Convert to target sequence ids.
        target_seq_group_metadata_list = self._create_scoring_model_input(
            spec_seqs, proposal_token_ids_list)
        num_scoring_tokens = len(target_seq_group_metadata_list)
        target_seq_group_metadata_list.extend(non_spec_seqs)

        return spec_indices, non_spec_indices, target_seq_group_metadata_list, num_scoring_tokens

    def _contract_batch(
        self,
        original_bs: int,
        target_sampler_output: List[SamplerOutput],
        proposals: SpeculativeProposals,
        num_scoring_tokens: int, non_spec_indices: List[int], spec_indices: List[int], k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (target_token_ids, target_probs,
         non_spec_target_token_ids, non_spec_target_probs) = self._split_scoring_output(
             target_sampler_output, num_scoring_tokens)

        # Map distinct sequences used to score each token
        # of shape [batch_size * k + 1] back to [batch_size, k + 1].
        batch_size, k = proposals.proposal_token_ids.shape

        target_token_ids = target_token_ids.squeeze().reshape(
            batch_size, k + 1)
        target_probs = target_probs.squeeze().reshape(batch_size, k + 1,
                                                      self._vocab_size)

        # shape: [batch_size, 1]
        bonus_token_ids = target_token_ids[:, -1:]

        # shape: [batch_size, k, vocab_size]
        proposal_scores = target_probs[:, :-1]
        
        all_tokens = torch.ones(original_bs, k + 1, device=self._device, dtype=torch.long) * -1
        all_probs = torch.zeros(original_bs, k + 1, self._vocab_size, device=self._device, dtype=torch.float32)
            
        if non_spec_indices:
            all_tokens[non_spec_indices, 0] = non_spec_target_token_ids
            all_probs[non_spec_indices, 1:, :] = non_spec_target_probs

        if spec_indices:
            all_tokens[spec_indices] = target_token_ids
            all_probs[spec_indices] = target_probs

        return all_tokens, all_probs


    def _create_scoring_model_input(
            self,
            seq_group_metadata_list: List[SequenceGroupMetadata],
            proposal_token_ids: List[List[int]],  # shape: [batch_size, k]
    ) -> List[SequenceGroupMetadata]:
        """Given the original input sequences and proposed tokens from the draft
        model, create a list of target sequences that can be used for scoring.
        """

        if not seq_group_metadata_list:
            return []

        target_seq_ids_iter = self._create_target_seq_id_iterator(
            get_all_seq_ids(seq_group_metadata_list))

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
        proposal_token_ids: List[int],  # shape: [batch_size, k]
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

        return SequenceGroupMetadata(
            request_id=seq_group_metadata.request_id,
            is_prompt=seq_group_metadata.is_prompt,
            seq_data={
                target_seq_id:
                SequenceData(
                    prompt_token_ids=prompt_token_ids,
                    output_token_ids=new_output_token_ids,
                ),
            },
            sampling_params=seq_group_metadata.sampling_params,
            block_tables={
                target_seq_id: seq_group_metadata.block_tables[seq_id],
            },
            lora_request=None,
        )

    def _split_scoring_output(
        self, sampler_output: SamplerOutput, num_scoring_tokens: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split the target model output into speculative and non-speculative
        output.
        """

        # First samples are from speculative scoring, latter samples are non-
        # speculative samples.
        split_sizes = [
            num_scoring_tokens,
            sampler_output.sampled_token_ids.numel() - num_scoring_tokens
        ]
        (spec_probs, non_spec_probs) = sampler_output.sampled_token_probs.split(split_sizes)
        (spec_sampled_tokens, non_spec_sampled_tokens
         ) = sampler_output.sampled_token_ids.flatten().split(split_sizes)

        # Convert scores to tensors.
        sampler_output.sampled_token_probs = spec_probs
        sampler_output.sampled_token_ids = spec_sampled_tokens
        target_token_ids, target_probs = sampler_output_to_torch(
            [sampler_output])

        # Convert non-speculative output tokens to tensors.
        sampler_output.sampled_token_probs = non_spec_probs
        sampler_output.sampled_token_ids = non_spec_sampled_tokens
        non_spec_target_token_ids, non_spec_target_probs = sampler_output_to_torch(
            [sampler_output])

        return target_token_ids, target_probs, non_spec_target_token_ids, non_spec_target_probs

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
            full_spec_token_ids: List[int]  # shape: [k]
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
        empty_token_ids = []

        token_ids_to_score = [empty_token_ids]
        token_ids_to_score.extend([
            full_spec_token_ids[:i + 1]
            for i in range(len(full_spec_token_ids))
        ])
        return token_ids_to_score


