from typing import Iterator, List, Tuple, Optional, Union, Dict
from itertools import chain, count
from functools import cached_property
import logging
import time
from dataclasses import dataclass

#import msgspec
import torch
import traceback

from vllm.worker.spec_decode.metrics import DraftTargetWorkerMetrics, AsyncMetricsCollector
#from vllm.anyscale.shm.msgspec_shm import SharedMsgspecBufferWithEvent
#from vllm.sequence import (SampleLogprob, SamplerOutput, SequenceGroupMetadata,
#                           ExecuteModelData, SequenceOutputs, SequenceData,
#                           SequenceGroupOutputs, DraftTargetWorkerMetrics)
from vllm.sequence import (SamplerOutput, SequenceGroupMetadata, SequenceData,
                           SequenceGroupOutput, SequenceOutput)
from vllm.worker.worker import Worker
from vllm.worker.spec_decode.multi_step_worker import MultiStepWorker
#from vllm.worker.prompt_lookup_worker import PromptLookupWorker
#from vllm.worker.single_tp_worker import SingleTpWorker
#from vllm.model_executor.layers.sampler import sampler_output_to_torch
from vllm.model_executor.layers.rejection_sampler import RejectionSampler
from vllm.model_executor.parallel_utils.parallel_state import get_tensor_model_parallel_group
from vllm.config import CacheConfig
#from vllm.worker.base_worker import BaseWorker
#from vllm.model_executor.layers.sampler import RawSamplerOutput
from vllm.utils import in_wsl
from vllm.worker.spec_decode.util import nvtx_range, sampler_output_to_torch

SeqId = int
TargetSeqId = int
TokenId = int

logger = logging.getLogger(__name__)

"""
Run spec step (ExecuteModelData -> AcceptedTokens)
- get proposals (ExecuteModelData -> Top1Proposals)
- score proposals ((ExecuteModelData, Top1Proposals) -> Top1Scores)
- verify proposals ((ExecuteModelData, Top1Proposals, Top1Scores) -> AcceptedTokens)

Missing pieces:
- get_proposals in DraftTargetWorker --> currently all mocked
- Tests cover vertical functionality.

I can start by writing get_proposals without batch expansion, write tests.
Then abstract out score proposals batch expansion. Move tests over to batch expander.
Then abstract out Top1Scores and rejection sampler interface. Fix tests.


"""

@dataclass
class Top1Proposals:
    proposal_token_ids: torch.Tensor
    proposal_token_probs: torch.Tensor

    # not sure
    proposal_lens: torch.Tensor

@dataclass
class Top1Scores:
    # TODO: how to represent bonus token ?
    # current thinking ->
    #   always have bonus token such that num_scored_tokens == k+1.
    token_probs: torch.Tensor

@dataclass
class AcceptedTokens:
    accepted_tokens: torch.Tensor
    logprobs: torch.Tensor

class Top1ProposerWorker:

    def init_model(self):
        pass
        # etc.

    def get_proposals(
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
        blocks_to_swap_in: Optional[Dict[int, int]],
        blocks_to_swap_out: Optional[Dict[int, int]],
        blocks_to_copy: Optional[Dict[int, List[int]]],
        num_spec_tokens: int,
    ) -> Top1Proposals:
        """
        - determine k
        - drop k=0
        - fwd pass
        - create output
        """
        pass
        """
        New thinking:
        This should go directly into the multi step worker.
        The multi step worker should be in charge of dropping k=0 sequences.
        The multi step worker should be in charge of returning blocks of output
            that have the correct format, e.g. sequences in same order.

        so, we have a MultiStepWorker, MultiStepTop1ProposerWorker

        how about only multi step worker.
        -> the return type is just proposals; top1 is a special case of topk
        -> to have variable k, the proposals should also know about proposal lens
        -> that means that the step output should be "padded" if a sequence ends
        -> / stops proposing
        -> 
        -> why do I need variable k now? because I want to refactor the k=0 case
        -> to be within the worker itself. why? because it will decouple proposals
        -> from batch expansion.
        ->
        -> do I need to figure out this new padding stuff now? can wait for Lily
        -> not really.. the alternative is to support only k=0 and k=k, then do
        -> similar indexing tricks to meet the interface. ("similar indexing tricks"
        -> means remove k=0 from batch, do fwd pass, place back in batch)
        ->
        -> So, I am focusing on expressing k=0 and k=k in a variable k framework, but
        -> not doing that work.
        ->
        -> So, what do I need? the multi step worker needs to have get_proposals.
        -> it will do the k=0 removal code, multi step fwd pass, then create proposal
        -> output (with k=0 and k=k).
        ->
        -> Last question, where does this logic live? it can live in multi step worker
        -> class, or it can be decoupled in separate class. Why would we decouple?
        -> code cleanliness (but complexity :(). any other reason?
        ->
        -> it should live in the worker as only the worker knows how many proposals
        -> there will be

        Now, next step:
        * write test for multi step worker get_proposals which ensures correct output
            format
        * write test for multi step worker get_proposals which ensures mixed
            k=0/k=k supported
        * write test for multi_step_worker get_proposals which ensures only k=0
            supported
        """

class Top1ScorerWorker:
    def init_model(self):
        pass
        # etc.

    def score_proposals(
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
        blocks_to_swap_in: Optional[Dict[int, int]],
        blocks_to_swap_out: Optional[Dict[int, int]],
        blocks_to_copy: Optional[Dict[int, List[int]]],
        num_spec_tokens: int,
        proposals: Top1Proposals,
    ) -> Top1Scores:
        """
        - extract k=0
        - batch expand k>0
        - combine
        - fwd pass
        - batch contract
        """
        pass

class Top1Verifier:
    def init_gpu_tensors(self):
        pass

    def verify_proposals(
        #seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
        #blocks_to_swap_in: Optional[Dict[int, int]],
        #blocks_to_swap_out: Optional[Dict[int, int]],
        #blocks_to_copy: Optional[Dict[int, List[int]]],
        #num_spec_tokens: int,
        proposals: Top1Proposals,
        scores: Top1Scores,
    ) -> AcceptedTokens:
        pass

class DraftTargetWorker:

    def __init__(
        self,
        draft_worker: MultiStepWorker,
        #draft_worker: Union[MultiStepWorker, SingleTpWorker,
        #                    PromptLookupWorker],
        target_worker: Worker,
        rejection_sampler: RejectionSampler,
        metrics_collector: Optional["AsyncMetricsCollector"] = None,
    ):
        """
        Create a DraftTargetWorker.

        Args:
            draft_worker: A draft worker that can run multiple steps
                in a row.
            target_worker: The normal worker that is used for scoring.
                It should contain the target model.
            rejection_sampler: A Torch module used to perform modified rejection
                sampling for speculative decoding.
        """
        self.draft_worker = draft_worker
        self.target_worker = target_worker
        self.rejection_sampler = rejection_sampler

        self.device = None
        self._metrics = AsyncMetricsCollector(
            rejection_sampler
        ) if metrics_collector is None else metrics_collector

        self.probs_dtype = self.rejection_sampler.probs_dtype
        self.token_id_dtype = self.rejection_sampler.token_id_dtype

        #self._profiler = TorchProfiler()

    def init_model(self) -> None:
        # Initialize the target model before the draft model.
        # This allows the draft model to have a smaller TP degree than the
        # larger model without refactors to parallel_state.
        self.target_worker.init_model()
        self.draft_worker.init_model()

        self._configure_samplers()

        self.device = self.target_worker.device

        self._metrics.init_gpu_tensors(self.rank)
        self.rejection_sampler.init_gpu_tensors(self.rank)

    def profile_num_available_blocks(self, block_size: int,
                                     gpu_memory_utilization: float,
                                     cpu_swap_space: int) -> Tuple[int, int]:
        num_gpu_blocks, num_cpu_blocks = (
            self.target_worker.profile_num_available_blocks(
                block_size, gpu_memory_utilization, cpu_swap_space))

        target_kv_size_bytes = self.target_worker.get_kv_size_bytes(block_size)
        draft_kv_size_bytes = self.draft_worker.get_kv_size_bytes(block_size)

        new_num_gpu_blocks = calculate_gpu_blocks(target_kv_size_bytes,
                                                  draft_kv_size_bytes,
                                                  num_gpu_blocks)
        return new_num_gpu_blocks, num_cpu_blocks

    def init_cache_engine(self, cache_config: CacheConfig):
        self.target_worker.init_cache_engine(cache_config)
        self.draft_worker.init_cache_engine(cache_config)

    def _configure_samplers(self):
        """Configure model samplers to return a probability tensor in the
        SamplerOutput. This simplifies the data wrangling logic in speculative
        decoding.
        """
        self.draft_worker.include_gpu_probs_tensor()
        self.target_worker.include_gpu_probs_tensor()

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
        blocks_to_swap_in: Optional[Dict[int, int]],
        blocks_to_swap_out: Optional[Dict[int, int]],
        blocks_to_copy: Optional[Dict[int, List[int]]],
        num_spec_tokens: int,
    ) -> Optional[SamplerOutput]:

        k = num_spec_tokens

        if k == 0 or len(seq_group_metadata_list) == 0:
            return self._run_no_spec(
                seq_group_metadata_list,
                blocks_to_swap_in,
                blocks_to_swap_out,
                blocks_to_copy,
            )

        return self._run_speculative_decoding_step(
            seq_group_metadata_list,
            blocks_to_swap_in,
            blocks_to_swap_out,
            blocks_to_copy,
            k,
        )

    @nvtx_range("draft_target_worker._run_no_spec")
    def _run_no_spec(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
        blocks_to_swap_in: Optional[Dict[int, int]],
        blocks_to_swap_out: Optional[Dict[int, int]],
        blocks_to_copy: Optional[Dict[int, List[int]]],
        #execute_model_data: ExecuteModelData
    ) -> List[SamplerOutput]:
        """Run a prefill step, without any speculation. The input is sent to the
        draft and target model so that prompt KV are stored in both caches.

        TODO update
        """

        self.draft_worker.execute_model(seq_group_metadata_list,
                                        blocks_to_swap_in,
                                        blocks_to_swap_out,
                                        blocks_to_copy,
                                        return_python_output=False)

        sampler_output = self.target_worker.execute_model(
            seq_group_metadata_list,
            blocks_to_swap_in,
            blocks_to_swap_out,
            blocks_to_copy,
        )

        # Do not want PyTorch tensors transferred back.
        sampler_output.probs = None
        sampler_output.sampled_tokens = None
        return [sampler_output]

    @nvtx_range("draft_target_worker._run_speculative_decoding_step")
    def _run_speculative_decoding_step(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
        blocks_to_swap_in: Optional[Dict[int, int]],
        blocks_to_swap_out: Optional[Dict[int, int]],
        blocks_to_copy: Optional[Dict[int, List[int]]],
        k: int,
    ) -> List[SamplerOutput]:
        """Execute a single step of speculative decoding.

        This runs the draft model k times, then scores each token using the
        target model. Rejection sampling is performed on the draft and target
        outputs to determine which tokens can be accepted without modifying the
        true distribution.

        Args:
            execute_model_data: The input sequences that will be speculated
                upon.
            k: A hyperparameter integer dictating how many tokens to speculate.
                Given some k, this will return a number of tokens per sequence
                in the interval [1, k+1], depending on how many tokens are
                accepted.

        Returns:
            A List of SamplerOutput, as if the target worker were simply called
            multiple times.
        """

        logger.debug(f"running draft model for {k=} steps")

        draft_max_model_len, target_max_model_len = self._get_max_model_len()
        max_model_len = min(draft_max_model_len, target_max_model_len)

        # Generate proposals using draft worker.
        proposals = self.draft_worker.get_spec_proposals(
            seq_group_metadata_list, blocks_to_swap_in, blocks_to_swap_out,
            blocks_to_copy, k, max_model_len)

        logger.debug("scoring draft tokens")
        (proposal_scores, bonus_token_ids,
         non_spec_token_ids) = self._score_proposals(
             seq_group_metadata_list,
             blocks_to_swap_in,
             blocks_to_swap_out,
             blocks_to_copy,
             k,
             #execute_model_data,
             proposals.proposal_token_ids,
             proposals.spec_seqs,
             proposals.non_spec_seqs)

        with nvtx_range("draft_target_worker.rejection_sampler"):
            accepted_token_ids = self.rejection_sampler(
                proposal_scores,
                bonus_token_ids,
                proposals.proposal_probs,
                proposals.proposal_token_ids,
            )

        # Append output tokens from non-speculative sequences to
        # the accepted token ids tensor.
        non_spec_token_ids = non_spec_token_ids.expand(-1, k + 1).clone()
        non_spec_token_ids[:, 1:] = -1
        accepted_token_ids = torch.cat(
            [accepted_token_ids, non_spec_token_ids])

        # Rearrange so that results are in the order of the original seq group
        # metadata.
        accepted_token_ids[
            proposals.original_indices] = accepted_token_ids.clone()

        # Construct output.
        seq_ids = self._get_all_seq_ids(proposals.all_seqs)
        sampler_output = self._create_output_sampler_list(
            seq_ids, accepted_token_ids)

        maybe_rejsample_metrics = self._metrics.maybe_collect_rejsample_metrics(
            k)
        if maybe_rejsample_metrics is not None:
            sampler_output[
                0].draft_target_worker_metrics = maybe_rejsample_metrics

        return sampler_output

    def _get_max_model_len(self) -> Tuple[int, int]:
        draft_max_model_len = self.draft_worker.max_model_len
        target_max_model_len = self.target_worker.max_model_len

        assert draft_max_model_len is not None
        assert target_max_model_len is not None

        return draft_max_model_len, target_max_model_len

    def _get_all_seq_ids(
            self, seq_group_metadata_list: List[SequenceGroupMetadata]
    ) -> List[SeqId]:
        """Given a list of SequenceGroupMetadata, create a list of all
        sequence ids.
        """
        return list(
            chain.from_iterable([
                seq_group_metadata.seq_data.keys()
                for seq_group_metadata in seq_group_metadata_list
            ]))

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

    @nvtx_range("draft_target_worker._score_proposals")
    def _score_proposals(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
        blocks_to_swap_in: Optional[Dict[int, int]],
        blocks_to_swap_out: Optional[Dict[int, int]],
        blocks_to_copy: Optional[Dict[int, List[int]]],
        k: int,

        #execute_model_data: ExecuteModelData,
        proposal_token_ids: torch.Tensor,  # shape: [batch_size, k]
        spec_seqs: List[SequenceGroupMetadata],
        non_spec_seqs: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score the proposed tokens via the target model.

        This converts each input sequence to a set of k+1 target sequences. The
        target sequences have the unique continuations to be scored and a
        unique sequence ID that is different from all input sequence ids.

        This adds overhead and should be removed. It is done because the sampler
        currently operates on sequences instead of queries.
        """
        # Convert to target sequence ids.
        target_seq_group_metadata_list = self._create_scoring_model_input(
            spec_seqs, proposal_token_ids)

        num_scoring_tokens = len(target_seq_group_metadata_list)
        target_seq_group_metadata_list.extend(non_spec_seqs)

        # Score proposal token ids.
        target_sampler_output = self.target_worker.execute_model(
            seq_group_metadata_list=target_seq_group_metadata_list,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            #ExecuteModelData(
            #    target_seq_group_metadata_list,
            #    execute_model_data.finished_request_ids_list,
            #    execute_model_data.blocks_to_swap_in,
            #    execute_model_data.blocks_to_swap_out,
            #    execute_model_data.blocks_to_copy,
            #    num_preallocated_slots=0,
            #),
            return_python_output=False)

        (target_token_ids, target_probs,
         non_spec_target_token_ids) = self._split_scoring_output(
             target_sampler_output, num_scoring_tokens)

        # Map distinct sequences used to score each token
        # of shape [batch_size * k + 1] back to [batch_size, k + 1].
        batch_size, k = proposal_token_ids.shape

        target_token_ids = target_token_ids.squeeze().reshape(
            batch_size, k + 1)
        target_probs = target_probs.squeeze().reshape(batch_size, k + 1,
                                                      self._vocab_size)

        # shape: [batch_size, 1]
        bonus_token_ids = target_token_ids[:, -1:]

        # shape: [batch_size, k, vocab_size]
        proposal_scores = target_probs[:, :-1]

        return proposal_scores, bonus_token_ids, non_spec_target_token_ids

    def _create_scoring_model_input(
            self,
            seq_group_metadata_list: List[SequenceGroupMetadata],
            proposal_token_ids: torch.Tensor,  # shape: [batch_size, k]
    ) -> List[SequenceGroupMetadata]:
        """Given the original input sequences and proposed tokens from the draft
        model, create a list of target sequences that can be used for scoring.
        """

        # TODO(cade) perform this on GPU to remove blocking call.
        proposal_token_ids = proposal_token_ids.tolist()

        if not seq_group_metadata_list:
            return []

        target_seq_ids_iter = self._create_target_seq_id_iterator(
            self._get_all_seq_ids(seq_group_metadata_list))

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

        # The first scoring sequence will include the normal number of
        # processed tokens. This allows it to write KV from the previous
        # iteration.
        #
        # Subsequent scoring sequences only include a single unprocessed token;
        # the token they score.
        #if len(token_ids) == 0:
        #    num_processed_token_ids = seq_data.get_num_processed_token_ids()
        #else:
        #    num_processed_token_ids = seq_data.get_len() + len(token_ids) - 1
        num_processed_token_ids = seq_data.get_len() + len(token_ids) - 1

        return SequenceGroupMetadata(
            request_id=seq_group_metadata.request_id,
            is_prompt=seq_group_metadata.is_prompt,
            #is_chunked_prefill=seq_group_metadata.is_chunked_prefill,
            seq_data={
                target_seq_id:
                SequenceData.from_anyscale_sd(
                    token_ids=prompt_token_ids + new_output_token_ids,
                    num_prompt_tokens=len(prompt_token_ids),
                    # Support for tracking cumulative logprob not yet
                    # implemented.
                    cumulative_logprob=0.0,
                    num_processed_token_ids=num_processed_token_ids,
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        non_spec_target_token_ids, _ = sampler_output_to_torch(
            [sampler_output])

        return target_token_ids, target_probs, non_spec_target_token_ids

    def _create_output_sampler_list(
        self,
        seq_ids: List[SeqId],
        accepted_token_ids: torch.Tensor  # shape: [batch_size, k+1]
    ) -> List[SamplerOutput]:
        """Given the accepted token ids, create a list of SamplerOutput.

        The output is padded with -1 tokens such that each sequence has
        the same number of outputs.
        """
        # shape: [k+1, batch_size]
        accepted_token_ids_by_step = accepted_token_ids.transpose(0,
                                                                  1).tolist()
        sampler_output_list = []
        for token_ids_by_step in accepted_token_ids_by_step:
            if all(token_id == -1 for token_id in token_ids_by_step):
                break

            step_output_token_ids = []
            for token_id, seq_id in zip(token_ids_by_step, seq_ids):
                step_output_token_ids.append(
                    SequenceGroupOutput(
                        samples=[
                            SequenceOutput(
                                parent_seq_id=seq_id,
                                output_token=token_id,
                                # TODO currently rejection sampling does not
                                # emit probs, so this value is meaningless.
                                logprobs={token_id: 0.0},
                            )
                        ],
                        prompt_logprobs=None,
                    ))
            sampler_output_list.append(
                SamplerOutput(outputs=step_output_token_ids))

        return sampler_output_list

    @cached_property
    def _vocab_size(self) -> int:
        """Get the vocab size of the model and make sure it's consistent between
        draft and target workers.
        """
        vocab_sizes = [
            worker.vocab_size
            for worker in [self.draft_worker, self.target_worker]
        ]
        assert all(vocab_sizes[0] == vocab_size for vocab_size in vocab_sizes)
        return vocab_sizes[0]

    @property
    def rank(self):
        return self.target_worker.rank


# TODO name
def calculate_gpu_blocks(target_kv_size_bytes: int, draft_kv_size_bytes: int,
                         total_num_gpu_blocks: int) -> int:
    """Given total_num_gpu_blocks, the number of GPU blocks that could be
    allocate to the target model, this function calculates how many blocks
    should be given to the draft and target model.

    Note that usually the block size, in bytes, of each model is different,
    as it's a function of number of KV/layer, number of heads, and hidden
    dimension size.

    Since the target and draft models allocate the same number of blocks, we
    simply calculate the number of blocks where if allocated by both models,
    the total memory usage from KV cache is no larger than the number of
    blocks allocatable by the target model alone.
    """
    new_num_gpu_blocks = int(total_num_gpu_blocks * target_kv_size_bytes /
                             (draft_kv_size_bytes + target_kv_size_bytes))

    return new_num_gpu_blocks
