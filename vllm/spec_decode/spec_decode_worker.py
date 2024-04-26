from functools import cached_property
from typing import Dict, List, Optional, Tuple

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.rejection_sampler import RejectionSampler
from vllm.sequence import (Logprob, SamplerOutput, SequenceGroupMetadata,
                           SequenceGroupOutput, SequenceOutput)
from vllm.spec_decode.batch_expansion import BatchExpansionTop1Scorer
from vllm.spec_decode.interfaces import (SpeculativeProposals,
                                         SpeculativeScorer, SpeculativeScores)
from vllm.spec_decode.metrics import AsyncMetricsCollector
from vllm.spec_decode.multi_step_worker import MultiStepWorker
from vllm.spec_decode.util import (get_all_seq_ids, nvtx_range,
                                   split_batch_by_proposal_len)
from vllm.worker.worker_base import LoraNotSupportedWorkerBase, WorkerBase

logger = init_logger(__name__)


class SpecDecodeWorker(LoraNotSupportedWorkerBase):
    """Worker which implements speculative decoding.

    Speculative decoding reduces decoding per-token latency by using a proposal
    method, such as a small draft model, to speculate ahead of a larger LLM. The
    probabilities of the speculative tokens are then determined by the larger
    LLM, after which some verification routine determines which (if any) of the
    speculative tokens are accepted by the larger LLM.

    See https://github.com/vllm-project/vllm/pull/2188 and
    https://github.com/vllm-project/vllm/pull/3103 for more info.

    The current implementation has the following limitations:
    * Only draft-model proposal is implemented (contributions for more forms are
        welcome!).
    * Only top-1 proposal and scoring are implemented. Tree-attention is left as
        future work.
    * Only lossless rejection sampling is supported. Contributions adding lossy
        verification routines are welcome (e.g. Medusa's typical acceptance).
    * All sequences in a batch must have the same proposal length, or zero. This
        can be improved by having per-sequence speculation in the future.
    * The scoring forward pass is done without an MQA kernel, which is
        suboptimal especially as the batch size, proposal length, and sequence
        lengths grow. Contributions to add a MQA scoring are welcome once
        correctness tests pass.
        More info here https://docs.google.com/document/d/1T-JaS2T1NRfdP51qzqpyakoCXxSXTtORppiwaj5asxA/edit.
    """

    @classmethod
    def from_workers(cls, proposer_worker: MultiStepWorker,
                     scorer_worker: WorkerBase) -> "SpecDecodeWorker":
        return SpecDecodeWorker(
            proposer_worker,
            scorer_worker,
            # TODO(cade) disable strict mode for speedup.
            rejection_sampler=RejectionSampler(strict_mode=True),
        )

    def __init__(
        self,
        proposer_worker: MultiStepWorker,
        scorer_worker: WorkerBase,
        rejection_sampler: RejectionSampler,
        metrics_collector: Optional[AsyncMetricsCollector] = None,
    ):
        """
        Create a SpecDecodeWorker.

        Args:
            proposer_worker: A worker that can produce speculative tokens for
                sequences.
            scorer_worker: A worker that produces probabilities of speculative
                tokens according to some base model. Typically a vanilla vLLM
                Worker.
            rejection_sampler: A Torch module used to perform modified rejection
                sampling for speculative decoding.
            metrics_collector: Helper class for collecting metrics; can be set
                for testing purposes.
        """
        self.proposer_worker = proposer_worker
        self.scorer_worker = scorer_worker
        self.rejection_sampler = rejection_sampler

        self._metrics = AsyncMetricsCollector(
            rejection_sampler
        ) if metrics_collector is None else metrics_collector

        self.probs_dtype = self.rejection_sampler.probs_dtype
        self.token_id_dtype = self.rejection_sampler.token_id_dtype

        # Lazy initiazliation.
        self.scorer: SpeculativeScorer

    def init_device(self) -> None:
        """Initialize both scorer and proposer models.
        """
        # The scorer worker model is initialized first in case the proposer
        # model has a smaller TP degree than the target worker.
        self.scorer_worker.init_device()
        self.proposer_worker.init_device()

        # NOTE(cade): load_model is not part of the WorkerBase interface.
        self.scorer_worker.load_model()
        self.proposer_worker.load_model()

        self._metrics.init_gpu_tensors(self.rank)
        self.rejection_sampler.init_gpu_tensors(self.rank)
        self.scorer = BatchExpansionTop1Scorer(
            scorer_worker=self.scorer_worker,
            device=self.device,
            vocab_size=self._vocab_size)

        self._configure_model_sampler_for_spec_decode()

    def _configure_model_sampler_for_spec_decode(self):
        """Configure model sampler to emit GPU tensors. This allows spec decode
        to keep data on device without transferring to CPU and serializing,
        which significantly reduces overhead of rejection sampling.

        NOTE(cade): This breaks abstraction boundaries pretty badly. The better
        design is to have the "move to CPU and serialize" sampling decision be
        done outside of the model/sampler; this way the "last-mile" worker
        object which interfaces with the scheduler can serialize and incur the
        performance hit as necessary. This allows us to run the worker several
        iterations in a row without incurring the "move to CPU and serialize"
        performance penalty.

        Since this requires a large change to vLLM, we defer it to later and
        temporarily accept this broken abstraction boundary.

        NOTE(cade): This will require a special check if the proposer worker
        does not have a sampler (e.g. ngram speculation).
        """
        (self.scorer_worker.model_runner.model.sampler.include_gpu_probs_tensor
         ) = True
        (self.proposer_worker.model_runner.model.sampler.
         include_gpu_probs_tensor) = True

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of cache blocks to use.

        This is done by profiling the scorer model (which is typically the
        larger of the two). Then the total memory which would be used by the
        scorer cache is divided evenly between the proposer and scorer model KV,
        such that the number of blocks is equal in both KV caches.
        """
        num_gpu_blocks, num_cpu_blocks = (
            self.scorer_worker.determine_num_available_blocks())

        scorer_cache_block_size_bytes = (
            self.scorer_worker.get_cache_block_size_bytes())
        proposer_cache_block_size_bytes = (
            self.proposer_worker.get_cache_block_size_bytes())

        new_num_gpu_blocks = split_num_cache_blocks_evenly(
            scorer_cache_block_size_bytes, proposer_cache_block_size_bytes,
            num_gpu_blocks)
        return new_num_gpu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Initialize the cache engine of the scorer and proposer workers.
        """
        self.scorer_worker.initialize_cache(num_gpu_blocks=num_gpu_blocks,
                                            num_cpu_blocks=num_cpu_blocks)
        self.proposer_worker.initialize_cache(num_gpu_blocks=num_gpu_blocks,
                                              num_cpu_blocks=num_cpu_blocks)

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Optional[Dict[int, int]],
        blocks_to_swap_out: Optional[Dict[int, int]],
        blocks_to_copy: Optional[Dict[int, List[int]]],
        num_lookahead_slots: int,
    ) -> List[SamplerOutput]:
        """Perform speculative decoding on the input batch.
        """

        assert seq_group_metadata_list is not None, (
            "speculative decoding "
            "requires non-None seq_group_metadata_list")

        logger.info(f"spec_decode_worker.execute_model {num_lookahead_slots=}")

        # If no spec tokens, call the proposer and scorer workers normally.
        # Used for prefill.
        if num_lookahead_slots == 0 or len(seq_group_metadata_list) == 0:
            return self._run_no_spec(
                seq_group_metadata_list=seq_group_metadata_list,
                blocks_to_swap_in=blocks_to_swap_in,
                blocks_to_swap_out=blocks_to_swap_out,
                blocks_to_copy=blocks_to_copy,
            )

        return self._run_speculative_decoding_step(
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            k=num_lookahead_slots,
        )

    @nvtx_range("spec_decode_worker._run_no_spec")
    def _run_no_spec(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Optional[Dict[int, int]],
        blocks_to_swap_out: Optional[Dict[int, int]],
        blocks_to_copy: Optional[Dict[int, List[int]]],
    ) -> List[SamplerOutput]:
        """Run a prefill step, without any speculation. The input is sent to the
        proposer and scorer model so that the KV cache is consistent between the
        two.
        """
        logger.info("run proposer worker no spec")

        self.proposer_worker.execute_model(
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
        )

        logger.info("run target worker no spec")
        sampler_output = self.scorer_worker.execute_model(
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
        )
        assert len(sampler_output) == 1
        sampler_output = sampler_output[0]

        # Clear device tensors from sampler output. This reduces communication
        # overhead when the engine runs in a different process than the workers.
        sampler_output.probs = None
        sampler_output.sampled_tokens = None
        return [sampler_output]

    @nvtx_range("spec_decode_worker._run_speculative_decoding_step")
    def _run_speculative_decoding_step(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Optional[Dict[int, int]],
        blocks_to_swap_out: Optional[Dict[int, int]],
        blocks_to_copy: Optional[Dict[int, List[int]]],
        k: int,
    ) -> List[SamplerOutput]:
        """Execute a single step of speculative decoding.

        This invokes the proposer worker to get k speculative tokens for each
        sequence, then scores each speculative token using the scoring worker.

        Returns a list of SamplerOutput, each containing a single token per
        sequence.
        """

        logger.info("get spec proposals")
        # Generate proposals using draft worker.
        assert blocks_to_swap_in is not None
        assert blocks_to_swap_out is not None
        assert blocks_to_copy is not None
        proposals = self.proposer_worker.get_spec_proposals(
            seq_group_metadata_list, blocks_to_swap_in, blocks_to_swap_out,
            blocks_to_copy, k)

        logger.info("score proposals")
        proposal_scores = self.scorer.score_proposals(
            seq_group_metadata_list,
            blocks_to_swap_in,
            blocks_to_swap_out,
            blocks_to_copy,
            k,
            proposals,
        )

        logger.info("verify proposals")
        accepted_token_ids = self._verify_tokens(seq_group_metadata_list,
                                                 proposal_scores, proposals, k)

        logger.info("create output list")
        return self._create_output_sampler_list(seq_group_metadata_list,
                                                accepted_token_ids, k)

    @nvtx_range("spec_decode_worker._verify_tokens")
    def _verify_tokens(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        proposal_scores: SpeculativeScores,
        proposals: SpeculativeProposals,
        max_proposal_len: int,
    ) -> torch.Tensor:
        """Determine which speculative tokens are accepted using the
        probabilities of each token according to the proposer and scorer models.
        """
        proposal_lens_list = proposals.proposal_lens.tolist()

        # vLLM currently only supports proposal lens equal to zero or the batch
        # proposal len. This adds some complexity (splitting the batch into spec
        # and non spec sequences) and should be removed in the future. It can be
        # done by supporting per-sequence proposal lens.
        _, spec_indices = split_batch_by_proposal_len(
            seq_group_metadata_list,
            proposal_lens_list,
            select_proposal_len_zero=False)
        _, non_spec_indices = split_batch_by_proposal_len(
            seq_group_metadata_list,
            proposal_lens_list,
            select_proposal_len_zero=True)
        original_indices = spec_indices + non_spec_indices

        # Get probabilities of target model, excluding bonus token.
        proposal_verifier_probs = proposal_scores.probs[spec_indices, :-1]

        # Get non-speculative sampled tokens from target model.
        non_spec_token_ids = proposal_scores.token_ids[non_spec_indices]

        # Get bonus tokens from target model.
        bonus_token_ids = proposal_scores.token_ids[spec_indices, -1:]

        # Get probabilities according to proposal method.
        proposal_probs = proposals.proposal_probs[spec_indices]

        # Get proposed tokens.
        proposal_token_ids = proposals.proposal_token_ids[spec_indices]

        accepted_token_ids = self.rejection_sampler(
            target_probs=proposal_verifier_probs,
            bonus_token_ids=bonus_token_ids,
            draft_probs=proposal_probs,
            draft_token_ids=proposal_token_ids,
        )

        # Append output tokens from non-speculative sequences to
        # the accepted token ids tensor.
        non_spec_token_ids = non_spec_token_ids.expand(-1, max_proposal_len +
                                                       1).clone()
        non_spec_token_ids[:, 1:] = -1
        accepted_token_ids = torch.cat(
            [accepted_token_ids, non_spec_token_ids])

        # TODO need to map the logprobs of the accepted tokens
        # need a way to determine prob batch index for a given sequence
        # I forget the order of the probs
        """
        (Pdb) accepted_token_ids
        tensor([[29889,    -1,    -1],
                [  697,  1058,    -1]], device='cuda:0')
        (Pdb) proposal_scores.probs
        tensor([[[0., 0., 0.,  ..., 0., 0., 0.],
                 [0., 0., 0.,  ..., 0., 0., 0.],
                 [0., 0., 0.,  ..., 0., 0., 0.]],
        
                [[0., 0., 0.,  ..., 0., 0., 0.],
                 [0., 0., 0.,  ..., 0., 0., 0.],
                 [0., 0., 0.,  ..., 0., 0., 0.]]], device='cuda:0')
        (Pdb) proposal_scores.probs.shape
        torch.Size([2, 3, 32000])
        (Pdb) accepted_token_ids.shape
        torch.Size([2, 3])
        (Pdb) proposal_scores.probs[0]
        tensor([[0., 0., 0.,  ..., 0., 0., 0.],
                [0., 0., 0.,  ..., 0., 0., 0.],
                [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0')
        (Pdb) proposal_scores.probs[0].shape
        torch.Size([3, 32000])
        (Pdb) proposal_scores.probs[0][0][29889]
        tensor(1., device='cuda:0')

        [batch][k][token]
        I think non spec sequences are in same order
        """

        #breakpoint()

        # Rearrange so that results are in the order of the original seq group
        # metadata.
        accepted_token_ids[original_indices] = accepted_token_ids.clone()

        return accepted_token_ids

    def _create_output_sampler_list(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        accepted_token_ids: torch.Tensor,  # shape: [batch_size, k+1]
        k: int,
    ) -> List[SamplerOutput]:
        """Given the accepted token ids, create a list of SamplerOutput.

        The output is padded with -1 tokens such that each sequence has
        the same number of outputs.
        """
        seq_ids = get_all_seq_ids(seq_group_metadata_list)

        # shape: [k+1, batch_size]
        accepted_token_ids_by_step = accepted_token_ids.transpose(0,
                                                                  1).tolist()
        # Need to get logprobs of accepted token ids according to target model
        # what is rank? how is order determined
        # 

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
                                # TODO Add verifier logprobs.
                                logprobs={token_id: Logprob(0.1)},
                            )
                        ],
                        prompt_logprobs=None,
                    ))
            sampler_output_list.append(
                SamplerOutput(outputs=step_output_token_ids))

        maybe_rejsample_metrics = (
            self._metrics.maybe_collect_rejsample_metrics(k))
        if maybe_rejsample_metrics is not None:
            sampler_output_list[
                0].spec_decode_worker_metrics = maybe_rejsample_metrics

        return sampler_output_list

    @cached_property
    def _vocab_size(self) -> int:
        """Get the vocab size of the model and make sure it's consistent between
        draft and target workers.
        """
        vocab_sizes = [
            worker.vocab_size
            for worker in [self.proposer_worker, self.scorer_worker]
        ]
        assert all(vocab_sizes[0] == vocab_size for vocab_size in vocab_sizes)
        return vocab_sizes[0]

    @property
    def rank(self):
        return self.scorer_worker.rank

    @property
    def device(self):
        return self.scorer_worker.device

    def get_cache_block_size_bytes(self):
        """Return the size of a cache block in bytes.
        
        This function is only used to compose workers within a SpecDecodeWorker.
        We leave composing a SpecDecodeWorker within a SpecDecodeWorker
        undefined for now, although it could be implemented in the future.
        See https://arxiv.org/abs/2308.04623.
        """
        raise NotImplementedError


def split_num_cache_blocks_evenly(scorer_cache_block_size_bytes: int,
                                  proposer_cache_block_size_bytes: int,
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
    new_num_gpu_blocks = int(
        total_num_gpu_blocks * scorer_cache_block_size_bytes /
        (proposer_cache_block_size_bytes + scorer_cache_block_size_bytes))

    return new_num_gpu_blocks
