from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple

import torch

from vllm.config import SpeculativeConfig
from vllm.distributed.communication_op import broadcast_tensor_dict
from vllm.logger import init_logger
from vllm.model_executor.layers.rejection_sampler import RejectionSampler
from vllm.sequence import (ExecuteModelRequest, SamplerOutput,
                           SequenceGroupMetadata)
from vllm.spec_decode.batch_expansion import BatchExpansionTop1Scorer
from vllm.spec_decode.interfaces import (SpeculativeProposals,
                                         SpeculativeScorer, SpeculativeScores)
from vllm.spec_decode.metrics import AsyncMetricsCollector
from vllm.spec_decode.multi_step_worker import MultiStepWorker
from vllm.spec_decode.ngram_worker import NGramWorker
from vllm.spec_decode.proposer_worker_base import ProposerWorkerBase
from vllm.spec_decode.util import (create_sequence_group_output,
                                   get_all_num_logprobs, get_all_seq_ids,
                                   get_sampled_token_logprobs, nvtx_range,
                                   split_batch_by_proposal_len)
from vllm.worker.worker import Worker
from vllm.worker.worker_base import LoraNotSupportedWorkerBase, WorkerBase

logger = init_logger(__name__)


def create_spec_worker(*args, **kwargs) -> "SpecDecodeWorker":
    """Helper method that is the entrypoint for Executors which use
    WorkerWrapper. It constructs a SpecDecodeWorker from the speculative config.
    """
    assert "speculative_config" in kwargs
    speculative_config: SpeculativeConfig = kwargs.get("speculative_config")
    assert speculative_config is not None

    target_worker = Worker(*args, **kwargs)

    draft_worker_kwargs = kwargs.copy()
    # Override draft-model specific worker args.
    draft_worker_kwargs.update(
        model_config=speculative_config.draft_model_config,
        parallel_config=speculative_config.draft_parallel_config,
        ngram_prompt_lookup_max=speculative_config.ngram_prompt_lookup_max,
        ngram_prompt_lookup_min=speculative_config.ngram_prompt_lookup_min,
        # TODO allow draft-model specific load config.
        #load_config=load_config,
    )

    spec_decode_worker = SpecDecodeWorker.create_worker(
        scorer_worker=target_worker,
        draft_worker_kwargs=draft_worker_kwargs,
        disable_by_batch_size=speculative_config.
        speculative_disable_by_batch_size,
    )

    return spec_decode_worker


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
    def create_worker(
        cls,
        scorer_worker: WorkerBase,
        draft_worker_kwargs: Dict[str, Any],
        disable_by_batch_size: Optional[int],
    ) -> "SpecDecodeWorker":

        ngram_prompt_lookup_max = (
            draft_worker_kwargs.pop("ngram_prompt_lookup_max"))
        ngram_prompt_lookup_min = (
            draft_worker_kwargs.pop("ngram_prompt_lookup_min"))

        disable_bonus_tokens = True
        if ngram_prompt_lookup_max > 0:
            disable_bonus_tokens = False
            proposer_worker = NGramWorker(**draft_worker_kwargs)
            proposer_worker.set_ngram_window_size(ngram_prompt_lookup_min,
                                                  ngram_prompt_lookup_max)
        else:
            proposer_worker = MultiStepWorker(**draft_worker_kwargs)

        logger.info("Configuring SpecDecodeWorker with proposer=%s",
                    type(proposer_worker))

        return SpecDecodeWorker(proposer_worker,
                                scorer_worker,
                                disable_by_batch_size=disable_by_batch_size,
                                rejection_sampler=RejectionSampler(
                                    disable_bonus_tokens=disable_bonus_tokens))

    def __init__(
        self,
        proposer_worker: ProposerWorkerBase,
        scorer_worker: WorkerBase,
        rejection_sampler: RejectionSampler,
        metrics_collector: Optional[AsyncMetricsCollector] = None,
        disable_by_batch_size: Optional[int] = None,
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
            disable_by_batch_size: If the batch size is larger than this,
                disable speculative decoding for new incoming requests.
            metrics_collector: Helper class for collecting metrics; can be set
                for testing purposes.
        """
        self.proposer_worker = proposer_worker
        self.scorer_worker = scorer_worker
        self.disable_by_batch_size = disable_by_batch_size or float("inf")
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

    def load_model(self, *args, **kwargs):
        pass

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
        self.proposer_worker.set_include_gpu_probs_tensor()

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
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> List[SamplerOutput]:
        """Perform speculative decoding on the input batch.
        """
        if self.rank != self._driver_rank:
            self._run_non_driver_rank()
            return []

        if execute_model_req is None:
            # This signals that there's no more requests to process for now.
            # All workers are running infinite loop with broadcast_tensor_dict,
            # and it stops the loop when the driver broadcasts an empty input.
            # Send an empty input to notify all other workers to stop their
            # execution loop.
            broadcast_tensor_dict({}, src=0)
            return []

        disable_all_speculation = self._should_disable_all_speculation(
            execute_model_req)
        num_lookahead_slots = execute_model_req.num_lookahead_slots

        # Broadcast how many lookahead slots are scheduled for this step, and
        # whether all speculation is disabled, to all non-driver workers.

        # This is required as if the number of draft model runs changes
        # dynamically, the non-driver workers won't know unless we perform a
        # communication to inform them.
        broadcast_dict = dict(
            num_lookahead_slots=num_lookahead_slots,
            disable_all_speculation=disable_all_speculation,
        )
        broadcast_tensor_dict(broadcast_dict, src=self._driver_rank)

        assert execute_model_req.seq_group_metadata_list is not None, (
            "speculative decoding requires non-None seq_group_metadata_list")

        self._maybe_disable_speculative_tokens(
            disable_all_speculation, execute_model_req.seq_group_metadata_list)

        # Speculative decoding is disabled in the following cases:
        # 1. Prefill phase: Speculative decoding is not
        #    used during the prefill phase.
        # 2. Auto-disable enabled: The running queue size exceeds
        #    the specified threshold.
        # 3. No request: There are no requests in the batch.
        # In any of these cases, the proposer and scorer workers
        # are called normally.
        if num_lookahead_slots == 0 or len(
                execute_model_req.seq_group_metadata_list
        ) == 0 or disable_all_speculation:
            return self._run_no_spec(execute_model_req,
                                     skip_proposer=disable_all_speculation)

        return self._run_speculative_decoding_step(execute_model_req,
                                                   num_lookahead_slots)

    @torch.inference_mode()
    def start_worker_execution_loop(self) -> None:
        """Execute model loop to perform speculative decoding
        in parallel worker."""
        while self._run_non_driver_rank():
            pass

    def _should_disable_all_speculation(
            self, execute_model_req: ExecuteModelRequest) -> bool:
        # When the batch size is too large, disable speculative decoding
        # to stop trading off throughput for latency.
        disable_all_speculation = (execute_model_req.running_queue_size >=
                                   self.disable_by_batch_size)

        return disable_all_speculation

    def _maybe_disable_speculative_tokens(
            self, disable_all_speculation: bool,
            seq_group_metadata_list: List[SequenceGroupMetadata]) -> None:
        if not disable_all_speculation:
            return

        for seq_group_metadata in seq_group_metadata_list:
            # Once num_speculative_tokens is set to 0, the spec decode
            # of this request will be disabled forever.
            # TODO(comaniac): We currently store spec decoding specific
            # state in the global data structure, but we should maintain
            # this state within spec decode worker.
            seq_group_metadata.num_speculative_tokens = 0

    @nvtx_range("spec_decode_worker._run_no_spec")
    def _run_no_spec(self, execute_model_req: ExecuteModelRequest,
                     skip_proposer: bool) -> List[SamplerOutput]:
        """Run a single generation step without any speculation. The input is
        sent to the proposer and scorer model so that the KV cache is consistent
        between the two. When skip_proposer is True, the proposer model is
        not called, meaning that the kv-cache in proposer for requests is not
        updated, so they cannot enable spec decode in the rest decoding.
        """
        if not skip_proposer:
            self.proposer_worker.execute_model(execute_model_req)

        sampler_output = self.scorer_worker.execute_model(execute_model_req)
        assert len(sampler_output) == 1
        sampler_output = sampler_output[0]

        # Clear device tensors from sampler output. This reduces communication
        # overhead when the engine runs in a different process than the workers.
        sampler_output.probs = None
        sampler_output.sampled_tokens = None
        sampler_output.logprobs = None
        return [sampler_output]

    def _run_non_driver_rank(self) -> bool:
        """Run proposer and verifier model in non-driver workers. This is used
        for both speculation cases (num_lookahead_slots>0) and non-speculation
        cases (e.g. prefill).

        Returns True iff there are remaining sequences to process.
        """
        assert self.rank != self._driver_rank

        data = broadcast_tensor_dict(src=self._driver_rank)
        if not data:
            return False
        num_lookahead_slots = data["num_lookahead_slots"]

        # Even if num_lookahead_slots is zero, we want to run the proposer model
        # as it may have KV.
        #
        # We run the proposer once per lookahead slot. In the future we should
        # delegate how many times it runs to the proposer.
        for _ in range(max(num_lookahead_slots, 1)):
            self.proposer_worker.execute_model()

        self.scorer_worker.execute_model()
        return True

    @nvtx_range("spec_decode_worker._run_speculative_decoding_step")
    def _run_speculative_decoding_step(
            self, execute_model_req: ExecuteModelRequest,
            num_lookahead_slots: int) -> List[SamplerOutput]:
        """Execute a single step of speculative decoding.

        This invokes the proposer worker to get k speculative tokens for each
        sequence, then scores each speculative token using the scoring worker.

        Returns a list of SamplerOutput, each containing a single token per
        sequence.
        """
        assert num_lookahead_slots == execute_model_req.num_lookahead_slots

        # Generate proposals using draft worker.
        proposals = self.proposer_worker.get_spec_proposals(execute_model_req)

        proposal_scores = self.scorer.score_proposals(
            execute_model_req,
            proposals,
        )

        accepted_token_ids, target_logprobs = self._verify_tokens(
            execute_model_req.seq_group_metadata_list, proposal_scores,
            proposals, execute_model_req.num_lookahead_slots)

        return self._create_output_sampler_list(
            execute_model_req.seq_group_metadata_list,
            accepted_token_ids,
            target_logprobs=target_logprobs,
            k=execute_model_req.num_lookahead_slots)

    @nvtx_range("spec_decode_worker._verify_tokens")
    def _verify_tokens(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        proposal_scores: SpeculativeScores,
        proposals: SpeculativeProposals,
        max_proposal_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Determine which speculative tokens are accepted using the
        probabilities of each token according to the proposer and scorer models.

        Returns a tuple of Tensors, one for the accepted token ids and one for
        the logprobs according to the scoring model.
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
        logprobs = proposal_scores.logprobs

        # Rearrange so that results are in the order of the original seq group
        # metadata.
        accepted_token_ids[original_indices] = accepted_token_ids.clone()

        return accepted_token_ids, logprobs

    def _create_output_sampler_list(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        accepted_token_ids: torch.Tensor,  # shape: [batch_size, k+1]
        target_logprobs: torch.Tensor,  # shape: [batch_size, k+1, vocab_size]
        k: int,
    ) -> List[SamplerOutput]:
        """Given the accepted token ids, create a list of SamplerOutput.

        The output is padded with -1 tokens such that each sequence has
        the same number of outputs.
        """
        batch_size, num_steps = accepted_token_ids.shape

        # Organize input tensors by step instead of by sequence.
        target_logprobs_by_step = target_logprobs.transpose(0, 1)
        accepted_token_ids_by_step = accepted_token_ids.transpose(0, 1)

        # Get the logprobs/rank of the accepted tokens.
        (accepted_token_id_ranks_by_step,
         accepted_token_id_logprobs_by_step) = get_sampled_token_logprobs(
             logprob_tensor=target_logprobs_by_step,
             sampled_token_ids=accepted_token_ids_by_step,
         )

        # Get the top-k logprobs (which may or may not include the logprob of
        # the accepted token).
        (topk_logprobs_by_step,
         topk_indices_by_step) = target_logprobs_by_step.topk(
             k=self.scorer_worker.model_config.max_logprobs,
             dim=-1,
         )

        # Get the sequence ids and num_logprobs (sampling parameter) in the
        # batch.
        seq_ids = get_all_seq_ids(seq_group_metadata_list)
        num_logprobs_per_seq = get_all_num_logprobs(seq_group_metadata_list)

        # Serialize all tensors to CPU Python lists.
        accepted_token_ids_by_step = accepted_token_ids_by_step.tolist()
        accepted_token_id_ranks_by_step = (
            accepted_token_id_ranks_by_step.tolist())
        accepted_token_id_logprobs_by_step = (
            accepted_token_id_logprobs_by_step.tolist())
        topk_logprobs_by_step = topk_logprobs_by_step.tolist()
        topk_indices_by_step = topk_indices_by_step.tolist()

        # Construct the output on a per-step, per-sequence basis.
        sampler_output_list = []
        for step_index in range(num_steps):
            if all(token_id == -1
                   for token_id in accepted_token_ids_by_step[step_index]):
                break

            step_output_token_ids = []
            for sequence_index in range(batch_size):
                # Each sequence may have a different num_logprobs; retrieve it.
                num_logprobs = num_logprobs_per_seq[sequence_index]

                step_output_token_ids.append(
                    create_sequence_group_output(
                        token_id=accepted_token_ids_by_step[step_index]
                        [sequence_index],
                        token_id_logprob_rank=accepted_token_id_ranks_by_step[
                            step_index][sequence_index],
                        token_id_logprob=accepted_token_id_logprobs_by_step[
                            step_index][sequence_index],
                        seq_id=seq_ids[sequence_index],
                        topk_token_ids=topk_indices_by_step[step_index]
                        [sequence_index][:num_logprobs],
                        topk_logprobs=topk_logprobs_by_step[step_index]
                        [sequence_index][:num_logprobs],
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

    @property
    def _driver_rank(self) -> int:
        return 0

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
