from collections import defaultdict
from functools import cached_property
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from vllm.config import ParallelConfig, SpeculativeConfig
from vllm.distributed.communication_op import broadcast_tensor_dict
from vllm.logger import init_logger
from vllm.model_executor.layers.rejection_sampler import RejectionSampler
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.layers.spec_decode_base_sampler import (
    SpecDecodeBaseSampler, SpecDecodeStochasticBaseSampler)
from vllm.model_executor.layers.typical_acceptance_sampler import (
    TypicalAcceptanceSampler)
from vllm.sequence import (CompletionSequenceGroupOutput, ExecuteModelRequest,
                           HiddenStates, SequenceGroupMetadata,
                           get_all_seq_ids, get_all_seq_ids_and_request_ids)
from vllm.spec_decode.batch_expansion import BatchExpansionTop1Scorer
from vllm.spec_decode.draft_model_runner import TP1DraftModelRunner
from vllm.spec_decode.interfaces import (SpeculativeProposals,
                                         SpeculativeScorer, SpeculativeScores)
from vllm.spec_decode.medusa_worker import MedusaWorker
from vllm.spec_decode.metrics import AsyncMetricsCollector
from vllm.spec_decode.mlp_speculator_worker import MLPSpeculatorWorker
from vllm.spec_decode.multi_step_worker import MultiStepWorker
from vllm.spec_decode.ngram_worker import NGramWorker
from vllm.spec_decode.proposer_worker_base import ProposerWorkerBase
from vllm.spec_decode.smaller_tp_proposer_worker import SmallerTpProposerWorker
from vllm.spec_decode.target_model_runner import TargetModelRunner
from vllm.spec_decode.util import (Timer, create_sequence_group_output,
                                   get_all_num_logprobs,
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

    draft_worker_kwargs = kwargs.copy()

    kwargs["model_runner_cls"] = TargetModelRunner
    target_worker = Worker(*args, **kwargs)
    # Set the disable_logprobs variable in the TargetModelRunner instance
    # as per its value specified in the SpeculativeConfig.
    target_worker.model_runner.disable_logprobs =\
         speculative_config.disable_logprobs

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
        draft_token_acceptance_method=speculative_config.
        draft_token_acceptance_method,
        typical_acceptance_sampler_posterior_threshold=speculative_config.
        typical_acceptance_sampler_posterior_threshold,
        typical_acceptance_sampler_posterior_alpha=speculative_config.
        typical_acceptance_sampler_posterior_alpha,
        disable_logprobs=speculative_config.disable_logprobs,
        disable_log_stats=speculative_config.disable_log_stats,
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
        scorer_worker: Worker,
        draft_worker_kwargs: Dict[str, Any],
        disable_by_batch_size: Optional[int],
        draft_token_acceptance_method: str,
        typical_acceptance_sampler_posterior_threshold: float,
        typical_acceptance_sampler_posterior_alpha: float,
        disable_logprobs: bool,
        disable_log_stats: bool,
    ) -> "SpecDecodeWorker":

        allow_zero_draft_token_step = True
        ngram_prompt_lookup_max = (
            draft_worker_kwargs.pop("ngram_prompt_lookup_max"))
        ngram_prompt_lookup_min = (
            draft_worker_kwargs.pop("ngram_prompt_lookup_min"))
        if ngram_prompt_lookup_max > 0:
            proposer_worker = NGramWorker(**draft_worker_kwargs)
            proposer_worker.set_ngram_window_size(ngram_prompt_lookup_min,
                                                  ngram_prompt_lookup_max)
        else:
            draft_parallel_config: ParallelConfig = draft_worker_kwargs[
                'parallel_config']
            draft_tp = draft_parallel_config.tensor_parallel_size
            target_tp = scorer_worker.parallel_config.tensor_parallel_size

            if draft_worker_kwargs[
                    "model_config"].hf_config.model_type == "mlp_speculator":
                proposer_worker = MLPSpeculatorWorker(**draft_worker_kwargs)
            elif draft_worker_kwargs[
                    "model_config"].hf_config.model_type == "medusa":
                proposer_worker = MedusaWorker(**draft_worker_kwargs)
            else:
                if draft_tp == 1:
                    draft_worker_kwargs[
                        "model_runner_cls"] = TP1DraftModelRunner
                else:
                    if draft_worker_kwargs[
                            "model_config"].hf_config.model_type == "eagle":
                        raise NotImplementedError(
                            "EAGLE does not support TP > 1 yet")

                    allow_zero_draft_token_step = False
                proposer_worker = MultiStepWorker(**draft_worker_kwargs)

            proposer_worker = SmallerTpProposerWorker.maybe_wrap_worker(
                proposer_worker, draft_tp, target_tp)

        logger.info("Configuring SpecDecodeWorker with proposer=%s",
                    type(proposer_worker))

        spec_decode_sampler: SpecDecodeBaseSampler = None
        if draft_token_acceptance_method == "rejection_sampler":
            spec_decode_sampler = RejectionSampler(
                disable_bonus_tokens=False, )
        elif draft_token_acceptance_method == "typical_acceptance_sampler":
            spec_decode_sampler = TypicalAcceptanceSampler(
                disable_bonus_tokens=False,
                posterior_threshold=\
                    typical_acceptance_sampler_posterior_threshold,
                posterior_alpha=typical_acceptance_sampler_posterior_alpha,
            )
        logger.info("Configuring SpecDecodeWorker with sampler=%s",
                    type(spec_decode_sampler))

        return SpecDecodeWorker(
            proposer_worker,
            scorer_worker,
            disable_logprobs=disable_logprobs,
            disable_log_stats=disable_log_stats,
            disable_by_batch_size=disable_by_batch_size,
            spec_decode_sampler=spec_decode_sampler,
            allow_zero_draft_token_step=allow_zero_draft_token_step)

    def __init__(
        self,
        proposer_worker: ProposerWorkerBase,
        scorer_worker: WorkerBase,
        spec_decode_sampler: SpecDecodeBaseSampler,
        disable_logprobs: bool = False,
        disable_log_stats: bool = False,
        metrics_collector: Optional[AsyncMetricsCollector] = None,
        disable_by_batch_size: Optional[int] = None,
        allow_zero_draft_token_step: Optional[bool] = True,
    ):
        """
        Create a SpecDecodeWorker.

        Args:
            proposer_worker: A worker that can produce speculative tokens for
                sequences.
            scorer_worker: A worker that produces probabilities of speculative
                tokens according to some base model. Typically a vanilla vLLM
                Worker.
            spec_decode_sampler: A Torch module used to perform acceptance
                sampling of the draft tokens in the verification step of
                speculative decoding. Currently we support two different 
                types of sampler namely RejectionSampler and
                TypicalAcceptanceSampler. 'spec_decode_sampler' is either an
                instance of RejectionSampler or TypicalAcceptanceSampler.
            disable_logprobs: If set to True, token log probabilities will
                not be output in both the draft worker and the target worker.
                If set to False, log probabilities will be output by both.
            disable_log_stats: If set to True, disable periodic printing of
                speculative stage times.
            disable_by_batch_size: If the batch size is larger than this,
                disable speculative decoding for new incoming requests.
            metrics_collector: Helper class for collecting metrics; can be set
                for testing purposes.
            allow_zero_draft_token_step: whether to allow a step where the draft
                model generates no draft token; should disallow when the tp of
                draft model is larger than 1 (TODO: #5814)
        """
        self.proposer_worker = proposer_worker
        self.scorer_worker = scorer_worker
        scorer_runner = getattr(self.scorer_worker, "model_runner", None)
        self.generators = scorer_runner.get_generators(
        ) if scorer_runner else None
        self.disable_by_batch_size = disable_by_batch_size or float("inf")
        self.spec_decode_sampler = spec_decode_sampler
        self._allow_zero_draft_token_step = allow_zero_draft_token_step
        self._metrics = AsyncMetricsCollector(
            self.spec_decode_sampler
        ) if metrics_collector is None else metrics_collector
        # Tracks the sequence IDs that received a bonus token ID in
        # their last forward pass. Needed only if KV cache is being
        # used for token generation such as in the case of MultiStepWorker.
        self._seq_with_bonus_token_in_last_step: Set[int] = set()
        # Tracks the currently active request ids and the sequence IDs
        # corresponding to them
        self._request_id_seq_id_mapping: Dict[str, Set[int]] = defaultdict(set)
        # Tracks if the proposer worker uses the KV cache or not.

        self.probs_dtype = self.spec_decode_sampler.probs_dtype
        self.token_id_dtype = self.spec_decode_sampler.token_id_dtype
        # Lazy initialization.
        self.scorer: SpeculativeScorer

        # Hidden states from target model to pass to proposer
        # in the subsequent step.
        self.previous_hidden_states: Optional[HiddenStates] = None
        self._disable_logprobs = disable_logprobs
        self._disable_log_stats = disable_log_stats

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
        self.spec_decode_sampler.init_gpu_tensors(self.rank)

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
        which significantly reduces overhead of sampling during verification.

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
        (self.scorer_worker.model_runner.model.sampler.
         should_modify_greedy_probs_inplace) = True
        self.proposer_worker.set_include_gpu_probs_tensor()
        self.proposer_worker.set_should_modify_greedy_probs_inplace()

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

        self._track_finished_requests(execute_model_req)
        disable_all_speculation = self._should_disable_all_speculation(
            execute_model_req)
        num_lookahead_slots = execute_model_req.num_lookahead_slots

        # Speculative decoding is disabled in the following cases:
        # 1. Prefill phase: Speculative decoding is not
        #    used during the prefill phase.
        # 2. Auto-disable enabled: The running queue size exceeds
        #    the specified threshold.
        # 3. No request: There are no requests in the batch, or
        #    none of the requests in the batch have spec decoding enabled.
        # In any of these cases, the proposer and scorer workers
        # are called normally.
        no_spec = num_lookahead_slots == 0 or disable_all_speculation or all(
            sgm.num_speculative_tokens == 0
            for sgm in execute_model_req.seq_group_metadata_list)

        # Broadcast how many lookahead slots are scheduled for this step, and
        # whether all speculation is disabled, to all non-driver workers.

        # This is required as if the number of draft model runs changes
        # dynamically, the non-driver workers won't know unless we perform a
        # communication to inform them.

        # no_spec is used to signal non-driver worker about prefill vs decode
        # stage. This is needed to ensure that order of execution of proposer
        # and scorer is same in both driver and non-driver workers (i.e.,
        # scorer -> proposer for prefill and proposer -> scorer in decode). This
        # order is needed to support models like EAGLE that take scorer states
        # as inputs.
        broadcast_dict = dict(
            num_lookahead_slots=num_lookahead_slots,
            no_spec=no_spec,
            disable_all_speculation=disable_all_speculation,
        )
        broadcast_tensor_dict(broadcast_dict, src=self._driver_rank)

        assert execute_model_req.seq_group_metadata_list is not None, (
            "speculative decoding requires non-None seq_group_metadata_list")

        self._maybe_disable_speculative_tokens(
            disable_all_speculation, execute_model_req.seq_group_metadata_list)

        if no_spec:
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
        return (execute_model_req.running_queue_size >=
                self.disable_by_batch_size)

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

    def _serialize_sampler_output_no_logprobs(
            self, execute_model_req: ExecuteModelRequest,
            sampler_output: SamplerOutput) -> SamplerOutput:
        """
        Creates and returns a `SamplerOutput` with only the sampled token IDs 
        being serialized to CPU & populated in `CompletionSequenceGroupOutput`.
        All other parameters in `CompletionSequenceGroupOutput` related to log 
        probabilities are skipped.

        Args:
            execute_model_req (ExecuteModelRequest): The model request that
            was executed.
            sampler_output (SamplerOutput): The output from the sampler with
            only GPU tensors populated.

        Returns:
            SamplerOutput: A new `SamplerOutput` instance containing a list of 
            `CompletionSequenceGroupOutput` objects with only sampled token
            IDs populated.
        """
        seq_ids = get_all_seq_ids(execute_model_req.seq_group_metadata_list)
        sampled_token_ids_list = sampler_output.sampled_token_ids.tolist()
        completion_seq_group_output_list: List[
            CompletionSequenceGroupOutput] = []
        for index, seq_id in enumerate(seq_ids):
            completion_seq_group_output_list.append(
                create_sequence_group_output(
                    token_id=sampled_token_ids_list[index][0],
                    token_id_logprob_rank=-1,
                    token_id_logprob=0.0,
                    seq_id=seq_id,
                    topk_token_ids=[],
                    topk_logprobs=[],
                ))
        return SamplerOutput(outputs=completion_seq_group_output_list)

    @nvtx_range("spec_decode_worker._run_no_spec")
    def _run_no_spec(self, execute_model_req: ExecuteModelRequest,
                     skip_proposer: bool) -> List[SamplerOutput]:
        """Run a single generation step without any speculation. The input is
        sent to the proposer and scorer model so that the KV cache is consistent
        between the two. When skip_proposer is True, the proposer model is
        not called, meaning that the kv-cache in proposer for requests is not
        updated, so they cannot enable spec decode in the rest decoding.
        """

        sampler_output = self.scorer_worker.execute_model(execute_model_req)
        assert len(sampler_output) == 1
        sampler_output = sampler_output[0]

        # Store hidden states from target model execution.
        hidden_states = sampler_output.hidden_states
        if hidden_states is not None:
            if self.previous_hidden_states is None:
                self.previous_hidden_states = HiddenStates(
                    hidden_states, execute_model_req.seq_group_metadata_list)
            else:
                self.previous_hidden_states.update(
                    hidden_states, execute_model_req.seq_group_metadata_list)

        if not skip_proposer:
            # We prepare the prefill hidden states here so that there no
            # additional complexity in worker for spec_decode vs non_spec_decode
            # flow and execute_model doesn't need additional modifications.
            execute_model_req.previous_hidden_states = \
                prepare_prefill_hidden_states(
                    sampler_output.prefill_hidden_states)

            self.proposer_worker.execute_model(execute_model_req)

        sampler_output_to_return = (self._serialize_sampler_output_no_logprobs(
            execute_model_req=execute_model_req, sampler_output=sampler_output)
                                    if self._disable_logprobs else
                                    sampler_output)

        # Clear device tensors from sampler output. This reduces communication
        # overhead when the engine runs in a different process than the workers.
        sampler_output.sampled_token_probs = None
        sampler_output.sampled_token_ids = None
        sampler_output.logprobs = None
        return [sampler_output_to_return]

    def _run_non_driver_rank(self) -> bool:
        """Run proposer and verifier model in non-driver workers. This is used
        for both speculation cases (num_lookahead_slots>0) and non-speculation
        cases (e.g. prefill).

        Returns True if there are remaining sequences to process.
        """
        assert self.rank != self._driver_rank

        data = broadcast_tensor_dict(src=self._driver_rank)
        if not data:
            return False
        num_lookahead_slots = data["num_lookahead_slots"]

        # In case of prefill, scorer_worker has to be run before proposer so
        # that the hidden states can be propagated to proposer when needed.
        if data["no_spec"]:
            self.scorer_worker.execute_model()

        if not data["disable_all_speculation"]:
            # Even if num_lookahead_slots is zero, we want to run the
            # proposer model as it may have KV.
            #
            # We run the proposer once per lookahead slot. In the future we
            # should delegate how many times it runs to the proposer.
            for _ in range(max(num_lookahead_slots, 1)):
                self.proposer_worker.execute_model()

        if not data["no_spec"]:
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

        # Pass last hidden states from target model to proposer
        execute_model_req.previous_hidden_states = self.previous_hidden_states
        self.previous_hidden_states = None

        with Timer() as proposal_timer:
            # Generate proposals using draft worker.
            proposals = self.proposer_worker.get_spec_proposals(
                execute_model_req, self._seq_with_bonus_token_in_last_step)

        if not self._allow_zero_draft_token_step and proposals.no_proposals:
            #TODO: Fix it #5814
            raise RuntimeError("Cannot handle cases where distributed draft "
                               "workers generate no tokens")

        execute_model_req.previous_hidden_states = None

        with Timer() as scoring_timer:
            proposal_scores = self.scorer.score_proposals(
                execute_model_req,
                proposals,
            )

        with Timer() as verification_timer:
            accepted_token_ids, target_logprobs = self._verify_tokens(
                execute_model_req.seq_group_metadata_list, proposal_scores,
                proposals, execute_model_req.num_lookahead_slots)

        stage_times = (proposal_timer.elapsed_time_ms / num_lookahead_slots,
                       scoring_timer.elapsed_time_ms,
                       verification_timer.elapsed_time_ms)

        return self._create_output_sampler_list(
            execute_model_req.seq_group_metadata_list,
            accepted_token_ids,
            target_logprobs=target_logprobs,
            k=execute_model_req.num_lookahead_slots,
            stage_times=stage_times)

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
        (_, spec_indices), (_, non_spec_indices) = split_batch_by_proposal_len(
            seq_group_metadata_list, proposal_lens_list)
        original_indices = spec_indices + non_spec_indices

        # Get probabilities of target model, including bonus tokens.
        proposal_verifier_probs = proposal_scores.probs[spec_indices]

        # Get non-speculative sampled tokens from target model.
        non_spec_token_ids = proposal_scores.token_ids[non_spec_indices]

        # Get bonus tokens from target model.
        bonus_token_ids = proposal_scores.token_ids[spec_indices, -1:]

        # Get probabilities according to proposal method.
        proposal_probs = proposals.proposal_probs[spec_indices]

        # Get proposed tokens.
        proposal_token_ids = proposals.proposal_token_ids[spec_indices]

        # Sampler arguments
        sampler_extra_kwargs: Dict[str, Any] = {}
        if self.generators and isinstance(self.spec_decode_sampler,
                                          SpecDecodeStochasticBaseSampler):
            sampler_extra_kwargs["seeded_seqs"] = {
                idx: self.generators[sgm.request_id]
                for idx, sgm in enumerate(seq_group_metadata_list)
                if sgm.sampling_params.seed is not None
            }

        accepted_token_ids = self.spec_decode_sampler(
            target_with_bonus_probs=proposal_verifier_probs,
            bonus_token_ids=bonus_token_ids,
            draft_probs=proposal_probs,
            draft_token_ids=proposal_token_ids,
            **sampler_extra_kwargs,
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

        hidden_states = proposal_scores.hidden_states
        if hidden_states is not None:
            # Contract hidden states based on accepted tokens
            hs_size = hidden_states.shape[-1]

            accepted_index = accepted_token_ids + 1  # Convert -1 to 0
            accepted_index = accepted_index.count_nonzero(dim=1).add_(-1)
            index = accepted_index[:, None, None].expand(-1, 1, hs_size)
            second_last_token_hidden_states = hidden_states[:, -2]  # b x d
            hidden_states = hidden_states.gather(1, index).squeeze(1)  # b x d
            # Store hidden states from target model for subsequent decode step
            self.previous_hidden_states = HiddenStates(
                hidden_states, seq_group_metadata_list,
                second_last_token_hidden_states)

        return accepted_token_ids, logprobs

    def _create_output_sampler_list(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        accepted_token_ids: torch.Tensor,  # shape: [batch_size, k+1]
        target_logprobs: torch.Tensor,  # shape: [batch_size, k+1, vocab_size]
        k: int,
        stage_times: Tuple[float, float, float],
    ) -> List[SamplerOutput]:
        """Given the accepted token ids, create a list of SamplerOutput.

        The output is padded with -1 tokens such that each sequence has
        the same number of outputs.
        """
        batch_size, num_steps = accepted_token_ids.shape
        accepted_token_ids_by_step = accepted_token_ids.transpose(0, 1)
        if self._disable_logprobs:
            # We are skipping the logprobs. Hence don't serialize the
            # logprobs related tensors from the GPU. Instead create
            # empty/dummy lists.
            (accepted_token_id_ranks_by_step,
            accepted_token_id_logprobs_by_step,
            topk_logprobs_by_step, topk_indices_by_step) =\
            self._create_dummy_logprob_lists(
                batch_size, num_steps,
                self.scorer_worker.model_config.max_logprobs)
        else:
            # Organize input tensors by step instead of by sequence.
            target_logprobs_by_step = target_logprobs.transpose(0, 1)
            # Serialize all tensors into Python lists.
            (accepted_token_id_ranks_by_step,
            accepted_token_id_logprobs_by_step,
            topk_logprobs_by_step, topk_indices_by_step) =\
                self._create_logprob_lists_from_tensors(
                    target_logprobs_by_step, accepted_token_ids_by_step,
                    self.scorer_worker.model_config.max_logprobs)

        # Get the sequence ids and num_logprobs (sampling parameter) in the
        # batch.
        seq_ids, request_ids_seq_ids_mapping = get_all_seq_ids_and_request_ids(
            seq_group_metadata_list)

        num_logprobs_per_seq = get_all_num_logprobs(seq_group_metadata_list)

        # Serialize tensor to CPU Python list.
        accepted_token_ids_by_step = accepted_token_ids_by_step.tolist()

        # Construct the output on a per-step, per-sequence basis.
        sampler_output_list: List[SamplerOutput] = []
        for step_index in range(num_steps):
            if all(token_id == -1
                   for token_id in accepted_token_ids_by_step[step_index]):
                break

            step_output_token_ids: List[CompletionSequenceGroupOutput] = []
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

        # Populate the data structures needed to keep track of sequences with
        # bonus tokens.
        self._track_sequences_with_bonus_tokens(seq_ids,
                                                request_ids_seq_ids_mapping,
                                                accepted_token_ids_by_step)
        maybe_rejsample_metrics = (
            self._metrics.maybe_collect_rejsample_metrics(k))
        if maybe_rejsample_metrics is not None:
            sampler_output_list[
                0].spec_decode_worker_metrics = maybe_rejsample_metrics

            # Log time spent in each stage periodically.
            # This is periodic because the rejection sampler emits metrics
            # periodically.
            self._maybe_log_stage_times(*stage_times)

        return sampler_output_list

    def _maybe_log_stage_times(self, average_time_per_proposal_tok_ms: float,
                               scoring_time_ms: float,
                               verification_time_ms: float) -> None:
        """Log the speculative stage times. If stat logging is disabled, do
        nothing.
        """
        if self._disable_log_stats:
            return

        logger.info(
            "SpecDecodeWorker stage times: "
            "average_time_per_proposal_tok_ms=%.02f "
            "scoring_time_ms=%.02f verification_time_ms=%.02f",
            average_time_per_proposal_tok_ms, scoring_time_ms,
            verification_time_ms)

    def _create_dummy_logprob_lists(
        self,
        batch_size: int,
        num_steps: int,
        num_top_k: int,
    ) -> Tuple[List[List[int]], List[List[float]],
               List[List[List[Optional[float]]]],
               List[List[List[Optional[int]]]]]:
        """
        Creates and returns four dummy lists representing token probabilities 
        and their ranks.

        This method initializes and returns:
            - The ranks of the accepted tokens, shaped (num_steps, batch_size)
            - The log probabilities of the accepted tokens,
              shaped (num_steps, batch_size)
            - The log probabilities of the top k tokens,
              shaped (num_steps, batch_size, num_top_k)
            - The token IDs of the top k tokens,
              shaped (num_steps, batch_size, num_top_k)

        Args:
            batch_size (int): The size of the batch.
            num_steps (int): The number of steps in the sequence.
            num_top_k (int): The number of top-k token log probabilities to
            return.
        
        Returns:
            A tuple containing four dummy lists as described above.
        """
        accepted_token_id_ranks_by_step = [[-1] * batch_size
                                           for _ in range(num_steps)]
        accepted_token_id_logprobs_by_step = [[0.0] * batch_size
                                              for _ in range(num_steps)]
        topk_logprobs_by_step: List[List[List[Optional[float]]]] = [[
            [None] * num_top_k for _ in range(batch_size)
        ] for _ in range(num_steps)]
        topk_indices_by_step: List[List[List[Optional[int]]]] = [[
            [None] * num_top_k for _ in range(batch_size)
        ] for _ in range(num_steps)]
        return (accepted_token_id_ranks_by_step,
                accepted_token_id_logprobs_by_step, topk_logprobs_by_step,
                topk_indices_by_step)

    def _create_logprob_lists_from_tensors(
        self,
        target_logprobs_by_step: torch.Tensor,
        accepted_token_ids_by_step: torch.Tensor,
        num_top_k: int,
    ) -> Tuple[List[List[int]], List[List[float]],
               List[List[List[Optional[float]]]],
               List[List[List[Optional[int]]]]]:
        """
        Creates and returns four lists representing token probabilities and
        their ranks.

        This method initializes and returns four lists containing:
            - The ranks of the accepted tokens, shaped (num_steps, batch_size)
            - The log probabilities of the accepted tokens,
              shaped (num_steps, batch_size)
            - The log probabilities of the top k tokens,
              shaped (num_steps, batch_size, num_top_k)
            - The token IDs of the top k tokens,
              shaped (num_steps, batch_size, num_top_k)

        Args:
            target_logprobs_by_step (torch.Tensor): Tensor representing the
            log probabilities of the target model,
            shaped (num_steps, batch_size, vocab_size)
            accepted_token_ids_by_step (torch.Tensor): Tensor representing
            the accepted  token_ids, shaped (num_steps, batch_size) 
            num_top_k (int): The number of top-k token log probabilities to
            return.
        
        Returns:
            A tuple containing the lists as described above.
        """
        # Serialize all tensors to CPU Python lists.
        # Get the logprobs/rank of the accepted tokens.
        (accepted_token_id_ranks_by_step_tensor,
         accepted_token_id_logprobs_by_step_tensor
         ) = get_sampled_token_logprobs(
             logprob_tensor=target_logprobs_by_step,
             sampled_token_ids=accepted_token_ids_by_step,
         )
        # Get the top-k logprobs (which may or may not include the
        # logprob of the accepted token).
        (topk_logprobs_by_step_tensor,
         topk_indices_by_step_tensor) = target_logprobs_by_step.topk(
             k=num_top_k,
             dim=-1,
         )
        accepted_token_id_ranks_by_step = (
            accepted_token_id_ranks_by_step_tensor.tolist())
        accepted_token_id_logprobs_by_step = (
            accepted_token_id_logprobs_by_step_tensor.tolist())
        topk_logprobs_by_step = topk_logprobs_by_step_tensor.tolist()
        topk_indices_by_step = topk_indices_by_step_tensor.tolist()
        return (accepted_token_id_ranks_by_step,
                accepted_token_id_logprobs_by_step, topk_logprobs_by_step,
                topk_indices_by_step)

    def _track_finished_requests(self, execute_model_req: ExecuteModelRequest):
        """
        Removes the finished requests and their associated sequence ids from
        internal book keeping data structures.
        """
        for finished_request in execute_model_req.finished_requests_ids:
            for seq_id in self._request_id_seq_id_mapping[finished_request]:
                self._seq_with_bonus_token_in_last_step.discard(seq_id)
            del self._request_id_seq_id_mapping[finished_request]

    def _track_sequences_with_bonus_tokens(
            self, seq_ids: List[int],
            request_ids_seq_ids_mapping: Dict[str, Set[int]],
            accepted_token_ids_by_step: List[List[int]]):
        """
        Updates the internal data structures which keep track of sequences
        which have been assigned bonus tokens in their last forward pass.
        """
        for seq_index, seq_id in enumerate(seq_ids):
            last_token_id = accepted_token_ids_by_step[-1][seq_index]
            if last_token_id == -1:
                self._seq_with_bonus_token_in_last_step.discard(seq_id)
            else:
                self._seq_with_bonus_token_in_last_step.add(seq_id)
        for request_id, sequences in request_ids_seq_ids_mapping.items():
            self._request_id_seq_id_mapping[request_id].update(sequences)

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


def prepare_prefill_hidden_states(
        prefill_hidden_states: torch.Tensor) -> HiddenStates:
    # For prefill step in proposer, we run the model for N-1 tokens
    # because Nth token will be processed in the first decode step. For
    # N-1 tokens, the input should be 0:N-1 hidden states which should
    # be concatanated with 1:N token (since output of scorer has to be
    # the input for proposer). Therefore, we shift the hidden states to
    # align n-1th hidden state with nth token.
    return HiddenStates(prefill_hidden_states.roll(
        shifts=1, dims=0)) if prefill_hidden_states is not None else None
