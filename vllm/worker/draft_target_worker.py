from typing import Iterator, List, Tuple, Optional, Union
from itertools import chain, count
from functools import cached_property
import logging
import time

import msgspec
import torch
import traceback

from vllm.anyscale.shm.msgspec_shm import SharedMsgspecBufferWithEvent
from vllm.sequence import (SamplerOutput, SequenceGroupMetadata,
                           ExecuteModelData, SequenceOutputs, SequenceData,
                           SequenceGroupOutputs, DraftTargetWorkerMetrics)
from vllm.worker.worker import Worker
from vllm.worker.multi_step_worker import MultiStepWorker
from vllm.worker.single_tp_worker import SingleTpWorker
from vllm.model_executor.layers.rejection_sampler import RejectionSampler
from vllm.model_executor.parallel_utils.parallel_state import get_tensor_model_parallel_group
from vllm.config import CacheConfig
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.base_worker import BaseWorker
from vllm.anyscale.profiler_utils import TorchProfiler, nvtx_range, Profilable
from vllm.model_executor.layers.sampler import RawSamplerOutput
from vllm.utils import in_wsl

SeqId = int
TargetSeqId = int
TokenId = int

logger = logging.getLogger(__name__)


class DraftTargetWorker(Profilable, BaseWorker):
    """Worker which implements speculative decoding via a draft model for
    proposing tokens and a target model for verifying tokens. A modified form of
    rejection sampling is applied to generate the output tokens.

    Scoring is done by querying the target model for a prefix context for each
    speculative token, and an additional prefix for a bonus token.

    For example, given the previously-generated sequence:
        [0, 1, 2, 3, 4, 5]

    and the continuation proposed by the draft model (k=3):
        [6, 7, 8]

    the DraftTargetWorker will query the target model for the probability
    distribution over token ids given the following contexts:
        [0, 1, 2, 3, 4, 5] # used to score [6]
        [0, 1, 2, 3, 4, 5, 6] # used to score [6, 7]
        [0, 1, 2, 3, 4, 5, 6, 7] # used to score [6, 7, 8]
        [0, 1, 2, 3, 4, 5, 6, 7, 8] # used to generate a bonus token

    The output of the first model for each context is then used to accept or
    reject the proposed draft tokens.
    """

    @classmethod
    def from_workers(cls, draft_worker: Union[MultiStepWorker, SingleTpWorker],
                     target_worker: Worker) -> "DraftTargetWorker":
        return cls(draft_worker, target_worker, RejectionSampler())

    def __init__(
        self,
        draft_worker: Union[MultiStepWorker, SingleTpWorker],
        target_worker: Worker,
        rejection_sampler: RejectionSampler,
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

        # We don't have a device set yet.
        self._copy_stream: Optional[torch.cuda.Stream] = None

        self.probs_dtype = self.rejection_sampler.probs_dtype
        self.token_id_dtype = self.rejection_sampler.token_id_dtype

        self._profiler = TorchProfiler()

        pin_memory = not in_wsl()
        self._aggregate_num_accepted_tokens = torch.tensor(
            0, dtype=torch.long, device="cpu", pin_memory=pin_memory)
        self._aggregate_num_emitted_tokens = torch.tensor(
            0, dtype=torch.long, device="cpu", pin_memory=pin_memory)
        self._aggregate_num_draft_tokens = 0

        self._rejsample_metrics_collect_interval_s = 5.0
        self._last_metrics_collect_time = 0

    def _configure_samplers(self):
        """Configure model samplers to return a probability tensor in the
        SamplerOutput. This simplifies the data wrangling logic in speculative
        decoding.
        """
        self.draft_worker.model.sampler.include_gpu_probs_tensor = (True)
        self.target_worker.model.sampler.include_gpu_probs_tensor = True

    def init_model(self):
        # Intitialize the target model before the draft model.
        # This allows the draft model to have a smaller TP degree than the
        # larger model without refactors to parallel_state.
        self.target_worker.init_model()
        self.draft_worker.init_model()
        self._configure_samplers()

        self.device = self.target_worker.device
        self._copy_stream = torch.cuda.Stream()

        self.rejection_sampler.init_gpu_tensors(self.rank)

    def profile_num_available_blocks(self, block_size: int,
                                     gpu_memory_utilization: float,
                                     cpu_swap_space: int):
        num_gpu_blocks, num_cpu_blocks = (
            self.target_worker.profile_num_available_blocks(
                block_size, gpu_memory_utilization, cpu_swap_space))

        new_num_gpu_blocks = self._calculate_gpu_blocks(
            block_size, num_gpu_blocks)
        return new_num_gpu_blocks, num_cpu_blocks

    def _calculate_gpu_blocks(self, block_size: int,
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
        target_kv_size_bytes = CacheEngine.get_cache_block_size(
            block_size,
            self.target_worker.model_config,
            self.target_worker.parallel_config,
        )

        draft_kv_size_bytes = CacheEngine.get_cache_block_size(
            block_size,
            self.draft_worker.model_config,
            self.draft_worker.parallel_config,
        )

        new_num_gpu_blocks = int(total_num_gpu_blocks * target_kv_size_bytes /
                                 (draft_kv_size_bytes + target_kv_size_bytes))

        return new_num_gpu_blocks

    def init_cache_engine(self, cache_config: CacheConfig):
        self.target_worker.init_cache_engine(cache_config)
        self.draft_worker.init_cache_engine(cache_config)

    @property
    def rank(self):
        return self.target_worker.rank

    def get_metadata_cache_len(self) -> int:
        """Metadata cache not currently supported.
        """
        return 0

    def get_runtime_context(self) -> Optional[dict]:
        return self.target_worker.get_runtime_context()

    def _get_max_model_len(self) -> Tuple[int, int]:
        draft_max_model_len = (self.draft_worker.model_config.max_model_len)
        target_max_model_len = (self.target_worker.model_config.max_model_len)

        assert draft_max_model_len is not None
        assert target_max_model_len is not None

        return draft_max_model_len, target_max_model_len

    @staticmethod
    def _get_k_from_execute_model_data(
            execute_model_data: ExecuteModelData) -> int:
        """Given an ExecuteModelData, determine the number of speculative
        tokens (k). This is equal to the number of preallocated slots as each
        speculative token requires a KV slot.
        """
        k = execute_model_data.num_preallocated_slots
        assert k >= 0, f"Expected {k=} >= 0"
        return k

    @staticmethod
    def _get_draft_num_preallocated_slots_from_k(k: int) -> int:
        """Given the number of speculative tokens, return the number of
        preallocated slots to give to the draft model. This is equal to k - 1,
        because the draft model will not store KV for the last of the k
        generated tokens.
        """
        num_preallocated_slots = k - 1
        assert num_preallocated_slots >= 0, ("Expected "
                                             f"{num_preallocated_slots=} >= 0")
        return num_preallocated_slots

    def execute_model_shared_memory(
            self,
            shared_memory_input: SharedMsgspecBufferWithEvent,
            shared_memory_output: SharedMsgspecBufferWithEvent,
            participant_id: int  # pylint: disable=unused-argument
    ):
        shared_memory_input.decoder = msgspec.msgpack.Decoder(ExecuteModelData)
        logger.info("Worker shared memory input buffer id: "
                    f"{shared_memory_input.participant_id}")
        logger.info("Worker shared memory output buffer id: "
                    f"{shared_memory_input.participant_id}")
        parallel_group = get_tensor_model_parallel_group()
        try:
            while True:
                shared_memory_input.wait_for_incoming_data()
                data = shared_memory_input.get_data()
                torch.distributed.barrier(group=parallel_group)
                shared_memory_input.clear()
                outputs = self.execute_model(data)
                if self.rank < 1:
                    shared_memory_output.set_data(outputs)
        except Exception:
            traceback.print_exc()
            shared_memory_output.set_error()
            raise

    @torch.inference_mode()
    @nvtx_range("draft_target_worker.execute_model")
    def execute_model(
        self,
        execute_model_data: ExecuteModelData,
        *,
        return_python_output: bool = True  # pylint: disable=unused-argument
    ) -> List[SamplerOutput]:

        k = self._get_k_from_execute_model_data(execute_model_data)
        if k == 0:
            return self._run_prefill(execute_model_data)

        if len(execute_model_data.seq_group_metadata_list) == 0:
            return self._run_for_empty_input(execute_model_data)

        return self._run_speculative_decoding_step(execute_model_data, k)

    @nvtx_range("draft_target_worker._run_prefill")
    def _run_prefill(
            self, execute_model_data: ExecuteModelData) -> List[SamplerOutput]:
        """Run a prefill step, without any speculation. The input is sent to the
        draft and target model so that prompt KV are stored in both caches.
        """
        assert self._is_prefill(execute_model_data)
        assert execute_model_data.num_preallocated_slots == 0, (
            "Expected "
            f"{execute_model_data.num_preallocated_slots=} to be zero during "
            "prefill.")

        logger.debug("draft prefill")
        self.draft_worker.execute_model(execute_model_data,
                                        return_python_output=False)

        logger.debug("target worker prefill")
        sampler_output, = self.target_worker.execute_model(execute_model_data)

        # Do not want PyTorch tensors transferred back.
        sampler_output.probs = None
        sampler_output.sampled_tokens = None
        return [sampler_output]

    def _is_prefill(self, execute_model_data: ExecuteModelData) -> bool:
        """Returns whether or not the input ExecuteModelData is prefill or not.
        """
        return any(seq_group_metadata.is_prompt for seq_group_metadata in
                   execute_model_data.seq_group_metadata_list)

    def _run_for_empty_input(
            self, execute_model_data: ExecuteModelData) -> List[SamplerOutput]:
        """If there are no sequences in the input, simply call the models with
        the inpiut. This allows them to process metadata, such as cleaning up
        after a request finishes.
        """
        self.draft_worker.execute_model(execute_model_data,
                                        return_python_output=False)
        target_output, = self.target_worker.execute_model(execute_model_data)
        return [target_output]

    @nvtx_range("draft_target_worker._run_speculative_decoding_step")
    def _run_speculative_decoding_step(
        self,
        execute_model_data: ExecuteModelData,
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

        (spec_seqs, non_spec_seqs, all_seqs, original_indices,
         original_indices_ready) = self._get_seqs_for_spec_decode(
             execute_model_data.seq_group_metadata_list, k)

        proposal_token_ids, proposal_probs = self._get_proposals(
            execute_model_data, spec_seqs, k)

        should_collect_rejsample_metrics = (
            self._should_collect_rejsample_metrics(time.time()))
        if should_collect_rejsample_metrics:
            aggregate_metrics_ready = self._copy_rejsample_metrics_async()

        logger.debug("scoring draft tokens")
        (proposal_scores, bonus_token_ids,
         non_spec_token_ids) = self._score_proposals(execute_model_data,
                                                     proposal_token_ids,
                                                     spec_seqs, non_spec_seqs)

        with nvtx_range("draft_target_worker.rejection_sampler"):
            accepted_token_ids = self.rejection_sampler(
                proposal_scores,
                bonus_token_ids,
                proposal_probs,
                proposal_token_ids,
            )

        # Append output tokens from non-speculative sequences to
        # the accepted token ids tensor.
        non_spec_token_ids = non_spec_token_ids.expand(-1, k + 1).clone()
        non_spec_token_ids[:, 1:] = -1
        accepted_token_ids = torch.cat(
            [accepted_token_ids, non_spec_token_ids])

        # Rearrange so that results are in the order of the original seq group
        # metadata.
        torch.cuda.current_stream().wait_event(original_indices_ready)
        accepted_token_ids = accepted_token_ids[original_indices]

        # Construct output.
        seq_ids = self._get_all_seq_ids(all_seqs)
        sampler_output = self._create_output_sampler_list(
            seq_ids, accepted_token_ids)

        if should_collect_rejsample_metrics:
            self._last_metrics_collect_time = time.time()
            metrics = self._collect_rejsample_metrics(k,
                                                      aggregate_metrics_ready)
            sampler_output[0].draft_target_worker_metrics = metrics

        return sampler_output

    def _get_seqs_for_spec_decode(
        self, seq_group_metadata_list: List[SequenceGroupMetadata], k: int
    ) -> Tuple[List[SequenceGroupMetadata], List[SequenceGroupMetadata],
               List[SequenceGroupMetadata], torch.Tensor, torch.cuda.Event]:
        """Determine which sequences are eligible for speculative decoding.

        Any sequence which would go over the model max len is ineligible.
        """

        all_seqs: List[SequenceGroupMetadata] = seq_group_metadata_list
        spec_seqs: List[SequenceGroupMetadata] = []
        non_spec_seqs: List[SequenceGroupMetadata] = []

        # Indices of each seq group in the original seq_group_metadata_list.
        non_spec_indices: List[int] = []
        spec_indices: List[int] = []

        # Maximum number of new tokens in speculative decoding.
        max_num_new_tokens = k + 1

        for i, seq_group_metadata in enumerate(seq_group_metadata_list):
            # Only one sequence per group in speculative decoding.
            seq_data = next(iter(seq_group_metadata.seq_data.values()))
            draft_max_model_len, target_max_model_len = self._get_max_model_len(
            )
            max_model_len = min(draft_max_model_len, target_max_model_len)
            seq_len = seq_data.get_len()

            # If the sequence would go over the model limit in speculative
            # decoding, then we should perform normal decoding.
            if seq_len + max_num_new_tokens > max_model_len:
                non_spec_seqs.append(seq_group_metadata)
                non_spec_indices.append(i)
                continue

            # Otherwise, we can do speculative decoding.
            spec_seqs.append(seq_group_metadata)
            spec_indices.append(i)

        # Return the original indices so that the output order can be preserved.
        # During scoring, the speculative sequences are placed first, then the
        # non-speculative sequences.
        #
        # This async copy is waited upon before constructing final output.
        with torch.cuda.stream(self._copy_stream):
            original_indices = torch.tensor(spec_indices + non_spec_indices,
                                            dtype=torch.long,
                                            pin_memory=True)
            original_indices = original_indices.to(device=self.device,
                                                   non_blocking=True)
            original_indices_ready = torch.cuda.Event()
            original_indices_ready.record()
        return (spec_seqs, non_spec_seqs, all_seqs, original_indices,
                original_indices_ready)

    @nvtx_range("draft_target_worker._get_proposals")
    def _get_proposals(self, execute_model_data: ExecuteModelData,
                       spec_seqs: List[SequenceGroupMetadata],
                       k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get proposal tokens from the draft model.
        """

        # If there are no sequences to generate draft tokens for, return empty
        # tensors.
        if not spec_seqs:
            return torch.zeros(0,
                               k,
                               dtype=self.token_id_dtype,
                               device=self.device), torch.zeros(
                                   0,
                                   k,
                                   self._vocab_size,
                                   dtype=self.probs_dtype,
                                   device=self.device)

        execute_model_data.num_preallocated_slots = (
            self._get_draft_num_preallocated_slots_from_k(k))
        execute_model_data.seq_group_metadata_list = spec_seqs

        sampler_output_list = self.draft_worker.execute_model(
            execute_model_data, return_python_output=False)

        # vLLM currently stores results in Python datastructures. We convert to
        # torch to use in rejection sampling and to simplify data
        # transformations.
        proposal_token_ids, proposal_probs = self._sampler_output_to_torch(
            sampler_output_list)

        return proposal_token_ids, proposal_probs

    @nvtx_range("draft_target_worker._score_proposals")
    def _score_proposals(
        self,
        execute_model_data: ExecuteModelData,
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
            ExecuteModelData(
                target_seq_group_metadata_list,
                execute_model_data.finished_request_ids_list,
                execute_model_data.blocks_to_swap_in,
                execute_model_data.blocks_to_swap_out,
                execute_model_data.blocks_to_copy,
                num_preallocated_slots=0,
            ),
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
                    SequenceGroupOutputs(
                        samples=[
                            SequenceOutputs(
                                parent_seq_id=seq_id,
                                output_token=token_id,
                                # TODO currently rejection sampling does not
                                # emit probs, so this value is meaningless.
                                logprobs={token_id: 0},
                            )
                        ],
                        prompt_logprobs=None,
                    ))
            sampler_output_list.append(
                SamplerOutput(outputs=step_output_token_ids))

        return sampler_output_list

    def _sampler_output_to_torch(
        self,
        sampler_output_list: List[RawSamplerOutput],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Utility function which converts a list of SamplerOutput to tensors.

        Returns:
            token_ids: torch.Tensor
                shape: [batch_size, len(sampler_output_list)]

            probs: torch.Tensor
                shape: [batch_size, len(sampler_output_list), vocab_size]
        """

        # shape: [batch_size, num_sampler_output, vocab_size]
        probs = torch.stack(
            [sampler_output.probs for sampler_output in sampler_output_list],
            dim=0,
        ).transpose(0, 1)

        # shape: [batch_size, num_sampler_output]
        token_ids = torch.stack(
            [
                sampler_output.sampled_tokens.flatten()
                for sampler_output in sampler_output_list
            ],
            dim=0,
        ).transpose(0, 1)

        return token_ids, probs

    def _should_collect_rejsample_metrics(self, now: float) -> bool:
        """Return whether or not this iteration should print rejection sampling
        metrics.
        """
        if self.rank != 0:
            return False

        if (now - self._last_metrics_collect_time <
                self._rejsample_metrics_collect_interval_s):
            return False
        return True

    def _copy_rejsample_metrics_async(self) -> torch.cuda.Event:
        """Copy rejection sampling metrics (number of accepted tokens, etc) to
        CPU asynchronously.

        Returns a CUDA event recording when the copy is complete.
        """
        self._copy_stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(self._copy_stream):
            self._aggregate_num_accepted_tokens.copy_(
                self.rejection_sampler.num_accepted_tokens, non_blocking=True)
            self._aggregate_num_emitted_tokens.copy_(
                self.rejection_sampler.num_emitted_tokens, non_blocking=True)
            # Number of draft tokens is calculated on CPU, so no copy is
            # required.
            self._aggregate_num_draft_tokens = (
                self.rejection_sampler.num_draft_tokens)

        aggregate_metrics_ready = torch.cuda.Event()
        aggregate_metrics_ready.record(self._copy_stream)

        return aggregate_metrics_ready

    def _collect_rejsample_metrics(
            self, k: int,
            ready_event: torch.cuda.Event) -> DraftTargetWorkerMetrics:
        """Create metrics object from statistics copied asynchronously.

        Args:
            k: int. The number of speculative tokens; used to determine system
                efficiency.
            ready_event: torch.cuda.Event. The CUDA event recording when the
                async GPU->CPU copy is complete.
        """

        ready_event.synchronize()
        accepted_tokens = self._aggregate_num_accepted_tokens.item()
        emitted_tokens = self._aggregate_num_emitted_tokens.item()
        draft_tokens = self._aggregate_num_draft_tokens

        # Divide by k since batch size can be variable.
        num_possible_tokens = (draft_tokens / k) * (k + 1)

        if draft_tokens > 0:
            draft_acceptance_rate = accepted_tokens / draft_tokens
        else:
            draft_acceptance_rate = float("nan")

        if num_possible_tokens > 0:
            system_efficiency = emitted_tokens / num_possible_tokens
        else:
            system_efficiency = float("nan")

        return DraftTargetWorkerMetrics(
            num_spec_tokens=k,
            draft_acceptance_rate=draft_acceptance_rate,
            system_efficiency=system_efficiency,
            accepted_tokens=accepted_tokens,
            draft_tokens=draft_tokens,
            emitted_tokens=emitted_tokens,
        )

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

    def _split_scoring_output(
        self, sampler_output: RawSamplerOutput, num_scoring_tokens: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split the target model output into speculative and non-speculative
        output.
        """

        # First samples are from speculative scoring, latter samples are non-
        # speculative samples.
        split_sizes = [
            num_scoring_tokens,
            sampler_output.sampled_tokens.numel() - num_scoring_tokens
        ]
        (spec_probs, non_spec_probs) = sampler_output.probs.split(split_sizes)
        (spec_sampled_tokens, non_spec_sampled_tokens
         ) = sampler_output.sampled_tokens.flatten().split(split_sizes)

        # Convert scores to tensors.
        sampler_output.probs = spec_probs
        sampler_output.sampled_tokens = spec_sampled_tokens
        target_token_ids, target_probs = self._sampler_output_to_torch(
            [sampler_output])

        # Convert non-speculative output tokens to tensors.
        sampler_output.probs = non_spec_probs
        sampler_output.sampled_tokens = non_spec_sampled_tokens
        non_spec_target_token_ids, _ = self._sampler_output_to_torch(
            [sampler_output])

        return target_token_ids, target_probs, non_spec_target_token_ids

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

        # The first scoring seqeuence will include the normal number of
        # processed tokens. This allows it to write KV from the previous
        # iteration.
        #
        # Subsequent scoring sequences only include a single unprocessed token;
        # the token they score.
        if len(token_ids) == 0:
            num_processed_token_ids = seq_data.get_num_processed_token_ids()
        else:
            num_processed_token_ids = seq_data.get_len() + len(token_ids) - 1

        return SequenceGroupMetadata(
            request_id=seq_group_metadata.request_id,
            is_prompt=seq_group_metadata.is_prompt,
            is_chunked_prefill=seq_group_metadata.is_chunked_prefill,
            seq_data={
                target_seq_id:
                SequenceData(
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

    @cached_property
    def _vocab_size(self) -> int:
        """Get the vocab size of the model and make sure it's consistent between
        draft and target workers.
        """
        vocab_sizes = [
            worker.model.config.vocab_size
            for worker in [self.draft_worker, self.target_worker]
        ]
        assert all(vocab_sizes[0] == vocab_size for vocab_size in vocab_sizes)
        return vocab_sizes[0]

    def start_profile(self, **kwargs) -> None:
        self._profiler.start_profile(**kwargs)

    def stop_profile(self) -> None:
        self._profiler.stop_profile()
