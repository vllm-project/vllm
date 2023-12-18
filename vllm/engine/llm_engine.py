import time
from functools import partial
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Tuple, Union

import msgspec

from vllm.anyscale.shm.msgspec_shm import RayEvent, SharedMsgspecBufferWithEvent, SharedMemoryManager
from vllm.anyscale.lora.utils import LoRARequest
from vllm.anyscale.tokenization import TransformersTokenizer, RayTokenizerPool
from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, LoadConfig, SpeculativeConfig,
                         LoRAConfig)
from vllm.core.scheduler import Scheduler, SchedulerOutputs
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.ray_utils import RayWorker, initialize_cluster, ray
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import (SamplerOutput, Sequence, SequenceGroup,
                           SequenceGroupMetadata, SequenceOutputs,
                           SequenceGroupOutputs, SequenceStatus,
                           ExecuteModelData, SequenceGroupMetadataDelta,
                           DraftTargetWorkerMetrics)
from vllm.transformers_utils.tokenizer import detokenize_incrementally
from vllm.utils import Counter
from vllm.worker.base_worker import BaseLoraWorker
from vllm.anyscale.profiler_utils import TorchProfiler

if ray:
    from ray.air.util.torch_dist import init_torch_dist_process_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy, NodeAffinitySchedulingStrategy

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup
    from vllm.worker.worker import Worker  # pylint: disable=ungrouped-imports

logger = init_logger(__name__)

_LOGGING_INTERVAL_SEC = 5
SHARED_MEMORY_BUFFER_SIZE = int(5e+7)  # 50 MB


class LLMEngine:
    """An LLM engine that receives requests and generates texts.

    This is the main class for the vLLM engine. It receives requests
    from clients and generates texts from the LLM. It includes a tokenizer, a
    language model (possibly distributed across multiple GPUs), and GPU memory
    space allocated for intermediate states (aka KV cache). This class utilizes
    iteration-level scheduling and efficient memory management to maximize the
    serving throughput.

    The `LLM` class wraps this class for offline batched inference and the
    `AsyncLLMEngine` class wraps this class for online serving.

    NOTE: The config arguments are derived from the `EngineArgs` class. For the
    comprehensive list of arguments, see `EngineArgs`.

    Args:
        model_config: The configuration related to the LLM model.
        cache_config: The configuration related to the KV cache memory
            management.
        parallel_config: The configuration related to distributed execution.
        scheduler_config: The configuration related to the request scheduler.
        distributed_init_method: The initialization method for distributed
            execution. See `torch.distributed.init_process_group` for details.
        placement_group: Ray placement group for distributed execution.
            Required for distributed execution.
        log_stats: Whether to log statistics.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        load_config: LoadConfig,
        speculative_config: SpeculativeConfig,
        lora_config: Optional[LoRAConfig],
        distributed_init_method: str,
        placement_group: Optional["PlacementGroup"],
        log_stats: bool,
    ) -> None:
        logger.info(
            "Initializing an LLM engine with config: "
            f"model={model_config.model!r}, "
            f"tokenizer={model_config.tokenizer!r}, "
            f"tokenizer_mode={model_config.tokenizer_mode}, "
            f"revision={model_config.revision}, "
            f"tokenizer_revision={model_config.tokenizer_revision}, "
            f"trust_remote_code={model_config.trust_remote_code}, "
            f"dtype={model_config.dtype}, "
            f"max_seq_len={model_config.max_model_len}, "
            f"download_dir={model_config.download_dir!r}, "
            f"load_format={model_config.load_format}, "
            f"tensor_parallel_size={parallel_config.tensor_parallel_size}, "
            f"quantization={model_config.quantization}, "
            f"seed={model_config.seed}")

        if load_config is not None:
            logger.info(
                "Try to initializing the model with"
                f" s3://{load_config.s3_bucket}/{load_config.s3_prefix}")

        # TODO(woosuk): Print more configs in debug mode.

        self.model_config = model_config
        self.cache_config = cache_config
        assert self.cache_config.sliding_window == getattr(
            self.model_config.hf_config, "sliding_window", None)
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.load_config = load_config
        self.lora_config = lora_config
        self.speculative_config = speculative_config
        self.log_stats = log_stats
        self._verify_args()

        self.seq_counter = Counter()
        self._init_tokenizer()

        self.shared_mem_manager = None
        self.shared_mem_event = None
        self.shared_mem_engine_to_worker = None
        self.shared_mem_worker_to_engine = None

        # Create the parallel GPU workers.
        if self.parallel_config.worker_use_ray:
            additional_ray_args = {}
            if self.parallel_config.ray_workers_use_nsight:
                logger.info("Configuring Ray workers to use nsight.")
                additional_ray_args = {"runtime_env": {"nsight": "default"}}

            self._init_workers_ray(placement_group, **additional_ray_args)
            runtime_contexts = self._run_workers("get_runtime_context",
                                                 get_all_outputs=True)
            # If engine and all workers are on the same node,
            # we can use shared memory.
            if (not self.parallel_config.disable_shared_memory
                    and all(runtime_context["node_id"] ==
                            ray.get_runtime_context().get_node_id()
                            for runtime_context in runtime_contexts)):
                logger.info("Using shared memory for communication between "
                            "engine and workers.")
                self.shared_mem_manager = SharedMemoryManager()
                self.shared_mem_manager.start()  # pylint: disable=consider-using-with
                # Reusing the same event for both buffers is fine, as there's
                # no situation in which we'd only want to wake up one buffer.
                self.shared_mem_event = RayEvent.options(
                    num_cpus=0,
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=ray.get_runtime_context().get_node_id(),
                        soft=False)).remote()
                self.shared_mem_engine_to_worker = SharedMsgspecBufferWithEvent(
                    size=SHARED_MEMORY_BUFFER_SIZE,
                    manager=self.shared_mem_manager,
                    encoder_init_fn=msgspec.msgpack.Encoder,
                    decoder_init_fn=lambda: msgspec.msgpack.Decoder(type=List[
                        SamplerOutput]),
                    ray_event=self.shared_mem_event,
                )
                self.shared_mem_worker_to_engine = SharedMsgspecBufferWithEvent(
                    size=SHARED_MEMORY_BUFFER_SIZE,
                    manager=self.shared_mem_manager,
                    encoder_init_fn=msgspec.msgpack.Encoder,
                    decoder_init_fn=lambda: msgspec.msgpack.Decoder(type=List[
                        SamplerOutput]),
                    ray_event=self.shared_mem_event,
                )
                logger.info(
                    "Engine shared memory input buffer id: "
                    f"{self.shared_mem_engine_to_worker.participant_id}")
                logger.info(
                    "Engine shared memory output buffer id: "
                    f"{self.shared_mem_worker_to_engine.participant_id}")
        else:
            self._init_workers(distributed_init_method)

        # Make sure the tokenizer actors are alive
        self.tokenizer.ping()

        # Profile the memory usage and initialize the cache.
        self._init_cache()

        # Create the scheduler.
        self.scheduler = Scheduler(scheduler_config, cache_config, lora_config)

        self._exceute_model_futures = None
        if self._uses_shared_memory:
            self._exceute_model_futures = self._run_workers(
                "execute_model_shared_memory",
                get_all_outputs=True,
                wait_for_workers=False,
                shared_memory_input=self.shared_mem_engine_to_worker,
                shared_memory_output=self.shared_mem_worker_to_engine,
                participant_id=self.shared_mem_engine_to_worker.participant_id)

        # Logging.
        self.last_logging_time = 0.0
        self.last_stats: Tuple[float, dict] = None
        # List of (timestamp, num_tokens)
        self.num_prompt_tokens: List[Tuple[float, int]] = []
        # List of (timestamp, num_tokens)
        self.num_generation_tokens: List[Tuple[float, int]] = []
        self.num_started_tasks = 0
        self.num_finished_tasks = 0
        self.num_aborted_tasks = 0
        self.num_iterations = 0

        self._last_draft_target_worker_metrics: Optional[
            DraftTargetWorkerMetrics] = None

        self._profiler = TorchProfiler()

    @property
    def _uses_shared_memory(self) -> bool:
        return self.shared_mem_engine_to_worker is not None

    def _init_tokenizer(self, **kwargs):
        init_kwargs = dict(
            enable_lora=bool(self.lora_config),
            max_num_seqs=self.scheduler_config.max_num_seqs,
            max_input_length=None,
            tokenizer_mode=self.model_config.tokenizer_mode,
            trust_remote_code=self.model_config.trust_remote_code,
            revision=self.model_config.tokenizer_revision)
        init_kwargs.update(kwargs)
        if self.parallel_config.num_tokenizer_actors > 0:
            ray_actor_options = (self.parallel_config.tokenizer_actor_options
                                 or {
                                     "num_cpus": 0
                                 })
            ray_actor_options[
                "scheduling_strategy"] = NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False)

            self.tokenizer: RayTokenizerPool = RayTokenizerPool(
                self.model_config.tokenizer,
                num_actors=self.parallel_config.num_tokenizer_actors,
                ray_actor_options=ray_actor_options,
                **init_kwargs)
        else:
            self.tokenizer: TransformersTokenizer = TransformersTokenizer(
                self.model_config.tokenizer, **init_kwargs)

    def _create_worker(
            self, rank: Optional[int],
            distributed_init_method: Optional[str]) -> BaseLoraWorker:
        # Lazy import the Worker classes to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        from vllm.worker.worker import Worker  # pylint: disable=import-outside-toplevel
        from vllm.worker.multi_step_worker import MultiStepWorker  # pylint: disable=import-outside-toplevel
        from vllm.worker.single_tp_worker import SingleTpWorker  # pylint: disable=import-outside-toplevel
        from vllm.worker.draft_target_worker import DraftTargetWorker  # pylint: disable=import-outside-toplevel

        if not self.speculative_config:
            return Worker(
                self.model_config,
                self.parallel_config,
                self.scheduler_config,
                rank,
                distributed_init_method,
                load_config=self.load_config,
                lora_config=self.lora_config,
            )

        target_worker = Worker(
            self.model_config,
            self.parallel_config,
            self.speculative_config.create_target_scheduler_config(
                self.scheduler_config),
            rank,
            distributed_init_method,
            load_config=self.load_config,
            lora_config=self.lora_config,
        )

        draft_worker = MultiStepWorker(
            self.speculative_config.draft_model_config,
            self.speculative_config.draft_parallel_config,
            self.speculative_config.create_draft_scheduler_config(
                self.scheduler_config),
            rank,
            distributed_init_method,
            load_config=self.load_config,
            lora_config=self.lora_config,
        )
        draft_worker = SingleTpWorker.maybe_wrap_worker(
            draft_worker, self.speculative_config.draft_parallel_config,
            self.parallel_config)
        return DraftTargetWorker.from_workers(draft_worker, target_worker)

    def _init_workers(self, distributed_init_method: str):
        assert self.parallel_config.world_size == 1, (
            "Ray is required if parallel_config.world_size > 1.")
        rank = 0

        self.workers: List[BaseLoraWorker] = [
            self._create_worker(rank, distributed_init_method)
        ]
        self._run_workers(
            "init_model",
            get_all_outputs=True,
        )

    def _init_workers_ray(self, placement_group: "PlacementGroup",
                          **ray_remote_kwargs):

        self.workers: List[BaseLoraWorker] = []
        for bundle in placement_group.bundle_specs:
            if not bundle.get("GPU", 0):
                continue
            worker = ray.remote(
                num_cpus=0,
                num_gpus=1,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=placement_group,
                    placement_group_capture_child_tasks=True),
                **ray_remote_kwargs,
            )(RayWorker).remote(self.model_config.trust_remote_code)
            self.workers.append(worker)

        # Initialize torch distributed process group for the workers.
        init_torch_dist_process_group(self.workers, backend="nccl")

        self._run_workers(
            "init_worker",
            get_all_outputs=True,
            worker_init_fn=partial(self._create_worker,
                                   rank=None,
                                   distributed_init_method=None),
        )
        self._run_workers(
            "init_model",
            get_all_outputs=True,
        )

    def _verify_args(self) -> None:
        self.model_config.verify_with_parallel_config(self.parallel_config)
        self.cache_config.verify_with_parallel_config(self.parallel_config)
        if self.lora_config:
            self.lora_config.verify_with_model_config(self.model_config)
            self.lora_config.verify_with_scheduler_config(
                self.scheduler_config)

    def _init_cache(self) -> None:
        """Profiles the memory usage and initializes the KV cache."""
        # Get the maximum number of blocks that can be allocated on GPU and CPU.
        num_blocks = self._run_workers(
            "profile_num_available_blocks",
            get_all_outputs=True,
            block_size=self.cache_config.block_size,
            gpu_memory_utilization=self.cache_config.gpu_memory_utilization,
            cpu_swap_space=self.cache_config.swap_space_bytes,
        )

        # Since we use a shared centralized controller, we take the minimum
        # number of blocks across all workers to make sure all the memory
        # operators can be applied to all workers.
        num_gpu_blocks = min(b[0] for b in num_blocks)
        num_cpu_blocks = min(b[1] for b in num_blocks)
        # FIXME(woosuk): Change to debug log.
        logger.info(f"# GPU blocks: {num_gpu_blocks}, "
                    f"# CPU blocks: {num_cpu_blocks}")

        if num_gpu_blocks <= 0:
            raise ValueError("No available memory for the cache blocks. "
                             "Try increasing `gpu_memory_utilization` when "
                             "initializing the engine.")
        max_seq_len = self.cache_config.block_size * num_gpu_blocks
        if self.model_config.max_model_len > max_seq_len:
            raise ValueError(
                f"The model's max seq len ({self.model_config.max_model_len}) "
                "is larger than the maximum number of tokens that can be "
                f"stored in KV cache ({max_seq_len}). Try increasing "
                "`gpu_memory_utilization` or decreasing `max_model_len` when "
                "initializing the engine.")

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        # Initialize the cache.
        self._run_workers("init_cache_engine", cache_config=self.cache_config)

    @classmethod
    def from_engine_args(cls, engine_args: EngineArgs) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_configs = engine_args.create_engine_configs()
        parallel_config = engine_configs[2]
        # Initialize the cluster.
        distributed_init_method, placement_group = initialize_cluster(
            parallel_config)
        # Create the LLM engine.
        engine = cls(*engine_configs,
                     distributed_init_method,
                     placement_group,
                     log_stats=not engine_args.disable_log_stats)
        return engine

    def encode_request(
        self,
        request_id: str,  # pylint: disable=unused-argument
        prompt: Optional[str],
        prompt_token_ids: Optional[List[int]] = None,
        lora_request: Optional[LoRARequest] = None,
    ):
        if prompt_token_ids is None:
            assert prompt is not None
            prompt_token_ids = self.tokenizer.encode(request_id=request_id,
                                                     prompt=prompt,
                                                     lora_request=lora_request)
        return prompt_token_ids

    def add_request(
        self,
        request_id: str,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
    ) -> None:
        """Add a request to the engine's request pool.

        The request is added to the request pool and will be processed by the
        scheduler as `engine.step()` is called. The exact scheduling policy is
        determined by the scheduler.

        Args:
            request_id: The unique ID of the request.
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            sampling_params: The sampling parameters for text generation.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.
            arrival_time: The arrival time of the request. If None, we use
                the current monotonic time.
        """
        if lora_request is not None and not self.lora_config:
            raise ValueError(f"Got lora_request {lora_request} but LoRA is "
                             "not enabled!")
        if arrival_time is None:
            arrival_time = time.monotonic()
        prompt_token_ids = self.encode_request(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            lora_request=lora_request)

        # Create the sequences.
        block_size = self.cache_config.block_size
        seq_id = next(self.seq_counter)
        seq = Sequence(seq_id, prompt, prompt_token_ids, block_size,
                       lora_request)

        # Create the sequence group.
        seq_group = SequenceGroup(request_id, [seq],
                                  sampling_params, arrival_time,
                                  time.perf_counter(), lora_request)

        # Add the sequence group to the scheduler.
        self.scheduler.add_seq_group(seq_group)
        self.num_started_tasks += 1

    def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        """Aborts a request(s) with the given ID.

        Args:
            request_id: The ID(s) of the request to abort.
        """
        self.num_aborted_tasks += self.scheduler.abort_seq_group(request_id)

    def get_model_config(self) -> ModelConfig:
        """Gets the model configuration."""
        return self.model_config

    def get_num_unfinished_requests(self) -> int:
        """Gets the number of unfinished requests."""
        return self.scheduler.get_num_unfinished_seq_groups()

    def has_unfinished_requests(self) -> bool:
        """Returns True if there are unfinished requests."""
        return self.scheduler.has_unfinished_seqs()

    def _schedule(
        self
    ) -> Tuple[List[Union[SequenceGroupMetadata, SequenceGroupMetadataDelta]],
               SchedulerOutputs, List[RequestOutput]]:
        seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()
        return seq_group_metadata_list, scheduler_outputs, [
            RequestOutput.from_seq_group(seq_group)
            for seq_group in scheduler_outputs.ignored_seq_groups
        ]

    # def _check_beam_search_early_stopping(
    #     self,
    #     early_stopping: Union[bool, str],
    #     sampling_params: SamplingParams,
    #     best_running_seq: Sequence,
    #     current_worst_seq: Sequence,
    # ) -> bool:
    #     assert sampling_params.use_beam_search
    #     length_penalty = sampling_params.length_penalty
    #     if early_stopping is True:
    #         return True

    #     current_worst_score = (current_worst_seq.get_beam_search_score(
    #         length_penalty=length_penalty,
    #         eos_token_id=self.tokenizer.get_lora_tokenizer(
    #             current_worst_seq.lora_request).eos_token_id))
    #     if early_stopping is False:
    #         highest_attainable_score = (
    # best_running_seq.get_beam_search_score(
    #             length_penalty=length_penalty,
    #             eos_token_id=self.tokenizer.get_lora_tokenizer(
    #                 best_running_seq.lora_request).eos_token_id))
    #     else:
    #         assert early_stopping == "never"
    #         if length_penalty > 0.0:
    #             # If length_penalty > 0.0, beam search will prefer longer
    #             # sequences. The highest attainable score calculation is
    #             # based on the longest possible sequence length in this case.
    #             max_possible_length = max(
    #                 best_running_seq.get_prompt_len() +
    #                 sampling_params.max_tokens,
    #                 self.scheduler_config.max_model_len)
    #             highest_attainable_score = (
    #                 best_running_seq.get_beam_search_score(
    #                     length_penalty=length_penalty,
    #                     eos_token_id=self.tokenizer.get_lora_tokenizer(
    #                         best_running_seq.lora_request).eos_token_id,
    #                     seq_len=max_possible_length))
    #         else:
    #             # Otherwise, beam search will prefer shorter sequences. The
    #             # highest attainable score calculation is based on the current
    #             # sequence length.
    #             highest_attainable_score = (
    #                 best_running_seq.get_beam_search_score(
    #                     length_penalty=length_penalty,
    #                     eos_token_id=self.tokenizer.get_lora_tokenizer(
    #                         best_running_seq.lora_request).eos_token_id))
    #     return current_worst_score >= highest_attainable_score

    def _process_spec_decode_sequence_group_outputs(
            self, seq_group: SequenceGroup,
            outputs: List[SequenceGroupOutputs]) -> None:
        """Process sequence group outputs when speculative decoding is enabled.

        This serves the same purpose as _process_sequence_group_outputs except
        without any of the beam search logic.
        """
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        assert len(seqs) == 1, ("Beam search not supported in speculative "
                                "decoding.")
        seq = seqs[0]

        # Since there's only one sequence per sequence group, we can take the
        # first sample.
        samples = [outputs[step].samples[0] for step in range(len(outputs))]

        # Draft target worker pads all outputs with -1 to have same length.
        output_token_ids = [
            sample.output_token for sample in samples
            if sample.output_token != -1
        ]
        output_logprobs = [sample.logprobs for sample in samples]

        # Truncate to max_tokens if necessary.
        remaining_tokens = seq_group.sampling_params.max_tokens - (
            seq.get_output_len() + len(output_token_ids))
        if remaining_tokens < 0:
            output_token_ids = output_token_ids[:remaining_tokens]
            output_logprobs = output_logprobs[:remaining_tokens]

        # Truncate any tokens after EOS. This is required as spec decode
        # generates tokens in fixed blocks, which may go beyond the EOS token.
        if not seq_group.sampling_params.ignore_eos:
            eos_token_id = self.tokenizer.get_lora_tokenizer(
                seq.lora_request).eos_token_id
            # Avoiding .index calls as exception throwing in the happy path
            # is expensive.
            for i in range(len(output_token_ids)):
                if output_token_ids[i] == eos_token_id:
                    output_token_ids = output_token_ids[:i + 1]
                    output_logprobs = output_logprobs[:i + 1]
                    break

        seq.append_token_ids(output_token_ids, output_logprobs)

        self._decode_sequence(seq, seq_group.sampling_params)
        self._check_stop(seq, seq_group.sampling_params)
        if seq.is_finished():
            self.scheduler.free_seq(seq)

    def _process_sequence_group_outputs(self, seq_group: SequenceGroup,
                                        outputs: SequenceGroupOutputs) -> None:
        # Process prompt logprobs
        prompt_logprobs = outputs.prompt_logprobs
        if prompt_logprobs is not None:
            seq_group.prompt_logprobs = prompt_logprobs

        # Process samples
        samples = outputs.samples
        parent_seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        existing_finished_seqs = seq_group.get_finished_seqs()
        parent_child_dict = {
            parent_seq.seq_id: []
            for parent_seq in parent_seqs
        }
        for sample in samples:
            parent_child_dict[sample.parent_seq_id].append(sample)
        # List of (child, parent)
        child_seqs: List[Tuple[Sequence, Sequence]] = []

        # Process the child samples for each parent sequence
        for parent in parent_seqs:
            child_samples: List[SequenceOutputs] = parent_child_dict[
                parent.seq_id]
            if len(child_samples) == 0:
                # This parent sequence has no children samples. Remove
                # the parent sequence from the sequence group since it will
                # not be used in the future iterations.
                parent.status = SequenceStatus.FINISHED_ABORTED
                seq_group.remove(parent.seq_id)
                self.scheduler.free_seq(parent)
                continue
            # Fork the parent sequence if there are multiple child samples.
            for child_sample in child_samples[:-1]:
                new_child_seq_id = next(self.seq_counter)
                child = parent.fork(new_child_seq_id)
                child.append_token_id(child_sample.output_token,
                                      child_sample.logprobs)
                child_seqs.append((child, parent))
            # Continue the parent sequence for the last child sample.
            # We reuse the parent sequence here to reduce redundant memory
            # copies, especially when using non-beam search sampling methods.
            last_child_sample = child_samples[-1]
            parent.append_token_id(last_child_sample.output_token,
                                   last_child_sample.logprobs)
            child_seqs.append((parent, parent))

        for seq, _ in child_seqs:
            self._decode_sequence(seq, seq_group.sampling_params)
            self._check_stop(seq, seq_group.sampling_params)

        # Non-beam search case
        if not seq_group.sampling_params.use_beam_search:
            # For newly created child sequences, add them to the sequence group
            # and fork them in block manager if they are not finished.
            for seq, parent in child_seqs:
                if seq is not parent:
                    seq_group.add(seq)
                    if not seq.is_finished():
                        self.scheduler.fork_seq(parent, seq)

            # Free the finished and selected parent sequences' memory in block
            # manager. Keep them in the sequence group as candidate output.
            # NOTE: we need to fork the new sequences before freeing the
            # old sequences.
            for seq, parent in child_seqs:
                if seq is parent and seq.is_finished():
                    self.scheduler.free_seq(seq)
            return

        # Beam search case
        # Select the child sequences to keep in the sequence group.
        selected_child_seqs = []
        unselected_child_seqs = []
        beam_width = seq_group.sampling_params.actual_best_of
        length_penalty = seq_group.sampling_params.length_penalty

        # Select the newly finished sequences with the highest scores
        # to replace existing finished sequences.
        # Tuple of (seq, parent, is_new)
        existing_finished_seqs = [(seq, None, False)
                                  for seq in existing_finished_seqs]
        new_finished_seqs = [(seq, parent, True) for seq, parent in child_seqs
                             if seq.is_finished()]
        all_finished_seqs = existing_finished_seqs + new_finished_seqs
        # Sort the finished sequences by their scores.
        all_finished_seqs.sort(key=lambda x: x[0].get_beam_search_score(
            length_penalty=length_penalty,
            eos_token_id=self.tokenizer.get_lora_tokenizer(x[0].lora_request
                                                           ).eos_token_id),
                               reverse=True)
        for seq, parent, is_new in all_finished_seqs[:beam_width]:
            if is_new:
                # A newly generated child sequence finishes and has a high
                # score, so we will add it into the sequence group.
                selected_child_seqs.append((seq, parent))
        for seq, parent, is_new in all_finished_seqs[beam_width:]:
            if is_new:
                # A newly generated child sequence finishes but has a low
                # score, so we will not add it into the sequence group.
                # Additionally, if this sequence is a continuation of a
                # parent sequence, we will need remove the parent sequence
                # from the sequence group.
                unselected_child_seqs.append((seq, parent))
            else:
                # An existing finished sequence has a low score, so we will
                # remove it from the sequence group.
                seq_group.remove(seq.seq_id)

        # select the top beam_width sequences from the running
        # sequences for the next iteration to continue the beam
        # search.
        running_child_seqs = [(seq, parent) for seq, parent in child_seqs
                              if not seq.is_finished()]
        # Sort the running sequences by their scores.
        running_child_seqs.sort(key=lambda x: x[0].get_beam_search_score(
            length_penalty=length_penalty,
            eos_token_id=self.tokenizer.get_lora_tokenizer(x[0].lora_request
                                                           ).eos_token_id),
                                reverse=True)

        # Check if we can stop the beam search.
        if len(running_child_seqs) == 0:
            # No running sequences, stop the beam search.
            stop_beam_search = True
        elif len(all_finished_seqs) < beam_width:
            # Not enough finished sequences, continue the beam search.
            stop_beam_search = False
        else:
            # Check the early stopping criteria
            best_running_seq = running_child_seqs[0][0]
            current_worst_seq = all_finished_seqs[beam_width - 1][0]
            stop_beam_search = self._check_beam_search_early_stopping(
                seq_group.sampling_params.early_stopping,
                seq_group.sampling_params, best_running_seq, current_worst_seq)

        if stop_beam_search:
            # Stop the beam search and remove all the running sequences from
            # the sequence group.
            unselected_child_seqs.extend(running_child_seqs)
        else:
            # Continue the beam search and select the top beam_width sequences
            # to continue the beam search.
            selected_child_seqs.extend(running_child_seqs[:beam_width])
            # The remaining running sequences will not be used in the next
            # iteration. Again, if these sequences are continuations of
            # parent sequences, we will need to remove the parent sequences
            # from the sequence group.
            unselected_child_seqs.extend(running_child_seqs[beam_width:])

        # For newly created child sequences, add them to the sequence group
        # and fork them in block manager if they are not finished.
        for seq, parent in selected_child_seqs:
            if seq is not parent:
                seq_group.add(seq)
                if not seq.is_finished():
                    self.scheduler.fork_seq(parent, seq)

        # Free the finished and selected parent sequences' memory in block
        # manager. Keep them in the sequence group as candidate output.
        for seq, parent in selected_child_seqs:
            if seq is parent and seq.is_finished():
                self.scheduler.free_seq(seq)

        # Remove the unselected parent sequences from the sequence group and
        # free their memory in block manager.
        for seq, parent in unselected_child_seqs:
            if seq is parent:
                # Remove the parent sequence if it is not selected for next
                # iteration
                seq_group.remove(seq.seq_id)
                self.scheduler.free_seq(seq)

    def _process_model_outputs(
            self, output: List[SamplerOutput],
            scheduler_outputs: SchedulerOutputs) -> List[RequestOutput]:
        # Update the scheduled sequence groups with the model outputs.
        now = time.time()
        scheduled_seq_groups = scheduler_outputs.scheduled_seq_groups

        # Organize list of sampler output by sequence group.
        output_by_sequence_group = [[] for _ in scheduled_seq_groups]
        for step in output:
            for i, sequence_group_output in enumerate(step):
                output_by_sequence_group[i].append(sequence_group_output)

        # combine all samples for zipping
        for i, (seq_group, outputs) in enumerate(
                zip(scheduled_seq_groups, output_by_sequence_group)):
            # Chunked prefill groups are not generation tokens. Their
            # outputs are ignored. For seq_group finished chunked
            # prefilling, it will be considered as prompting.
            if i < scheduler_outputs.num_chunked_prefill_groups:
                continue
            if seq_group.first_token_time is None:
                seq_group.first_token_time = now

            if self.speculative_config:
                self._process_spec_decode_sequence_group_outputs(
                    seq_group, outputs)
            else:
                assert len(outputs) == 1
                self._process_sequence_group_outputs(seq_group, outputs[0])

        # Free the finished sequence groups.
        self.scheduler.free_finished_seq_groups()

        # Create the outputs.
        request_outputs: List[RequestOutput] = []
        for seq_group in (scheduled_seq_groups +
                          scheduler_outputs.ignored_seq_groups):
            request_output = RequestOutput.from_seq_group(seq_group)
            request_outputs.append(request_output)
            self.num_finished_tasks += int(request_output.finished)

        # Write logits to request outputs if present in sampler outputs.
        for i, step in enumerate(output):
            if step and step.logits is not None:
                request_outputs[i].logits = step.logits

        # If worker metrics are provided, store locally.
        if (self.speculative_config and output
                and output[0].draft_target_worker_metrics is not None):
            self._last_draft_target_worker_metrics = output[
                0].draft_target_worker_metrics

        if self.log_stats:
            # Log the system stats.
            self._log_system_stats(scheduler_outputs.num_prompt_groups,
                                   scheduler_outputs.num_batched_tokens)
        self.num_iterations += 1
        return request_outputs

    def step(self, return_logits: bool = False) -> List[RequestOutput]:
        """Performs one decoding iteration and returns newly generated results.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration and the
        token blocks to be swapped in/out/copy. Then, it executes the model
        and updates the scheduler with the model outputs. Finally, it decodes
        the sequences and returns the newly generated results.

        Args:
            return_logits: Whether to return the logits from the model for
                quality evaluation purposes.
        """
        seq_group_metadata_list, scheduler_outputs, ignored = self._schedule()
        if scheduler_outputs.is_empty():
            return ignored

        data = ExecuteModelData(
            seq_group_metadata_list=seq_group_metadata_list,
            finished_request_ids_list=list(
                scheduler_outputs.done_seq_group_ids),
            blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
            blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
            blocks_to_copy=scheduler_outputs.blocks_to_copy,
            num_preallocated_slots=scheduler_outputs.num_preallocated_slots,
            return_logits=return_logits,
        )
        # Execute the model.
        now = time.perf_counter()
        output = self._run_workers("execute_model",
                                   data,
                                   use_shared_memory=self._uses_shared_memory)
        logger.debug(f"model execution takes{time.perf_counter() - now}")

        outputs = self._process_model_outputs(output, scheduler_outputs)

        if self._uses_shared_memory:
            if not outputs or all(out.finished for out in outputs):
                self.shared_mem_engine_to_worker.clear()
                self.shared_mem_worker_to_engine.clear()
                self.shared_mem_engine_to_worker.put_to_sleep(block=False)
                self.shared_mem_worker_to_engine.put_to_sleep(block=False)

        return outputs

    def _log_system_stats(
        self,
        prompt_run: bool,
        num_batched_tokens: int,
    ) -> None:
        now = time.monotonic()
        # Log the number of batched input tokens.
        if prompt_run:
            self.num_prompt_tokens.append((now, num_batched_tokens))
        else:
            self.num_generation_tokens.append((now, num_batched_tokens))

        elapsed_time = now - self.last_logging_time
        if elapsed_time < _LOGGING_INTERVAL_SEC:
            return

        # Discard the old stats.
        self.num_prompt_tokens = [(t, n) for t, n in self.num_prompt_tokens
                                  if now - t < _LOGGING_INTERVAL_SEC]
        self.num_generation_tokens = [(t, n)
                                      for t, n in self.num_generation_tokens
                                      if now - t < _LOGGING_INTERVAL_SEC]

        if len(self.num_prompt_tokens) > 1:
            total_num_tokens = sum(n for _, n in self.num_prompt_tokens[:-1])
            window = now - self.num_prompt_tokens[0][0]
            avg_prompt_throughput = total_num_tokens / window
        else:
            avg_prompt_throughput = 0.0
        if len(self.num_generation_tokens) > 1:
            total_num_tokens = sum(n
                                   for _, n in self.num_generation_tokens[:-1])
            window = now - self.num_generation_tokens[0][0]
            avg_generation_throughput = total_num_tokens / window
        else:
            avg_generation_throughput = 0.0

        total_num_gpu_blocks = self.cache_config.num_gpu_blocks
        num_free_gpu_blocks = (
            self.scheduler.block_manager.get_num_free_gpu_blocks())
        num_used_gpu_blocks = total_num_gpu_blocks - num_free_gpu_blocks
        gpu_cache_usage = num_used_gpu_blocks / total_num_gpu_blocks

        total_num_cpu_blocks = self.cache_config.num_cpu_blocks
        if total_num_cpu_blocks > 0:
            num_free_cpu_blocks = (
                self.scheduler.block_manager.get_num_free_cpu_blocks())
            num_used_cpu_blocks = total_num_cpu_blocks - num_free_cpu_blocks
            cpu_cache_usage = num_used_cpu_blocks / total_num_cpu_blocks
        else:
            cpu_cache_usage = 0.0

        logger.info("Avg prompt throughput: "
                    f"{avg_prompt_throughput:.1f} tokens/s, "
                    "Avg generation throughput: "
                    f"{avg_generation_throughput:.1f} tokens/s, "
                    f"Running: {len(self.scheduler.running)} reqs, "
                    f"Swapped: {len(self.scheduler.swapped)} reqs, "
                    f"Pending: {len(self.scheduler.waiting)} reqs, "
                    f"GPU KV cache usage: {gpu_cache_usage * 100:.1f}%, "
                    f"CPU KV cache usage: {cpu_cache_usage * 100:.1f}%")
        self.last_logging_time = now

        self._record_system_stats(avg_prompt_throughput,
                                  avg_generation_throughput, gpu_cache_usage,
                                  cpu_cache_usage)

        if self._last_draft_target_worker_metrics is not None:
            metrics = self._last_draft_target_worker_metrics
            logger.info(
                "Speculative metrics: "
                f"Draft acceptance rate: {metrics.draft_acceptance_rate:.3f}, "
                f"System efficiency: {metrics.system_efficiency:.3f}, "
                f"Number of speculative tokens: {metrics.num_spec_tokens}, "
                f"Number of accepted tokens: {metrics.accepted_tokens}, "
                f"Number of draft tokens tokens: {metrics.draft_tokens}, "
                f"Number of emitted tokens tokens: {metrics.emitted_tokens}.")

    def _record_system_stats(self, avg_prompt_throughput: float,
                             avg_generation_throughput: float,
                             gpu_cache_usage: float,
                             cpu_cache_usage: float) -> Tuple[float, dict]:
        self.last_stats = (self.last_logging_time, {
            "avg_prompt_throughput": avg_prompt_throughput,
            "avg_generation_throughput": avg_generation_throughput,
            "gpu_cache_usage": gpu_cache_usage,
            "cpu_cache_usage": cpu_cache_usage,
        })
        return self.last_stats

    def _decode_sequence(self, seq: Sequence,
                         sampling_params: SamplingParams) -> None:
        """Decodes new token(s) for a sequence."""
        unseen_token_ids = seq.get_new_token_ids()
        token_ids = seq.get_token_ids()[:-len(unseen_token_ids)]

        for new_token_id in unseen_token_ids:
            token_ids.append(new_token_id)

            (new_tokens, new_output_text, prefix_offset,
             read_offset) = detokenize_incrementally(
                 self.tokenizer.get_lora_tokenizer(seq.lora_request),
                 all_input_ids=token_ids,
                 prev_tokens=seq.tokens,
                 prefix_offset=seq.prefix_offset,
                 read_offset=seq.read_offset,
                 skip_special_tokens=sampling_params.skip_special_tokens,
                 spaces_between_special_tokens=sampling_params.
                 spaces_between_special_tokens,
             )
            if seq.tokens is None:
                seq.tokens = new_tokens
            else:
                seq.tokens.extend(new_tokens)
            seq.prefix_offset = prefix_offset
            seq.read_offset = read_offset
            seq.output_text += new_output_text

    def _check_stop(self, seq: Sequence,
                    sampling_params: SamplingParams) -> None:
        """Stop the finished sequences."""
        for stop_str in sampling_params.stop:
            if seq.output_text.endswith(stop_str):
                # Truncate the output text so that the stop string is
                # not included in the output.
                seq.output_text = seq.output_text[:-len(stop_str)]
                seq.status = SequenceStatus.FINISHED_STOPPED
                return
        if set(seq.get_new_token_ids()).intersection(
                sampling_params.stop_token_ids):
            seq.status = SequenceStatus.FINISHED_STOPPED
            return

        # Check if the sequence has reached max_model_len.
        if seq.get_len() > self.scheduler_config.max_model_len:
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            return

        # Check if the sequence has reached max_tokens.
        if seq.get_output_len() >= sampling_params.max_tokens:
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            return

        # Check if the sequence has generated the EOS token.
        if ((not sampling_params.ignore_eos)
                and self.tokenizer.get_lora_tokenizer(
                    seq.lora_request).eos_token_id in seq.get_new_token_ids()):
            seq.status = SequenceStatus.FINISHED_STOPPED
            return

    def _run_workers(
        self,
        method: str,
        *args,
        get_all_outputs: bool = False,
        wait_for_workers: bool = True,
        use_shared_memory: bool = False,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""
        if use_shared_memory:
            try:
                logger.debug(f"Set data to shared memory: {args[0]}")
                self.shared_mem_engine_to_worker.set_data(args[0])
            except RuntimeError:
                # Raise underlying exception
                ray.get(self._exceute_model_futures, timeout=5)
                raise
            logger.debug("Waiting for incoming data...")
            self.shared_mem_worker_to_engine.wait_for_incoming_data()
            try:
                output = self.shared_mem_worker_to_engine.get_data()
            except RuntimeError:
                # Raise underlying exception
                ray.get(self._exceute_model_futures, timeout=5)
                raise
            logger.debug(f"Got data {output}")
            self.shared_mem_worker_to_engine.clear()
            return output
        else:
            all_outputs = []
            start = time.time()

            for worker in self.workers:
                if self.parallel_config.worker_use_ray:
                    executor = partial(worker.execute_method.remote, method)
                else:
                    executor = getattr(worker, method)

                output = executor(*args, **kwargs)
                all_outputs.append(output)

            if self.parallel_config.worker_use_ray:
                if wait_for_workers:
                    all_outputs = ray.get(all_outputs)

            end = time.time()

            if method == "init_model":
                logger.info("{} used {:.3f} seconds".format(
                    method, end - start))

            if get_all_outputs:
                return all_outputs

            # Make sure all workers have the same results.
            output = all_outputs[0]
            if wait_for_workers:
                for other_output in all_outputs[1:]:
                    assert output == other_output
            return output

    def add_lora(self, lora_request: LoRARequest) -> bool:
        assert lora_request.lora_int_id > 0, "lora_id must be greater than 0."
        return self._run_workers(
            "add_lora",
            lora_request=lora_request,
        )

    def remove_lora(self, lora_id: int) -> bool:
        assert lora_id > 0, "lora_id must be greater than 0."
        return self._run_workers(
            "remove_lora",
            lora_id=lora_id,
        )

    def list_loras(self) -> List[int]:
        return self._run_workers("list_loras")

    def get_metadata_cache_len(self) -> int:
        return self._run_workers("get_metadata_cache_len", )

    def _check_if_any_actor_is_dead(self):
        workers = (self.workers
                   or []) + (self.tokenizer.tokenizer_actors if isinstance(
                       self.tokenizer, RayTokenizerPool) else [])
        if workers:
            dead_actors = []
            for actor in workers:
                actor_state = ray.state.actors(actor._ray_actor_id.hex())  # pylint: disable=protected-access
                if actor_state["State"] == "DEAD":
                    dead_actors.append(actor)
            if dead_actors:
                raise RuntimeError("At least one Worker is dead. "
                                   f"Dead Workers: {dead_actors}. ")

    def check_health(self) -> None:
        if not self.parallel_config.worker_use_ray:
            return

        self._check_if_any_actor_is_dead()
        if self._exceute_model_futures:
            ready, _ = ray.wait(self._exceute_model_futures, timeout=0)
            if ready:
                # Raise any exception
                ray.get(ready, timeout=1)
                raise RuntimeError("At least one Worker is dead.")

    def __del__(self):
        if getattr(self, "shared_mem_manager", None) is not None:
            self.shared_mem_manager.shutdown()

    def start_profile(self, profile_ray_workers: bool, **kwargs):
        """Start profiling. Can optionally run profiling in Ray workers.
        """
        self._profiler.start_profile(**kwargs)
        if profile_ray_workers:
            if not self.parallel_config.worker_use_ray:
                raise ValueError(
                    "Cannot profile ray workers: "
                    f" worker_use_ray={self.parallel_config.worker_use_ray:}")

            if not self.parallel_config.disable_shared_memory:
                raise ValueError("Cannot profile ray workers: shared memory "
                                 "must be disabled")
            self._run_workers("start_profile", **kwargs)

    def stop_profile(self, profile_ray_workers: bool):
        if self.parallel_config.worker_use_ray and profile_ray_workers:
            self._run_workers("stop_profile")

        self._profiler.stop_profile()
