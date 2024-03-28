import time
from typing import Iterable, List, Optional, Tuple, Type, Union

from transformers import PreTrainedTokenizer

import vllm
from vllm.config import (CacheConfig, DeviceConfig, LoRAConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig, VisionLanguageConfig)
from vllm.core.scheduler import Scheduler, SchedulerOutputs
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.metrics import StatLogger, Stats
from vllm.engine.ray_utils import initialize_ray_cluster
from vllm.executor.executor_base import ExecutorBase
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import (MultiModalData, SamplerOutput, Sequence,
                           SequenceGroup, SequenceGroupOutput, SequenceOutput,
                           SequenceStatus)
from vllm.transformers_utils.detokenizer import Detokenizer
from vllm.transformers_utils.tokenizer_group import (BaseTokenizerGroup,
                                                     get_tokenizer_group)
from vllm.utils import Counter

logger = init_logger(__name__)
_LOCAL_LOGGING_INTERVAL_SEC = 5


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
        device_config: The configuration related to the device.
        executor_class: The model executor class for managing distributed
            execution.
        log_stats: Whether to log statistics.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        lora_config: Optional[LoRAConfig],
        vision_language_config: Optional["VisionLanguageConfig"],
        executor_class: Type[ExecutorBase],
        log_stats: bool,
    ) -> None:
        logger.info(
            f"Initializing an LLM engine (v{vllm.__version__}) with config: "
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
            f"disable_custom_all_reduce="
            f"{parallel_config.disable_custom_all_reduce}, "
            f"quantization={model_config.quantization}, "
            f"enforce_eager={model_config.enforce_eager}, "
            f"kv_cache_dtype={cache_config.cache_dtype}, "
            f"device_config={device_config.device}, "
            f"seed={model_config.seed})")
        # TODO(woosuk): Print more configs in debug mode.

        self.model_config = model_config
        self.cache_config = cache_config
        self.lora_config = lora_config
        self.vision_language_config = vision_language_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.log_stats = log_stats
        self._verify_args()

        self._init_tokenizer()
        self.detokenizer = Detokenizer(self.tokenizer)
        self.seq_counter = Counter()

        self.model_executor = executor_class(model_config, cache_config,
                                             parallel_config, scheduler_config,
                                             device_config, lora_config,
                                             vision_language_config)

        # Ping the tokenizer to ensure liveness if it runs in a
        # different process.
        self.tokenizer.ping()

        # Create the scheduler.
        # NOTE: the cache_config here have been updated with the numbers of
        # GPU and CPU blocks, which are profiled in the distributed executor.
        self.scheduler = Scheduler(scheduler_config, cache_config, lora_config)

        # Metric Logging.
        if self.log_stats:
            self.stat_logger = StatLogger(
                local_interval=_LOCAL_LOGGING_INTERVAL_SEC,
                labels=dict(model_name=model_config.model))
            self.stat_logger.info("cache_config", self.cache_config)

    @classmethod
    def from_engine_args(cls, engine_args: EngineArgs) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_configs = engine_args.create_engine_configs()
        parallel_config = engine_configs[2]
        device_config = engine_configs[4]

        # Initialize the cluster and specify the executor class.
        if device_config.device_type == "neuron":
            from vllm.executor.neuron_executor import NeuronExecutor
            executor_class = NeuronExecutor
        elif parallel_config.worker_use_ray:
            initialize_ray_cluster(parallel_config)
            from vllm.executor.ray_gpu_executor import RayGPUExecutor
            executor_class = RayGPUExecutor
        else:
            assert parallel_config.world_size == 1, (
                "Ray is required if parallel_config.world_size > 1.")
            from vllm.executor.gpu_executor import GPUExecutor
            executor_class = GPUExecutor

        # Create the LLM engine.
        engine = cls(*engine_configs,
                     executor_class=executor_class,
                     log_stats=not engine_args.disable_log_stats)
        return engine

    def __reduce__(self):
        # This is to ensure that the LLMEngine is not referenced in
        # the closure used to initialize Ray worker actors
        raise RuntimeError("LLMEngine should not be pickled!")

    def get_tokenizer(self) -> "PreTrainedTokenizer":
        return self.tokenizer.get_lora_tokenizer(None)

    def get_tokenizer_for_seq(self,
                              sequence: Sequence) -> "PreTrainedTokenizer":
        return self.tokenizer.get_lora_tokenizer(sequence.lora_request)

    def _init_tokenizer(self, **tokenizer_init_kwargs):
        init_kwargs = dict(
            tokenizer_id=self.model_config.tokenizer,
            enable_lora=bool(self.lora_config),
            max_num_seqs=self.scheduler_config.max_num_seqs,
            max_input_length=None,
            tokenizer_mode=self.model_config.tokenizer_mode,
            trust_remote_code=self.model_config.trust_remote_code,
            revision=self.model_config.tokenizer_revision)
        init_kwargs.update(tokenizer_init_kwargs)
        self.tokenizer: BaseTokenizerGroup = get_tokenizer_group(
            self.parallel_config.tokenizer_pool_config, **init_kwargs)

        if len(self.get_tokenizer()) != self.model_config.get_vocab_size():
            logger.warning(
                f"The tokenizer's vocabulary size {len(self.get_tokenizer())}"
                f" does not match the model's vocabulary size "
                f"{self.model_config.get_vocab_size()}. This might "
                f"cause an error in decoding. Please change config.json "
                "to match the tokenizer's vocabulary size.")

    def _verify_args(self) -> None:
        self.model_config.verify_with_parallel_config(self.parallel_config)
        self.cache_config.verify_with_parallel_config(self.parallel_config)
        if self.lora_config:
            self.lora_config.verify_with_model_config(self.model_config)
            self.lora_config.verify_with_scheduler_config(
                self.scheduler_config)

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
        multi_modal_data: Optional[MultiModalData] = None,
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
            multi_modal_data: Multi modal data per request.

        Details:
            - Set arrival_time to the current time if it is None.
            - Set prompt_token_ids to the encoded prompt if it is None.
            - Create `best_of` number of :class:`~vllm.Sequence` objects.
            - Create a :class:`~vllm.SequenceGroup` object
              from the list of :class:`~vllm.Sequence`.
            - Add the :class:`~vllm.SequenceGroup` object to the scheduler.

        Example:
            >>> # initialize engine
            >>> engine = LLMEngine.from_engine_args(engine_args)
            >>> # set request arguments
            >>> example_prompt = "Who is the president of the United States?"
            >>> sampling_params = SamplingParams(temperature=0.0)
            >>> request_id = 0
            >>>
            >>> # add the request to the engine
            >>> engine.add_request(
            >>>    str(request_id),
            >>>    example_prompt,
            >>>    SamplingParams(temperature=0.0))
            >>> # continue the request processing
            >>> ...
        """
        if lora_request is not None and not self.lora_config:
            raise ValueError(f"Got lora_request {lora_request} but LoRA is "
                             "not enabled!")
        max_logprobs = self.get_model_config().max_logprobs
        if (sampling_params.logprobs
                and sampling_params.logprobs > max_logprobs) or (
                    sampling_params.prompt_logprobs
                    and sampling_params.prompt_logprobs > max_logprobs):
            raise ValueError(f"Cannot request more than "
                             f"{max_logprobs} logprobs.")
        if arrival_time is None:
            arrival_time = time.time()
        prompt_token_ids = self.encode_request(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            lora_request=lora_request)

        # Create the sequences.
        block_size = self.cache_config.block_size
        seq_id = next(self.seq_counter)
        eos_token_id = self.tokenizer.get_lora_tokenizer(
            lora_request).eos_token_id
        seq = Sequence(seq_id, prompt, prompt_token_ids, block_size,
                       eos_token_id, lora_request)

        # Defensive copy of SamplingParams, which are used by the sampler,
        # this doesn't deep-copy LogitsProcessor objects
        sampling_params = sampling_params.clone()
        # inject the eos token id into the sampling_params to support min_tokens
        # processing
        sampling_params.eos_token_id = seq.eos_token_id

        # Create the sequence group.
        seq_group = SequenceGroup(request_id, [seq], sampling_params,
                                  arrival_time, lora_request, multi_modal_data)

        # Add the sequence group to the scheduler.
        self.scheduler.add_seq_group(seq_group)

    def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        """Aborts a request(s) with the given ID.

        Args:
            request_id: The ID(s) of the request to abort.

        Details:
            - Refer to the
              :meth:`~vllm.core.scheduler.Scheduler.abort_seq_group`
              from class :class:`~vllm.core.scheduler.Scheduler`.

        Example:
            >>> # initialize engine and add a request with request_id
            >>> request_id = str(0)
            >>> # abort the request
            >>> engine.abort_request(request_id)
        """
        self.scheduler.abort_seq_group(request_id)

    def get_model_config(self) -> ModelConfig:
        """Gets the model configuration."""
        return self.model_config

    def get_num_unfinished_requests(self) -> int:
        """Gets the number of unfinished requests."""
        return self.scheduler.get_num_unfinished_seq_groups()

    def has_unfinished_requests(self) -> bool:
        """Returns True if there are unfinished requests."""
        return self.scheduler.has_unfinished_seqs()

    def _check_beam_search_early_stopping(
        self,
        early_stopping: Union[bool, str],
        sampling_params: SamplingParams,
        best_running_seq: Sequence,
        current_worst_seq: Sequence,
    ) -> bool:
        assert sampling_params.use_beam_search
        length_penalty = sampling_params.length_penalty
        if early_stopping is True:
            return True

        current_worst_score = current_worst_seq.get_beam_search_score(
            length_penalty=length_penalty,
            eos_token_id=current_worst_seq.eos_token_id)
        if early_stopping is False:
            highest_attainable_score = best_running_seq.get_beam_search_score(
                length_penalty=length_penalty,
                eos_token_id=best_running_seq.eos_token_id)
        else:
            assert early_stopping == "never"
            if length_penalty > 0.0:
                # If length_penalty > 0.0, beam search will prefer longer
                # sequences. The highest attainable score calculation is
                # based on the longest possible sequence length in this case.
                max_possible_length = max(
                    best_running_seq.get_prompt_len() +
                    sampling_params.max_tokens,
                    self.scheduler_config.max_model_len)
                highest_attainable_score = (
                    best_running_seq.get_beam_search_score(
                        length_penalty=length_penalty,
                        eos_token_id=best_running_seq.eos_token_id,
                        seq_len=max_possible_length))
            else:
                # Otherwise, beam search will prefer shorter sequences. The
                # highest attainable score calculation is based on the current
                # sequence length.
                highest_attainable_score = (
                    best_running_seq.get_beam_search_score(
                        length_penalty=length_penalty,
                        eos_token_id=best_running_seq.eos_token_id))
        return current_worst_score >= highest_attainable_score

    def _process_sequence_group_outputs(self, seq_group: SequenceGroup,
                                        outputs: SequenceGroupOutput) -> None:

        # Process prompt logprobs
        prompt_logprobs = outputs.prompt_logprobs
        if prompt_logprobs is not None:
            self.detokenizer.decode_prompt_logprobs_inplace(
                seq_group, prompt_logprobs)
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
            child_samples: List[SequenceOutput] = parent_child_dict[
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
            self.detokenizer.decode_sequence_inplace(seq,
                                                     seq_group.sampling_params)
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
        beam_width = seq_group.sampling_params.best_of
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
            length_penalty=length_penalty, eos_token_id=x[0].eos_token_id),
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
            length_penalty=length_penalty, eos_token_id=x[0].eos_token_id),
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
            self, output: SamplerOutput,
            scheduler_outputs: SchedulerOutputs) -> List[RequestOutput]:
        now = time.time()
        # Update the scheduled sequence groups with the model outputs.
        scheduled_seq_groups = scheduler_outputs.scheduled_seq_groups

        for scheduled_seq_group, outputs in zip(scheduled_seq_groups, output):
            seq_group = scheduled_seq_group.seq_group
            token_chunk_size = scheduled_seq_group.token_chunk_size
            seq_group.update_num_computed_tokens(token_chunk_size)
            self._process_sequence_group_outputs(seq_group, outputs)

        # Free the finished sequence groups.
        self.scheduler.free_finished_seq_groups()

        # Create the outputs.
        request_outputs: List[RequestOutput] = []
        for scheduled_seq_group in scheduled_seq_groups:
            seq_group = scheduled_seq_group.seq_group
            seq_group.maybe_set_first_token_time(now)
            request_output = RequestOutput.from_seq_group(seq_group)
            request_outputs.append(request_output)
        for seq_group in scheduler_outputs.ignored_seq_groups:
            request_output = RequestOutput.from_seq_group(seq_group)
            request_outputs.append(request_output)

        # Log stats.
        if self.log_stats:
            self.stat_logger.log(self._get_stats(scheduler_outputs))
        return request_outputs

    def step(self) -> List[RequestOutput]:
        """Performs one decoding iteration and returns newly generated results.

        .. figure:: https://i.imgur.com/sv2HssD.png
            :alt: Overview of the step function
            :align: center

            Overview of the step function.

        Details:
            - Step 1: Schedules the sequences to be executed in the next
              iteration and the token blocks to be swapped in/out/copy.

                - Depending on the scheduling policy,
                  sequences may be `preempted/reordered`.
                - A Sequence Group (SG) refer to a group of sequences
                  that are generated from the same prompt.

            - Step 2: Calls the distributed executor to execute the model.
            - Step 3: Processes the model output. This mainly includes:

                - Decodes the relevant outputs.
                - Updates the scheduled sequence groups with model outputs
                  based on its `sampling parameters` (`use_beam_search` or not).
                - Frees the finished sequence groups.

            - Finally, it creates and returns the newly generated results.

        Example:
            >>> # Please see the example/ folder for more detailed examples.
            >>>
            >>> # initialize engine and request arguments
            >>> engine = LLMEngine.from_engine_args(engine_args)
            >>> example_inputs = [(0, "What is LLM?",
            >>>    SamplingParams(temperature=0.0))]
            >>>
            >>> # Start the engine with an event loop
            >>> while True:
            >>>     if example_inputs:
            >>>         req_id, prompt, sampling_params = example_inputs.pop(0)
            >>>         engine.add_request(str(req_id), prompt, sampling_params)
            >>>
            >>>     # continue the request processing
            >>>     request_outputs = engine.step()
            >>>     for request_output in request_outputs:
            >>>         if request_output.finished:
            >>>             # return or show the request output
            >>>
            >>>     if not (engine.has_unfinished_requests() or example_inputs):
            >>>         break
        """
        seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()

        if not scheduler_outputs.is_empty():
            output = self.model_executor.execute_model(
                seq_group_metadata_list, scheduler_outputs.blocks_to_swap_in,
                scheduler_outputs.blocks_to_swap_out,
                scheduler_outputs.blocks_to_copy)
        else:
            output = []

        return self._process_model_outputs(output, scheduler_outputs)

    def do_log_stats(self) -> None:
        """Forced log when no requests active."""
        if self.log_stats:
            self.stat_logger.log(self._get_stats(scheduler_outputs=None))

    def _get_stats(self,
                   scheduler_outputs: Optional[SchedulerOutputs]) -> Stats:
        """Get Stats to be Logged to Prometheus."""
        now = time.time()

        # KV Cache Usage in %.
        num_total_gpu = self.cache_config.num_gpu_blocks
        num_free_gpu = self.scheduler.block_manager.get_num_free_gpu_blocks()
        gpu_cache_usage = 1.0 - (num_free_gpu / num_total_gpu)

        num_total_cpu = self.cache_config.num_cpu_blocks
        cpu_cache_usage = 0.
        if num_total_cpu > 0:
            num_free_cpu = self.scheduler.block_manager.get_num_free_cpu_blocks(
            )
            cpu_cache_usage = 1.0 - (num_free_cpu / num_total_cpu)

        # Scheduler State
        num_running = len(self.scheduler.running)
        num_swapped = len(self.scheduler.swapped)
        num_waiting = len(self.scheduler.waiting)

        # Iteration stats if we have scheduler output.
        num_prompt_tokens = 0
        num_generation_tokens = 0
        time_to_first_tokens = []
        time_per_output_tokens = []
        time_e2e_requests = []
        if scheduler_outputs is not None:
            prompt_run = scheduler_outputs.prompt_run

            # Number of Tokens.
            if prompt_run:
                num_prompt_tokens = sum(
                    len(scheduled_seq_group.seq_group.prompt_token_ids)
                    for scheduled_seq_group in
                    scheduler_outputs.scheduled_seq_groups)
                num_generation_tokens = sum(
                    scheduled_seq_group.seq_group.num_seqs()
                    for scheduled_seq_group in
                    scheduler_outputs.scheduled_seq_groups)
            else:
                num_generation_tokens = scheduler_outputs.num_batched_tokens

            # Latency Timings.
            time_last_iters = []
            for scheduled_seq_group in scheduler_outputs.scheduled_seq_groups:
                seq_group = scheduled_seq_group.seq_group
                # Time since last token.
                # (n.b. updates seq_group.metrics.last_token_time)
                time_last_iters.append(seq_group.get_last_latency(now))
                # Time since arrival for all finished requests.
                if seq_group.is_finished():
                    time_e2e_requests.append(now -
                                             seq_group.metrics.arrival_time)

            time_to_first_tokens = time_last_iters if prompt_run else []
            time_per_output_tokens = [] if prompt_run else time_last_iters

        return Stats(
            now=now,
            num_running=num_running,
            num_swapped=num_swapped,
            num_waiting=num_waiting,
            gpu_cache_usage=gpu_cache_usage,
            cpu_cache_usage=cpu_cache_usage,
            num_prompt_tokens=num_prompt_tokens,
            num_generation_tokens=num_generation_tokens,
            time_to_first_tokens=time_to_first_tokens,
            time_per_output_tokens=time_per_output_tokens,
            time_e2e_requests=time_e2e_requests,
        )

    def _check_stop(self, seq: Sequence,
                    sampling_params: SamplingParams) -> None:
        """Stop the finished sequences."""
        # Check if the sequence has reached max_model_len.
        if seq.get_len() > self.scheduler_config.max_model_len:
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            return

        # Check if the sequence has reached max_tokens.
        if seq.get_output_len() == sampling_params.max_tokens:
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            return

        # Check if the minimum number of tokens has been generated yet;
        # skip the stop string/token checks if not
        if seq.get_output_len() < sampling_params.min_tokens:
            return

        for stop_str in sampling_params.stop:
            if seq.output_text.endswith(stop_str):
                self._finalize_sequence(seq, sampling_params, stop_str)
                seq.status = SequenceStatus.FINISHED_STOPPED
                seq.stop_reason = stop_str
                return
        last_token_id = seq.get_last_token_id()
        if last_token_id in sampling_params.stop_token_ids:
            stop_str = self.get_tokenizer_for_seq(seq).convert_ids_to_tokens(
                last_token_id)
            self._finalize_sequence(seq, sampling_params, stop_str)
            seq.status = SequenceStatus.FINISHED_STOPPED
            seq.stop_reason = last_token_id
            return

        # Check if the sequence has generated the EOS token.
        if ((not sampling_params.ignore_eos)
                and seq.get_last_token_id() == seq.eos_token_id):
            seq.status = SequenceStatus.FINISHED_STOPPED
            return

    def _finalize_sequence(self, seq: Sequence,
                           sampling_params: SamplingParams,
                           stop_string: str) -> None:
        if sampling_params.include_stop_str_in_output:
            return

        if stop_string and seq.output_text.endswith(stop_string):
            # Truncate the output text so that the stop string is
            # not included in the output.
            seq.output_text = seq.output_text[:-len(stop_string)]

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_executor.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_executor.remove_lora(lora_id)

    def list_loras(self) -> List[int]:
        return self.model_executor.list_loras()

    def check_health(self) -> None:
        self.model_executor.check_health()
