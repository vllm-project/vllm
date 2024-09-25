import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import (TYPE_CHECKING, Any, Callable, ClassVar, Deque, Dict,
                    Iterable, List, Mapping, NamedTuple, Optional)
from typing import Sequence as GenericSequence
from typing import Set, Type, Union

import torch
from typing_extensions import TypeVar

import vllm.envs as envs
from vllm.config import (CacheConfig, DecodingConfig, DeviceConfig,
                         EngineConfig, LoadConfig, LoRAConfig, ModelConfig,
                         ObservabilityConfig, ParallelConfig,
                         PromptAdapterConfig, SchedulerConfig,
                         SpeculativeConfig)
from vllm.core.scheduler_v2 import Scheduler
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.metrics_types import StatLoggerBase, Stats
from vllm.executor.executor_base import ExecutorBase
from vllm.executor.ray_utils import initialize_ray_cluster
from vllm.inputs import (INPUT_REGISTRY, EncoderDecoderLLMInputs,
                         InputRegistry, LLMInputs, PromptType)
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.outputs import (EmbeddingRequestOutput, RequestOutput,
                          RequestOutputFactory)
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.request import Request
from vllm.transformers_utils.config import try_get_generation_config
from vllm.transformers_utils.detokenizer import Detokenizer
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.transformers_utils.tokenizer_group import (
    BaseTokenizerGroup, init_tokenizer_from_configs)
from vllm.usage.usage_lib import (UsageContext, is_usage_stats_enabled,
                                  usage_message)
from vllm.version import __version__ as VLLM_VERSION
from vllm.request import Request

logger = init_logger(__name__)
_LOCAL_LOGGING_INTERVAL_SEC = 5


def _load_generation_config_dict(model_config: ModelConfig) -> Dict[str, Any]:
    config = try_get_generation_config(
        model_config.model,
        trust_remote_code=model_config.trust_remote_code,
        revision=model_config.revision,
    )

    if config is None:
        return {}

    return config.to_diff_dict()


_G = TypeVar("_G", bound=BaseTokenizerGroup, default=BaseTokenizerGroup)
_O = TypeVar("_O", RequestOutput, EmbeddingRequestOutput)


class LLMEngine:
    """An LLM engine that receives requests and generates texts.

    This is the main class for the vLLM engine. It receives requests
    from clients and generates texts from the LLM. It includes a tokenizer, a
    language model (possibly distributed across multiple GPUs), and GPU memory
    space allocated for intermediate states (aka KV cache). This class utilizes
    iteration-level scheduling and efficient memory management to maximize the
    serving throughput.

    The :class:`~vllm.LLM` class wraps this class for offline batched inference
    and the :class:`AsyncLLMEngine` class wraps this class for online serving.

    The config arguments are derived from :class:`~vllm.EngineArgs`. (See
    :ref:`engine_args`)

    Args:
        model_config: The configuration related to the LLM model.
        cache_config: The configuration related to the KV cache memory
            management.
        parallel_config: The configuration related to distributed execution.
        scheduler_config: The configuration related to the request scheduler.
        device_config: The configuration related to the device.
        lora_config (Optional): The configuration related to serving multi-LoRA.
        speculative_config (Optional): The configuration related to speculative
            decoding.
        executor_class: The model executor class for managing distributed
            execution.
        prompt_adapter_config (Optional): The configuration related to serving
            prompt adapters.
        log_stats: Whether to log statistics.
        usage_context: Specified entry point, used for usage info collection.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        load_config: LoadConfig,
        lora_config: Optional[LoRAConfig],
        speculative_config: Optional[SpeculativeConfig],
        decoding_config: Optional[DecodingConfig],
        observability_config: Optional[ObservabilityConfig],
        prompt_adapter_config: Optional[PromptAdapterConfig],
        executor_class: Type[ExecutorBase],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
        input_registry: InputRegistry = INPUT_REGISTRY,
    ) -> None:
        logger.info(
            "Initializing an LLM engine (v%s) with config: "
            "model=%r, speculative_config=%r, tokenizer=%r, "
            "skip_tokenizer_init=%s, tokenizer_mode=%s, revision=%s, "
            "override_neuron_config=%s, "
            "rope_scaling=%r, rope_theta=%r, tokenizer_revision=%s, "
            "trust_remote_code=%s, dtype=%s, max_seq_len=%d, "
            "download_dir=%r, load_format=%s, tensor_parallel_size=%d, "
            "pipeline_parallel_size=%d, "
            "disable_custom_all_reduce=%s, quantization=%s, "
            "enforce_eager=%s, kv_cache_dtype=%s, "
            "quantization_param_path=%s, device_config=%s, "
            "decoding_config=%r, observability_config=%r, "
            "seed=%d, served_model_name=%s, use_v2_block_manager=%s, "
            "num_scheduler_steps=%d, enable_prefix_caching=%s, "
            "use_async_output_proc=%s, mm_processor_kwargs=%s)",
            VLLM_VERSION,
            model_config.model,
            speculative_config,
            model_config.tokenizer,
            model_config.skip_tokenizer_init,
            model_config.tokenizer_mode,
            model_config.revision,
            model_config.override_neuron_config,
            model_config.rope_scaling,
            model_config.rope_theta,
            model_config.tokenizer_revision,
            model_config.trust_remote_code,
            model_config.dtype,
            model_config.max_model_len,
            load_config.download_dir,
            load_config.load_format,
            parallel_config.tensor_parallel_size,
            parallel_config.pipeline_parallel_size,
            parallel_config.disable_custom_all_reduce,
            model_config.quantization,
            model_config.enforce_eager,
            cache_config.cache_dtype,
            model_config.quantization_param_path,
            device_config.device,
            decoding_config,
            observability_config,
            model_config.seed,
            model_config.served_model_name,
            scheduler_config.use_v2_block_manager,
            scheduler_config.num_scheduler_steps,
            cache_config.enable_prefix_caching,
            model_config.use_async_output_proc,
            model_config.mm_processor_kwargs,
        )
        # TODO(woosuk): Print more configs in debug mode.
        from vllm.plugins import load_general_plugins
        load_general_plugins()

        self.model_config = model_config
        self.cache_config = cache_config
        self.lora_config = lora_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.speculative_config = speculative_config
        self.load_config = load_config
        self.decoding_config = decoding_config or DecodingConfig()
        self.prompt_adapter_config = prompt_adapter_config
        self.observability_config = observability_config or ObservabilityConfig(
        )
        self.log_stats = log_stats

        if not self.model_config.skip_tokenizer_init:
            self.tokenizer = self._init_tokenizer()
            self.detokenizer = Detokenizer(self.tokenizer)
            tokenizer_group = self.get_tokenizer_group()
        else:
            self.tokenizer = None
            self.detokenizer = None
            tokenizer_group = None

        self.generation_config_fields = _load_generation_config_dict(
            model_config)

        self.input_preprocessor = InputPreprocessor(model_config,
                                                    self.tokenizer)

        self.input_registry = input_registry
        self.input_processor = input_registry.create_input_processor(
            model_config)

        self.model_executor = executor_class(
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            device_config=device_config,
            lora_config=lora_config,
            speculative_config=speculative_config,
            load_config=load_config,
            prompt_adapter_config=prompt_adapter_config,
            observability_config=self.observability_config,
        )

        if not self.model_config.embedding_mode:
            self._initialize_kv_caches()

        # If usage stat is enabled, collect relevant info.
        if is_usage_stats_enabled():
            from vllm.model_executor.model_loader import (
                get_architecture_class_name)
            usage_message.report_usage(
                get_architecture_class_name(model_config),
                usage_context,
                extra_kvs={
                    # Common configuration
                    "dtype":
                    str(model_config.dtype),
                    "tensor_parallel_size":
                    parallel_config.tensor_parallel_size,
                    "block_size":
                    cache_config.block_size,
                    "gpu_memory_utilization":
                    cache_config.gpu_memory_utilization,

                    # Quantization
                    "quantization":
                    model_config.quantization,
                    "kv_cache_dtype":
                    str(cache_config.cache_dtype),

                    # Feature flags
                    "enable_lora":
                    bool(lora_config),
                    "enable_prompt_adapter":
                    bool(prompt_adapter_config),
                    "enable_prefix_caching":
                    cache_config.enable_prefix_caching,
                    "enforce_eager":
                    model_config.enforce_eager,
                    "disable_custom_all_reduce":
                    parallel_config.disable_custom_all_reduce,
                })

        if self.tokenizer:
            # Ping the tokenizer to ensure liveness if it runs in a
            # different process.
            self.tokenizer.ping()

        # Create the scheduler.
        # NOTE: the cache_config here have been updated with the numbers of
        # GPU and CPU blocks, which are profiled in the distributed executor.
        self.scheduler = Scheduler(
            scheduler_config, cache_config, lora_config)

        # Metric Logging.
        if self.log_stats:
            if stat_loggers is not None:
                self.stat_loggers = stat_loggers
            else:
                # Lazy import for prometheus multiprocessing.
                # We need to set PROMETHEUS_MULTIPROC_DIR environment variable
                # before prometheus_client is imported.
                # See https://prometheus.github.io/client_python/multiprocess/
                from vllm.engine.metrics import (LoggingStatLogger,
                                                 PrometheusStatLogger)

                self.stat_loggers = {
                    "logging":
                    LoggingStatLogger(
                        local_interval=_LOCAL_LOGGING_INTERVAL_SEC),
                    "prometheus":
                    PrometheusStatLogger(
                        local_interval=_LOCAL_LOGGING_INTERVAL_SEC,
                        labels=dict(model_name=model_config.served_model_name),
                        max_model_len=self.model_config.max_model_len),
                }
                self.stat_loggers["prometheus"].info("cache_config",
                                                     self.cache_config)

    def _initialize_kv_caches(self) -> None:
        """Initialize the KV cache in the worker(s).

        The workers will determine the number of blocks in both the GPU cache
        and the swap CPU cache.
        """
        num_gpu_blocks, num_cpu_blocks = (
            self.model_executor.determine_num_available_blocks())

        if self.cache_config.num_gpu_blocks_override is not None:
            num_gpu_blocks_override = self.cache_config.num_gpu_blocks_override
            logger.info(
                "Overriding num_gpu_blocks=%d with "
                "num_gpu_blocks_override=%d", num_gpu_blocks,
                num_gpu_blocks_override)
            num_gpu_blocks = num_gpu_blocks_override

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        self.model_executor.initialize_cache(num_gpu_blocks, num_cpu_blocks)

    @classmethod
    def from_engine_args(
        cls,
        engine_args: EngineArgs,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
    ) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_config = engine_args.create_engine_config()
        executor_class = cls._get_executor_cls(engine_config)
        # Create the LLM engine.
        engine = cls(
            **engine_config.to_dict(),
            executor_class=executor_class,
            log_stats=not engine_args.disable_log_stats,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
        )

        return engine

    def __reduce__(self):
        # This is to ensure that the LLMEngine is not referenced in
        # the closure used to initialize Ray worker actors
        raise RuntimeError("LLMEngine should not be pickled!")

    def __del__(self):
        # Shutdown model executor when engine is garbage collected
        # Use getattr since __init__ can fail before the field is set
        if model_executor := getattr(self, "model_executor", None):
            model_executor.shutdown()

    def get_tokenizer_group(
        self,
        group_type: Type[_G] = BaseTokenizerGroup,
    ) -> _G:
        tokenizer_group = self.tokenizer

        if tokenizer_group is None:
            raise ValueError("Unable to get tokenizer because "
                             "skip_tokenizer_init is True")
        if not isinstance(tokenizer_group, group_type):
            raise TypeError("Invalid type of tokenizer group. "
                            f"Expected type: {group_type}, but "
                            f"found type: {type(tokenizer_group)}")

        return tokenizer_group

    def get_tokenizer(
        self,
        lora_request: Optional[LoRARequest] = None,
    ) -> AnyTokenizer:
        return self.get_tokenizer_group().get_lora_tokenizer(lora_request)

    def _init_tokenizer(self) -> BaseTokenizerGroup:
        return init_tokenizer_from_configs(
            model_config=self.model_config,
            scheduler_config=self.scheduler_config,
            parallel_config=self.parallel_config,
            enable_lora=bool(self.lora_config))

    def _verify_args(self) -> None:
        self.model_config.verify_with_parallel_config(self.parallel_config)
        self.cache_config.verify_with_parallel_config(self.parallel_config)
        if self.lora_config:
            self.lora_config.verify_with_model_config(self.model_config)
            self.lora_config.verify_with_scheduler_config(
                self.scheduler_config)
        if self.prompt_adapter_config:
            self.prompt_adapter_config.verify_with_model_config(
                self.model_config)

    def _add_processed_request(
        self,
        request_id: str,
        processed_inputs: Union[LLMInputs, EncoderDecoderLLMInputs],
        params: Union[SamplingParams, PoolingParams],
        arrival_time: float,
        lora_request: Optional[LoRARequest],
        prompt_adapter_request: Optional[PromptAdapterRequest],
        trace_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._validate_model_inputs(processed_inputs)
        eos_token_id = self.input_preprocessor.get_eos_token_id(lora_request)

        # TODO(woosuk): Support embedding mode.
        assert isinstance(params, SamplingParams)
        sampling_params = params.clone()
        sampling_params.update_from_generation_config(
            self.generation_config_fields, eos_token_id)

        # TODO(woosuk): Check max_logprobs
        # TODO(woosuk): Support encoder-decoder models.
        req = Request(request_id, processed_inputs, arrival_time, sampling_params=params)
        self.scheduler.add_req(req)

    def stop_remote_worker_execution_loop(self) -> None:
        self.model_executor.stop_remote_worker_execution_loop()

    def add_request(
        self,
        request_id: str,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    ) -> None:
        """Add a request to the engine's request pool.

        The request is added to the request pool and will be processed by the
        scheduler as `engine.step()` is called. The exact scheduling policy is
        determined by the scheduler.

        Args:
            request_id: The unique ID of the request.
            prompt: The prompt to the LLM. See :class:`~vllm.inputs.PromptType`
                for more details about the format of each input.
            params: Parameters for sampling or pooling.
                :class:`~vllm.SamplingParams` for text generation.
                :class:`~vllm.PoolingParams` for pooling.
            arrival_time: The arrival time of the request. If None, we use
                the current monotonic time.
            trace_headers: OpenTelemetry trace headers.

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
        if arrival_time is None:
            arrival_time = time.time()

        preprocessed_inputs = self.input_preprocessor.preprocess(
            prompt,
            request_id=request_id,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
        )
        processed_inputs = self.input_processor(preprocessed_inputs)

        self._add_processed_request(
            request_id=request_id,
            processed_inputs=processed_inputs,
            params=params,
            arrival_time=arrival_time,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
            trace_headers=trace_headers,
        )

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
        self.scheduler.abort_reqs(request_id)

    def get_model_config(self) -> ModelConfig:
        """Gets the model configuration."""
        return self.model_config

    def get_parallel_config(self) -> ParallelConfig:
        """Gets the parallel configuration."""
        return self.parallel_config

    def get_decoding_config(self) -> DecodingConfig:
        """Gets the decoding configuration."""
        return self.decoding_config

    def get_scheduler_config(self) -> SchedulerConfig:
        """Gets the scheduler configuration."""
        return self.scheduler_config

    def get_lora_config(self) -> LoRAConfig:
        """Gets the LoRA configuration."""
        return self.lora_config

    def get_num_unfinished_requests(self) -> int:
        """Gets the number of unfinished requests."""
        return self.scheduler.get_num_unfinished_reqs()

    def has_unfinished_requests(self) -> bool:
        """Returns True if there are unfinished requests."""
        return self.scheduler.has_unfinished_req()

    def step(self) -> List[Union[RequestOutput, EmbeddingRequestOutput]]:
        scheduler_output = self.scheduler.schedule()
        sampler_output = self.model_executor.execute_model(scheduler_output)
        self._process_model_outputs(sampler_output)

        if not self.has_unfinished_requests():
            # Stop the execute model loop in parallel workers until there are
            # more requests to process. This avoids waiting indefinitely in
            # torch.distributed ops which may otherwise timeout, and unblocks
            # the RPC thread in the workers so that they can process any other
            # queued control plane messages, such as add/remove lora adapters.
            logger.debug("Stopping remote worker execution loop.")
            self.model_executor.stop_remote_worker_execution_loop()
        return sampler_output

    def add_logger(self, logger_name: str, logger: StatLoggerBase) -> None:
        if not self.log_stats:
            raise RuntimeError(
                "Stat logging is disabled. Set `disable_log_stats=False` "
                "argument to enable.")
        if logger_name in self.stat_loggers:
            raise KeyError(f"Logger with name {logger_name} already exists.")
        self.stat_loggers[logger_name] = logger

    def remove_logger(self, logger_name: str) -> None:
        if not self.log_stats:
            raise RuntimeError(
                "Stat logging is disabled. Set `disable_log_stats=False` "
                "argument to enable.")
        if logger_name not in self.stat_loggers:
            raise KeyError(f"Logger with name {logger_name} does not exist.")
        del self.stat_loggers[logger_name]

    def check_health(self) -> None:
        if self.tokenizer:
            self.tokenizer.check_health()
        self.model_executor.check_health()

    def _validate_model_inputs(self, inputs: Union[LLMInputs,
                                                   EncoderDecoderLLMInputs]):
        if self.is_encoder_decoder_model():
            prompt_ids = inputs.get("encoder_prompt_token_ids")
        else:
            prompt_ids = inputs.get("prompt_token_ids")

        if prompt_ids is None or len(prompt_ids) == 0:
            raise ValueError("Prompt cannot be empty")

        if self.model_config.is_multimodal_model:
            max_prompt_len = self.model_config.max_model_len

            if len(prompt_ids) > max_prompt_len:
                raise ValueError(
                    f"The prompt (total length {len(prompt_ids)}) is too long "
                    f"to fit into the model (context length {max_prompt_len}). "
                    "Make sure that `max_model_len` is no smaller than the "
                    "number of text tokens plus multimodal tokens. For image "
                    "inputs, the number of image tokens depends on the number "
                    "of images, and possibly their aspect ratios as well.")

            # TODO: Find out how many placeholder tokens are there so we can
            # check that chunked prefill does not truncate them
            # max_batch_len = self.scheduler_config.max_num_batched_tokens


def _get_executor_cls(engine_config: EngineConfig) -> Type[ExecutorBase]:
    distributed_executor_backend = (
        engine_config.parallel_config.distributed_executor_backend)
    # Initialize the cluster and specify the executor class.
    if isinstance(distributed_executor_backend, type):
        if not issubclass(distributed_executor_backend, ExecutorBase):
            raise TypeError(
                "distributed_executor_backend must be a subclass of "
                f"ExecutorBase. Got {distributed_executor_backend}.")
        if distributed_executor_backend.uses_ray:  # type: ignore
            initialize_ray_cluster(engine_config.parallel_config)
        executor_class = distributed_executor_backend
    elif engine_config.device_config.device_type == "neuron":
        from vllm.executor.neuron_executor import NeuronExecutor
        executor_class = NeuronExecutor
    elif engine_config.device_config.device_type == "tpu":
        if distributed_executor_backend == "ray":
            initialize_ray_cluster(engine_config.parallel_config)
            from vllm.executor.ray_tpu_executor import RayTPUExecutor
            executor_class = RayTPUExecutor
        else:
            assert distributed_executor_backend is None
            from vllm.executor.tpu_executor import TPUExecutor
            executor_class = TPUExecutor
    elif engine_config.device_config.device_type == "cpu":
        from vllm.executor.cpu_executor import CPUExecutor
        executor_class = CPUExecutor
    elif engine_config.device_config.device_type == "openvino":
        from vllm.executor.openvino_executor import OpenVINOExecutor
        executor_class = OpenVINOExecutor
    elif engine_config.device_config.device_type == "xpu":
        if distributed_executor_backend == "ray":
            initialize_ray_cluster(engine_config.parallel_config)
            from vllm.executor.ray_xpu_executor import RayXPUExecutor
            executor_class = RayXPUExecutor
        elif distributed_executor_backend == "mp":
            # FIXME(kunshang):
            # spawn needs calling `if __name__ == '__main__':``
            # fork is not supported for xpu start new process.
            logger.error(
                "Both start methods (spawn and fork) have issue "
                "on XPU if you use mp backend, Please try ray instead.")
        else:
            from vllm.executor.xpu_executor import XPUExecutor
            executor_class = XPUExecutor
    elif distributed_executor_backend == "ray":
        initialize_ray_cluster(engine_config.parallel_config)
        from vllm.executor.ray_gpu_executor import RayGPUExecutor
        executor_class = RayGPUExecutor
    elif distributed_executor_backend == "mp":
        from vllm.executor.multiproc_gpu_executor import (
            MultiprocessingGPUExecutor)
        assert not envs.VLLM_USE_RAY_SPMD_WORKER, (
            "multiprocessing distributed executor backend does not "
            "support VLLM_USE_RAY_SPMD_WORKER=1")
        executor_class = MultiprocessingGPUExecutor
    else:
        from vllm.executor.gpu_executor import GPUExecutor
        executor_class = GPUExecutor
    return executor_class
