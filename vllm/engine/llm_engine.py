# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import time
from collections import Counter as collectionsCounter
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import (TYPE_CHECKING, Any, Callable, ClassVar, Deque, Dict,
                    Iterable, List, Literal, Mapping, NamedTuple, Optional)
from typing import Sequence as GenericSequence
from typing import Set, Type, Union, cast

import torch
from typing_extensions import TypeVar

import vllm.envs as envs
from vllm.config import (DecodingConfig, LoRAConfig, ModelConfig,
                         ObservabilityConfig, ParallelConfig, SchedulerConfig,
                         VllmConfig)
from vllm.core.scheduler import ScheduledSequenceGroup, SchedulerOutputs
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.metrics_types import StatLoggerBase, Stats
from vllm.engine.output_processor.interfaces import (
    SequenceGroupOutputProcessor)
from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.engine.output_processor.util import create_output_by_sequence_group
from vllm.entrypoints.openai.logits_processors import (
    get_logits_processors as get_openai_logits_processors)
from vllm.executor.executor_base import ExecutorBase
from vllm.inputs import ProcessorInputs, PromptType, SingletonInputs
from vllm.inputs.parse import split_enc_dec_inputs
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.logits_process import get_bad_words_logits_processors
from vllm.lora.request import LoRARequest
from vllm.model_executor.guided_decoding import (
    get_local_guided_decoding_logits_processor)
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.multimodal.processing import EncDecMultiModalProcessor
from vllm.outputs import (PoolingRequestOutput, RequestOutput,
                          RequestOutputFactory)
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.sequence import (ExecuteModelRequest, ParallelSampleSequenceGroup,
                           PoolingSequenceGroupOutput, Sequence, SequenceGroup,
                           SequenceGroupBase, SequenceGroupMetadata,
                           SequenceGroupOutput, SequenceStatus)
from vllm.tracing import (SpanAttributes, SpanKind, extract_trace_context,
                          init_tracer)
from vllm.transformers_utils.detokenizer import Detokenizer
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.transformers_utils.tokenizer_group import (
    TokenizerGroup, init_tokenizer_from_configs)
from vllm.usage.usage_lib import (UsageContext, is_usage_stats_enabled,
                                  usage_message)
from vllm.utils import Counter, Device, resolve_obj_by_qualname, weak_bind
from vllm.version import __version__ as VLLM_VERSION
from vllm.worker.model_runner_base import InputProcessingError

logger = init_logger(__name__)
_LOCAL_LOGGING_INTERVAL_SEC = 5

_O = TypeVar("_O", RequestOutput, PoolingRequestOutput)
_R = TypeVar("_R", default=Any)


@dataclass
class SchedulerOutputState:
    """Caches the scheduler outputs for a virtual engine. Used for Multi-Step"""
    seq_group_metadata_list: Optional[List[SequenceGroupMetadata]] = None
    scheduler_outputs: Optional[SchedulerOutputs] = None
    allow_async_output_proc: bool = False
    last_output: Optional[SamplerOutput] = None


class OutputData(NamedTuple):
    outputs: List[SamplerOutput]
    seq_group_metadata_list: List[SequenceGroupMetadata]
    scheduler_outputs: SchedulerOutputs
    is_async: bool
    is_last_step: bool
    # Indicates if this output is from the first step of the
    # multi-step. When multi-step is disabled, this is always
    # set to True.
    # is_first_step_output is invalid when `outputs` has
    # outputs from multiple steps.
    is_first_step_output: Optional[bool]
    skip: List[int]


class SchedulerContext:

    def __init__(self, multi_step_stream_outputs: bool = False):
        self.output_queue: Deque[OutputData] = deque()
        self.request_outputs: List[Union[RequestOutput,
                                         PoolingRequestOutput]] = []
        self.seq_group_metadata_list: Optional[
            List[SequenceGroupMetadata]] = None
        self.scheduler_outputs: Optional[SchedulerOutputs] = None

        self.multi_step_stream_outputs: bool = multi_step_stream_outputs

    def append_output(self, outputs: List[SamplerOutput],
                      seq_group_metadata_list: List[SequenceGroupMetadata],
                      scheduler_outputs: SchedulerOutputs, is_async: bool,
                      is_last_step: bool,
                      is_first_step_output: Optional[bool]):
        self.output_queue.append(
            OutputData(outputs=outputs,
                       seq_group_metadata_list=seq_group_metadata_list,
                       scheduler_outputs=scheduler_outputs,
                       is_async=is_async,
                       is_last_step=is_last_step,
                       is_first_step_output=is_first_step_output,
                       skip=[]))


class LLMEngine:
    """An LLM engine that receives requests and generates texts.

    This is the main class for the vLLM engine. It receives requests
    from clients and generates texts from the LLM. It includes a tokenizer, a
    language model (possibly distributed across multiple GPUs), and GPU memory
    space allocated for intermediate states (aka KV cache). This class utilizes
    iteration-level scheduling and efficient memory management to maximize the
    serving throughput.

    The [`LLM`][vllm.LLM] class wraps this class for offline batched inference
    and the [`AsyncLLMEngine`][vllm.engine.async_llm_engine.AsyncLLMEngine]
    class wraps this class for online serving.

    The config arguments are derived from [`EngineArgs`][vllm.EngineArgs].

    Args:
        vllm_config: The configuration for initializing and running vLLM.
        executor_class: The model executor class for managing distributed
            execution.
        log_stats: Whether to log statistics.
        usage_context: Specified entry point, used for usage info collection.
    """

    DO_VALIDATE_OUTPUT: ClassVar[bool] = False
    """A flag to toggle whether to validate the type of request output."""

    @classmethod
    @contextmanager
    def enable_output_validation(cls):
        cls.DO_VALIDATE_OUTPUT = True

        yield

        cls.DO_VALIDATE_OUTPUT = False

    @classmethod
    def validate_output(
        cls,
        output: object,
        output_type: Type[_O],
    ) -> _O:
        do_validate = cls.DO_VALIDATE_OUTPUT

        if ((TYPE_CHECKING or do_validate)
                and not isinstance(output, output_type)):
            raise TypeError(f"Expected output of type {output_type}, "
                            f"but found type {type(output)}")

        return cast(_O, output)

    @classmethod
    def validate_outputs(
        cls,
        outputs: GenericSequence[object],
        output_type: Type[_O],
    ) -> List[_O]:
        do_validate = cls.DO_VALIDATE_OUTPUT

        outputs_: List[_O]
        if TYPE_CHECKING or do_validate:
            outputs_ = []
            for output in outputs:
                if not isinstance(output, output_type):
                    raise TypeError(f"Expected output of type {output_type}, "
                                    f"but found type {type(output)}")

                outputs_.append(output)
        else:
            outputs_ = outputs

        return outputs_

    tokenizer: Optional[TokenizerGroup]

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: Type[ExecutorBase],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        use_cached_outputs: bool = False,
    ) -> None:
        if envs.VLLM_USE_V1:
            raise ValueError(
                "Using V0 LLMEngine, but envs.VLLM_USE_V1=True. "
                "This should not happen. As a workaround, try using "
                "LLMEngine.from_vllm_config(...) or explicitly set "
                "VLLM_USE_V1=0 or 1 and report this issue on Github.")

        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config  # noqa
        self.load_config = vllm_config.load_config
        self.decoding_config = vllm_config.decoding_config or DecodingConfig(  # noqa
        )
        self.prompt_adapter_config = vllm_config.prompt_adapter_config  # noqa
        self.observability_config = vllm_config.observability_config or ObservabilityConfig(  # noqa
        )

        logger.info(
            "Initializing a V0 LLM engine (v%s) with config: %s, "
            "use_cached_outputs=%s, ",
            VLLM_VERSION,
            vllm_config,
            use_cached_outputs,
        )

        self.log_stats = log_stats
        self.use_cached_outputs = use_cached_outputs

        if not self.model_config.skip_tokenizer_init:
            self.tokenizer = self._init_tokenizer()
            self.detokenizer = Detokenizer(self.tokenizer)
            tokenizer_group = self.get_tokenizer_group()
        else:
            self.tokenizer = None
            self.detokenizer = None
            tokenizer_group = None

        # Ensure that the function doesn't contain a reference to self,
        # to avoid engine GC issues
        def get_tokenizer_for_seq(sequence: Sequence) -> AnyTokenizer:
            assert tokenizer_group, ("tokenizer_group cannot be None, "
                                     "make sure skip_tokenizer_init is False")
            return tokenizer_group.get_lora_tokenizer(sequence.lora_request)

        self.seq_counter = Counter()
        self.generation_config_fields = (
            self.model_config.try_get_generation_config())

        self.input_preprocessor = InputPreprocessor(self.model_config,
                                                    self.tokenizer,
                                                    mm_registry)

        self.model_executor = executor_class(vllm_config=vllm_config)

        if self.model_config.runner_type != "pooling":
            self._initialize_kv_caches()

        # If usage stat is enabled, collect relevant info.
        if is_usage_stats_enabled():
            from vllm.model_executor.model_loader import (
                get_architecture_class_name)
            usage_message.report_usage(
                get_architecture_class_name(self.model_config),
                usage_context,
                extra_kvs={
                    # Common configuration
                    "dtype":
                    str(self.model_config.dtype),
                    "tensor_parallel_size":
                    self.parallel_config.tensor_parallel_size,
                    "block_size":
                    self.cache_config.block_size,
                    "gpu_memory_utilization":
                    self.cache_config.gpu_memory_utilization,

                    # Quantization
                    "quantization":
                    self.model_config.quantization,
                    "kv_cache_dtype":
                    str(self.cache_config.cache_dtype),

                    # Feature flags
                    "enable_lora":
                    bool(self.lora_config),
                    "enable_prompt_adapter":
                    bool(self.prompt_adapter_config),
                    "enable_prefix_caching":
                    self.cache_config.enable_prefix_caching,
                    "enforce_eager":
                    self.model_config.enforce_eager,
                    "disable_custom_all_reduce":
                    self.parallel_config.disable_custom_all_reduce,
                })

        self.cached_scheduler_outputs = [
            SchedulerOutputState()
            for _ in range(self.parallel_config.pipeline_parallel_size)
        ]

        self.scheduler_contexts = [
            SchedulerContext(multi_step_stream_outputs=self.scheduler_config.
                             multi_step_stream_outputs)
            for _ in range(self.parallel_config.pipeline_parallel_size)
        ]

        if self.model_config.use_async_output_proc:
            process_model_outputs = weak_bind(self._process_model_outputs)

            self.async_callbacks = [
                partial(process_model_outputs,
                        ctx=self.scheduler_contexts[v_id])
                for v_id in range(self.parallel_config.pipeline_parallel_size)
            ]
        else:
            self.async_callbacks = []

        # Currently used by AsyncLLMEngine to ensure quick append
        # of request outputs to asyncio queues
        self.process_request_outputs_callback: Optional[Callable] = None

        # Create the scheduler.
        # NOTE: the cache_config here have been updated with the numbers of
        # GPU and CPU blocks, which are profiled in the distributed executor.
        if isinstance(self.vllm_config.scheduler_config.scheduler_cls, str):
            Scheduler = resolve_obj_by_qualname(
                self.vllm_config.scheduler_config.scheduler_cls)
        else:
            Scheduler = self.vllm_config.scheduler_config.scheduler_cls
        self.scheduler = [
            Scheduler(
                self.scheduler_config, self.cache_config, self.lora_config,
                self.parallel_config.pipeline_parallel_size,
                self.async_callbacks[v_id]
                if self.model_config.use_async_output_proc else None)
            for v_id in range(self.parallel_config.pipeline_parallel_size)
        ]

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
                        local_interval=_LOCAL_LOGGING_INTERVAL_SEC,
                        vllm_config=vllm_config),
                    "prometheus":
                    PrometheusStatLogger(
                        local_interval=_LOCAL_LOGGING_INTERVAL_SEC,
                        labels=dict(
                            model_name=self.model_config.served_model_name),
                        vllm_config=vllm_config),
                }
                self.stat_loggers["prometheus"].info("cache_config",
                                                     self.cache_config)

        self.tracer = None
        if self.observability_config.otlp_traces_endpoint:
            self.tracer = init_tracer(
                "vllm.llm_engine",
                self.observability_config.otlp_traces_endpoint)

        # Create sequence output processor, e.g. for beam search or
        # speculative decoding.
        self.output_processor = (
            SequenceGroupOutputProcessor.create_output_processor(
                self.scheduler_config,
                self.detokenizer,
                self.scheduler,
                self.seq_counter,
                get_tokenizer_for_seq,
                stop_checker=StopChecker(self.scheduler_config.max_model_len,
                                         get_tokenizer_for_seq),
            ))

        self.seq_id_to_seq_group: Dict[str, SequenceGroupBase] = {}

        # Flag to set when an input fails to process and the engine should run
        # the next step without re-scheduling.
        self._skip_scheduling_next_step = False

        # Don't keep the dummy data in memory
        self.reset_mm_cache()

    def _initialize_kv_caches(self) -> None:
        """Initialize the KV cache in the worker(s).

        The workers will determine the number of blocks in both the GPU cache
        and the swap CPU cache.
        """
        start = time.time()
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
        elapsed = time.time() - start
        logger.info(("init engine (profile, create kv cache, "
                     "warmup model) took %.2f seconds"), elapsed)

    @classmethod
    def _get_executor_cls(cls,
                          engine_config: VllmConfig) -> Type[ExecutorBase]:
        # distributed_executor_backend must be set in VllmConfig.__post_init__
        distributed_executor_backend = (
            engine_config.parallel_config.distributed_executor_backend)
        # Initialize the cluster and specify the executor class.
        if isinstance(distributed_executor_backend, type):
            if not issubclass(distributed_executor_backend, ExecutorBase):
                raise TypeError(
                    "distributed_executor_backend must be a subclass of "
                    f"ExecutorBase. Got {distributed_executor_backend}.")
            executor_class = distributed_executor_backend
        elif distributed_executor_backend == "ray":
            from vllm.executor.ray_distributed_executor import (
                RayDistributedExecutor)
            executor_class = RayDistributedExecutor
        elif distributed_executor_backend == "mp":
            from vllm.executor.mp_distributed_executor import (
                MultiprocessingDistributedExecutor)
            assert not envs.VLLM_USE_RAY_SPMD_WORKER, (
                "multiprocessing distributed executor backend does not "
                "support VLLM_USE_RAY_SPMD_WORKER=1")
            executor_class = MultiprocessingDistributedExecutor
        elif distributed_executor_backend == "uni":
            # JAX-style, single-process, multi-device executor.
            from vllm.executor.uniproc_executor import UniProcExecutor
            executor_class = UniProcExecutor
        elif distributed_executor_backend == "external_launcher":
            # executor with external launcher
            from vllm.executor.uniproc_executor import (  # noqa
                ExecutorWithExternalLauncher)
            executor_class = ExecutorWithExternalLauncher
        else:
            raise ValueError("unrecognized distributed_executor_backend: "
                             f"{distributed_executor_backend}")
        return executor_class

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
        disable_log_stats: bool = False,
    ) -> "LLMEngine":
        return cls(
            vllm_config=vllm_config,
            executor_class=cls._get_executor_cls(vllm_config),
            log_stats=(not disable_log_stats),
            usage_context=usage_context,
            stat_loggers=stat_loggers,
        )

    @classmethod
    def from_engine_args(
        cls,
        engine_args: EngineArgs,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
    ) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        vllm_config = engine_args.create_engine_config(usage_context)

        engine_cls = cls
        if envs.VLLM_USE_V1:
            from vllm.v1.engine.llm_engine import LLMEngine as V1LLMEngine
            engine_cls = V1LLMEngine

        return engine_cls.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            disable_log_stats=engine_args.disable_log_stats,
        )

    def __reduce__(self):
        # This is to ensure that the LLMEngine is not referenced in
        # the closure used to initialize Ray worker actors
        raise RuntimeError("LLMEngine should not be pickled!")

    def __del__(self):
        # Shutdown model executor when engine is garbage collected
        # Use getattr since __init__ can fail before the field is set
        if model_executor := getattr(self, "model_executor", None):
            model_executor.shutdown()

    def get_tokenizer_group(self) -> TokenizerGroup:
        if self.tokenizer is None:
            raise ValueError("Unable to get tokenizer because "
                             "skip_tokenizer_init is True")

        return self.tokenizer

    def get_tokenizer(
        self,
        lora_request: Optional[LoRARequest] = None,
    ) -> AnyTokenizer:
        return self.get_tokenizer_group().get_lora_tokenizer(lora_request)

    def _init_tokenizer(self) -> TokenizerGroup:
        return init_tokenizer_from_configs(
            model_config=self.model_config,
            scheduler_config=self.scheduler_config,
            lora_config=self.lora_config)

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
        processed_inputs: ProcessorInputs,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: float,
        lora_request: Optional[LoRARequest],
        prompt_adapter_request: Optional[PromptAdapterRequest],
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
    ) -> Optional[SequenceGroup]:
        """Add a processed request to the engine's request pool.
        return the created sequence group.
        """
        if isinstance(params, SamplingParams) and params.n > 1:
            ParallelSampleSequenceGroup.add_request(
                request_id,
                self,
                params,
                processed_inputs=processed_inputs,
                arrival_time=arrival_time,
                lora_request=lora_request,
                trace_headers=trace_headers,
                prompt_adapter_request=prompt_adapter_request,
                priority=priority,
            )
            return None

        self._validate_model_inputs(processed_inputs, lora_request)
        # Create the sequences.
        block_size = self.cache_config.block_size
        seq_id = next(self.seq_counter)
        eos_token_id = self.input_preprocessor.get_eos_token_id(lora_request)

        encoder_inputs, decoder_inputs = split_enc_dec_inputs(processed_inputs)

        seq = Sequence(seq_id, decoder_inputs, block_size, eos_token_id,
                       lora_request, prompt_adapter_request)

        encoder_seq = (None if encoder_inputs is None else Sequence(
            seq_id, encoder_inputs, block_size, eos_token_id, lora_request,
            prompt_adapter_request))

        # Create a SequenceGroup based on SamplingParams or PoolingParams
        if isinstance(params, SamplingParams):
            seq_group = self._create_sequence_group_with_sampling(
                request_id,
                seq,
                params,
                arrival_time=arrival_time,
                lora_request=lora_request,
                trace_headers=trace_headers,
                prompt_adapter_request=prompt_adapter_request,
                encoder_seq=encoder_seq,
                priority=priority)
        elif isinstance(params, PoolingParams):
            seq_group = self._create_sequence_group_with_pooling(
                request_id,
                seq,
                params,
                arrival_time=arrival_time,
                lora_request=lora_request,
                prompt_adapter_request=prompt_adapter_request,
                encoder_seq=encoder_seq,
                priority=priority)
        else:
            raise ValueError(
                "Either SamplingParams or PoolingParams must be provided.")

        # Add the sequence group to the scheduler with least unfinished seqs.
        costs = [
            scheduler.get_num_unfinished_seq_groups()
            for scheduler in self.scheduler
        ]
        min_cost_scheduler = self.scheduler[costs.index(min(costs))]
        min_cost_scheduler.add_seq_group(seq_group)

        return seq_group

    def stop_remote_worker_execution_loop(self) -> None:
        self.model_executor.stop_remote_worker_execution_loop()

    def add_request(
        self,
        request_id: str,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        tokenization_kwargs: Optional[dict[str, Any]] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> None:
        """Add a request to the engine's request pool.

        The request is added to the request pool and will be processed by the
        scheduler as `engine.step()` is called. The exact scheduling policy is
        determined by the scheduler.

        Args:
            request_id: The unique ID of the request.
            prompt: The prompt to the LLM. See
                [PromptType][vllm.inputs.PromptType]
                for more details about the format of each input.
            params: Parameters for sampling or pooling.
                [SamplingParams][vllm.SamplingParams] for text generation.
                [PoolingParams][vllm.PoolingParams] for pooling.
            arrival_time: The arrival time of the request. If None, we use
                the current monotonic time.
            lora_request: The LoRA request to add.
            trace_headers: OpenTelemetry trace headers.
            prompt_adapter_request: The prompt adapter request to add.
            priority: The priority of the request.
                Only applicable with priority scheduling.

        Details:
            - Set arrival_time to the current time if it is None.
            - Set prompt_token_ids to the encoded prompt if it is None.
            - Create `n` number of [Sequence][vllm.Sequence] objects.
            - Create a [SequenceGroup][vllm.SequenceGroup] object
              from the list of [Sequence][vllm.Sequence].
            - Add the [SequenceGroup][vllm.SequenceGroup] object to the
              scheduler.

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
        if not isinstance(request_id, str):
            raise TypeError(
                f"request_id must be a string, got {type(request_id)}")

        if lora_request is not None and not self.lora_config:
            raise ValueError(f"Got lora_request {lora_request} but LoRA is "
                             "not enabled!")

        if priority != 0 and not self.scheduler_config.policy == "priority":
            raise ValueError(f"Got priority {priority} but "
                             "Priority scheduling is not enabled.")

        if isinstance(params, SamplingParams) \
            and (params.guided_decoding or params.logits_processors) \
            and self.scheduler_config.num_scheduler_steps > 1:
            raise ValueError(
                "Guided decoding and logits processors are not supported "
                "in multi-step decoding")

        if arrival_time is None:
            arrival_time = time.time()

        if (isinstance(prompt, dict)
                and prompt.get("prompt_embeds", None) is not None
                and not prompt.get("prompt_token_ids", None)):
            seq_len = prompt["prompt_embeds"].shape[0]
            prompt["prompt_token_ids"] = [0] * seq_len

        processed_inputs = self.input_preprocessor.preprocess(
            prompt,
            tokenization_kwargs=tokenization_kwargs,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
        )

        self._add_processed_request(
            request_id=request_id,
            processed_inputs=processed_inputs,
            params=params,
            arrival_time=arrival_time,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
            trace_headers=trace_headers,
            priority=priority,
        )

    def _create_sequence_group_with_sampling(
        self,
        request_id: str,
        seq: Sequence,
        sampling_params: SamplingParams,
        arrival_time: float,
        lora_request: Optional[LoRARequest],
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        encoder_seq: Optional[Sequence] = None,
        priority: int = 0,
    ) -> SequenceGroup:
        """Creates a SequenceGroup with SamplingParams."""
        max_logprobs = self.get_model_config().max_logprobs
        if (sampling_params.logprobs
                and sampling_params.logprobs > max_logprobs) or (
                    sampling_params.prompt_logprobs
                    and sampling_params.prompt_logprobs > max_logprobs):
            raise ValueError(f"Cannot request more than "
                             f"{max_logprobs} logprobs.")

        sampling_params = self._build_logits_processors(
            sampling_params, lora_request)

        # Defensive copy of SamplingParams, which are used by the sampler,
        # this doesn't deep-copy LogitsProcessor objects
        sampling_params = sampling_params.clone()

        sampling_params.update_from_generation_config(
            self.generation_config_fields, seq.eos_token_id)

        # Create the sequence group.
        draft_size = 1
        if self.vllm_config.speculative_config is not None:
            draft_size = \
                self.vllm_config.speculative_config.num_speculative_tokens + 1
        seq_group = SequenceGroup(
            request_id=request_id,
            seqs=[seq],
            arrival_time=arrival_time,
            sampling_params=sampling_params,
            lora_request=lora_request,
            trace_headers=trace_headers,
            prompt_adapter_request=prompt_adapter_request,
            encoder_seq=encoder_seq,
            priority=priority,
            draft_size=draft_size)

        return seq_group

    def _create_sequence_group_with_pooling(
        self,
        request_id: str,
        seq: Sequence,
        pooling_params: PoolingParams,
        arrival_time: float,
        lora_request: Optional[LoRARequest],
        prompt_adapter_request: Optional[PromptAdapterRequest],
        encoder_seq: Optional[Sequence] = None,
        priority: int = 0,
    ) -> SequenceGroup:
        """Creates a SequenceGroup with PoolingParams."""
        # Defensive copy of PoolingParams, which are used by the pooler
        pooling_params = pooling_params.clone()
        # Create the sequence group.
        seq_group = SequenceGroup(
            request_id=request_id,
            seqs=[seq],
            arrival_time=arrival_time,
            lora_request=lora_request,
            pooling_params=pooling_params,
            prompt_adapter_request=prompt_adapter_request,
            encoder_seq=encoder_seq,
            priority=priority)
        return seq_group

    def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        """Aborts a request(s) with the given ID.

        Args:
            request_id: The ID(s) of the request to abort.

        Details:
            - Refer to [vllm.core.scheduler.Scheduler.abort_seq_group][].

        Example:
            >>> # initialize engine and add a request with request_id
            >>> request_id = str(0)
            >>> # abort the request
            >>> engine.abort_request(request_id)
        """
        for scheduler in self.scheduler:
            scheduler.abort_seq_group(
                request_id, seq_id_to_seq_group=self.seq_id_to_seq_group)

    def get_vllm_config(self) -> VllmConfig:
        """Gets the vllm configuration."""
        return self.vllm_config

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
        return sum(scheduler.get_num_unfinished_seq_groups()
                   for scheduler in self.scheduler)

    def has_unfinished_requests(self) -> bool:
        """Returns True if there are unfinished requests."""
        return any(scheduler.has_unfinished_seqs()
                   for scheduler in self.scheduler)

    def has_unfinished_requests_for_virtual_engine(
            self, virtual_engine: int) -> bool:
        """
        Returns True if there are unfinished requests for the virtual engine.
        """
        return self.scheduler[virtual_engine].has_unfinished_seqs()

    def reset_mm_cache(self) -> bool:
        """Reset the multi-modal cache."""
        return self.input_preprocessor.mm_registry.reset_processor_cache()

    def reset_prefix_cache(self, device: Optional[Device] = None) -> bool:
        """Reset prefix cache for all devices."""

        success = True
        for scheduler in self.scheduler:
            success = success and scheduler.reset_prefix_cache(device)
        return success

    @staticmethod
    def _process_sequence_group_outputs(
        seq_group: SequenceGroup,
        outputs: List[PoolingSequenceGroupOutput],
    ) -> None:
        seq_group.pooled_data = outputs[0].data

        for seq in seq_group.get_seqs():
            seq.status = SequenceStatus.FINISHED_STOPPED

        return

    def _update_num_computed_tokens_for_multi_step_prefill(
            self, seq_group: SequenceGroup,
            seq_group_meta: SequenceGroupMetadata,
            is_first_step_output: Optional[bool]):
        """
        This function updates num_computed_tokens for prompt sequences
        when Multi-Step is enabled.

        seq_group: SequenceGroup to update the num_computed_tokens for.
        seq_group_meta: Metadata of the given SequenceGroup.
        is_first_step_output: Optional[bool] -
            When available, is_first_step_output indicates if the appended
            output token is the output of the first-step in multi-step.
            A value of None indicates that outputs from all steps in
            in multi-step are submitted in a single burst.
        """

        assert self.scheduler_config.is_multi_step

        if not seq_group_meta.is_prompt:
            # num_computed_token updates for multi-step decodes happen after
            # the tokens are appended to the sequence.
            return

        do_update: bool = False
        if self.scheduler_config.chunked_prefill_enabled:
            # In multi-step + chunked-prefill case, the prompt sequences
            # that are scheduled are fully processed in the first step.
            do_update = is_first_step_output is None or is_first_step_output
        else:
            # Normal multi-step decoding case. In this case prompt-sequences
            # are actually single-stepped. Always update in this case.
            assert seq_group.state.num_steps == 1
            do_update = True

        if do_update:
            seq_group.update_num_computed_tokens(
                seq_group_meta.token_chunk_size)

    def _process_model_outputs(self,
                               ctx: SchedulerContext,
                               request_id: Optional[str] = None) -> None:
        """Apply the model output to the sequences in the scheduled seq groups
        and return responses.

        ctx: The virtual engine context to work on
        request_id: If provided, then only this request is going to be processed
        """

        now = time.time()

        if len(ctx.output_queue) == 0:
            return None

        # Get pending async postprocessor
        if request_id:
            # When we process only one request, no pop is required
            # (since later we will process all of the rest)
            (outputs, seq_group_metadata_list, scheduler_outputs, is_async,
             is_last_step, is_first_step_output, skip) = ctx.output_queue[0]
        else:
            (outputs, seq_group_metadata_list, scheduler_outputs, is_async,
             is_last_step, is_first_step_output,
             skip) = ctx.output_queue.popleft()

        # Sanity check
        assert len(seq_group_metadata_list) == len(
            scheduler_outputs.scheduled_seq_groups)

        has_multiple_outputs: bool = len(outputs) > 1
        outputs_by_sequence_group: List[List[SequenceGroupOutput]]
        if has_multiple_outputs:
            assert self.scheduler_config.is_multi_step or \
                     self.speculative_config
            # Organize outputs by [step][sequence group] instead of
            # [sequence group][step].
            if self.scheduler_config.is_multi_step:
                outputs_by_sequence_group = create_output_by_sequence_group(
                    outputs, len(seq_group_metadata_list))
            elif self.speculative_config:
                # Decodes are multi-steps while prefills are not, outputting at
                # most 1 token. Separate them so that we can trigger chunk
                # processing without having to pad or copy over prompts K times
                # to match decodes structure (costly with prompt_logprobs).
                num_prefills = sum(sg.is_prompt
                                   for sg in seq_group_metadata_list)
                prefills, decodes = outputs[:num_prefills], outputs[
                    num_prefills:]
                outputs_by_sequence_group = create_output_by_sequence_group(
                    decodes,
                    num_seq_groups=len(seq_group_metadata_list) - num_prefills)
                outputs_by_sequence_group = [p.outputs for p in prefills
                                             ] + outputs_by_sequence_group
            # We have outputs for multiple steps submitted in a single burst,
            # so invalidate is_first_step_output.
            is_first_step_output = None
        else:
            outputs_by_sequence_group = outputs

        # Determine the requests we need to operate on
        if request_id:
            indices = []
            for i, seq_group_meta in enumerate(seq_group_metadata_list):
                if seq_group_meta.request_id == request_id:
                    assert i not in skip  # Cannot be called twice
                    indices.append(i)
                    break

            # If the request_id was not found, then it means that
            # this is a new request that has no pending async
            # postprocessor
            if not indices:
                return
        else:
            indices = range(len(seq_group_metadata_list))  # type: ignore

        finished_before: List[int] = []
        finished_now: List[int] = []
        for i in indices:
            if i in skip:
                continue

            seq_group_meta = seq_group_metadata_list[i]
            scheduled_seq_group = scheduler_outputs.scheduled_seq_groups[i]

            seq_group: SequenceGroup = scheduled_seq_group.seq_group

            if seq_group.is_finished():
                finished_before.append(i)
                continue

            output: List[SequenceGroupOutput]
            if has_multiple_outputs:
                output = outputs_by_sequence_group[i]
            else:
                output = [outputs_by_sequence_group[0][i]]

            if not is_async:
                if self.scheduler_config.is_multi_step:
                    # Updates happen only if the sequence is prefill
                    self._update_num_computed_tokens_for_multi_step_prefill(
                        seq_group, seq_group_meta, is_first_step_output)
                else:
                    seq_group.update_num_computed_tokens(
                        seq_group_meta.token_chunk_size or 0)

            if outputs:
                for o in outputs:
                    if (isinstance(o, SamplerOutput)
                            and seq_group.metrics is not None):
                        if seq_group.metrics.model_forward_time is not None:
                            seq_group.metrics.model_forward_time += (
                                o.model_forward_time or 0)
                        else:
                            seq_group.metrics.model_forward_time = (
                                o.model_forward_time)
                        if seq_group.metrics.model_execute_time is not None:
                            seq_group.metrics.model_execute_time += (
                                o.model_execute_time or 0)
                        else:
                            seq_group.metrics.model_execute_time = (
                                o.model_execute_time)

            if self.model_config.runner_type == "pooling":
                self._process_sequence_group_outputs(seq_group, output)
            else:
                self.output_processor.process_prompt_logprob(seq_group, output)
                if seq_group_meta.do_sample:
                    self.output_processor.process_outputs(
                        seq_group, output, is_async)

            if seq_group.is_finished():
                finished_now.append(i)

        # Generate outputs for the requests that finished this iteration
        for i in finished_now:
            scheduled_seq_group = scheduler_outputs.scheduled_seq_groups[i]

            seq_group = scheduled_seq_group.seq_group
            seq_group.maybe_set_first_token_time(now)
            if not seq_group.is_prefill():
                seq_group.set_last_token_time(now)
            request_output = RequestOutputFactory.create(
                seq_group,
                self.seq_id_to_seq_group,
                use_cache=self.use_cached_outputs)
            if request_output:
                ctx.request_outputs.append(request_output)

        # When we process a single request, we skip it for the next time,
        # and invoke the request output callback (if there was final output)
        if request_id:
            assert len(indices) == 1
            skip.append(indices[0])

            if (finished_now
                    and self.process_request_outputs_callback is not None):
                self.process_request_outputs_callback(ctx.request_outputs)
                ctx.request_outputs.clear()
            return

        # Free currently finished requests
        if finished_now:
            for scheduler in self.scheduler:
                scheduler.free_finished_seq_groups()

        # For multi-step without streaming, don't create outputs each iteration
        if not is_last_step and not ctx.multi_step_stream_outputs:
            # Immediately process request outputs here (if callback is given)
            if (finished_now
                    and self.process_request_outputs_callback is not None):
                self.process_request_outputs_callback(ctx.request_outputs)
                ctx.request_outputs.clear()
            return

        # Create the outputs
        for i in indices:
            if i in skip or i in finished_before or i in finished_now:
                continue  # Avoids double processing

            scheduled_seq_group = scheduler_outputs.scheduled_seq_groups[i]

            seq_group = scheduled_seq_group.seq_group
            seq_group.maybe_set_first_token_time(now)
            if not seq_group.is_prefill():
                seq_group.set_last_token_time(now)
            request_output = RequestOutputFactory.create(
                seq_group,
                self.seq_id_to_seq_group,
                use_cache=self.use_cached_outputs)
            if request_output:
                ctx.request_outputs.append(request_output)

        # For multi-step with streaming, create outputs each iteration
        if not is_last_step and ctx.multi_step_stream_outputs:
            # Immediately process request outputs here (if callback is given)
            if self.process_request_outputs_callback is not None:
                self.process_request_outputs_callback(ctx.request_outputs)
                ctx.request_outputs.clear()
            return

        for seq_group in scheduler_outputs.ignored_seq_groups:
            params = seq_group.sampling_params
            if params is not None and params.output_kind == (
                    RequestOutputKind.DELTA) and not seq_group.is_finished():
                continue

            request_output = RequestOutputFactory.create(
                seq_group,
                self.seq_id_to_seq_group,
                use_cache=self.use_cached_outputs,
            )
            if request_output:
                ctx.request_outputs.append(request_output)

        # Immediately process request outputs here (if callback is given)
        if (ctx.request_outputs
                and self.process_request_outputs_callback is not None):
            self.process_request_outputs_callback(ctx.request_outputs)
            ctx.request_outputs.clear()

        # For async case, we need to record the stats here.
        # For non-async case, the stats are done in the
        # LLMEngine/AsyncLLMEngine directly
        if is_async:
            # Log stats.
            self.do_log_stats(scheduler_outputs, outputs, finished_before,
                              skip)

            # Tracing
            self.do_tracing(scheduler_outputs, finished_before)

        return None

    def _advance_to_next_step(
            self, output: SamplerOutput,
            seq_group_metadata_list: List[SequenceGroupMetadata],
            scheduled_seq_groups: List[ScheduledSequenceGroup]) -> None:
        """Given model output from a single run, append the tokens to the
        sequences. This is normally done inside output processor, but it is
        required if the worker is to perform async forward pass to next step.
        """
        for seq_group_metadata, sequence_group_outputs, scheduled_seq_group in \
            zip(seq_group_metadata_list, output, scheduled_seq_groups):
            seq_group = scheduled_seq_group.seq_group

            if seq_group.is_finished():
                continue

            if self.scheduler_config.is_multi_step:
                # Updates happen only if the sequence is prefill
                self._update_num_computed_tokens_for_multi_step_prefill(
                    seq_group, seq_group_metadata,
                    seq_group.state.num_steps == 1)
            else:
                token_chunk_size = (seq_group_metadata.token_chunk_size
                                    if seq_group_metadata.token_chunk_size
                                    is not None else 0)
                seq_group.update_num_computed_tokens(token_chunk_size)

            if seq_group_metadata.do_sample:
                assert len(sequence_group_outputs.samples) == 1, (
                    "Async output processor expects a single sample"
                    " (i.e sampling_params.n == 1)")
                sample = sequence_group_outputs.samples[0]

                assert len(seq_group.seqs) == 1
                seq = seq_group.seqs[0]

                if self.scheduler_config.is_multi_step:
                    is_prefill_append = seq.data.get_num_uncomputed_tokens(
                    ) == 0
                    seq.append_token_id(sample.output_token, sample.logprobs,
                                        sample.output_embed)
                    if not is_prefill_append:
                        seq_group.update_num_computed_tokens(1)
                else:
                    seq.append_token_id(sample.output_token, sample.logprobs,
                                        sample.output_embed)

    def step(self) -> List[Union[RequestOutput, PoolingRequestOutput]]:
        """Performs one decoding iteration and returns newly generated results.

        <figure markdown="span">
        ![Overview of the step function](https://i.imgur.com/sv2HssD.png)
        <figcaption>Overview of the step function</figcaption>
        </figure>

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
        ```
        # Please see the example/ folder for more detailed examples.

        # initialize engine and request arguments
        engine = LLMEngine.from_engine_args(engine_args)
        example_inputs = [(0, "What is LLM?",
        SamplingParams(temperature=0.0))]
    
        # Start the engine with an event loop
        while True:
            if example_inputs:
                req_id, prompt, sampling_params = example_inputs.pop(0)
                engine.add_request(str(req_id),prompt,sampling_params)

            # continue the request processing
            request_outputs = engine.step()
            for request_output in request_outputs:
                if request_output.finished:
                    # return or show the request output

            if not (engine.has_unfinished_requests() or example_inputs):
                break
        ```
        """
        if self.parallel_config.pipeline_parallel_size > 1:
            raise NotImplementedError(
                "Pipeline parallelism is only supported through AsyncLLMEngine "
                "as performance will be severely degraded otherwise.")

        # For llm_engine, there is no pipeline parallel support, so the engine
        # used is always 0.
        virtual_engine = 0

        # These are cached outputs from previous iterations. None if on first
        # iteration
        cached_outputs = self.cached_scheduler_outputs[virtual_engine]
        seq_group_metadata_list = cached_outputs.seq_group_metadata_list
        scheduler_outputs = cached_outputs.scheduler_outputs
        allow_async_output_proc = cached_outputs.allow_async_output_proc

        ctx = self.scheduler_contexts[virtual_engine]

        # Clear outputs for each new scheduler iteration
        ctx.request_outputs.clear()

        # Skip the scheduler if there are any remaining steps in the seq groups.
        # This ensures that the scheduler is only called again when the current
        # batch has completed.
        # The scheduler is also skipped if a single request caused the last
        # engine step to fail, and the previous schedule needs to be rerun.
        if not self._has_remaining_steps(
                seq_group_metadata_list
        ) and not self._skip_scheduling_next_step:
            # Schedule iteration
            (seq_group_metadata_list, scheduler_outputs,
             allow_async_output_proc
             ) = self.scheduler[virtual_engine].schedule()

            ctx.seq_group_metadata_list = seq_group_metadata_list
            ctx.scheduler_outputs = scheduler_outputs

            finished_requests_ids = self.scheduler[
                virtual_engine].get_and_reset_finished_requests_ids()
            # When n>1, elements in self.seq_id_to_seq_group should be deleted
            # here, otherwise memory leaks.
            for finished_request_id in finished_requests_ids:
                if finished_request_id in self.seq_id_to_seq_group:
                    del self.seq_id_to_seq_group[finished_request_id]

            # Maybe switch from async mode to sync mode
            if not allow_async_output_proc and len(ctx.output_queue) > 0:
                self._process_model_outputs(ctx=ctx)

            if (self.scheduler_config.is_multi_step
                    and scheduler_outputs.num_lookahead_slots > 0):
                # cache the scheduler outputs for the next iteration if we have
                # lookahead slots
                self._cache_scheduler_outputs_for_multi_step(
                    virtual_engine, seq_group_metadata_list, scheduler_outputs,
                    allow_async_output_proc)
        else:
            finished_requests_ids = list()

        assert seq_group_metadata_list is not None
        assert scheduler_outputs is not None

        if not scheduler_outputs.is_empty():

            # Check if we have a cached last_output from the previous iteration.
            # For supporting PP this is probably the best way to pass the
            # sampled_token_ids, as a separate broadcast over all the PP stages
            # will cause one virtual engine's microbatch to block the pipeline.
            last_sampled_token_ids = \
                self._get_last_sampled_token_ids(virtual_engine)

            execute_model_req = ExecuteModelRequest(
                seq_group_metadata_list=seq_group_metadata_list,
                blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
                blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
                blocks_to_copy=scheduler_outputs.blocks_to_copy,
                num_lookahead_slots=scheduler_outputs.num_lookahead_slots,
                running_queue_size=scheduler_outputs.running_queue_size,
                finished_requests_ids=finished_requests_ids,
                # We use ExecuteModelRequest to pass the last sampled_token_ids
                # to each of the non-last PP stages for in-place prepare_input.
                last_sampled_token_ids=last_sampled_token_ids)

            if allow_async_output_proc:
                execute_model_req.async_callback = self.async_callbacks[
                    virtual_engine]

            try:
                outputs = self.model_executor.execute_model(
                    execute_model_req=execute_model_req)
                self._skip_scheduling_next_step = False
            except InputProcessingError as e:
                # The input for this request cannot be processed, so we must
                # abort it. If there are remaining requests in the batch that
                # have been scheduled, they will be retried on the next step.
                invalid_request_id = e.request_id
                self._abort_and_cache_schedule(
                    request_id=invalid_request_id,
                    virtual_engine=virtual_engine,
                    seq_group_metadata_list=seq_group_metadata_list,
                    scheduler_outputs=scheduler_outputs,
                    allow_async_output_proc=allow_async_output_proc)
                # Raise so the caller is notified that this request failed
                raise

            # We need to do this here so that last step's sampled_token_ids can
            # be passed to the next iteration for PP.
            if self.scheduler_config.is_multi_step:
                self._update_cached_scheduler_output(virtual_engine, outputs)
        else:
            # Nothing scheduled => If there is pending async postprocessor,
            # then finish it here.
            if len(ctx.output_queue) > 0:
                self._process_model_outputs(ctx=ctx)
            # No outputs in this case
            outputs = []

        # Finish the current step for all the sequence groups.
        if self.scheduler_config.is_multi_step:
            for seq_group in seq_group_metadata_list:
                seq_group.finish_step()

        if not self._has_remaining_steps(seq_group_metadata_list):
            # clear the cache if we have finished all the steps.
            if self.scheduler_config.is_multi_step:
                self.cached_scheduler_outputs[0] = SchedulerOutputState()

            # is_first_step_output is True only when the num_steps of all
            # the sequences are 1. When the num_steps > 1,
            # multi_step_model_runner does the first-step output append.
            is_first_step_output: bool = False if not seq_group_metadata_list \
                else seq_group_metadata_list[0].state.num_steps == 1

            # Add results to the output_queue
            ctx.append_output(outputs=outputs,
                              seq_group_metadata_list=seq_group_metadata_list,
                              scheduler_outputs=scheduler_outputs,
                              is_async=allow_async_output_proc,
                              is_last_step=True,
                              is_first_step_output=is_first_step_output)

            if outputs and allow_async_output_proc:
                assert len(outputs) == 1, (
                    "Async postprocessor expects only a single output set")

                self._advance_to_next_step(
                    outputs[0], seq_group_metadata_list,
                    scheduler_outputs.scheduled_seq_groups)

            # Check if need to run the usual non-async path
            if not allow_async_output_proc:
                self._process_model_outputs(ctx=ctx)

                # Log stats.
                self.do_log_stats(scheduler_outputs, outputs)

                # Tracing
                self.do_tracing(scheduler_outputs)
        else:
            # Multi-step case
            return ctx.request_outputs

        if not self.has_unfinished_requests():
            # Drain async postprocessor (if exists)
            if len(ctx.output_queue) > 0:
                self._process_model_outputs(ctx=ctx)
            assert len(ctx.output_queue) == 0

            # Stop the execute model loop in parallel workers until there are
            # more requests to process. This avoids waiting indefinitely in
            # torch.distributed ops which may otherwise timeout, and unblocks
            # the RPC thread in the workers so that they can process any other
            # queued control plane messages, such as add/remove lora adapters.
            logger.debug("Stopping remote worker execution loop.")
            self.model_executor.stop_remote_worker_execution_loop()

        return ctx.request_outputs

    def _abort_and_cache_schedule(
            self, request_id: str, virtual_engine: int,
            seq_group_metadata_list: List[SequenceGroupMetadata],
            scheduler_outputs: SchedulerOutputs,
            allow_async_output_proc: bool) -> None:
        """Aborts a single request, and caches the scheduler outputs minus that
        request. This allows the next step to continue processing the remaining
        requests without having to re-run the scheduler."""

        # Abort the request and remove its sequence group from the current
        # schedule
        self.abort_request(request_id)
        for i, metadata in enumerate(seq_group_metadata_list):
            if metadata.request_id == request_id:
                del seq_group_metadata_list[i]
                break
        for i, group in enumerate(scheduler_outputs.scheduled_seq_groups):
            if group.seq_group.request_id == request_id:
                del scheduler_outputs.scheduled_seq_groups[i]
                break

        # If there are still other sequence groups left in the schedule, cache
        # them and flag the engine to reuse the schedule.
        if len(seq_group_metadata_list) > 0:
            self._skip_scheduling_next_step = True
            # Reuse multi-step caching logic
            self._cache_scheduler_outputs_for_multi_step(
                virtual_engine=virtual_engine,
                scheduler_outputs=scheduler_outputs,
                seq_group_metadata_list=seq_group_metadata_list,
                allow_async_output_proc=allow_async_output_proc)

    def _has_remaining_steps(
        self, seq_group_metadata_list: Optional[List[SequenceGroupMetadata]]
    ) -> bool:
        if (not self.scheduler_config.is_multi_step
                or not seq_group_metadata_list):
            return False

        # TODO(will) this is a sanity check for nowto make sure that all the
        # seqs are on the same steps. Eventually we will want to do some sort of
        # dynamic scheduling when doing multi-step decoding.
        ref_remaining_steps = seq_group_metadata_list[0].state.remaining_steps
        if any([
                seq_group.state.remaining_steps != ref_remaining_steps
                for seq_group in seq_group_metadata_list[1:]
        ]):
            raise AssertionError("All running sequence groups should "
                                 "have the same remaining steps.")

        return ref_remaining_steps > 0

    def _cache_scheduler_outputs_for_multi_step(
            self, virtual_engine: int,
            seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
            scheduler_outputs: SchedulerOutputs,
            allow_async_output_proc: bool) -> None:
        co = self.cached_scheduler_outputs[virtual_engine]

        co.seq_group_metadata_list = seq_group_metadata_list
        co.scheduler_outputs = scheduler_outputs
        co.allow_async_output_proc = allow_async_output_proc
        co.last_output = None

    def _update_cached_scheduler_output(
            self, virtual_engine: int,
            output: List[Optional[SamplerOutput]]) -> None:
        if (self.parallel_config.pipeline_parallel_size > 1 and len(output) > 0
                and output[0] is not None):
            last_output = output[-1]
            assert last_output is not None
            assert last_output.sampled_token_ids_cpu is not None
            assert last_output.sampled_token_ids is None
            assert last_output.sampled_token_probs is None
            self.cached_scheduler_outputs[
                virtual_engine].last_output = last_output

    def _get_last_sampled_token_ids(
            self, virtual_engine: int) -> Optional[torch.Tensor]:
        cached_last_output = self.cached_scheduler_outputs[
            virtual_engine].last_output
        if (self.scheduler_config.is_multi_step
                and self.parallel_config.pipeline_parallel_size > 1
                and cached_last_output is not None
                and cached_last_output.sampled_token_ids_cpu is not None):
            return cached_last_output.sampled_token_ids_cpu
        return None

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

    def do_log_stats(self,
                     scheduler_outputs: Optional[SchedulerOutputs] = None,
                     model_output: Optional[List[SamplerOutput]] = None,
                     finished_before: Optional[List[int]] = None,
                     skip: Optional[List[int]] = None) -> None:
        """Forced log when no requests active."""
        if self.log_stats:
            stats = self._get_stats(scheduler_outputs, model_output,
                                    finished_before, skip)
            for logger in self.stat_loggers.values():
                logger.log(stats)

    def _get_stats(self,
                   scheduler_outputs: Optional[SchedulerOutputs],
                   model_output: Optional[List[SamplerOutput]] = None,
                   finished_before: Optional[List[int]] = None,
                   skip: Optional[List[int]] = None) -> Stats:
        """Get Stats to be Logged to Prometheus.

        Args:
            scheduler_outputs: Optional, used to populate metrics related to
                the scheduled batch,
            model_output: Optional, used to emit speculative decoding metrics
                which are created by the workers.
            finished_before: Optional, indices of sequences that were finished
                before. These sequences will be ignored.
            skip: Optional, indices of sequences that were preempted. These
                sequences will be ignored.
        """
        now = time.time()

        # System State
        #   Scheduler State
        num_running_sys = sum(
            len(scheduler.running) for scheduler in self.scheduler)
        num_swapped_sys = sum(
            len(scheduler.swapped) for scheduler in self.scheduler)
        num_waiting_sys = sum(
            len(scheduler.waiting) for scheduler in self.scheduler)

        # KV Cache Usage in %
        num_total_gpu = self.cache_config.num_gpu_blocks
        gpu_cache_usage_sys = 0.
        if num_total_gpu:  # Guard against both None and 0
            num_free_gpu = sum(
                scheduler.block_manager.get_num_free_gpu_blocks()
                for scheduler in self.scheduler)
            gpu_cache_usage_sys = 1.0 - (num_free_gpu / num_total_gpu)

        num_total_cpu = self.cache_config.num_cpu_blocks
        cpu_cache_usage_sys = 0.
        if num_total_cpu:  # Guard against both None and 0
            num_free_cpu = sum(
                scheduler.block_manager.get_num_free_cpu_blocks()
                for scheduler in self.scheduler)
            cpu_cache_usage_sys = 1.0 - (num_free_cpu / num_total_cpu)

        # Prefix Cache Hit Rate. Note that we always use
        # the cache hit rate of the first virtual engine.
        cpu_prefix_cache_hit_rate = self.scheduler[
            0].get_prefix_cache_hit_rate(Device.CPU)
        gpu_prefix_cache_hit_rate = self.scheduler[
            0].get_prefix_cache_hit_rate(Device.GPU)

        # Exchange the uasge and cache hit stats between gpu and cpu when
        # running on cpu because the cpu_worker.py intentionally reports the
        # number of cpu blocks as gpu blocks in favor of cache management.
        if self.device_config.device_type == "cpu":
            num_total_gpu, num_total_cpu = num_total_cpu, num_total_gpu
            gpu_cache_usage_sys, cpu_cache_usage_sys = (
                cpu_cache_usage_sys,
                gpu_cache_usage_sys,
            )
            gpu_prefix_cache_hit_rate, cpu_prefix_cache_hit_rate = (
                cpu_prefix_cache_hit_rate,
                gpu_prefix_cache_hit_rate,
            )

        # Iteration stats
        num_prompt_tokens_iter = 0
        num_generation_tokens_iter = 0
        num_tokens_iter = 0
        time_to_first_tokens_iter: List[float] = []
        time_per_output_tokens_iter: List[float] = []
        num_preemption_iter = (0 if scheduler_outputs is None else
                               scheduler_outputs.preempted)

        # Request stats
        #   Latency
        time_e2e_requests: List[float] = []
        time_queue_requests: List[float] = []
        time_inference_requests: List[float] = []
        time_prefill_requests: List[float] = []
        time_decode_requests: List[float] = []
        #   Metadata
        num_prompt_tokens_requests: List[int] = []
        num_generation_tokens_requests: List[int] = []
        n_requests: List[int] = []
        max_num_generation_tokens_requests: List[int] = []
        max_tokens_requests: List[int] = []
        finished_reason_requests: List[str] = []

        # LoRA requests
        running_lora_adapters = dict(
            collectionsCounter([
                running_request.lora_request.lora_name
                for scheduler in self.scheduler
                for running_request in scheduler.running
                if running_request.lora_request
            ]))
        waiting_lora_adapters = dict(
            collectionsCounter([
                waiting_request.lora_request.lora_name
                for scheduler in self.scheduler
                for waiting_request in scheduler.waiting
                if waiting_request.lora_request
            ]))
        max_lora_stat = "0"
        if self.lora_config:
            max_lora_stat = str(self.lora_config.max_loras)

        # NOTE: This loop assumes prefill seq_groups are before
        # decode seq_groups in scheduled_seq_groups.
        if scheduler_outputs is not None:
            # For async postprocessor, already finished sequences need to be
            # not counted (to avoid double counting)
            actual_num_batched_tokens = scheduler_outputs.num_batched_tokens  # type: ignore

            num_generation_tokens_from_prefill_groups = 0
            # NOTE: if scheduler_outputs.num_prefill_groups > 0 and
            # the len of scheduler_outputs.scheduled_seq_groups is !=
            # scheduler_outputs.num_prefill_groups, this means that
            # chunked prefills have been detected.

            for idx, scheduled_seq_group in enumerate(
                    scheduler_outputs.scheduled_seq_groups):
                # Skip double logging when using async output proc
                if finished_before and idx in finished_before:
                    actual_num_batched_tokens -= 1
                    continue

                # Currently, skip == preempted sequences, so we need to skip
                # their log stats
                if skip and idx in skip:
                    continue

                group_was_prefill = idx < scheduler_outputs.num_prefill_groups
                seq_group = scheduled_seq_group.seq_group

                # NOTE: a seq_group that completed all of its prefill tokens
                # in the last iteration will have seq_group.is_prefill() = False
                # with group_was_prefill = True
                if group_was_prefill:
                    # Number of prompt tokens.
                    num_prompt_tokens_iter += (
                        scheduled_seq_group.token_chunk_size)

                    # If the seq_group just finished the prefill state
                    # get TTFT.
                    if not seq_group.is_prefill():
                        latency = seq_group.get_last_token_latency()
                        time_to_first_tokens_iter.append(latency)

                        # One generation token per finished prefill.
                        num_generation_tokens_from_prefill_groups += (
                            seq_group.num_seqs())
                else:
                    # TPOTs.
                    latency = seq_group.get_last_token_latency()
                    time_per_output_tokens_iter.append(latency)
                    if seq_group.state.current_step == 0:
                        # For async_output_proc, the do_log_stats()
                        # is called following init_multi_step(), which
                        # sets the current_step to zero.
                        actual_num_batched_tokens +=\
                            seq_group.state.num_steps - 1
                    else:
                        actual_num_batched_tokens +=\
                            seq_group.state.current_step - 1

                # Because of chunked prefill, we can have a single sequence
                # group that does multiple prompt_runs. To prevent logging
                # the same metadata more than once per request, we standardize
                # on logging request level information for finished requests,
                # which can only happen once.
                if seq_group.is_finished():
                    # Latency timings
                    time_e2e_requests.append(now -
                                             seq_group.metrics.arrival_time)
                    if (seq_group.metrics.first_scheduled_time is not None and
                            seq_group.metrics.first_token_time is not None):
                        time_queue_requests.append(
                            seq_group.metrics.first_scheduled_time -
                            seq_group.metrics.arrival_time)
                        time_prefill_requests.append(
                            seq_group.metrics.first_token_time -
                            seq_group.metrics.first_scheduled_time)
                        time_decode_requests.append(
                            now - seq_group.metrics.first_token_time)
                        time_inference_requests.append(
                            now - seq_group.metrics.first_scheduled_time)
                    # Metadata
                    num_prompt_tokens_requests.append(
                        len(seq_group.prompt_token_ids))
                    num_generation_tokens_requests.extend([
                        seq.get_output_len()
                        for seq in seq_group.get_finished_seqs()
                    ])
                    max_num_generation_tokens_requests.append(
                        max(seq.get_output_len()
                            for seq in seq_group.get_seqs()))
                    if seq_group.sampling_params is not None:
                        n_requests.append(seq_group.sampling_params.n)
                        max_tokens_requests.append(
                            seq_group.sampling_params.max_tokens)
                    finished_reason_requests.extend([
                        SequenceStatus.get_finished_reason(seq.status)
                        for seq in seq_group.get_finished_seqs()
                    ])

            # Number of generation tokens.
            #   num_batched_tokens equals the number of prompt_tokens plus the
            #   number of decode_tokens in a single iteration. So,
            #   num_generation_tokens = num_batched_tokens - num_prompt_tokens
            #   + num_generation_tokens_from_prefill_groups (since we generate
            #   one token on prefills on iters where the prefill finishes).
            num_generation_tokens_iter = (
                actual_num_batched_tokens - num_prompt_tokens_iter +
                num_generation_tokens_from_prefill_groups)
            num_tokens_iter = (num_generation_tokens_iter +
                               num_prompt_tokens_iter)
        # Spec decode, if enabled, emits specialized metrics from the worker in
        # sampler output.
        if model_output and isinstance(model_output[0], SamplerOutput) and (
                model_output[0].spec_decode_worker_metrics is not None):
            spec_decode_metrics = model_output[0].spec_decode_worker_metrics
        else:
            spec_decode_metrics = None

        return Stats(
            now=now,
            # System stats
            #   Scheduler State
            num_running_sys=num_running_sys,
            num_swapped_sys=num_swapped_sys,
            num_waiting_sys=num_waiting_sys,
            #   KV Cache Usage in %
            gpu_cache_usage_sys=gpu_cache_usage_sys,
            cpu_cache_usage_sys=cpu_cache_usage_sys,
            #   Prefix Cache Hit Rate
            cpu_prefix_cache_hit_rate=cpu_prefix_cache_hit_rate,
            gpu_prefix_cache_hit_rate=gpu_prefix_cache_hit_rate,

            # Iteration stats
            num_prompt_tokens_iter=num_prompt_tokens_iter,
            num_generation_tokens_iter=num_generation_tokens_iter,
            num_tokens_iter=num_tokens_iter,
            time_to_first_tokens_iter=time_to_first_tokens_iter,
            time_per_output_tokens_iter=time_per_output_tokens_iter,
            spec_decode_metrics=spec_decode_metrics,
            num_preemption_iter=num_preemption_iter,

            # Request stats
            #   Latency
            time_e2e_requests=time_e2e_requests,
            time_queue_requests=time_queue_requests,
            time_inference_requests=time_inference_requests,
            time_prefill_requests=time_prefill_requests,
            time_decode_requests=time_decode_requests,
            #   Metadata
            num_prompt_tokens_requests=num_prompt_tokens_requests,
            num_generation_tokens_requests=num_generation_tokens_requests,
            max_num_generation_tokens_requests=
            max_num_generation_tokens_requests,
            n_requests=n_requests,
            max_tokens_requests=max_tokens_requests,
            finished_reason_requests=finished_reason_requests,
            max_lora=str(max_lora_stat),
            waiting_lora_adapters=list(waiting_lora_adapters.keys()),
            running_lora_adapters=list(running_lora_adapters.keys()))

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_executor.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_executor.remove_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.model_executor.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_executor.pin_lora(lora_id)

    def add_prompt_adapter(
            self, prompt_adapter_request: PromptAdapterRequest) -> bool:
        return self.model_executor.add_prompt_adapter(prompt_adapter_request)

    def remove_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        return self.model_executor.remove_prompt_adapter(prompt_adapter_id)

    def list_prompt_adapters(self) -> List[int]:
        return self.model_executor.list_prompt_adapters()

    def start_profile(self) -> None:
        self.model_executor.start_profile()

    def stop_profile(self) -> None:
        self.model_executor.stop_profile()

    def sleep(self, level: int = 1) -> None:
        assert self.vllm_config.model_config.enable_sleep_mode, (
            "Sleep mode is not enabled in the model config")
        self.model_executor.sleep(level=level)

    def wake_up(self, tags: Optional[list[str]] = None) -> None:
        assert self.vllm_config.model_config.enable_sleep_mode, (
            "Sleep mode is not enabled in the model config")
        self.model_executor.wake_up(tags)

    def is_sleeping(self) -> bool:
        return self.model_executor.is_sleeping

    def check_health(self) -> None:
        self.model_executor.check_health()

    def is_tracing_enabled(self) -> bool:
        return self.tracer is not None

    def do_tracing(self,
                   scheduler_outputs: SchedulerOutputs,
                   finished_before: Optional[List[int]] = None) -> None:
        if self.tracer is None:
            return

        for idx, scheduled_seq_group in enumerate(
                scheduler_outputs.scheduled_seq_groups):
            # Skip double tracing when using async output proc
            if finished_before and idx in finished_before:
                continue

            seq_group = scheduled_seq_group.seq_group
            if seq_group.is_finished():
                self.create_trace_span(seq_group)

    def create_trace_span(self, seq_group: SequenceGroup) -> None:
        if self.tracer is None or seq_group.sampling_params is None:
            return
        arrival_time_nano_seconds = int(seq_group.metrics.arrival_time * 1e9)

        trace_context = extract_trace_context(seq_group.trace_headers)

        with self.tracer.start_as_current_span(
                "llm_request",
                kind=SpanKind.SERVER,
                context=trace_context,
                start_time=arrival_time_nano_seconds) as seq_span:
            metrics = seq_group.metrics
            ttft = metrics.first_token_time - metrics.arrival_time
            e2e_time = metrics.finished_time - metrics.arrival_time
            seq_span.set_attribute(SpanAttributes.GEN_AI_RESPONSE_MODEL,
                                   self.model_config.model)
            seq_span.set_attribute(SpanAttributes.GEN_AI_REQUEST_ID,
                                   seq_group.request_id)
            seq_span.set_attribute(SpanAttributes.GEN_AI_REQUEST_TEMPERATURE,
                                   seq_group.sampling_params.temperature)
            seq_span.set_attribute(SpanAttributes.GEN_AI_REQUEST_TOP_P,
                                   seq_group.sampling_params.top_p)
            seq_span.set_attribute(SpanAttributes.GEN_AI_REQUEST_MAX_TOKENS,
                                   seq_group.sampling_params.max_tokens)
            seq_span.set_attribute(SpanAttributes.GEN_AI_REQUEST_N,
                                   seq_group.sampling_params.n)
            seq_span.set_attribute(SpanAttributes.GEN_AI_USAGE_NUM_SEQUENCES,
                                   seq_group.num_seqs())
            seq_span.set_attribute(SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS,
                                   len(seq_group.prompt_token_ids))
            seq_span.set_attribute(
                SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS,
                sum([
                    seq.get_output_len()
                    for seq in seq_group.get_finished_seqs()
                ]))
            seq_span.set_attribute(SpanAttributes.GEN_AI_LATENCY_TIME_IN_QUEUE,
                                   metrics.time_in_queue)
            seq_span.set_attribute(
                SpanAttributes.GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN, ttft)
            seq_span.set_attribute(SpanAttributes.GEN_AI_LATENCY_E2E, e2e_time)
            if metrics.scheduler_time is not None:
                seq_span.set_attribute(
                    SpanAttributes.GEN_AI_LATENCY_TIME_IN_SCHEDULER,
                    metrics.scheduler_time)
            if metrics.model_forward_time is not None:
                seq_span.set_attribute(
                    SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_FORWARD,
                    metrics.model_forward_time / 1000.0)
            if metrics.model_execute_time is not None:
                seq_span.set_attribute(
                    SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_EXECUTE,
                    metrics.model_execute_time)

    def _validate_model_inputs(self, inputs: ProcessorInputs,
                               lora_request: Optional[LoRARequest]):
        encoder_inputs, decoder_inputs = split_enc_dec_inputs(inputs)

        if encoder_inputs is not None:
            self._validate_model_input(encoder_inputs,
                                       lora_request,
                                       prompt_type="encoder")

        self._validate_model_input(decoder_inputs,
                                   lora_request,
                                   prompt_type="decoder")

    def _validate_model_input(
        self,
        prompt_inputs: SingletonInputs,
        lora_request: Optional[LoRARequest],
        *,
        prompt_type: Literal["encoder", "decoder"],
    ):
        model_config = self.model_config
        tokenizer = (None if self.tokenizer is None else
                     self.tokenizer.get_lora_tokenizer(lora_request))

        prompt_ids = prompt_inputs.get("prompt_token_ids", [])
        if not prompt_ids:
            if prompt_type == "encoder" and model_config.is_multimodal_model:
                pass  # Mllama may have empty encoder inputs for text-only data
            elif prompt_inputs["type"] == "embeds":
                pass
            else:
                raise ValueError(f"The {prompt_type} prompt cannot be empty")

        if tokenizer is not None:
            max_input_id = max(prompt_ids, default=0)
            if max_input_id > tokenizer.max_token_id:
                raise ValueError(
                    f"Token id {max_input_id} is out of vocabulary")

        max_prompt_len = self.model_config.max_model_len
        if len(prompt_ids) > max_prompt_len:
            if prompt_type == "encoder" and model_config.is_multimodal_model:
                mm_registry = self.input_preprocessor.mm_registry
                mm_processor = mm_registry.create_processor(
                    model_config,
                    tokenizer=tokenizer or object(),  # Dummy if no tokenizer
                )
                assert isinstance(mm_processor, EncDecMultiModalProcessor)

                if mm_processor.pad_dummy_encoder_prompt:
                    return  # Skip encoder length check for Whisper

            if model_config.is_multimodal_model:
                suggestion = (
                    "Make sure that `max_model_len` is no smaller than the "
                    "number of text tokens plus multimodal tokens. For image "
                    "inputs, the number of image tokens depends on the number "
                    "of images, and possibly their aspect ratios as well.")
            else:
                suggestion = (
                    "Make sure that `max_model_len` is no smaller than the "
                    "number of text tokens.")

            raise ValueError(
                f"The {prompt_type} prompt (length {len(prompt_ids)}) is "
                f"longer than the maximum model length of {max_prompt_len}. "
                f"{suggestion}")

            # TODO: Find out how many placeholder tokens are there so we can
            # check that chunked prefill does not truncate them
            # max_batch_len = self.scheduler_config.max_num_batched_tokens

    def _build_logits_processors(
            self, sampling_params: SamplingParams,
            lora_request: Optional[LoRARequest]) -> SamplingParams:
        """Constructs logits processors based on the guided_decoding,
        logits_bias, and allowed_token_ids fields in sampling_params. Deletes
        those fields and adds the constructed logits processors to the
        logits_processors field. Returns the modified sampling params."""

        logits_processors = []

        if sampling_params.guided_decoding is not None:
            # Defensively copy sampling params since guided decoding logits
            # processors can have different state for each request
            sampling_params = copy.copy(sampling_params)
            guided_decoding = sampling_params.guided_decoding

            logger.debug(
                "Building guided decoding logits processor in "
                "LLMEngine. Params: %s", guided_decoding)

            tokenizer = self.get_tokenizer(lora_request=lora_request)
            guided_decoding.backend = guided_decoding.backend or \
                self.decoding_config.backend

            if self.decoding_config.reasoning_backend:
                logger.debug("Building with reasoning backend %s",
                             self.decoding_config.reasoning_backend)

            processor = get_local_guided_decoding_logits_processor(
                guided_params=guided_decoding,
                tokenizer=tokenizer,
                model_config=self.model_config,
                reasoning_backend=self.decoding_config.reasoning_backend,
            )
            if processor:
                logits_processors.append(processor)

            # Unset so this doesn't get passed down to the model
            sampling_params.guided_decoding = None

        if (sampling_params.logit_bias or sampling_params.allowed_token_ids):
            tokenizer = self.get_tokenizer(lora_request=lora_request)

            processors = get_openai_logits_processors(
                logit_bias=sampling_params.logit_bias,
                allowed_token_ids=sampling_params.allowed_token_ids,
                tokenizer=tokenizer)
            logits_processors.extend(processors)

            # Unset so these don't get passed down to the model
            sampling_params.logit_bias = None
            sampling_params.allowed_token_ids = None

        if len(sampling_params.bad_words) > 0:
            tokenizer = self.get_tokenizer(lora_request)
            processors = get_bad_words_logits_processors(
                bad_words=sampling_params.bad_words, tokenizer=tokenizer)
            logits_processors.extend(processors)

        if logits_processors:
            if sampling_params.logits_processors is None:
                sampling_params.logits_processors = logits_processors
            else:
                sampling_params.logits_processors.extend(logits_processors)

        return sampling_params

    def collective_rpc(self,
                       method: Union[str, Callable[..., _R]],
                       timeout: Optional[float] = None,
                       args: tuple = (),
                       kwargs: Optional[dict[str, Any]] = None) -> list[_R]:
        return self.model_executor.collective_rpc(method, timeout, args,
                                                  kwargs)


if envs.is_set("VLLM_USE_V1") and envs.VLLM_USE_V1:
    from vllm.v1.engine.llm_engine import LLMEngine as V1LLMEngine
    LLMEngine = V1LLMEngine  # type: ignore
