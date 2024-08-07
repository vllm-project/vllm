import time
from contextlib import contextmanager
from typing import (TYPE_CHECKING, Any, ClassVar, Dict, Iterable, List,
                    Mapping, Optional)
from typing import Sequence as GenericSequence
from typing import Set, Tuple, Type, TypeVar, Union

import vllm.envs as envs
from vllm.config import (CacheConfig, DecodingConfig, DeviceConfig,
                         EngineConfig, LoadConfig, LoRAConfig, ModelConfig,
                         MultiModalConfig, ObservabilityConfig, ParallelConfig,
                         PromptAdapterConfig, SchedulerConfig,
                         SpeculativeConfig)
from vllm.core.scheduler import (ScheduledSequenceGroup, Scheduler,
                                 SchedulerOutputs)
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.metrics import (LoggingStatLogger, PrometheusStatLogger,
                                 StatLoggerBase, Stats)
from vllm.engine.output_processor.interfaces import (
    SequenceGroupOutputProcessor)
from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.engine.output_processor.util import create_output_by_sequence_group
from vllm.executor.executor_base import ExecutorBase
from vllm.executor.ray_utils import initialize_ray_cluster
from vllm.inputs import (INPUT_REGISTRY, LLMInputs, PromptInputs,
                         get_prompt_type)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import (EmbeddingRequestOutput, RequestOutput,
                          RequestOutputFactory)
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.sequence import (EmbeddingSequenceGroupOutput, ExecuteModelRequest,
                           PoolerOutput, SamplerOutput, Sequence,
                           SequenceGroup, SequenceGroupMetadata,
                           SequenceStatus)
from vllm.tracing import (SpanAttributes, SpanKind, extract_trace_context,
                          init_tracer)
from vllm.transformers_utils.config import try_get_generation_config
from vllm.transformers_utils.detokenizer import Detokenizer
from vllm.transformers_utils.tokenizer_group import (
    AnyTokenizer, BaseTokenizerGroup, init_tokenizer_from_configs)
from vllm.usage.usage_lib import (UsageContext, is_usage_stats_enabled,
                                  usage_message)
from vllm.utils import (Counter, is_embedding_model_config,
                        is_encoder_decoder_model_config)
from vllm.version import __version__ as VLLM_VERSION

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
        multimodal_config (Optional): The configuration related to multimodal 
            models.
        speculative_config (Optional): The configuration related to speculative
            decoding.
        executor_class: The model executor class for managing distributed
            execution.
        prompt_adapter_config (Optional): The configuration related to serving 
            prompt adapters.
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

        return output

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

    tokenizer: Optional[BaseTokenizerGroup]

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        load_config: LoadConfig,
        lora_config: Optional[LoRAConfig],
        multimodal_config: Optional[MultiModalConfig],
        speculative_config: Optional[SpeculativeConfig],
        decoding_config: Optional[DecodingConfig],
        observability_config: Optional[ObservabilityConfig],
        prompt_adapter_config: Optional[PromptAdapterConfig],
        executor_class: Type[ExecutorBase],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
    ) -> None:
        logger.info(
            "Initializing an LLM engine (v%s) with config: "
            "model=%r, speculative_config=%r, tokenizer=%r, "
            "skip_tokenizer_init=%s, tokenizer_mode=%s, revision=%s, "
            "rope_scaling=%r, rope_theta=%r, tokenizer_revision=%s, "
            "trust_remote_code=%s, dtype=%s, max_seq_len=%d, "
            "download_dir=%r, load_format=%s, tensor_parallel_size=%d, "
            "pipeline_parallel_size=%d, "
            "disable_custom_all_reduce=%s, quantization=%s, "
            "enforce_eager=%s, kv_cache_dtype=%s, "
            "quantization_param_path=%s, device_config=%s, "
            "decoding_config=%r, observability_config=%r, "
            "seed=%d, served_model_name=%s, use_v2_block_manager=%s, "
            "enable_prefix_caching=%s)",
            VLLM_VERSION,
            model_config.model,
            speculative_config,
            model_config.tokenizer,
            model_config.skip_tokenizer_init,
            model_config.tokenizer_mode,
            model_config.revision,
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
            cache_config.enable_prefix_caching,
        )
        # TODO(woosuk): Print more configs in debug mode.

        self.model_config = model_config
        self.cache_config = cache_config
        self.lora_config = lora_config
        self.multimodal_config = multimodal_config
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
        else:
            self.tokenizer = None
            self.detokenizer = None

        self.seq_counter = Counter()
        self.generation_config_fields = _load_generation_config_dict(
            model_config)

        self.input_processor = INPUT_REGISTRY.create_input_processor(
            self.model_config)

        self.model_executor = executor_class(
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            device_config=device_config,
            lora_config=lora_config,
            multimodal_config=multimodal_config,
            speculative_config=speculative_config,
            load_config=load_config,
            prompt_adapter_config=prompt_adapter_config,
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
        self.scheduler = [
            Scheduler(scheduler_config, cache_config, lora_config,
                      parallel_config.pipeline_parallel_size)
            for _ in range(parallel_config.pipeline_parallel_size)
        ]

        # Metric Logging.
        if self.log_stats:
            if stat_loggers is not None:
                self.stat_loggers = stat_loggers
            else:
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
                self.get_tokenizer_for_seq,
                stop_checker=StopChecker(
                    self.scheduler_config.max_model_len,
                    self.get_tokenizer_for_seq,
                ),
            ))

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
    def _get_executor_cls(cls,
                          engine_config: EngineConfig) -> Type[ExecutorBase]:
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

    MISSING_TOKENIZER_GROUP_MSG = ("Unable to get tokenizer because "
                                   "skip_tokenizer_init is True")

    def get_tokenizer_group(
            self,
            fail_msg: str = MISSING_TOKENIZER_GROUP_MSG) -> BaseTokenizerGroup:
        if self.tokenizer is None:
            raise ValueError(fail_msg)

        return self.tokenizer

    def get_tokenizer(
        self,
        lora_request: Optional[LoRARequest] = None,
    ) -> AnyTokenizer:
        return self.get_tokenizer_group().get_lora_tokenizer(lora_request)

    def get_tokenizer_for_seq(self, sequence: Sequence) -> AnyTokenizer:
        return self.get_tokenizer_group().get_lora_tokenizer(
            sequence.lora_request)

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

    def _get_bos_token_id(self,
                          lora_request: Optional[LoRARequest] = None
                          ) -> Optional[int]:
        if self.tokenizer is None:
            logger.warning("Using None for BOS token id because tokenizer "
                           "is not initialized")
            return None

        return self.tokenizer.get_lora_tokenizer(lora_request).bos_token_id

    def _get_eos_token_id(self,
                          lora_request: Optional[LoRARequest] = None
                          ) -> Optional[int]:
        if self.tokenizer is None:
            logger.warning("Using None for EOS token id because tokenizer "
                           "is not initialized")
            return None

        return self.tokenizer.get_lora_tokenizer(lora_request).eos_token_id

    def _get_decoder_start_token_id(self, ) -> Optional[int]:
        '''
        Obtain the decoder start token id employed by an encoder/decoder
        model. Returns None for non-encoder/decoder models or if the
        model config is unavailable.
        '''

        if not self.is_encoder_decoder_model():
            logger.warning("Using None for decoder start token id because "
                           "this is not an encoder/decoder model.")
            return None

        if (self.model_config is None or self.model_config.hf_config is None):
            logger.warning("Using None for decoder start token id because "
                           "model config is not available.")
            return None

        dec_start_token_id = getattr(self.model_config.hf_config,
                                     'decoder_start_token_id', None)
        if dec_start_token_id is None:
            logger.warning("Falling back on <BOS> for decoder start token id "
                           "because decoder start token id is not available.")
            dec_start_token_id = self._get_bos_token_id()

        return dec_start_token_id

    def _add_processed_request(
        self,
        request_id: str,
        processed_inputs: LLMInputs,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: float,
        lora_request: Optional[LoRARequest],
        prompt_adapter_request: Optional[PromptAdapterRequest],
        trace_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        # Create the sequences.
        block_size = self.cache_config.block_size
        seq_id = next(self.seq_counter)
        eos_token_id = self._get_eos_token_id(lora_request)

        seq = Sequence(seq_id, processed_inputs, block_size, eos_token_id,
                       lora_request, prompt_adapter_request)

        encoder_seq = None
        if 'encoder_prompt_token_ids' in processed_inputs:
            encoder_seq = Sequence(seq_id,
                                   processed_inputs,
                                   block_size,
                                   eos_token_id,
                                   lora_request,
                                   prompt_adapter_request,
                                   from_decoder_prompt=False)

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
                encoder_seq=encoder_seq)
        elif isinstance(params, PoolingParams):
            seq_group = self._create_sequence_group_with_pooling(
                request_id,
                seq,
                params,
                arrival_time=arrival_time,
                lora_request=lora_request,
                prompt_adapter_request=prompt_adapter_request,
                encoder_seq=encoder_seq)
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

    def stop_remote_worker_execution_loop(self) -> None:
        self.model_executor.stop_remote_worker_execution_loop()

    _LLMInputComponentsType = Tuple[str, List[int], ]

    def _prepare_decoder_input_ids_for_generation(
        self,
        decoder_input_ids: Optional[List[int]] = None,
    ) -> List[int]:
        """
        Prepares `decoder_input_ids` for generation with encoder-decoder models.

        Based on

        https://github.com/huggingface/transformers/blob/
        4037a2b5b1278736e566aec12e169100275545ea/
        src/transformers/generation/utils.py

        specifically GenerationMixin._prepare_decoder_input_ids_for_generation()

        Arguments:

        * decoder_input_ids: input token ids to preprocess

        Returns:

        * Processed token list
        """

        decoder_start_token_id: Optional[int] = (
            self._get_decoder_start_token_id())
        assert decoder_start_token_id is not None

        if decoder_input_ids is None:
            # no decoder prompt input ->
            # use decoder_start_token_id as decoder_input_ids
            (decoder_input_ids) = self._get_default_enc_dec_decoder_prompt()

        if (len(decoder_input_ids) == 0
                or decoder_input_ids[0] != decoder_start_token_id):
            decoder_input_ids = [decoder_start_token_id] + decoder_input_ids

        return decoder_input_ids

    def _tokenize_prompt(
        self,
        prompt: str,
        request_id: Optional[str] = None,
        lora_request: Optional[str] = None,
    ) -> List[int]:
        '''
        Wrapper around application of the model's
        tokenizer.

        Arguments:

        * prompt
        * request_id
        * lora_request

        Returns:

        * prompt token ids
        '''

        tokenizer = self.get_tokenizer_group("prompts must be None if "
                                             "skip_tokenizer_init is True")

        prompt_token_ids = tokenizer.encode(request_id=request_id,
                                            prompt=prompt,
                                            lora_request=lora_request)

        return prompt_token_ids

    def _extract_single_prompt_for_enc_dec_input(
        self,
        inputs: Optional[PromptInputs],
        request_id: Optional[str] = None,
        ptype: Optional[str] = None,
        is_encoder_prompt: bool = False,
    ) -> Tuple[Optional[str], List[int]]:
        '''
        Only for encoder/decoder models:
        Extract prompt & prompt_token_ids from any single
        encoder or decoder input prompt. For encoder input prompts
        in particular, also extract multi-modal data.

        This function handles the following scenarios:
        1. The user supplied a singleton encoder prompt
          & the prompt/prompt-token-ids must be extracted.
        2. The user supplied an explicit encoder/decoder
          prompt & the prompt/prompt-token-ids must be
          extracted from either the encoder and decoder prompts.

        For decoder prompts in particular (scenario 2), special
        processing is applied to the returned decoder token ids.

        Arguments:

        * request_id
        * ptype: str representation of the input prompt type.
                 If `ptype` is `None`, assume that the prompt
                 type is unknown and must be inferred. This is the
                 case for ExplicitEncoderDecoder sub-prompts.
        * inputs: single encoder or decoder input prompt
        * is_encoder_prompt: True if encoder input prompt.
                             If False, decoder prompt tokens
                             are preprocessed.

        Returns:

        * prompt
        * prompt_token_ids
        '''
        prompt_token_ids = None
        ptype = (get_prompt_type(inputs) if ptype is None else ptype)

        if inputs is None:
            prompt = None
        elif ptype == 'str':
            prompt = inputs
            prompt_token_ids = self._tokenize_prompt(
                prompt,
                request_id=request_id,
            )
        elif ptype == 'TokensPrompt':
            prompt = None
            prompt_token_ids = inputs['prompt_token_ids']
        else:
            prompt = inputs['prompt']
            prompt_token_ids = self._tokenize_prompt(
                prompt,
                request_id=request_id,
            )

        if not is_encoder_prompt:
            # Apply special pre-processing to
            # decoder prompts
            prompt_token_ids = (self._prepare_decoder_input_ids_for_generation(
                prompt_token_ids, ))

        assert prompt_token_ids is not None

        return (
            prompt,
            prompt_token_ids,
        )

    def _get_default_enc_dec_decoder_prompt(self, ) -> List[int]:
        '''
        Specifically for encoder/decoder models:
        generate a default decoder prompt for when
        the user specifies only the encoder prompt.

        Encoder/decoder models utilize the decoder
        prompt in different ways; as new models are
        added, it is intended that this function
        will be extended to produce differing
        default decoder prompts, depending on the
        model variety.

        Absent a special case, the default behavior
        of this method is to mirror the behavior of
        the HuggingFace (HF) GenerationMixin for a None
        decoder prompt, which is to employ a logit processor
        setting to force the first decoded token to be <BOS>.
        Here, this behavior is approximated by having the
        "default" decoder prompt be <BOS>.

        However, it is possible that in the future
        other models may have different or more 
        complex logic for the default decoder prompt.
        This motivates having a special helper method
        for default decoder prompts.

        Returns:

        * prompt_token_ids
        '''

        bos_token_id = self._get_bos_token_id()
        assert bos_token_id is not None
        prompt_token_ids: List[int] = [bos_token_id]
        return prompt_token_ids

    def _process_encoder_decoder_prompt(
        self,
        inputs: PromptInputs,
        request_id: Optional[str] = None,
    ) -> LLMInputs:
        '''
        For encoder/decoder models only:
        Process an input prompt
        into an `LLMInputs` instance.

        There are two types of input prompts:
        singleton prompts which carry only the
        encoder prompt, and explicit encoder/decoder
        prompts which carry both the encoder and the
        decoder prompts as member variables.

        This function handles the following scenarios:
        * Singleton encoder prompt: extract encoder prompt
          token ids & infer default decoder prompt token ids
        * Explicit encoder/decoder prompt: extract encoder
          and decoder prompt token ids

        Note that for Explicit encoder/decoder prompts,
        each sub-prompt (encoder or decoder prompt) can
        have any possible singleton type; thus this
        method relies on helper functions to obtain
        token ids for the sub-prompts.
        
        Arguments:

        * inputs: an input prompt
        * request_id

        Returns:

        * `LLMInputs` instance
        '''

        ptype = get_prompt_type(inputs)

        # Obtain encoder and decoder prompt tokens. Note
        # that, no matter what, the decoder
        # prompt type is unknown.
        if ptype == "ExplicitEncoderDecoder":
            # If input is explicit encoder/decoder prompt,
            # then it remains to be determined what type
            # of encoder prompt we have
            extracted_encoder_prompt = inputs.get('encoder_prompt')
            encoder_ptype = None
            # Extract decoder prompt from explicit
            # encoder/decoder prompt
            extracted_decoder_prompt = inputs.get('decoder_prompt')
        else:
            # If input is singleton encoder prompt, then
            # we know the encoder prompt type
            extracted_encoder_prompt = inputs
            encoder_ptype = ptype
            # Decoder prompt is always unknown if
            # encoder/decoder prompt is not explicit
            extracted_decoder_prompt = None

        # Invoke helper function to obtain encoder
        # prompt and prompt token ids, either from
        # singleton encoder prompt or from the
        # encoder sub-prompt of an explicit
        # encoder/decode scenario 2), special
        # processing is applied to the returned decoder token ids
        (
            encoder_prompt,
            encoder_prompt_token_ids,
        ) = self._extract_single_prompt_for_enc_dec_input(
            extracted_encoder_prompt,
            request_id=request_id,
            ptype=encoder_ptype,
            is_encoder_prompt=True,
        )

        # Invoke helper method to obtain
        # decoder prompt and prompt token ids.
        #
        # The helper method will detect the decoder
        # prompt type.
        #
        # Helper method will also apply special
        # preprocessing unique to decoder prompts.
        (
            decoder_prompt,
            decoder_prompt_token_ids,
        ) = self._extract_single_prompt_for_enc_dec_input(
            extracted_decoder_prompt,
            request_id=request_id,
            ptype=None,
            is_encoder_prompt=False,
        )

        return LLMInputs(
            prompt_token_ids=decoder_prompt_token_ids,
            prompt=decoder_prompt,
            encoder_prompt_token_ids=encoder_prompt_token_ids,
            encoder_prompt=encoder_prompt,
        )

    def _process_decoder_only_prompt(
        self,
        inputs: PromptInputs,
        lora_request: Optional[LoRARequest] = None,
        request_id: Optional[str] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    ) -> LLMInputs:
        '''
        For decoder-only models:
        Process an input prompt
        into an `LLMInputs` instance.

        Arguments:

        * inputs: input prompt
        * lora_request
        * request_id
        * prompt_adapter_request

        Returns:

        * `LLMInputs` instance
        '''

        if isinstance(inputs, str):
            inputs = {"prompt": inputs}
        prompt = inputs.get("prompt")

        if "prompt_token_ids" not in inputs:
            prompt_token_ids = self._tokenize_prompt(
                prompt,
                request_id=request_id,
                lora_request=lora_request,
            )
        else:
            prompt_token_ids = inputs["prompt_token_ids"]

        if prompt_adapter_request:
            prompt_token_ids = (
                [0] * prompt_adapter_request.prompt_adapter_num_virtual_tokens
                + prompt_token_ids)

        return LLMInputs(prompt_token_ids=prompt_token_ids,
                         prompt=prompt,
                         multi_modal_data=inputs.get("multi_modal_data"))

    def process_model_inputs(
        self,
        request_id: str,
        inputs: PromptInputs,
        lora_request: Optional[LoRARequest] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    ) -> LLMInputs:

        if self.is_encoder_decoder_model():
            # Encoder-decoder model requires special mapping of
            # input prompts to encoder & decoder

            model_inputs = self._process_encoder_decoder_prompt(
                inputs,
                request_id=request_id,
            )
        else:
            # Decoder-only operation
            model_inputs = self._process_decoder_only_prompt(
                inputs,
                request_id=request_id,
                lora_request=lora_request,
                prompt_adapter_request=prompt_adapter_request,
            )

        return self.input_processor(model_inputs)

    def add_request(
        self,
        request_id: str,
        inputs: PromptInputs,
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
            inputs: The inputs to the LLM. See
                :class:`~vllm.inputs.PromptInputs`
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

        processed_inputs = self.process_model_inputs(
            request_id=request_id,
            inputs=inputs,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request)

        self._add_processed_request(
            request_id=request_id,
            processed_inputs=processed_inputs,
            params=params,
            arrival_time=arrival_time,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
            trace_headers=trace_headers,
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
    ) -> SequenceGroup:
        """Creates a SequenceGroup with SamplingParams."""
        max_logprobs = self.get_model_config().max_logprobs
        if (sampling_params.logprobs
                and sampling_params.logprobs > max_logprobs) or (
                    sampling_params.prompt_logprobs
                    and sampling_params.prompt_logprobs > max_logprobs):
            raise ValueError(f"Cannot request more than "
                             f"{max_logprobs} logprobs.")

        # Defensive copy of SamplingParams, which are used by the sampler,
        # this doesn't deep-copy LogitsProcessor objects
        sampling_params = sampling_params.clone()

        sampling_params.update_from_generation_config(
            self.generation_config_fields, seq.eos_token_id)

        # Create the sequence group.
        seq_group = SequenceGroup(
            request_id=request_id,
            seqs=[seq],
            arrival_time=arrival_time,
            sampling_params=sampling_params,
            lora_request=lora_request,
            trace_headers=trace_headers,
            prompt_adapter_request=prompt_adapter_request,
            encoder_seq=encoder_seq)

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
            encoder_seq=encoder_seq)
        return seq_group

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
        for scheduler in self.scheduler:
            scheduler.abort_seq_group(request_id)

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

    def _process_sequence_group_outputs(
        self,
        seq_group: SequenceGroup,
        outputs: List[EmbeddingSequenceGroupOutput],
    ) -> None:
        seq_group.embeddings = outputs[0].embeddings

        for seq in seq_group.get_seqs():
            seq.status = SequenceStatus.FINISHED_STOPPED

        return

    def _process_model_outputs(
        self,
        output: GenericSequence[Union[SamplerOutput, PoolerOutput]],
        scheduled_seq_groups: List[ScheduledSequenceGroup],
        ignored_seq_groups: List[SequenceGroup],
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> List[Union[RequestOutput, EmbeddingRequestOutput]]:
        """Apply the model output to the sequences in the scheduled seq groups.

        Returns RequestOutputs that can be returned to the client.
        """

        now = time.time()

        # Organize outputs by [sequence group][step] instead of
        # [step][sequence group].
        output_by_sequence_group = create_output_by_sequence_group(
            output, num_seq_groups=len(scheduled_seq_groups))

        # Update the scheduled sequence groups with the model outputs.
        for scheduled_seq_group, outputs, seq_group_meta in zip(
                scheduled_seq_groups, output_by_sequence_group,
                seq_group_metadata_list):
            seq_group = scheduled_seq_group.seq_group
            seq_group.update_num_computed_tokens(
                scheduled_seq_group.token_chunk_size)
            if self.model_config.embedding_mode:
                self._process_sequence_group_outputs(seq_group, outputs)
                continue

            self.output_processor.process_prompt_logprob(seq_group, outputs)
            if seq_group_meta.do_sample:
                self.output_processor.process_outputs(seq_group, outputs)

        # Free the finished sequence groups.
        for scheduler in self.scheduler:
            scheduler.free_finished_seq_groups()

        # Create the outputs.
        request_outputs: List[Union[RequestOutput,
                                    EmbeddingRequestOutput]] = []
        for scheduled_seq_group in scheduled_seq_groups:
            seq_group = scheduled_seq_group.seq_group
            seq_group.maybe_set_first_token_time(now)
            request_output = RequestOutputFactory.create(seq_group)
            request_outputs.append(request_output)
        for seq_group in ignored_seq_groups:
            request_output = RequestOutputFactory.create(seq_group)
            request_outputs.append(request_output)
        return request_outputs

    def step(self) -> List[Union[RequestOutput, EmbeddingRequestOutput]]:
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
            >>>         engine.add_request(str(req_id),prompt,sampling_params)
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
        if self.parallel_config.pipeline_parallel_size > 1:
            raise NotImplementedError(
                "Pipeline parallelism is only supported through AsyncLLMEngine "
                "as performance will be severely degraded otherwise.")
        seq_group_metadata_list, scheduler_outputs = self.scheduler[
            0].schedule()

        if not scheduler_outputs.is_empty():
            finished_requests_ids = self.scheduler[
                0].get_and_reset_finished_requests_ids()
            execute_model_req = ExecuteModelRequest(
                seq_group_metadata_list=seq_group_metadata_list,
                blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
                blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
                blocks_to_copy=scheduler_outputs.blocks_to_copy,
                num_lookahead_slots=scheduler_outputs.num_lookahead_slots,
                running_queue_size=scheduler_outputs.running_queue_size,
                finished_requests_ids=finished_requests_ids)
            output = self.model_executor.execute_model(
                execute_model_req=execute_model_req)
        else:
            output = []

        request_outputs = self._process_model_outputs(
            output, scheduler_outputs.scheduled_seq_groups,
            scheduler_outputs.ignored_seq_groups, seq_group_metadata_list)

        # Log stats.
        self.do_log_stats(scheduler_outputs, output)

        # Tracing
        self.do_tracing(scheduler_outputs)

        if not self.has_unfinished_requests():
            # Stop the execute model loop in parallel workers until there are
            # more requests to process. This avoids waiting indefinitely in
            # torch.distributed ops which may otherwise timeout, and unblocks
            # the RPC thread in the workers so that they can process any other
            # queued control plane messages, such as add/remove lora adapters.
            self.model_executor.stop_remote_worker_execution_loop()

        return request_outputs

    def add_logger(self, logger_name: str, logger: StatLoggerBase) -> None:
        if logger_name in self.stat_loggers:
            raise KeyError(f"Logger with name {logger_name} already exists.")
        self.stat_loggers[logger_name] = logger

    def remove_logger(self, logger_name: str) -> None:
        if logger_name not in self.stat_loggers:
            raise KeyError(f"Logger with name {logger_name} does not exist.")
        del self.stat_loggers[logger_name]

    def do_log_stats(
            self,
            scheduler_outputs: Optional[SchedulerOutputs] = None,
            model_output: Optional[List[SamplerOutput]] = None) -> None:
        """Forced log when no requests active."""
        if self.log_stats:
            stats = self._get_stats(scheduler_outputs, model_output)
            for logger in self.stat_loggers.values():
                logger.log(stats)

    def _get_stats(
            self,
            scheduler_outputs: Optional[SchedulerOutputs],
            model_output: Optional[List[SamplerOutput]] = None) -> Stats:
        """Get Stats to be Logged to Prometheus.

        Args:
            scheduler_outputs: Optional, used to populate metrics related to
                the scheduled batch,
            model_output: Optional, used to emit speculative decoding metrics
                which are created by the workers.
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
        if num_total_gpu is not None:
            num_free_gpu = sum(
                scheduler.block_manager.get_num_free_gpu_blocks()
                for scheduler in self.scheduler)
            gpu_cache_usage_sys = 1.0 - (num_free_gpu / num_total_gpu)

        num_total_cpu = self.cache_config.num_cpu_blocks
        cpu_cache_usage_sys = 0.
        if num_total_cpu is not None and num_total_cpu > 0:
            num_free_cpu = sum(
                scheduler.block_manager.get_num_free_cpu_blocks()
                for scheduler in self.scheduler)
            cpu_cache_usage_sys = 1.0 - (num_free_cpu / num_total_cpu)

        # Iteration stats
        num_prompt_tokens_iter = 0
        num_generation_tokens_iter = 0
        time_to_first_tokens_iter: List[float] = []
        time_per_output_tokens_iter: List[float] = []
        num_preemption_iter = (0 if scheduler_outputs is None else
                               scheduler_outputs.preempted)

        # Request stats
        #   Latency
        time_e2e_requests: List[float] = []
        #   Metadata
        num_prompt_tokens_requests: List[int] = []
        num_generation_tokens_requests: List[int] = []
        best_of_requests: List[int] = []
        n_requests: List[int] = []
        finished_reason_requests: List[str] = []

        # NOTE: This loop assumes prefill seq_groups are before
        # decode seq_groups in scheduled_seq_groups.
        if scheduler_outputs is not None:
            num_generation_tokens_from_prefill_groups = 0.
            # NOTE: if scheduler_outputs.num_prefill_groups > 0 and
            # the len of scheduler_outputs.scheduled_seq_groups is !=
            # scheduler_outputs.num_prefill_groups, this means that
            # chunked prefills have been detected.

            for idx, scheduled_seq_group in enumerate(
                    scheduler_outputs.scheduled_seq_groups):
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
                        latency = seq_group.get_last_latency(now)
                        time_to_first_tokens_iter.append(latency)

                        # One generation token per finished prefill.
                        num_generation_tokens_from_prefill_groups += (
                            seq_group.num_seqs())
                else:
                    # TPOTs.
                    latency = seq_group.get_last_latency(now)
                    time_per_output_tokens_iter.append(latency)

                # Because of chunked prefill, we can have a single sequence
                # group that does multiple prompt_runs. To prevent logging
                # the same metadata more than once per request, we standardize
                # on logging request level information for finished requests,
                # which can only happen once.
                if seq_group.is_finished():
                    # Latency timings
                    time_e2e_requests.append(now -
                                             seq_group.metrics.arrival_time)

                    # Metadata
                    num_prompt_tokens_requests.append(
                        len(seq_group.prompt_token_ids))
                    num_generation_tokens_requests.extend([
                        seq.get_output_len()
                        for seq in seq_group.get_finished_seqs()
                    ])
                    if seq_group.sampling_params is not None:
                        best_of_requests.append(
                            seq_group.sampling_params.best_of)
                        n_requests.append(seq_group.sampling_params.n)
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
                scheduler_outputs.num_batched_tokens - num_prompt_tokens_iter +
                num_generation_tokens_from_prefill_groups)

        # Spec decode, if enabled, emits specialized metrics from the worker in
        # sampler output.
        if model_output and (model_output[0].spec_decode_worker_metrics
                             is not None):
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

            # Iteration stats
            num_prompt_tokens_iter=num_prompt_tokens_iter,
            num_generation_tokens_iter=num_generation_tokens_iter,
            time_to_first_tokens_iter=time_to_first_tokens_iter,
            time_per_output_tokens_iter=time_per_output_tokens_iter,
            spec_decode_metrics=spec_decode_metrics,
            num_preemption_iter=num_preemption_iter,

            # Request stats
            #   Latency
            time_e2e_requests=time_e2e_requests,
            #   Metadata
            num_prompt_tokens_requests=num_prompt_tokens_requests,
            num_generation_tokens_requests=num_generation_tokens_requests,
            best_of_requests=best_of_requests,
            n_requests=n_requests,
            finished_reason_requests=finished_reason_requests,
        )

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

    def check_health(self) -> None:
        if self.tokenizer:
            self.tokenizer.check_health()
        self.model_executor.check_health()

    def is_tracing_enabled(self) -> bool:
        return self.tracer is not None

    def do_tracing(self, scheduler_outputs: SchedulerOutputs) -> None:
        if self.tracer is None:
            return

        for scheduled_seq_group in scheduler_outputs.scheduled_seq_groups:
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
            # attribute names are based on
            # https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/llm-spans.md
            seq_span.set_attribute(SpanAttributes.LLM_RESPONSE_MODEL,
                                   self.model_config.model)
            seq_span.set_attribute(SpanAttributes.LLM_REQUEST_ID,
                                   seq_group.request_id)
            seq_span.set_attribute(SpanAttributes.LLM_REQUEST_TEMPERATURE,
                                   seq_group.sampling_params.temperature)
            seq_span.set_attribute(SpanAttributes.LLM_REQUEST_TOP_P,
                                   seq_group.sampling_params.top_p)
            seq_span.set_attribute(SpanAttributes.LLM_REQUEST_MAX_TOKENS,
                                   seq_group.sampling_params.max_tokens)
            seq_span.set_attribute(SpanAttributes.LLM_REQUEST_BEST_OF,
                                   seq_group.sampling_params.best_of)
            seq_span.set_attribute(SpanAttributes.LLM_REQUEST_N,
                                   seq_group.sampling_params.n)
            seq_span.set_attribute(SpanAttributes.LLM_USAGE_NUM_SEQUENCES,
                                   seq_group.num_seqs())
            seq_span.set_attribute(SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
                                   len(seq_group.prompt_token_ids))
            seq_span.set_attribute(
                SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
                sum([
                    seq.get_output_len()
                    for seq in seq_group.get_finished_seqs()
                ]))
            seq_span.set_attribute(SpanAttributes.LLM_LATENCY_TIME_IN_QUEUE,
                                   metrics.time_in_queue)
            seq_span.set_attribute(
                SpanAttributes.LLM_LATENCY_TIME_TO_FIRST_TOKEN, ttft)
            seq_span.set_attribute(SpanAttributes.LLM_LATENCY_E2E, e2e_time)

    def is_encoder_decoder_model(self):
        return is_encoder_decoder_model_config(self.model_config)

    def is_embedding_model(self):
        return is_embedding_model_config(self.model_config)
