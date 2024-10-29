import time
from typing import (Any, Dict, Iterable, List, Mapping, Optional, Tuple, Type,
                    Union)

import msgspec
import zmq
import zmq.asyncio

from vllm.config import (CacheConfig, DecodingConfig, DeviceConfig,
                         EngineConfig, LoadConfig, LoRAConfig, ModelConfig,
                         ObservabilityConfig, ParallelConfig,
                         PromptAdapterConfig, SchedulerConfig,
                         SpeculativeConfig)
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.metrics_types import StatLoggerBase
from vllm.inputs import (INPUT_REGISTRY, DecoderOnlyInputs,
                         EncoderDecoderLLMInputs, InputRegistry, PromptType)
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.config import try_get_generation_config
from vllm.transformers_utils.tokenizer_group import (
    BaseTokenizerGroup, init_tokenizer_from_configs)
from vllm.usage.usage_lib import UsageContext
from vllm.utils import get_open_zmq_ipc_path
from vllm.v1.engine import (LLM_ENGINE_CORE_READY_STR, POLLING_TIMEOUT_MS,
                            DetokenizerRequest, EngineCoreOutputs,
                            EngineCoreRequest)
from vllm.v1.engine.detokenizer import Detokenizer
from vllm.v1.engine.llm_engine_core import LLMEngineCore
from vllm.v1.executor.gpu_executor import GPUExecutor
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)


class LLMEngine:

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
        executor_class: Type[GPUExecutor],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
        input_registry: InputRegistry = INPUT_REGISTRY,
        use_cached_outputs: bool = False,
        use_async_sockets: bool = False,
    ) -> None:
        # Override the configs for V1.
        # FIXME
        if usage_context == UsageContext.LLM_CLASS:
            scheduler_config.max_num_seqs = 1024
            scheduler_config.max_num_batched_tokens = 8192
        elif usage_context == UsageContext.OPENAI_API_SERVER:
            scheduler_config.max_num_seqs = 1024
            scheduler_config.max_num_batched_tokens = 2048

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
            "seed=%d, served_model_name=%s, "
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
            scheduler_config.num_scheduler_steps,
            cache_config.enable_prefix_caching,
            model_config.use_async_output_proc,
            model_config.mm_processor_kwargs,
        )

        self.model_config = model_config
        assert self.model_config.task != "embedding"
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

        assert not self.model_config.skip_tokenizer_init
        self.tokenizer = self._init_tokenizer()
        if self.tokenizer:
            # Ping the tokenizer to ensure liveness if it runs in a
            # different process.
            self.tokenizer.ping()
        self.detokenizer = Detokenizer(self.model_config.tokenizer,
                                       stream_mode=use_async_sockets)

        self.generation_config_fields = _load_generation_config_dict(
            model_config)
        self.input_preprocessor = InputPreprocessor(model_config,
                                                    self.tokenizer)
        self.input_registry = input_registry
        self.input_processor = input_registry.create_input_processor(
            model_config)

        # IPC serialization / deserialization
        self.encoder = msgspec.msgpack.Encoder()
        self.decoder = msgspec.msgpack.Decoder(EngineCoreOutputs)

        # IPC path setup
        if use_async_sockets:
            import zmq.asyncio
            self.ctx = zmq.asyncio.Context()  # type: ignore[attr-defined]
        else:
            self.ctx = zmq.Context()  # type: ignore[attr-defined]
        from_core_ipc_path = get_open_zmq_ipc_path()
        to_core_ipc_path = get_open_zmq_ipc_path()
        core_ready_ipc_path = get_open_zmq_ipc_path()

        # Get output (EngineCoreOutput) from LLMEngineCore.
        self.from_core = self.ctx.socket(zmq.constants.PULL)
        self.from_core.connect(from_core_ipc_path)

        # Send input (new Requests) to LLMEngineCore.
        self.to_core = self.ctx.socket(zmq.constants.PUSH)
        self.to_core.bind(to_core_ipc_path)

        # TODO: some of these configs will be mutated by
        # EngineCore (in a separate process), e.g. cache_config.
        # It would be better if we could prune down what is needed
        # for EngineCore and prevent having two sources of truth
        # or perhaps made these immutable?
        self.engine_core = LLMEngineCore(
            input_path=to_core_ipc_path,
            output_path=from_core_ipc_path,
            core_ready_path=core_ready_ipc_path,
            executor_class=executor_class,
            model_config=self.model_config,
            cache_config=self.cache_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
            device_config=self.device_config,
            load_config=self.load_config,
            lora_config=self.lora_config,
            speculative_config=self.speculative_config,
            observability_config=self.observability_config,
            prompt_adapter_config=self.prompt_adapter_config,
        )
        self.engine_core.start()

        if not use_async_sockets:
            self._wait_for_engine_core(core_ready_ipc_path)
        else:
            self.readiness_ipc_path = core_ready_ipc_path

    def __del__(self):
        logger.debug("Shutting down LLMEngineCore.")

        if hasattr(self, "ctx"):
            self.ctx.destroy(linger=0)

        if hasattr(self, "engine_core"):
            # TODO: do this more gracefully by sending a message?
            # or at least using .terminate()
            self.engine_core.kill()

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

    def _wait_for_engine_core(self, ipc_path: str):
        try:
            ready_socket = self.ctx.socket(zmq.constants.PULL)
            ready_socket.connect(ipc_path)
            while ready_socket.poll(timeout=POLLING_TIMEOUT_MS) == 0:
                logger.debug("Waiting for LLMEngineCore to startup.")
                if not self.engine_core.is_alive():
                    raise RuntimeError("LLMEngineCore process failed to start")

            message = ready_socket.recv_string()
            assert message == LLM_ENGINE_CORE_READY_STR

        finally:
            ready_socket.close(linger=0)

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

    def stop_remote_worker_execution_loop(self) -> None:
        raise NotImplementedError("TP not implemented yet.")

    def add_request(
        self,
        request_id: str,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> None:

        # 1) Process raw inputs into the request.
        detokenizer_request, engine_core_request = self._process_inputs(
            request_id, prompt, params, arrival_time, lora_request,
            trace_headers, prompt_adapter_request, priority)

        # 2) Add the request to Detokenizer (this process).
        self.detokenizer.add_request(detokenizer_request)

        # 3) Add the request to EngineCore (separate process).
        self.to_core.send_multipart(
            (self.encoder.encode(engine_core_request), ),
            copy=False,
            flags=zmq.NOBLOCK)

    def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        # TODO: send to EngineCore
        # TODO: send to Deoktenizer
        pass

    # NOTE: a significant drawback of this design is now we have two
    # trackers of running state (the Detokenizer and the Scheduler).
    # Is there a better way to do this?
    # Unfortunately we need need to send state over IPC?
    # Maybe we could get back the scheduler state with EngineCoreOutput?
    # Such that state is explicitly in sync rather than implicitly?

    def get_num_unfinished_requests(self) -> int:
        return self.detokenizer.get_num_unfinished_requests()

    def has_unfinished_requests(self) -> bool:
        return self.detokenizer.has_unfinished_requests()

    def step(self) -> List[RequestOutput]:
        # NOTE: This method returns an empty list if the EngineCore
        # step is running. This is not the end of the generation process.
        if self.from_core.poll(timeout=0) != 0:
            frames = self.from_core.recv_multipart(copy=False)
            engine_core_outputs = self.decoder.decode(frames[0].buffer).outputs
            request_outputs = self.detokenizer.step(engine_core_outputs)
            return request_outputs

        return []

    def check_health(self) -> None:
        if self.tokenizer:
            self.tokenizer.check_health()
        # self.model_executor.check_health()
        # TODO: send health check to EngineCore.

    def _process_inputs(
        self,
        request_id: str,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: float,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> Tuple[DetokenizerRequest, EngineCoreRequest]:

        # TODO(woosuk): Support embedding mode.
        # TODO(woosuk): Check max_logprobs
        # TODO(woosuk): Support encoder-decoder models.

        if lora_request is not None and not self.lora_config:
            raise ValueError(f"Got lora_request {lora_request} but LoRA is "
                             "not enabled!")
        if arrival_time is None:
            arrival_time = time.time()
        assert priority == 0, "vLLM V1 does not support priority at the moment."
        assert trace_headers is None, "vLLM V1 does not support tracing yet."

        # Process inputs.
        preprocessed_inputs = self.input_preprocessor.preprocess(
            prompt,
            request_id=request_id,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
        )
        processed_inputs = self.input_processor(preprocessed_inputs)
        self._validate_model_inputs(processed_inputs)
        eos_token_id = self.input_preprocessor.get_eos_token_id(lora_request)

        assert isinstance(params, SamplingParams)
        sampling_params = params.clone()
        sampling_params.update_from_generation_config(
            self.generation_config_fields, eos_token_id)

        # Make Request for Detokenizer.
        detokenizer_request = DetokenizerRequest(
            request_id, processed_inputs.get("prompt"),
            processed_inputs.get("prompt_token_ids"),
            sampling_params.skip_special_tokens,
            sampling_params.spaces_between_special_tokens,
            sampling_params.output_kind)

        # Make Request for EngineCore.
        engine_core_request = EngineCoreRequest(
            request_id, processed_inputs.get("prompt"),
            processed_inputs.get("prompt_token_ids"), sampling_params,
            eos_token_id, arrival_time, lora_request)

        return detokenizer_request, engine_core_request

    def _validate_model_inputs(self, inputs: Union[DecoderOnlyInputs,
                                                   EncoderDecoderLLMInputs]):
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

    @classmethod
    def validate_outputs(cls, outputs, output_type):
        return outputs

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

    @classmethod
    def _get_executor_cls(cls, engine_config: EngineConfig):
        return GPUExecutor

    def is_tracing_enabled(self) -> bool:
        return False

    def do_log_stats(self, *args, **kwargs) -> None:
        pass

    def is_encoder_decoder_model(self) -> bool:
        return False

    def start_profile(self) -> None:
        pass

    def stop_profile(self) -> None:
        pass

    def get_tokenizer_group(self, *args, **kwargs):
        return self.tokenizer


def _load_generation_config_dict(model_config: ModelConfig) -> Dict[str, Any]:
    config = try_get_generation_config(
        model_config.model,
        trust_remote_code=model_config.trust_remote_code,
        revision=model_config.revision,
    )

    if config is None:
        return {}

    return config.to_diff_dict()
