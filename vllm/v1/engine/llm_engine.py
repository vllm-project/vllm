from typing import Dict, Iterable, List, Mapping, Optional, Type, Union

from vllm.config import (CacheConfig, DecodingConfig, DeviceConfig,
                         EngineConfig, LoadConfig, LoRAConfig, ModelConfig,
                         ObservabilityConfig, ParallelConfig,
                         PromptAdapterConfig, SchedulerConfig,
                         SpeculativeConfig)
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.metrics_types import StatLoggerBase
from vllm.inputs import INPUT_REGISTRY, InputRegistry, PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.core import EngineCore, EngineCoreClient
from vllm.v1.engine.detokenizer import Detokenizer
from vllm.v1.engine.processor import Processor
from vllm.v1.executor.gpu_executor import GPUExecutor

logger = init_logger(__name__)


class LLMEngine:
    """Legacy LLMEngine for backwards compatibility."""

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
        use_multiprocessing: bool = False,
    ) -> None:

        # TODO: Can we avoid this?
        self.model_config = model_config

        # Tokenizer
        self.tokenizer = init_tokenizer_from_configs(
            model_config=model_config,
            scheduler_config=scheduler_config,
            parallel_config=parallel_config,
            enable_lora=bool(lora_config))
        # Ensure liveness if running in another process.
        self.tokenizer.ping()

        # Processor (convert Inputs --> EngineCoreRequests)
        self.processor = Processor(model_config, lora_config,
                                   self.tokenizer, input_registry)

        # Detokenizer (converts EngineCoreOutputs --> RequestOutput)
        self.detokenizer = Detokenizer(self.tokenizer)

        if use_multiprocessing:
            self.engine_core = EngineCoreClient(
                executor_class,
                model_config,
                cache_config,
                parallel_config,
                scheduler_config,
                device_config,
                load_config,
                lora_config,
                speculative_config,
                decoding_config or DecodingConfig(),
                observability_config or ObservabilityConfig(),
                prompt_adapter_config,
                usage_context,
                asyncio_mode=False,
            )

        else:
            # EngineCore
            self.engine_core = EngineCore(
                executor_class=executor_class,
                model_config=model_config,
                cache_config=cache_config,
                parallel_config=parallel_config,
                scheduler_config=scheduler_config,
                device_config=device_config,
                load_config=load_config,
                lora_config=lora_config,
                speculative_config=speculative_config,
                decoding_config=(decoding_config or DecodingConfig()),
                observability_config=(observability_config
                                    or ObservabilityConfig()),
                prompt_adapter_config=prompt_adapter_config,
                usage_context=usage_context,
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
        engine_config = engine_args.create_engine_config()
        executor_class = cls._get_executor_cls(engine_config)
        # Create the LLMEngine.
        engine = cls(
            **engine_config.to_dict(),
            executor_class=executor_class,
            log_stats=not engine_args.disable_log_stats,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
        )
        return engine

    @classmethod
    def _get_executor_cls(cls, engine_config: EngineConfig):
        return GPUExecutor

    def stop_remote_worker_execution_loop(self) -> None:
        raise NotImplementedError("TP not implemented yet.")

    def get_num_unfinished_requests(self) -> int:
        return self.detokenizer.get_num_unfinished_requests()

    def has_unfinished_requests(self) -> bool:
        return self.detokenizer.has_unfinished_requests()

    @classmethod
    def validate_outputs(cls, outputs, output_type):
        return outputs

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
        detokenizer_req, engine_core_req = self.processor.process_inputs(
            request_id, prompt, params, arrival_time, lora_request,
            trace_headers, prompt_adapter_request, priority)

        # 2) Add the request to Detokenizer.
        self.detokenizer.add_request(detokenizer_req)

        # 3) Add the request to EngineCore.
        self.engine_core.add_request(engine_core_req)

    def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        # TODO: send to EngineCore
        # TODO: send to Detokenizer
        raise NotImplementedError

    def step(self) -> List[RequestOutput]:

        # 1) Step the EngineCore.
        engine_core_outputs = self.engine_core.step()

        # 2) Step the Detokenizer.
        request_outputs, to_abort = self.detokenizer.step(engine_core_outputs)

        # 3) Abort requests that finished due to stop criteria
        self.abort_request(to_abort)

        return request_outputs
