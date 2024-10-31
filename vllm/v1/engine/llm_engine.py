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
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.detokenizer import Detokenizer
from vllm.v1.engine.llm_engine_core import LLMEngineCore
from vllm.v1.engine.processor import Processor
from vllm.v1.executor.gpu_executor import GPUExecutor

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
    ) -> None:
        # Override the configs for V1.
        # FIXME
        if usage_context == UsageContext.LLM_CLASS:
            scheduler_config.max_num_seqs = 1024
            scheduler_config.max_num_batched_tokens = 8192
        elif usage_context == UsageContext.OPENAI_API_SERVER:
            scheduler_config.max_num_seqs = 1024
            scheduler_config.max_num_batched_tokens = 2048

        # Processor (convert Inputs --> EngineCoreRequests)
        self.processor = Processor(model_config, parallel_config,
                                   scheduler_config, lora_config,
                                   input_registry)

        # Detokenizer (converts EngineCoreOutputs --> RequestOutput)
        self.detokenizer = Detokenizer(model_config.tokenizer)

        # LLMEngineCore (core engine)
        self.engine_core = LLMEngineCore(
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
        # Create the LLM engine.
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

        # 1) Step the LLMEngineCore.
        engine_core_outputs = self.engine_core.step()

        # 2) Step the Detokenizer.
        request_outputs = self.detokenizer.step(engine_core_outputs)

        return request_outputs

    def get_num_unfinished_requests(self) -> int:
        return self.detokenizer.get_num_unfinished_requests()

    def has_unfinished_requests(self) -> bool:
        return self.detokenizer.has_unfinished_requests()
