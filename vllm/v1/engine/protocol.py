from abc import ABC
import time
from typing import Dict, Optional, Type

from vllm.config import (CacheConfig, DecodingConfig, DeviceConfig,
                         EngineConfig, LoadConfig, LoRAConfig, ModelConfig,
                         ObservabilityConfig, ParallelConfig,
                         PromptAdapterConfig, SchedulerConfig,
                         SpeculativeConfig)
from vllm.engine.metrics_types import StatLoggerBase
from vllm.inputs import INPUT_REGISTRY, InputRegistry
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.detokenizer import Detokenizer
from vllm.v1.engine.processor import Processor
from vllm.v1.executor.gpu_executor import GPUExecutor
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)


class LLMEngineProtocol(ABC):
    """Protocol for LLMEngine and AsyncLLMEngine"""

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
    ):
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

        self.processor = Processor(model_config, parallel_config,
                                   scheduler_config, lora_config,
                                   input_registry)
        self.detokenizer = Detokenizer(self.model_config.tokenizer)

    def stop_remote_worker_execution_loop(self) -> None:
        raise NotImplementedError("TP not implemented yet.")

    def get_num_unfinished_requests(self) -> int:
        return self.detokenizer.get_num_unfinished_requests()

    def has_unfinished_requests(self) -> bool:
        return self.detokenizer.has_unfinished_requests()

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
        print("do_log_stats")

    def is_encoder_decoder_model(self) -> bool:
        return False

    def start_profile(self) -> None:
        print("start_profile")

    def stop_profile(self) -> None:
        print("stop_profile")

    def get_tokenizer_group(self, *args, **kwargs):
        return self.processor.tokenizer
