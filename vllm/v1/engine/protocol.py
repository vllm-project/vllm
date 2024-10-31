import multiprocessing

from abc import ABC, abstractmethod
from typing import Union

from vllm.config import (DecodingConfig, EngineConfig, LoRAConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig)
from vllm.logger import init_logger
from vllm.v1.executor.gpu_executor import GPUExecutor
from vllm.v1.engine.detokenizer import Detokenizer
from vllm.v1.engine.processor import Processor
from vllm.v1.engine.llm_engine_core import LLMEngineCore

logger = init_logger(__name__)


class LLMEngineProtocol(ABC):
    """Protocol for LLMEngine and AsyncLLMEngine"""

    engine_core: Union[LLMEngineCore, multiprocessing.Process]
    detokenizer: Detokenizer
    processor: Processor

    # TODO: These are needed for the get_xxx_config methods
    # I think these are basically dead code. Will see if this
    # can be removed

    model_config: ModelConfig
    parallel_config: ParallelConfig
    decoding_config: DecodingConfig
    scheduler_config: SchedulerConfig
    lora_config: LoRAConfig

    def stop_remote_worker_execution_loop(self) -> None:
        raise NotImplementedError("TP not implemented yet.")

    @abstractmethod
    def get_num_unfinished_requests(self) -> int:
        return self.detokenizer.get_num_unfinished_requests()

    @abstractmethod
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
