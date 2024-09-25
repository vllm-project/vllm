from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.wde.decode_only.arg_utils import (DecodeOnlyEngineArgs,
                                            DecodeOnlyEngineConfig,
                                            filter_unexpected_fields)

logger = init_logger(__name__)


@filter_unexpected_fields
@dataclass
class Qwen2EngineArgs(DecodeOnlyEngineArgs):
    switch_to_gte_Qwen2: bool = False

    def create_engine_config(self) -> DecodeOnlyEngineConfig:
        if "gte" in self.model and not self.switch_to_gte_Qwen2:
            logger.warning("Because gte-Qwen2 and Qwen2 use the "
                           "same architecture name Qwen2ForCausalLM, "
                           "So you need to manually switch to "
                           "gte-Qwen2 using switch_to_gte_Qwen2.")

        if self.switch_to_gte_Qwen2:
            self.output_last_hidden_states = True
        config = super().create_engine_config()
        config.model_config.switch_to_gte_Qwen2 = self.switch_to_gte_Qwen2
        return config
