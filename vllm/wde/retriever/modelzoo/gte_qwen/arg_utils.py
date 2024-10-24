from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.wde.decode_only.arg_utils import (DecodeOnlyEngineArgs,
                                            DecodeOnlyEngineConfig,
                                            filter_unexpected_fields)

logger = init_logger(__name__)


@filter_unexpected_fields
@dataclass
class Qwen2EngineArgs(DecodeOnlyEngineArgs):

    def create_engine_config(self) -> DecodeOnlyEngineConfig:
        # gte-Qwen2 and Qwen2 use the same architecture name，Qwen2ForCausalLM.
        # gte-Qwen2 family may have multiple different architectures.
        # gte-Qwen2-1.5B-instruct, does not use enable bidirectional.
        #     I'm not sure if this is a bug
        # gte-Qwen2-7B-instruct use enable bidirectional
        if "gte-Qwen2-1.5B-instruct" in self.model:
            self.output_last_hidden_states = True
        elif "gte-Qwen2-7B-instruct" in self.model:
            self.output_last_hidden_states = True
            self.enable_bidirectional = True

        config = super().create_engine_config()
        return config
