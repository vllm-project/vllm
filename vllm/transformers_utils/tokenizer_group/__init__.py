# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from vllm.config import LoRAConfig, ModelConfig, SchedulerConfig

from .base_tokenizer_group import AnyTokenizer, BaseTokenizerGroup
from .tokenizer_group import TokenizerGroup


def init_tokenizer_from_configs(model_config: ModelConfig,
                                scheduler_config: SchedulerConfig,
                                lora_config: Optional[LoRAConfig]):
    return TokenizerGroup(
        tokenizer_id=model_config.tokenizer,
        enable_lora=bool(lora_config),
        max_num_seqs=scheduler_config.max_num_seqs,
        max_loras=lora_config.max_loras if lora_config else 0,
        max_input_length=None,
        tokenizer_mode=model_config.tokenizer_mode,
        trust_remote_code=model_config.trust_remote_code,
        revision=model_config.tokenizer_revision,
        truncation_side=model_config.truncation_side)


__all__ = ["AnyTokenizer", "BaseTokenizerGroup"]
