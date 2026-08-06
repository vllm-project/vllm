# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.models.qwen3_vl import Qwen3VLProcessingInfo
from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5Config
from vllm.transformers_utils.configs.qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
)


class Qwen3_5ProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3_5Config)


class Qwen3_5MoeProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self):
        # transformers 5.x renames the top-level Qwen3.5-MoE config class to
        # Qwen3_5MoeTextConfig for text-only models, while transformers ≤4.x
        # returns Qwen3_5MoeConfig (the multimodal wrapper).
        return self.ctx.get_hf_config((Qwen3_5MoeConfig, Qwen3_5MoeTextConfig))
