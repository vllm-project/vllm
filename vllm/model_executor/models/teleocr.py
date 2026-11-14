# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix


class TeleOCRForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    """TeleOCR combines a Qwen2.5-VL vision tower with a Qwen3 decoder."""

    def _init_language_model(self, vllm_config: VllmConfig, prefix: str) -> nn.Module:
        return init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=vllm_config.model_config.hf_text_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen3ForCausalLM"],
        )
