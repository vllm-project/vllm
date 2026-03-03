# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from pydantic import ConfigDict

from vllm.config.utils import config
from vllm.logger import init_logger

logger = init_logger(__name__)


@config(config=ConfigDict(arbitrary_types_allowed=True))
class ModelArchitectureConfig:
    """
    Configuration for model architecture that required by vLLM runtime
    """

    architectures: list[str] | None = None
    """List of model architecture class names (e.g., ['LlamaForCausalLM']).
       It can be None upon calling `vllm_config.with_hf_config(config.text_config)`"""

    model_type: str = ""
    """Model type identifier (e.g., 'llama', 'gpt_oss')."""

    text_model_type: str | None = None
    """Text model type identifier (e.g., 'llama4_text')."""

    hidden_size: int = 0
    """Hidden size of the model."""

    total_num_hidden_layers: int = 0
    """Number of hidden layers in the model."""

    total_num_attention_heads: int = 0
    """Number of attention heads in the model."""

    head_size: int = 0
    """Head dimension of the model."""

    vocab_size: int = 0
    """Vocabulary size of the model."""

    total_num_kv_heads: int = 0
    """Number of key value heads in the model."""

    num_experts: int = 0
    """Number of experts in the model."""

    quantization_config: dict[str, Any] | None = None
    """Quantization configuration dictionary containing quantization parameters."""

    is_deepseek_mla: bool = False
    """Whether the model is a DeepSeek MLA model."""

    derived_max_model_len_and_key: tuple[float, str | None] | None = None
    """Derived maximum model length and key from the hf config."""
