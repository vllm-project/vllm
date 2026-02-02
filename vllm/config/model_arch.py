# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ModelArchitectureConfig:
    """
    Configuration for model architecture that required by vLLM runtime
    """

    architectures: list[str] | None
    """List of model architecture class names (e.g., ['LlamaForCausalLM']).
       It can be None upon calling `vllm_config.with_hf_config(config.text_config)`"""

    model_type: str
    """Model type identifier (e.g., 'llama', 'gpt_oss')."""

    text_model_type: str | None
    """Text model type identifier (e.g., 'llama4_text')."""

    hidden_size: int
    """Hidden size of the model."""

    total_num_hidden_layers: int
    """Number of hidden layers in the model."""

    total_num_attention_heads: int
    """Number of attention heads in the model."""

    head_size: int
    """Head dimension of the model."""

    vocab_size: int
    """Vocabulary size of the model."""

    total_num_kv_heads: int
    """Number of key value heads in the model."""

    num_experts: int
    """Number of experts in the model."""

    quantization_config: dict[str, Any] | None
    """Quantization configuration dictionary containing quantization parameters."""

    is_deepseek_mla: bool
    """Whether the model is a DeepSeek MLA model."""

    uses_ngram_embedding: bool
    """Whether the model uses N-gram embeddings."""

    ngram_context_len: int
    """Context length for N-gram embeddings (N-1)."""

    derived_max_model_len_and_key: tuple[float, str | None]
    """Derived maximum model length and key from the hf config."""
