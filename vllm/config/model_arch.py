# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, NamedTuple

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from vllm.logger import init_logger

logger = init_logger(__name__)


class MaxModelLenInfo(NamedTuple):
    """Information about the maximum model length."""

    derived: float
    """Maximum supported sequence length after RoPE scaling.
    Used for:
    1. Validation - user-specified max_model_len cannot exceed this.
    2. Default for non-LongRoPE models (with sliding_window/tokenizer caps)."""

    derived_key: str | None
    """The config key used to derive the max length (for error messages)."""

    default: float | None
    """For LongRoPE models only: original_max_position_embeddings.
    Used as the default max_model_len to avoid performance degradation.
    None for non-LongRoPE models (derived is used instead)."""

    model_max_length: int | None
    """The model_max_length from hf_config. Used as a fallback for validation
    when user-specified max_model_len exceeds derived."""


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

    max_model_len_info: MaxModelLenInfo
    """Derived maximum model length information including RoPE scaling."""

    # RoPE-related fields
    uses_mrope: bool
    """Whether the model uses M-RoPE (multi-dimensional rotary position embedding)."""

    uses_xdrope_dim: int
    """Number of dimensions for XD-RoPE. 0 if not used."""
