# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""GptOss configuration."""

from typing import Any, Dict, Optional

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class GptOssConfig(PretrainedConfig):
    """Configuration class for GptOss.

    [`GptOssModel`]. It is used to instantiate a GptOss model according to the
    specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the GptOss model.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the embeddings and hidden states.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to None):
            Number of key-value heads for each attention layer in the Transformer encoder. Will default to
            `num_attention_heads` if not set.
        head_dim (`int`, *optional*, defaults to None):
            The attention head dimension. If not specified, it will default to `hidden_size // num_attention_heads`.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings.
        rope_ntk_alpha (`float`, *optional*, defaults to 1.0):
            The alpha parameter for NTK-aware RoPE scaling.
        rope_ntk_beta (`float`, *optional*, defaults to 32.0):
            The beta parameter for NTK-aware RoPE scaling.
        sliding_window (`int`, *optional*):
            Sliding window attention window size. If not specified, will default to 4096.
        num_local_experts (`int`, *optional*, defaults to 8):
            Number of experts in the mixture of experts layer.
        num_experts_per_token (`int`, *optional*, defaults to 2):
            Number of experts to use per token in the mixture of experts layer.
        num_experts_per_tok (`int`, *optional*, defaults to None):
            Alias for `num_experts_per_token`. If specified, will override `num_experts_per_token`.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions.
        **kwargs:
            Additional keyword arguments passed to the parent class.
    """

    model_type = "gpt_oss"

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[Dict[str, Any]] = None,
        rope_ntk_alpha: float = 1.0,
        rope_ntk_beta: float = 32.0,
        sliding_window: Optional[int] = 4096,
        num_local_experts: int = 8,
        num_experts_per_token: int = 2,
        num_experts_per_tok: Optional[int] = None,
        use_cache: bool = True,
        **kwargs: Any,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # Set num_key_value_heads to num_attention_heads if not specified
        if num_key_value_heads is None:
            self.num_key_value_heads = num_attention_heads
        else:
            self.num_key_value_heads = num_key_value_heads

        # Set head_dim if not specified
        if head_dim is None:
            self.head_dim = hidden_size // num_attention_heads
        else:
            self.head_dim = head_dim

        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta

        # Set default rope_scaling if not specified
        if rope_scaling is None:
            self.rope_scaling = {
                "factor": 1.0,
                "original_max_position_embeddings": max_position_embeddings,
            }
        else:
            self.rope_scaling = rope_scaling

        self.rope_ntk_alpha = rope_ntk_alpha
        self.rope_ntk_beta = rope_ntk_beta
        self.sliding_window = sliding_window
        self.num_local_experts = num_local_experts

        # Handle the num_experts_per_tok alias
        if num_experts_per_tok is not None:
            self.num_experts_per_token = num_experts_per_tok
            self.num_experts_per_tok = num_experts_per_tok
        else:
            self.num_experts_per_token = num_experts_per_token
            self.num_experts_per_tok = num_experts_per_token

        self.use_cache = use_cache

        super().__init__(
            use_cache=use_cache,
            **kwargs,
        )
