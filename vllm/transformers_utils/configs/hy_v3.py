# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from transformers.configuration_utils import PretrainedConfig


class HYV3Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`HYV3Model`].
    It is used to instantiate a HYV3 model (HY V3 MoE language model) according to
    the specified arguments.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to
    control the model outputs. Read the documentation from [`PretrainedConfig`]
    for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 120832):
            Vocabulary size of the model.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 13312):
            Dimension of the dense FFN intermediate representations.
        num_hidden_layers (`int`, *optional*, defaults to 80):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 64):
            Number of attention heads for each attention layer.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key-value heads for grouped-query attention.
        head_dim (`int`, *optional*, defaults to 128):
            Dimension per attention head.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            Activation function used in FFN layers.
        max_position_embeddings (`int`, *optional*, defaults to 131072):
            Maximum sequence length supported by the model.
        initializer_range (`float`, *optional*, defaults to 0.006):
            Standard deviation of the truncated normal initializer for weight
            initialization.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            Epsilon for RMS normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to use KV cache for decoding.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*):
            Beginning-of-sequence token id.
        eos_token_id (`int` or `List[int]`, *optional*):
            End-of-sequence token id(s).
        rope_parameters (`dict`, *optional*):
            The parameters of the RoPE embeddings.
        qk_norm (`bool`, *optional*, defaults to `True`):
            Whether to apply RMSNorm to query and key states before attention.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie input and output embedding weights.
        enable_attention_fp32_softmax (`bool`, *optional*, defaults to `False`):
            Whether to upcast attention softmax to float32. Note: the eager attention
            path always computes softmax in float32 regardless of this setting; this
            flag is reserved for future use with custom attention backends.
        enable_lm_head_fp32 (`bool`, *optional*, defaults to `True`):
            Whether to upcast the LM head computation to float32.
        num_experts (`int`, *optional*, defaults to 192):
            Total number of MoE experts.
        num_experts_per_tok (`int`, *optional*, defaults to 8):
            Number of experts selected per token (top-k routing).
        num_shared_experts (`int`, *optional*, defaults to 1):
            Number of always-active shared experts combined into a single MLP.
        expert_hidden_dim (`int`, *optional*, defaults to 1536):
            Intermediate dimension of each individual MoE expert.
        moe_router_enable_expert_bias (`bool`, *optional*, defaults to `True`):
            Whether to use per-expert load-balancing bias in the router.
        moe_router_use_sigmoid (`bool`, *optional*, defaults to `True`):
            Whether to use sigmoid (instead of softmax) for router scoring.
        route_norm (`bool`, *optional*, defaults to `True`):
            Whether to normalize routing scores when using sigmoid routing.
        router_scaling_factor (`float`, *optional*):
            Optional multiplicative scaling factor applied to routing scores.
        use_grouped_mm (`bool`, *optional*, defaults to `False`):
            Whether to use grouped GEMM for expert computation (not yet implemented).
        enable_moe_fp32_combine (`bool`, *optional*, defaults to `False`):
            Whether to accumulate expert outputs in float32.
        first_k_dense_replace (`int`, *optional*, defaults to 1):
            Number of initial decoder layers that use a dense FFN instead of MoE.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether to output router logits from each MoE layer. Useful for computing
            auxiliary load-balancing loss during training. Disabled by default to avoid
            the memory overhead of storing per-layer router tensors during inference.

    Example:
        ```python
        >>> from transformers import HYV3Config, HYV3Model

        >>> config = HYV3Config()
        >>> model = HYV3Model(config)
        ```
    """

    model_type = "hy_v3"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=120832,
        hidden_size=4096,
        intermediate_size=13312,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=131072,
        initializer_range=0.006,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        rope_parameters: dict[str, Any] | None = None,
        qk_norm=True,
        tie_word_embeddings=False,
        enable_attention_fp32_softmax=False,
        enable_lm_head_fp32=True,
        # MoE specific
        num_experts=192,
        num_experts_per_tok=8,
        num_shared_experts=1,
        expert_hidden_dim=1536,
        moe_router_enable_expert_bias=True,
        moe_router_use_sigmoid=True,
        route_norm=True,
        router_scaling_factor=None,
        use_grouped_mm=False,
        enable_moe_fp32_combine=False,
        # Dense/MoE layer control
        first_k_dense_replace=1,
        output_router_logits=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        rope_theta = kwargs.pop("rope_theta", 11158840.0)
        if rope_parameters is None:
            rope_parameters = {"rope_type": "default", "rope_theta": rope_theta}
        self.rope_parameters = rope_parameters
        self.qk_norm = qk_norm
        self.tie_word_embeddings = tie_word_embeddings
        self.enable_lm_head_fp32 = enable_lm_head_fp32
        self.enable_attention_fp32_softmax = enable_attention_fp32_softmax

        # MoE specific
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_shared_experts = num_shared_experts
        self.expert_hidden_dim = expert_hidden_dim
        self.moe_router_enable_expert_bias = moe_router_enable_expert_bias
        self.moe_router_use_sigmoid = moe_router_use_sigmoid
        self.route_norm = route_norm
        self.use_grouped_mm = use_grouped_mm
        self.router_scaling_factor = router_scaling_factor
        self.enable_moe_fp32_combine = enable_moe_fp32_combine

        # Dense/MoE layer control
        self.first_k_dense_replace = first_k_dense_replace
        self.output_router_logits = output_router_logits

        if eos_token_id is not None and isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
