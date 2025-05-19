# SPDX-License-Identifier: Apache-2.0

# yapf: disable
# ruff: noqa: E501
# coding=utf-8
# Copied from
# https://huggingface.co/Snowflake/snowflake-arctic-instruct/blob/main/configuration_arctic.py
""" Arctic model configuration"""

from dataclasses import asdict, dataclass
from typing import Any

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

ARCTIC_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "arctic": "https://huggingface.co/Snowflake/snowflake-arctic-instruct/tree/main/config.json",
}


@dataclass
class ArcticLoRAConfig:
    lora_r: int = 64
    lora_alpha: float = 16
    shard_base_weights: bool = False


@dataclass
class ArcticQuantizationConfig:
    q_bits: int = 8
    rounding: str = "nearest"
    mantissa_bits: int = 3
    group_size: int = 128


class ArcticConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ArcticModel`]. It is used to instantiate an
    Arctic model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the #TODO(rsamdani): add what model has the default config..


    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Arctic model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`ArcticModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 14336):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `8`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to `4096*32`):
            The maximum sequence length that this model might ever be used with. Arctic's sliding window attention
            allows sequence of up to 4096*32 tokens.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            The id of the padding token.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the "beginning-of-sequence" token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the "end-of-sequence" token.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 1000000.0):
            The base period of the RoPE embeddings.
        sliding_window (`int`, *optional*):
            Sliding window attention window size. If not specified, will default to `4096`.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        num_experts_per_tok (`int`, *optional*, defaults to 2):
            The number of experts to root per-token, can be also interpreted as the `top-p` routing
            parameter
        num_local_experts (`int`, *optional*, defaults to 8):
            Number of experts per Sparse MLP layer.
        router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
            The aux loss factor for the total loss.

    ```python
    >>> from transformers import ArcticModel, ArcticConfig

    >>> # Initializing a Arctic 7B style configuration TODO(rsamdani): verify which model does the default configuration correspond to.
    >>> configuration = ArcticConfig()

    >>> # Initializing a model from the Arctic 7B style configuration
    >>> model = ArcticModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "arctic"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=1e6,
        sliding_window=None,
        attention_dropout=0.0,
        num_experts_per_tok=1,
        num_local_experts=8,
        router_aux_loss_coef=0.001,
        moe_layer_frequency=2,
        parallel_attn_mlp_res=False,
        moe_train_capacity_factor=1,
        moe_eval_capacity_factor=1,
        enable_expert_tensor_parallelism=False,
        moe_min_capacity=0,
        moe_token_dropping=True,
        quantization=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout

        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.router_aux_loss_coef = router_aux_loss_coef
        self.moe_layer_frequency = moe_layer_frequency
        self.moe_train_capacity_factor = moe_train_capacity_factor
        self.moe_eval_capacity_factor = moe_eval_capacity_factor
        self.enable_expert_tensor_parallelism = enable_expert_tensor_parallelism
        self.moe_min_capacity = moe_min_capacity
        self.moe_token_dropping = moe_token_dropping
        self.parallel_attn_mlp_res = parallel_attn_mlp_res
        if isinstance(quantization, dict):
            self.quantization = ArcticQuantizationConfig(**quantization)
        else:
            self.quantization = quantization

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs) -> "ArcticConfig":
        result = super().from_dict(config_dict, **kwargs)
        config = result[0] if isinstance(result, tuple) else result
        if isinstance(config.quantization, dict):
            config.quantization = ArcticQuantizationConfig(**config.quantization)
        return result

    def to_dict(self) -> dict[str, Any]:
        ret = super().to_dict()
        if isinstance(ret["quantization"], ArcticQuantizationConfig):
            ret["quantization"] = asdict(ret["quantization"])
        return ret
