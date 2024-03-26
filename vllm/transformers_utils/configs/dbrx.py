# coding=utf-8
# Copyright 2023 HuggingFace Inc. team and Dbrx team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DBRX configuration"""
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from typing import Optional, Any

logger = logging.get_logger(__name__)

DBRX_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class DbrxAttentionConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`DbrxAttention`] class. It is used to instantiate
    attention layers according to the specified arguments, defining the layers architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        attn_pdrop (`float`, *optional*, defaults to 0.0):
            The dropout probability for the attention layers.
        clip_qkv (`float`, *optional*, defaults to None):
            If not `None`, clip the queries, keys, and values in the attention layer to this value.
        kv_n_heads (Optional[int]): For grouped_query_attention only, allow user to specify number of kv heads.
        rope_theta (float): The base frequency for rope.
    """

    def __init__(
        self,
        attn_pdrop: float = 0,
        clip_qkv: Optional[float] = None,
        kv_n_heads: int = 1,
        rope_theta: float = 10000.0,
        **kwargs: Any,
    ):
        super().__init__()
        self.attn_pdrop = attn_pdrop
        self.clip_qkv = clip_qkv
        self.kv_n_heads = kv_n_heads
        self.rope_theta = rope_theta

        for k in ['model_type']:
            if k in kwargs:
                kwargs.pop(k)
        if len(kwargs) != 0:
            raise ValueError(f"Found unknown {kwargs=}")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str,
                        **kwargs: Any) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs)

        if config_dict.get("model_type") == "dbrx":
            config_dict = config_dict["attn_config"]

        if "model_type" in config_dict and hasattr(
                cls,
                "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class DbrxFFNConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`DbrxFFN`] class. It is used to instantiate
    feedforward layers according to the specified arguments, defining the layers architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:

    """

    def __init__(
        self,
        ffn_act_fn: str = 'silu',
        ffn_hidden_size: int = 3584,
        moe_num_experts: int = 4,
        moe_top_k: int = 1,
        moe_jitter_eps: Optional[float] = None,
        moe_loss_weight: float = 0.01,
        moe_normalize_expert_weights: Optional[float] = 1,
        uniform_expert_assignment: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        self.ffn_act_fn = ffn_act_fn
        self.ffn_hidden_size = ffn_hidden_size
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.moe_jitter_eps = moe_jitter_eps
        self.moe_loss_weight = moe_loss_weight
        self.moe_normalize_expert_weights = moe_normalize_expert_weights
        self.uniform_expert_assignment = uniform_expert_assignment

        for k in ['model_type']:
            if k in kwargs:
                kwargs.pop(k)
        if len(kwargs) != 0:
            raise ValueError(f"Found unknown {kwargs=}")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str,
                        **kwargs: Any) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs)

        if config_dict.get("model_type") == "dbrx":
            config_dict = config_dict["ffn_config"]

        if "model_type" in config_dict and hasattr(
                cls,
                "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class DbrxConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`DbrxModel`]. It is used to instantiate a Dbrx model
    according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        d_model (`int`, *optional*, defaults to 6144):
            Dimensionality of the embeddings and hidden states.
        n_heads (`int`, *optional*, defaults to 48):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_layers (`int`, *optional*, defaults to 40):
            Number of hidden layers in the Transformer encoder.
        max_seq_len (`int`, *optional*, defaults to 32768):
            The maximum sequence length of the model.
        vocab_size (`int`, *optional*, defaults to 100352):
            Vocabulary size of the Dbrx model. Defines the maximum number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`DbrxModel`].
        resid_pdrop (`float`, *optional*, defaults to 0.0):
            The dropout probability applied to the attention output before combining with residual.
        emb_pdrop (`float`, *optional*, defaults to 0.0):
            The dropout probability for the embedding layer.
        attn_config (`dict`, *optional*):
            A dictionary used to configure the model's attention module.
        ffn_config (`dict`, *optional*):
            A dictionary used to configure the model's FFN module.
        use_cache (`bool`, *optional*, defaults to `False`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        pad_token_id (`int`, *optional*): # TODO
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1): # TODO
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2): # TODO
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.


    Example:

    ```python
    >>> from transformers import DbrxConfig, DbrxModel

    >>> # Initializing a Dbrx configuration
    >>> configuration = DbrxConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = DbrxModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "dbrx"
    attribute_map = {
        "num_attention_heads": "n_heads",
        "hidden_size": "d_model",
        "num_hidden_layers": "n_layers",
        "max_position_embeddings": "max_seq_len"
    }

    def __init__(
        self,
        d_model: int = 2048,
        n_heads: int = 16,
        n_layers: int = 24,
        max_seq_len: int = 2048,
        vocab_size: int = 32000,
        resid_pdrop: float = 0.0,
        emb_pdrop: float = 0.0,
        attn_config: Optional[DbrxAttentionConfig] = None,
        ffn_config: Optional[DbrxFFNConfig] = None,
        use_cache: bool = True,
        pad_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = 1,
        eos_token_id: Optional[int] = 2,
        tie_word_embeddings: bool = False,
        initializer_range: float = 0.02,
        **kwargs: Any,
    ):
        if attn_config is None:
            self.attn_config = DbrxAttentionConfig()
        elif isinstance(attn_config, dict):
            self.attn_config = DbrxAttentionConfig(**attn_config)
        else:
            self.attn_config = attn_config

        if ffn_config is None:
            self.ffn_config = DbrxFFNConfig()
        elif isinstance(ffn_config, dict):
            self.ffn_config = DbrxFFNConfig(**ffn_config)
        else:
            self.ffn_config = ffn_config

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.resid_pdrop = resid_pdrop
        self.emb_pdrop = emb_pdrop
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )