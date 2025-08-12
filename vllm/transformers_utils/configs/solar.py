# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Solar model configuration"""

from transformers import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class SolarConfig(PretrainedConfig):
    r"""
    This is the configuration class to store
    the configuration of a [`SolarModel`].
    It is used to instantiate an LLaMA model
    according to the specified arguments,
    defining the model architecture.
    Instantiating a configuration with the
    defaults will yield a similar
    configuration to that of the LLaMA-7B.
    Configuration objects inherit from [`PretrainedConfig`]
    and can be used to control the model outputs.
    Read the documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the LLaMA model.
            Defines the number of different tokens
            that can be represented by the `inputs_ids`
            passed when calling [`SolarModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer
            in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that
            should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`,
            the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model
            will use Multi Query Attention (MQA)
            otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint,
            each group key and value head should be constructed
            by meanpooling all the original heads within that group.
            For more details checkout [this paper]
            (https://arxiv.org/pdf/2305.13245.pdf).
            If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string)
            in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with.
            Solar 1 supports up to 2048 tokens,
            Solar 2 up to 4096, CodeSolar up to 16384.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of
            the truncated_normal_initializer for initializing
            all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return
            the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank
            used during pretraining.
            Please refer to [this
            document](https://huggingface.co/docs/
            transformers/main/
            perf_train_gpu_many#tensor-parallelism)
             to understand more about it. This value is
            necessary to ensure exact reproducibility
            of the pretraining results.
            Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for
            the RoPE embeddings.
            Currently supports two scaling
            strategies: linear and dynamic.
            Their scaling factor must be a float greater than 1.
            The expected format is
            `{"type": strategy name, "factor": scaling factor}`.
            When using this flag, don't update
            `max_position_embeddings` to the expected new maximum.
            See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/
            dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking
            API changes in future versions.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value
            and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in up_proj, down_proj and gate_proj
            layers in the MLP layers.
        sliding_window (`int`, *optional*, defaults to 2047):
            Sliding window attention window size. If not specified,
            will default to `2047`.
    ```python
    >>> from transformers import SolarModel, SolarConfig
    >>> # Initializing a Solar-pro style configuration
    >>> configuration = SolarConfig()
    >>> # Initializing a model from the Solar-pro style configuration
    >>> model = SolarModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "solar"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        sliding_window=2047,
        bskcn_1=None,
        bskcn_2=None,
        bskcn_3=None,
        bskcn_4=None,
        bskcn_tv=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.sliding_window = sliding_window
        self.bskcn_1 = bskcn_1 if bskcn_1 is not None else [12, 20, 32, 44]
        self.bskcn_2 = bskcn_2 if bskcn_2 is not None else [20, 32]
        self.bskcn_3 = bskcn_3 if bskcn_3 is not None else [16, 24, 36, 48]
        self.bskcn_4 = bskcn_4 if bskcn_4 is not None else [28, 40]
        self.bskcn_tv = bskcn_tv if bskcn_tv is not None else [0.9, 0.8]

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if (not isinstance(self.rope_scaling, dict)
                or len(self.rope_scaling) != 2):
            raise ValueError(
                "`rope_scaling` must be a dictionary with two fields,"
                " `type` and `factor`, "
                f"got {self.rope_scaling}")
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in [
                "linear",
                "dynamic",
        ]:
            raise ValueError(f"`rope_scaling`'s type field must be one of "
                             f"['linear', 'dynamic'], got {rope_scaling_type}")
        if (rope_scaling_factor is None
                or not isinstance(rope_scaling_factor, float)
                or rope_scaling_factor <= 1.0):
            raise ValueError(
                f"`rope_scaling`'s factor field must be a float > 1,"
                f" got {rope_scaling_factor}")
