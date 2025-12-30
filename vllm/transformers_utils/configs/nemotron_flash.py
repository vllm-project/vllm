# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2024 HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
"""NemotronFlash model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class NemotronFlashConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`NemotronFlashModel`]. It is used to instantiate a NemotronFlash model
    according to the specified arguments, defining the model architecture.

    NemotronFlash is a hybrid architecture combining:
    - Sequential attention layers
    - Mamba2 layers
    - DeltaNet layers
    - Meta tokens for enhanced context

    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the NemotronFlash model.
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 5632):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the model.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key_value heads for Grouped Query Attention.
        head_dim (`int`, *optional*, defaults to 128):
            Dimension of each attention head.
        max_position_embeddings (`int`, *optional*, defaults to 8192):
            The maximum sequence length that this model might ever be used with.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the RMSNorm layers.
        num_meta_tokens (`int`, *optional*, defaults to 0):
            Number of learnable meta tokens prepended to each sequence.
            These tokens provide additional context for the model.
        num_memory_tokens (`int`, *optional*, defaults to 0):
            Alias for num_meta_tokens (used by some models).
        layer_types (`list`, *optional*):
            List defining layer types for each layer.

        # Mamba2 parameters
        mamba_num_heads (`int`, *optional*, defaults to 64):
            Number of heads in Mamba2 layers.
        mamba_head_dim (`int`, *optional*, defaults to 64):
            Dimension of each Mamba2 head.
        mamba_n_groups (`int`, *optional*, defaults to 8):
            Number of groups in Mamba2 layers.
        ssm_state_size (`int`, *optional*, defaults to 128):
            The dimension of the mamba state space latents.
        d_conv (`int`, *optional*, defaults to 4):
            The size of the mamba convolution kernel.
        mamba_hidden_act (`str`, *optional*, defaults to "silu"):
            The non-linear activation function in Mamba layers.
        use_conv_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in Mamba convolution layers.
        use_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in Mamba projection layers.

        # DeltaNet parameters
        linear_num_key_heads (`int`, *optional*):
            Number of key heads for DeltaNet layers.
        linear_num_value_heads (`int`, *optional*):
            Number of value heads for DeltaNet layers.
        linear_key_head_dim (`int`, *optional*):
            Dimension of each DeltaNet key head.
        linear_value_head_dim (`int`, *optional*):
            Dimension of each DeltaNet value head.
        linear_conv_kernel_dim (`int`, *optional*, defaults to 4):
            Convolution kernel size for DeltaNet.

        # Sequential Attention parameters
        sequential_attention_type (`str`, *optional*, defaults to "causal"):
            Type of sequential attention mechanism.
        attention_window_size (`int`, *optional*):
            Window size for sequential attention (if applicable).
    """

    model_type = "nemotron_flash"
    is_composition = False
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=2048,
        intermediate_size=5632,
        num_hidden_layers=32,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=8192,
        rms_norm_eps=1e-5,
        initializer_range=0.02,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        # Meta tokens
        num_meta_tokens=0,
        num_memory_tokens=None,  # Alias for num_meta_tokens
        # Hybrid pattern
        hybrid_layer_pattern=None,
        layer_types=None,  # Alternative to hybrid_layer_pattern
        # Mamba2 parameters
        mamba_num_heads=None,  # Will be calculated from d_inner / head_dim
        mamba_head_dim=64,
        mamba2_headdim=None,  # Alternative name
        mamba_n_groups=1,
        ssm_state_size=128,
        mamba_d_state=None,
        conv_kernel=4,
        d_conv=None,  # Alternative name
        mamba_d_conv=None,  # Another alternative
        mamba_expand=2,  # Default value
        mamba_hidden_act="silu",
        mamba_dt_rank=None,
        mamba_dt_min=0.001,
        mamba_dt_max=0.1,
        mamba_dt_init_floor=1e-4,
        mamba_conv_bias=None,  # Alternative name
        use_conv_bias=True,
        mamba_proj_bias=None,  # Alternative name
        use_bias=False,
        # DeltaNet parameters
        linear_num_key_heads=None,
        linear_num_value_heads=None,
        linear_key_head_dim=None,
        linear_value_head_dim=None,
        linear_conv_kernel_dim=4,
        # Sequential Attention parameters
        sequential_attention_type="causal",
        attention_window_size=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        # Meta tokens - handle both num_meta_tokens and num_memory_tokens
        if num_memory_tokens is not None:
            self.num_meta_tokens = num_memory_tokens
            self.num_memory_tokens = num_memory_tokens
        else:
            self.num_meta_tokens = num_meta_tokens
            self.num_memory_tokens = num_meta_tokens

        # Hybrid architecture - handle both layer_types and hybrid_layer_pattern
        self.layer_types = layer_types
        self.hybrid_layer_pattern = hybrid_layer_pattern or (
            ("*" * num_hidden_layers) if layer_types is None else None
        )

        # Mamba2 configuration - handle alternative parameter names
        self.mamba_num_heads = mamba_num_heads
        self.mamba_head_dim = (
            mamba2_headdim if mamba2_headdim is not None else mamba_head_dim
        )
        self.mamba2_headdim = self.mamba_head_dim
        self.n_groups = mamba_n_groups
        self.mamba_n_groups = mamba_n_groups
        self.ssm_state_size = (
            mamba_d_state if mamba_d_state is not None else ssm_state_size
        )
        self.mamba_d_state = self.ssm_state_size

        # Handle d_conv / conv_kernel / mamba_d_conv
        if d_conv is not None:
            self.conv_kernel = d_conv
        elif mamba_d_conv is not None:
            self.conv_kernel = mamba_d_conv
        else:
            self.conv_kernel = conv_kernel
        self.d_conv = self.conv_kernel
        self.mamba_d_conv = self.conv_kernel

        self.mamba_expand = mamba_expand

        # Set defaults for missing mamba parameters
        if self.mamba_num_heads is None:
            # Default: d_inner / head_dim = (hidden_size * expand) / head_dim
            d_inner = int(self.hidden_size * self.mamba_expand)  # 6144
            self.mamba_num_heads = d_inner // self.mamba_head_dim  # 6144/64=96

        self.mamba_hidden_act = mamba_hidden_act
        self.mamba_dt_rank = mamba_dt_rank or (self.hidden_size // 16)
        self.mamba_dt_min = mamba_dt_min
        self.mamba_dt_max = mamba_dt_max
        self.mamba_dt_init_floor = mamba_dt_init_floor

        # Handle conv_bias / mamba_conv_bias
        if mamba_conv_bias is not None:
            self.use_conv_bias = mamba_conv_bias
        else:
            self.use_conv_bias = use_conv_bias
        self.mamba_conv_bias = self.use_conv_bias

        # Handle proj_bias / mamba_proj_bias
        if mamba_proj_bias is not None:
            self.use_bias = mamba_proj_bias
        else:
            self.use_bias = use_bias
        self.mamba_proj_bias = self.use_bias

        # DeltaNet configuration - provide defaults if not specified
        self.linear_num_key_heads = (
            linear_num_key_heads
            if linear_num_key_heads is not None
            else num_attention_heads
        )
        self.linear_num_value_heads = (
            linear_num_value_heads
            if linear_num_value_heads is not None
            else num_attention_heads
        )
        self.linear_key_head_dim = (
            linear_key_head_dim if linear_key_head_dim is not None else head_dim
        )
        self.linear_value_head_dim = (
            linear_value_head_dim
            if linear_value_head_dim is not None
            else (linear_key_head_dim if linear_key_head_dim is not None else head_dim)
        )
        self.linear_conv_kernel_dim = linear_conv_kernel_dim

        # Sequential Attention configuration
        self.sequential_attention_type = sequential_attention_type
        self.attention_window_size = attention_window_size

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @classmethod
    def from_pretrained_hf_config(
        cls, hf_config: "PretrainedConfig"
    ) -> "NemotronFlashConfig":
        """
        Convert a HuggingFace config (like FastSLMConfig) to NemotronFlashConfig.
        Normalizes configs to provide a clean, single interface.

        Key normalizations:
        - num_memory_tokens (HF) → num_meta_tokens (vLLM)
        - Adds layers_block_type property for layer pattern parsing
        - Standardizes all mamba/attention/deltanet config attributes
        - Provides sensible defaults for missing DeltaNet parameters
        """
        # Extract all config dict
        config_dict = hf_config.to_dict()

        # Normalize attribute names that differ between HF and vLLM
        # HF uses num_memory_tokens, we use num_meta_tokens
        if "num_memory_tokens" in config_dict and "num_meta_tokens" not in config_dict:
            config_dict["num_meta_tokens"] = config_dict["num_memory_tokens"]

        # Provide sensible defaults for DeltaNet parameters if not present
        if (
            "linear_num_key_heads" not in config_dict
            or config_dict.get("linear_num_key_heads") is None
        ):
            # Default to num_attention_heads (same as value heads for DeltaNet/GLA)
            config_dict["linear_num_key_heads"] = config_dict.get(
                "num_attention_heads", 24
            )

        if (
            "linear_num_value_heads" not in config_dict
            or config_dict.get("linear_num_value_heads") is None
        ):
            # Default to num_attention_heads
            config_dict["linear_num_value_heads"] = config_dict.get(
                "num_attention_heads", 24
            )

        if (
            "linear_key_head_dim" not in config_dict
            or config_dict.get("linear_key_head_dim") is None
        ):
            # Default to hidden_size // num_attention_heads
            head_dim = config_dict.get("hidden_size", 3072) // config_dict.get(
                "num_attention_heads", 24
            )
            config_dict["linear_key_head_dim"] = head_dim

        if (
            "linear_value_head_dim" not in config_dict
            or config_dict.get("linear_value_head_dim") is None
        ):
            # Default to same as key head dim
            config_dict["linear_value_head_dim"] = config_dict["linear_key_head_dim"]

        if (
            "linear_conv_kernel_dim" not in config_dict
            or config_dict.get("linear_conv_kernel_dim") is None
        ):
            # Default to 4 (same as mamba_d_conv)
            config_dict["linear_conv_kernel_dim"] = config_dict.get("mamba_d_conv", 4)

        # Sensible defaults for Mamba2 parameters if not present
        if (
            "mamba_num_heads" not in config_dict
            or config_dict.get("mamba_num_heads") is None
        ):
            # NOTE: Nemotron-Flash-3B checkpoint structure:
            # - in_proj: 12640 = 2 * 6320
            # - conv1d: 6400 (slightly more padding)
            # - norm, out_proj: 6144
            # - A_log, D, dt_bias: 96 heads
            #
            # The checkpoint uses 96 heads with different padding schemes:
            # - Core d_inner = 96 * 64 = 6144 (for norm, out_proj)
            # - in_proj uses 6320 per part (12640 total)
            # - conv1d uses 6400 (extra padding)
            #
            # We'll use 96 heads (6144) and let weight loaders handle padding
            mamba2_headdim = config_dict.get(
                "mamba2_headdim", config_dict.get("mamba_head_dim", 64)
            )

            # Use 96 heads to match A_log, D, dt_bias from checkpoint
            config_dict["mamba_num_heads"] = 96
            d_inner = 96 * mamba2_headdim  # 6144

            hidden_size = config_dict.get("hidden_size", 3072)
            config_dict["mamba_expand"] = d_inner / hidden_size  # 2.0

        if (
            "mamba_head_dim" not in config_dict
            or config_dict.get("mamba_head_dim") is None
        ):
            config_dict["mamba_head_dim"] = config_dict.get("mamba2_headdim", 64)

        if (
            "ssm_state_size" not in config_dict
            or config_dict.get("ssm_state_size") is None
        ):
            config_dict["ssm_state_size"] = 128

        if (
            "mamba_n_groups" not in config_dict
            or config_dict.get("mamba_n_groups") is None
        ):
            # Nemotron-Flash-3B checkpoint uses n_groups=1 (no conv1d merging)
            config_dict["mamba_n_groups"] = 1

        if "n_groups" not in config_dict or config_dict.get("n_groups") is None:
            config_dict["n_groups"] = config_dict.get("mamba_n_groups", 1)

        # Fix mamba_expand to match checkpoint (6400 / 3072 ≈ 2.0833)
        # The checkpoint has d_inner=6400, not 6144 (which would be expand=2.0)
        # Force override to match actual checkpoint dimensions
        hidden_size = config_dict.get("hidden_size", 3072)
        # Checkpoint has d_inner=6400, config incorrectly says expand=2
        config_dict["mamba_expand"] = 6400 / hidden_size  # ≈ 2.0833

        # Recalculate mamba_num_heads based on actual d_inner
        # Force recalculation to match checkpoint
        mamba_expand = config_dict.get("mamba_expand")
        d_inner = int(hidden_size * mamba_expand)  # 6400
        mamba2_headdim = config_dict.get("mamba2_headdim") or config_dict.get(
            "mamba_head_dim", 64
        )
        config_dict["mamba_num_heads"] = d_inner // mamba2_headdim  # 6400 / 64 = 100

        if "conv_kernel" not in config_dict or config_dict.get("conv_kernel") is None:
            config_dict["conv_kernel"] = config_dict.get(
                "mamba_d_conv", config_dict.get("d_conv", 4)
            )

        # Create our config from the normalized dict
        return cls(**config_dict)

    @property
    def layers_block_type(self):
        """
        Map layer types to block types.

        Supports two formats:
        1. layer_types list: ["deltanet", "f", "m2", "a", ...]
        2. hybrid_layer_pattern string: "M-D-*-M-D-*"

        Maps to vLLM block types: mamba, deltanet, attention, mlp
        """
        # Priority 1: Use layer_types if available (nvidia/Nemotron-Flash-3B format)
        if self.layer_types is not None:
            result = []
            for layer_type in self.layer_types:
                if layer_type in ["deltanet"]:
                    result.append("deltanet")
                elif layer_type in ["m2", "m", "mamba"]:
                    result.append("mamba")
                elif layer_type in ["a", "attention", "full_attention"]:
                    result.append("attention")
                elif layer_type in ["mlp", "-"]:
                    result.append("mlp")
                elif layer_type in ["f"]:
                    result.append("f")
                else:
                    # Default to attention for unknown types
                    result.append("attention")
            return result

        # Priority 2: Use hybrid_layer_pattern if available
        if self.hybrid_layer_pattern is not None:
            type_map = {
                "M": "mamba",
                "D": "deltanet",
                "*": "attention",
                "-": "mlp",
            }
            return [
                type_map.get(self.hybrid_layer_pattern[i], "attention")
                for i in range(self.num_hidden_layers)
            ]

        # Fallback: All attention layers
        return ["attention"] * self.num_hidden_layers


# Alias for compatibility with nvidia/Nemotron-Flash-3B which uses FastSLMConfig
FastSLMConfig = NemotronFlashConfig
