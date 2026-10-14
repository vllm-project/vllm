# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for the normalized latent recurrent Dory architecture."""

from huggingface_hub.dataclasses import strict
from transformers.configuration_utils import PreTrainedConfig


@strict
class DoryConfig(PreTrainedConfig):
    model_type = "dory"
    attribute_map = {"layer_types": "layers_block_type"}

    dory_format_version: int = 1
    vocab_size: int = 131072
    hidden_size: int = 2560
    num_hidden_layers: int = 72
    intermediate_size: int = 7168
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    max_position_embeddings: int = 8192
    tie_word_embeddings: bool = False
    hidden_act: str = "silu"
    # Also normalizes the learned nGPT scales at inference time.
    initializer_range: float = 0.02
    sliding_window: int | None = 1024
    rope_parameters: dict | None = None
    layers_block_type: list[str] | None = None
    ngpt: bool = True
    fngpt: bool = True
    ngpt_alpha_init: float = 0.05
    ngpt_enable_final_norm: bool = False
    attention_output_gate: bool = True
    attention_output_gate_logit_scale_init: float | None = None
    mlp_bias: bool = False
    attention_bias: bool = False
    n_input_layers: int = 0
    n_recurrent_layers: int | None = None
    n_output_layers: int = 0
    n_recurrent_loops: int = 1
    # "last_loop" reuses one cache, overwritten by each recurrent loop.
    recurrent_kv_cache_mode: str = "per_loop"
    initial_state: str = "input_copy"
    input_transform: str = "residual"
    output_transform: str = "replace"

    def __post_init__(self, **kwargs):
        if self.layers_block_type is None:
            self.layers_block_type = [
                "mlp"
                if i % 2
                else "full_attention"
                if i % 8 == 6
                else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]
        if self.rope_parameters is None:
            self.rope_parameters = {
                kind: {
                    "rope_type": "default",
                    "rope_theta": 10000,
                    "partial_rotary_factor": 1.0,
                }
                for kind in ("sliding_attention", "full_attention")
            }
        if self.n_recurrent_layers is None:
            self.n_recurrent_layers = (
                self.num_hidden_layers - self.n_input_layers - self.n_output_layers
            )
        super().__post_init__(**kwargs)

    def validate_layer_type(self):
        # Each physical block is attention-only or MLP-only.
        valid_types = {"sliding_attention", "full_attention", "mlp"}
        if not isinstance(self.layers_block_type, list) or any(
            not isinstance(kind, str) or kind not in valid_types
            for kind in self.layers_block_type
        ):
            raise ValueError(
                "layers_block_type must be a list of sliding_attention, "
                "full_attention, or mlp"
            )
        if len(self.layers_block_type) != self.num_hidden_layers:
            raise ValueError("layers_block_type must have num_hidden_layers entries")

    def validate_architecture(self):
        if self.dory_format_version != 1:
            raise ValueError("Only Dory format version 1 is supported")
        self.validate_layer_type()
        if self.layers_block_type is None or self.n_recurrent_layers is None:
            raise ValueError("Dory layer layout must be initialized")
        partitions = (
            self.n_input_layers,
            self.n_recurrent_layers,
            self.n_output_layers,
        )
        if (
            any(type(n) is not int or n < 0 for n in partitions)
            or sum(partitions) != self.num_hidden_layers
        ):
            raise ValueError(
                "Dory input, recurrent, and output layers must sum to num_hidden_layers"
            )
        if type(self.n_recurrent_loops) is not int or self.n_recurrent_loops < 1:
            raise ValueError("n_recurrent_loops must be a positive integer")
        if self.recurrent_kv_cache_mode not in ("per_loop", "last_loop"):
            raise ValueError("recurrent_kv_cache_mode must be per_loop or last_loop")
        if any(
            type(n) is not int or n <= 0
            for n in (self.head_dim, self.num_attention_heads, self.num_key_value_heads)
        ):
            raise ValueError("Attention dimensions must be positive integers")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.initializer_range <= 0:
            raise ValueError("initializer_range must be positive")
        if "sliding_attention" in self.layers_block_type and (
            type(self.sliding_window) is not int or self.sliding_window <= 0
        ):
            raise ValueError("sliding_window must be a positive integer")
        attention_types = set(self.layers_block_type) - {"mlp"}
        if not isinstance(self.rope_parameters, dict) or (
            set(self.rope_parameters) - {"sliding_attention", "full_attention"}
            or attention_types - self.rope_parameters.keys()
        ):
            raise ValueError("rope_parameters must define each attention block type")
        for rope in self.rope_parameters.values():
            if (
                not isinstance(rope, dict)
                or rope.get("rope_type") != "default"
                or rope.get("rope_theta", 0) <= 0
            ):
                raise ValueError(
                    "Dory requires default RoPE with a positive rope_theta"
                )
            fraction = rope.get("partial_rotary_factor", 1.0)
            rotary_dim = int(self.head_dim * fraction)
            if not 0 < fraction <= 1 or rotary_dim <= 0 or rotary_dim % 2:
                raise ValueError(
                    "Rotary dimensions must be positive, even, and <= head_dim"
                )
        if not self.ngpt or not self.fngpt or self.ngpt_enable_final_norm:
            raise ValueError("Dory requires ngpt=True, fngpt=True, and no final norm")
        if not self.attention_output_gate or self.attention_bias or self.mlp_bias:
            raise ValueError("Dory requires gated attention and bias-free projections")
        if self.hidden_act != "silu" or self.tie_word_embeddings:
            raise ValueError("Dory requires SiLU and untied embeddings")
        if self.initial_state != "input_copy" or self.input_transform != "residual":
            raise ValueError(
                "Dory requires input_copy initialization and residual input transform"
            )
        if self.output_transform != "replace":
            raise ValueError("Dory requires the replace output transform")
