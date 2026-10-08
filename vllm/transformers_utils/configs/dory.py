# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for the normalized latent recurrent Dory architecture."""

from transformers.configuration_utils import PreTrainedConfig


class DoryConfig(PreTrainedConfig):
    model_type = "dory"

    def __init__(
        self,
        dory_format_version=1,
        vocab_size=131072,
        hidden_size=2560,
        num_hidden_layers=72,
        intermediate_size=7168,
        num_attention_heads=16,
        num_key_value_heads=4,
        attention_head_dim=256,
        max_position_embeddings=8192,
        tie_word_embeddings=False,
        hybrid_layer_pattern=None,
        swa_window_size=1024,
        swa_num_attention_heads=None,
        swa_num_key_value_heads=None,
        swa_head_dim=None,
        rope_theta=10000,
        partial_rotary_factor=1.0,
        rope_theta_2=None,
        partial_rotary_factor_2=None,
        rope_profile_layers=None,
        no_rope_layers=None,
        ngpt=True,
        fngpt=True,
        init_method_std=0.02,
        ngpt_alpha_init=0.05,
        ngpt_enable_final_norm=False,
        attention_output_gate=True,
        attention_output_gate_logit_scale_init=None,
        mlp_hidden_act="silu",
        mlp_bias=False,
        attention_bias=False,
        n_input_layers=0,
        n_recurrent_layers=None,
        n_output_layers=0,
        n_recurrent_loops=1,
        recurrent_kv_cache_mode="per_loop",
        initial_state="input_copy",
        input_transform="residual",
        output_transform="replace",
        mixer_type=None,
        **kwargs,
    ):
        self.dory_format_version = dory_format_version
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.attention_head_dim = attention_head_dim
        self.head_dim = attention_head_dim
        self.max_position_embeddings = max_position_embeddings
        self.hybrid_layer_pattern = (
            "S-S-S-*-" * (num_hidden_layers // 8)
            if hybrid_layer_pattern is None
            else hybrid_layer_pattern
        )
        self.swa_window_size = swa_window_size
        self.swa_num_attention_heads = swa_num_attention_heads
        self.swa_num_key_value_heads = swa_num_key_value_heads
        self.swa_head_dim = swa_head_dim
        self.rope_theta = rope_theta
        self.partial_rotary_factor = partial_rotary_factor
        # Per-layer RoPE profile: 1 = rope_theta, 2 = rope_theta_2.
        # Absent in legacy single-RoPE artifacts, which are all profile 1.
        self.rope_profile_layers = (
            [1] * num_hidden_layers
            if rope_profile_layers is None
            else rope_profile_layers
        )
        self.rope_theta_2 = rope_theta if rope_theta_2 is None else rope_theta_2
        self.partial_rotary_factor_2 = (
            partial_rotary_factor
            if partial_rotary_factor_2 is None
            else partial_rotary_factor_2
        )
        self.no_rope_layers = (
            [0] * num_hidden_layers if no_rope_layers is None else no_rope_layers
        )
        self.ngpt = ngpt
        self.fngpt = fngpt
        self.init_method_std = init_method_std
        self.ngpt_alpha_init = ngpt_alpha_init
        self.ngpt_enable_final_norm = ngpt_enable_final_norm
        self.attention_output_gate = attention_output_gate
        self.attention_output_gate_logit_scale_init = (
            attention_output_gate_logit_scale_init
        )
        self.mlp_hidden_act = mlp_hidden_act
        self.mlp_bias = mlp_bias
        self.attention_bias = attention_bias
        self.n_input_layers = n_input_layers
        self.n_recurrent_layers = (
            num_hidden_layers - n_input_layers - n_output_layers
            if n_recurrent_layers is None
            else n_recurrent_layers
        )
        self.n_output_layers = n_output_layers
        self.n_recurrent_loops = n_recurrent_loops
        # Recurrent attention KV: "per_loop" keeps one cache per loop. "last_loop"
        # keeps one cache that each loop overwrites.
        self.recurrent_kv_cache_mode = recurrent_kv_cache_mode
        self.initial_state = initial_state
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.mixer_type = mixer_type
        # Older exports use attention_head_dim, rather than HF's head_dim.
        head_dim = kwargs.pop("head_dim", attention_head_dim)
        if head_dim != attention_head_dim:
            raise ValueError("head_dim and attention_head_dim must agree")
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.validate_architecture()

    def validate_architecture(self):
        if self.dory_format_version != 1:
            raise ValueError("Only Dory format version 1 is supported")
        if len(self.hybrid_layer_pattern) != self.num_hidden_layers or set(
            self.hybrid_layer_pattern
        ) - {"S", "*", "-"}:
            raise ValueError(
                "hybrid_layer_pattern must contain one S, *, or - per layer"
            )
        if min(
            self.n_input_layers, self.n_recurrent_layers, self.n_output_layers
        ) < 0 or (
            self.n_input_layers + self.n_recurrent_layers + self.n_output_layers
            != self.num_hidden_layers
        ):
            raise ValueError(
                "Dory input, recurrent, and output layers must sum to num_hidden_layers"
            )
        if not isinstance(self.n_recurrent_loops, int) or self.n_recurrent_loops < 1:
            raise ValueError("n_recurrent_loops must be a positive integer")
        if self.recurrent_kv_cache_mode not in ("per_loop", "last_loop"):
            raise ValueError("recurrent_kv_cache_mode must be per_loop or last_loop")
        for field, allowed in (
            ("rope_profile_layers", {0, 1, 2}),
            ("no_rope_layers", {0, 1}),
        ):
            values = getattr(self, field)
            if len(values) != self.num_hidden_layers or set(values) - allowed:
                raise ValueError(f"{field} must contain one valid value per layer")
        if 0 in self.rope_profile_layers or any(self.no_rope_layers):
            raise ValueError(
                "Dory does not support NoPE layers: rope_profile_layers must use "
                "profiles 1 or 2 and no_rope_layers must be all zeros"
            )
        if "S" in self.hybrid_layer_pattern and (
            self.swa_window_size is None or self.swa_window_size <= 0
        ):
            raise ValueError("S layers require a positive swa_window_size")
        for swa, full in (
            ("swa_num_attention_heads", "num_attention_heads"),
            ("swa_num_key_value_heads", "num_key_value_heads"),
            ("swa_head_dim", "attention_head_dim"),
        ):
            if getattr(self, swa) not in (None, getattr(self, full)):
                raise ValueError(
                    "Dory requires matching sliding and full attention dimensions"
                )
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.init_method_std <= 0:
            raise ValueError("init_method_std must be positive")
        for fraction in (self.partial_rotary_factor, self.partial_rotary_factor_2):
            rotary_dim = int(self.attention_head_dim * fraction)
            if not 0 < fraction <= 1 or rotary_dim <= 0 or rotary_dim % 2:
                raise ValueError(
                    "Rotary dimensions must be positive, even, and <= head_dim"
                )
        if not self.ngpt or not self.fngpt or self.ngpt_enable_final_norm:
            raise ValueError("Dory requires ngpt=True, fngpt=True, and no final norm")
        if not self.attention_output_gate or self.attention_bias or self.mlp_bias:
            raise ValueError("Dory requires gated attention and bias-free projections")
        if self.mlp_hidden_act != "silu" or self.tie_word_embeddings:
            raise ValueError("Dory requires SiLU and untied embeddings")
        if self.initial_state != "input_copy" or self.input_transform != "residual":
            raise ValueError(
                "Dory requires input_copy initialization and residual input transform"
            )
        if self.output_transform != "replace":
            raise ValueError("Dory requires the replace output transform")
        if self.mixer_type is not None:
            raise ValueError("Unsupported Dory mixer")
