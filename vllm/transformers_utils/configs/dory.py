# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for the normalized latent recurrent Dory architecture."""

from typing import Any

from huggingface_hub.dataclasses import strict
from transformers.configuration_utils import PreTrainedConfig


@strict
class DoryConfig(PreTrainedConfig):
    model_type = "dory"
    default_theta = 10_000.0

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
    layer_types: list[str] | None = None
    ngpt_alpha_init: float = 0.05
    attention_output_gate_logit_scale_init: float | None = None
    mlp_bias: bool = False
    attention_bias: bool = False
    n_input_layers: int = 0
    n_recurrent_layers: int | None = None
    n_output_layers: int = 0
    n_recurrent_loops: int = 1
    # Requests may set their own loop count, 1..max_recurrent_loops, with
    # SamplingParams.extra_args["loops"]; n_recurrent_loops is the default. The
    # per-loop KV cache is sized for max_recurrent_loops (None: n_recurrent_loops,
    # resolved at use so an hf_overrides change of n_recurrent_loops follows).
    max_recurrent_loops: int | None = None
    # "last_loop" reuses one cache, overwritten by each recurrent loop.
    recurrent_kv_cache_mode: str = "per_loop"

    def __post_init__(self, **kwargs):
        if self.layer_types is None:
            self.layer_types = [
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
                    "partial_rotary_factor": 1.0,
                }
                for kind in ("sliding_attention", "full_attention")
                if kind in self.layer_types
            }
        if self.n_recurrent_layers is None:
            self.n_recurrent_layers = (
                self.num_hidden_layers - self.n_input_layers - self.n_output_layers
            )
        super().__post_init__(**kwargs)

    def validate_layer_type(self):
        # Each physical block is attention-only or MLP-only.
        valid_types = {"sliding_attention", "full_attention", "mlp"}
        if self.layer_types is None or any(
            kind not in valid_types for kind in self.layer_types
        ):
            raise ValueError(
                "layer_types must be a list of sliding_attention, "
                "full_attention, or mlp"
            )
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must have num_hidden_layers entries")

    def validate_architecture(self):
        if self.layer_types is None or self.n_recurrent_layers is None:
            raise ValueError("Dory layer layout must be initialized")
        partitions = (
            self.n_input_layers,
            self.n_recurrent_layers,
            self.n_output_layers,
        )
        if any(n < 0 for n in partitions) or sum(partitions) != self.num_hidden_layers:
            raise ValueError(
                "Dory input, recurrent, and output layers must sum to num_hidden_layers"
            )
        if self.n_recurrent_loops < 1:
            raise ValueError("n_recurrent_loops must be a positive integer")
        max_loops = self.max_recurrent_loops or self.n_recurrent_loops
        if max_loops < self.n_recurrent_loops:
            raise ValueError("max_recurrent_loops must be >= n_recurrent_loops")
        if (
            self.recurrent_kv_cache_mode == "last_loop"
            and max_loops != self.n_recurrent_loops
        ):
            # One shared cache per layer would mix the loops of different counts.
            raise ValueError(
                "per-request loop counts require recurrent_kv_cache_mode=per_loop"
            )
        if self.recurrent_kv_cache_mode not in ("per_loop", "last_loop"):
            raise ValueError("recurrent_kv_cache_mode must be per_loop or last_loop")
        if any(
            n <= 0
            for n in (self.head_dim, self.num_attention_heads, self.num_key_value_heads)
        ):
            raise ValueError("Attention dimensions must be positive integers")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.initializer_range <= 0:
            raise ValueError("initializer_range must be positive")
        if "sliding_attention" in self.layer_types and (
            self.sliding_window is None or self.sliding_window <= 0
        ):
            raise ValueError("sliding_window must be a positive integer")
        if self.attention_bias or self.mlp_bias:
            raise ValueError("Dory requires bias-free projections")
        if self.hidden_act != "silu" or self.tie_word_embeddings:
            raise ValueError("Dory requires SiLU and untied embeddings")

    def validate_rope(self):
        if self.layer_types is None:
            raise ValueError("Dory layer layout must be initialized")
        attention_types = set(self.layer_types) - {"mlp"}
        if self.rope_parameters is None or (
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

    def request_cache_salt(
        self, extra_args: dict[str, Any], cache_salt: str | None
    ) -> str | None:
        """Validate a request's loop count (``extra_args["loops"]``) and return its
        prefix-cache salt. KV computed with one loop count is only valid for that
        count, so requests with a non-default count get their own salt."""
        raw = extra_args.get("loops")
        if raw is None:
            return cache_salt
        max_loops = self.max_recurrent_loops or self.n_recurrent_loops
        loops = raw if isinstance(raw, int) and not isinstance(raw, bool) else None
        if isinstance(raw, str) and raw.strip().isdigit():
            loops = int(raw)
        if loops is None or not 1 <= loops <= max_loops:
            raise ValueError(
                f"loops must be an integer from 1 to {max_loops} "
                f"(default {self.n_recurrent_loops}), got {raw!r}"
            )
        extra_args["loops"] = loops
        if loops == self.n_recurrent_loops:
            return cache_salt
        tag = f"dory-loops={loops}"
        return f"{cache_salt}|{tag}" if cache_salt else tag
