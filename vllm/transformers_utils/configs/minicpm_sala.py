# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from transformers.configuration_utils import PretrainedConfig

_LIGHTNING_MIXER_TYPES = frozenset({"lightning", "lightning_attn", "lightning-attn"})
_SPARSE_MIXER_TYPE = "minicpm4"


class MiniCPMSALAConfig(PretrainedConfig):
    """Hugging Face-compatible config for MiniCPM-SALA (hybrid sparse + linear attn)."""

    model_type = "minicpm_sala"
    keys_to_ignore_at_inference = ["past_key_values"]
    # Prevent transformers' `to_diff_dict` from calling `cls()` with no
    # arguments to compute default values — our __init__ requires `mixer_types`
    # and would raise ValueError if called without it.
    has_no_defaults_at_init = True

    def __init__(
        self,
        vocab_size: int = 73448,
        hidden_size: int = 4096,
        intermediate_size: int = 16384,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 524288,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        rope_scaling: dict[str, Any] | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.1,
        use_cache: bool = True,
        scale_emb: float = 12.0,
        dim_model_base: int = 256,
        scale_depth: float = 1.4,
        mup_denominator: int = 32,
        sparse_config: dict[str, Any] | None = None,
        mixer_types: Sequence[str] | None = None,
        head_dim: int | None = 128,
        lightning_head_dim: int | None = None,
        use_output_gate: bool = True,
        use_output_norm: bool = True,
        attn_use_output_gate: bool = True,
        lightning_use_rope: bool = True,
        lightning_nkv: int | None = None,
        lightning_nh: int | None = None,
        qk_norm: bool = True,
        lightning_scale: str = "1/sqrt(d)",
        attn_use_rope: bool = False,
        rand_init: bool = False,
        pad_token_id: int | None = 2,
        bos_token_id: int | None = 1,
        eos_token_id: int | list[int] | None = None,
        tie_word_embeddings: bool = False,
        **kwargs: Any,
    ) -> None:
        rope_scaling_kw = kwargs.pop("rope_scaling", rope_scaling)
        resolved_eos = (
            eos_token_id
            if eos_token_id is not None
            else kwargs.pop("eos_token_id", [2, 73440])
        )

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=resolved_eos,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        if mixer_types is None:
            raise ValueError("MiniCPMSALAConfig requires `mixer_types`.")

        if len(mixer_types) != num_hidden_layers:
            raise ValueError(
                "len(mixer_types) must equal num_hidden_layers: "
                f"got {len(mixer_types)} vs {num_hidden_layers}"
            )

        merged_sparse = sparse_config.copy() if isinstance(sparse_config, dict) else {}

        # Match HF default: fall back to num_attention_heads if not set
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling_kw
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.scale_emb = scale_emb
        self.dim_model_base = dim_model_base
        self.scale_depth = scale_depth
        self.mup_denominator = mup_denominator
        self.sparse_config = merged_sparse if merged_sparse else (sparse_config or {})
        self.mixer_types = list(mixer_types)
        computed_head_dim = (
            hidden_size // num_attention_heads if head_dim is None else head_dim
        )
        self.head_dim = computed_head_dim
        self.lightning_head_dim = (
            lightning_head_dim if lightning_head_dim is not None else computed_head_dim
        )
        self.use_output_gate = use_output_gate
        self.use_output_norm = use_output_norm
        self.attn_use_output_gate = attn_use_output_gate
        self.lightning_use_rope = lightning_use_rope
        self.lightning_nh = lightning_nh
        self.lightning_nkv = lightning_nkv
        self.qk_norm = qk_norm
        self.lightning_scale = lightning_scale
        self.attn_use_rope = attn_use_rope
        self.rand_init = rand_init

        if self.lightning_nh is None:
            self.lightning_nh = num_attention_heads
        # Match HF: lightning_nkv falls back to num_key_value_heads (not lightning_nh)
        if self.lightning_nkv is None:
            self.lightning_nkv = num_key_value_heads

    @staticmethod
    def _is_lightning_mixer(mixer_type: str) -> bool:
        return mixer_type in _LIGHTNING_MIXER_TYPES

    def is_lightning_layer(self, layer_idx: int) -> bool:
        return self._is_lightning_mixer(self.mixer_types[layer_idx])

    def is_sparse_layer(self, layer_idx: int) -> bool:
        return self.mixer_types[layer_idx] == _SPARSE_MIXER_TYPE

    def get_lightning_layer_indices(self) -> list[int]:
        return [i for i in range(len(self.mixer_types)) if self.is_lightning_layer(i)]

    def get_sparse_layer_indices(self) -> list[int]:
        return [i for i in range(len(self.mixer_types)) if self.is_sparse_layer(i)]

    @property
    def layers_block_type(self) -> list[str]:
        """Canonical hybrid block kinds for `ModelConfig.get_num_layers_by_block_type()`.

        Sparse InfLLM/dense-attention slabs use `"attention"`; recurrent linear-attention
        slabs use `"mamba"` regardless of Lightning mixer spelling.
        """
        return [
            "attention" if mt == _SPARSE_MIXER_TYPE else "mamba"
            for mt in self.mixer_types
        ]
