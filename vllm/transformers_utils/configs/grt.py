# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GRT model configuration."""

from transformers.configuration_utils import PreTrainedConfig


class GRTConfig(PreTrainedConfig):
    """Configuration for the GRT (Geometric Rotational Transformer) causal LM,
    as Transformers' GRTConfig saves it. The defaults are matterwaveai/grt-30m's.
    Initialization-only fields are kept as extra attributes.
    """

    model_type = "grt"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 256,
        intermediate_size: int | None = None,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 4,
        max_position_embeddings: int = 2048,
        ffn_n_rotations: int = 21,
        n_qkv_rotations: int = 11,
        n_mixer_rotations: int = 1,
        pos_freq_base: float = 10000.0,
        learnable_rope: bool = True,
        learnable_residual_gates: bool = True,
        residual_gate_mode: str = "bounded_depth",
        residual_norm_mode: str = "none",
        peri_ln: bool = False,
        tie_word_embeddings: bool = True,
        pad_token_id: int | None = 151643,
        bos_token_id: int | None = 151643,
        eos_token_id: int | None = 151643,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size or 4 * hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.ffn_n_rotations = ffn_n_rotations
        self.n_qkv_rotations = n_qkv_rotations
        self.n_mixer_rotations = n_mixer_rotations
        self.pos_freq_base = pos_freq_base
        self.learnable_rope = learnable_rope
        self.learnable_residual_gates = learnable_residual_gates
        self.residual_gate_mode = residual_gate_mode
        self.residual_norm_mode = residual_norm_mode
        self.peri_ln = peri_ln

        kwargs.update(
            tie_word_embeddings=tie_word_embeddings,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )
        super().__init__(**kwargs)


__all__ = ["GRTConfig"]
