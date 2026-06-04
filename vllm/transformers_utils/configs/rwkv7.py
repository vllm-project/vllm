# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang, Zhiyuan Li
#
# Adapted from
# https://github.com/fla-org/flash-linear-attention/blob/main/fla/models/rwkv7/configuration_rwkv7.py
# Original code licensed under the MIT license.

from transformers.configuration_utils import PretrainedConfig


class RWKV7Config(PretrainedConfig):
    """RWKV-7 "Goose" config.

    Vendored from `flash-linear-attention` so that vLLM can parse
    `fla-hub/rwkv7-*` checkpoints without `trust_remote_code` being needed
    just to read the config (their `auto_map` points at fla's modeling file
    which imports the full fla package).
    """

    model_type = "rwkv7"
    keys_to_ignore_at_inference = ["past_key_values"]

    # vLLM uses ``model_config.get_mamba_chunk_size()`` to pick a
    # ``mamba_block_size`` aligned to the recurrent kernel's chunk granularity.
    # The chunk dispatch in ``RWKV7Attention`` uses ``chunk_size=64``, so the
    # cache block size must be a multiple of 64 for prefix caching to scatter
    # per-chunk states into the right blocks.
    mamba_chunk_size = 64

    def __init__(
        self,
        attn_mode: str = "chunk",
        hidden_size: int = 2048,
        hidden_ratio: int | None = 4,
        intermediate_size: int | None = None,
        num_hidden_layers: int = 24,
        head_dim: int | None = 64,
        num_heads: int | None = None,
        decay_low_rank_dim: int = 64,
        gate_low_rank_dim: int = 128,
        a_low_rank_dim: int = 64,
        v_low_rank_dim: int = 16,
        hidden_act: str = "sqrelu",
        max_position_embeddings: int = 2048,
        norm_first: bool = True,
        norm_bias: bool = True,
        norm_eps: float = 1e-5,
        attn: dict | None = None,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        initializer_range: float = 0.02,
        fuse_norm: bool = True,
        fuse_cross_entropy: bool = True,
        fuse_linear_cross_entropy: bool = False,
        use_l2warp: bool = True,
        vocab_size: int = 32000,
        value_dim: int | list[int] | None = None,
        **kwargs,
    ) -> None:
        self.attn_mode = attn_mode
        self.hidden_size = hidden_size
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.norm_first = norm_first
        self.num_hidden_layers = num_hidden_layers

        if head_dim is None and num_heads is not None:
            head_dim = int(hidden_size // num_heads)
        elif head_dim is not None and num_heads is None:
            num_heads = int(hidden_size // head_dim)

        if value_dim is None:
            value_dim = [hidden_size] * num_hidden_layers
        elif isinstance(value_dim, int):
            assert value_dim >= hidden_size
            assert value_dim % hidden_size == 0
            value_dim = [value_dim] * num_hidden_layers
        else:
            assert len(value_dim) == num_hidden_layers

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.value_dim = value_dim

        self.decay_low_rank_dim = decay_low_rank_dim
        self.gate_low_rank_dim = gate_low_rank_dim
        self.a_low_rank_dim = a_low_rank_dim
        self.v_low_rank_dim = v_low_rank_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.norm_bias = norm_bias
        self.norm_eps = norm_eps
        self.attn = attn
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        self.fuse_norm = fuse_norm
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy
        self.use_l2warp = use_l2warp
        self.vocab_size = vocab_size

        if attn is not None:
            if not isinstance(attn, dict):
                raise ValueError("attn must be a dictionary")
            if "layers" not in attn:
                raise ValueError(
                    "Layer indices must be provided to initialize "
                    "hybrid attention layers"
                )
            if "num_heads" not in attn:
                raise ValueError(
                    "Number of heads must be provided to initialize "
                    "hybrid attention layers"
                )
            attn["num_kv_heads"] = attn.get("num_kv_heads", attn["num_heads"])
            attn["qkv_bias"] = attn.get("qkv_bias", False)
            attn["window_size"] = attn.get("window_size", None)
            attn["rope_theta"] = attn.get("rope_theta", 10000.0)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
