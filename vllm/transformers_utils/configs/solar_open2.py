# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers.configuration_utils import PretrainedConfig


class SolarOpen2Config(PretrainedConfig):
    """Configuration for Solar Open 2 models served by vLLM."""

    model_type = "solar_open2"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_local_experts": "n_routed_experts"}

    def __init__(
        self,
        vocab_size: int = 196608,
        hidden_size: int = 4096,
        intermediate_size: int = 10240,
        num_hidden_layers: int = 48,
        num_attention_heads: int = 64,
        head_dim: int = 128,
        num_key_value_heads: int = 8,
        hidden_act: str = "silu",
        max_position_embeddings: int = 1_048_576,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_parameters: dict | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        moe_intermediate_size: int = 1280,
        num_experts_per_tok: int = 8,
        n_shared_experts: int = 1,
        n_routed_experts: int = 320,
        routed_scaling_factor: float = 1.0,
        n_group: int | None = 1,
        topk_group: int | None = 1,
        first_k_dense_replace: int = 0,
        norm_topk_prob: bool = True,
        use_qk_norm: bool = False,
        use_rope: bool = False,
        gqa_interval: int = 3,
        gqa_layers: list[int] | None = None,
        use_gqa_gate: bool = True,
        use_gqa_gate_bias: bool = False,
        linear_attn_config: dict | None = None,
        kda_use_full_proj: bool = False,
        kda_allow_neg_eigval: bool = True,
        layer_types: list[str] | None = None,
        pad_token_id: int | None = 2,
        bos_token_id: int | None = 1,
        eos_token_id: int | list[int] | None = 2,
        **kwargs,
    ) -> None:
        # Fold the pre-v5 rope keys into `rope_parameters` and drop them, so
        # they cannot survive as a second, disagreeing source of truth.
        rope_scaling = kwargs.pop("rope_scaling", None)
        rope_theta = kwargs.pop("rope_theta", 10000.0)
        partial_rotary_factor = kwargs.pop("partial_rotary_factor", 1.0)
        if rope_parameters is None:
            rope_parameters = dict(rope_scaling or {})
            if "type" in rope_parameters and "rope_type" not in rope_parameters:
                rope_parameters["rope_type"] = rope_parameters.pop("type")
            rope_parameters.setdefault("rope_type", "default")
        rope_parameters.setdefault("rope_theta", rope_theta)
        # NoPE by default; keep the full rotary dim available when use_rope=True.
        rope_parameters.setdefault("partial_rotary_factor", partial_rotary_factor)
        # Set before `super().__init__()` so that Transformers standardizes and
        # validates the rope params, which includes seeding
        # `original_max_position_embeddings` for scaled rope types.
        self.max_position_embeddings = max_position_embeddings
        self.rope_parameters = rope_parameters

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        if (
            not isinstance(gqa_interval, int)
            or isinstance(gqa_interval, bool)
            or gqa_interval < 1
        ):
            raise ValueError(
                f"`gqa_interval` must be a positive integer, got {gqa_interval!r}."
            )
        if gqa_layers is not None:
            gqa_layers = list(gqa_layers)
            if any(
                not isinstance(i, int)
                or isinstance(i, bool)
                or not 0 <= i < num_hidden_layers
                for i in gqa_layers
            ):
                raise ValueError(
                    "`gqa_layers` entries must be integer layer indices in "
                    f"[0, {num_hidden_layers}), got {gqa_layers}."
                )

        # Explicit layer_types takes priority over gqa_layers / gqa_interval.
        # gqa_interval=N: one full-attention layer followed by N
        # linear-attention layers, starting at layer 0.
        if layer_types is None:
            if gqa_layers is not None:
                full_attention_layers = set(gqa_layers)
                layer_types = [
                    "full_attention"
                    if layer in full_attention_layers
                    else "linear_attention"
                    for layer in range(num_hidden_layers)
                ]
            else:
                period = gqa_interval + 1
                layer_types = [
                    "full_attention" if layer % period == 0 else "linear_attention"
                    for layer in range(num_hidden_layers)
                ]
        else:
            if len(layer_types) != num_hidden_layers or any(
                layer_type not in {"full_attention", "linear_attention"}
                for layer_type in layer_types
            ):
                raise ValueError(
                    "layer_types must contain one valid attention type per layer"
                )
            layer_types = list(layer_types)
        if "full_attention" not in layer_types:
            raise ValueError(
                "SolarOpen2 requires at least one full-attention layer; "
                "check `gqa_layers` / `gqa_interval`."
            )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = 1 if n_group is None else n_group
        self.topk_group = 1 if topk_group is None else topk_group
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob

        self.use_qk_norm = use_qk_norm
        self.use_rope = use_rope
        self.gqa_interval = gqa_interval
        self.gqa_layers = gqa_layers
        self.use_gqa_gate = use_gqa_gate
        self.use_gqa_gate_bias = use_gqa_gate_bias
        self.linear_attn_config = linear_attn_config or {
            "short_conv_kernel_size": 4,
            "head_dim": head_dim,
            "num_heads": num_attention_heads,
            "num_kv_heads": None,
        }
        missing_keys = {
            "short_conv_kernel_size",
            "head_dim",
            "num_heads",
        } - self.linear_attn_config.keys()
        if missing_keys:
            raise ValueError(
                "`linear_attn_config` is missing required keys: "
                f"{sorted(missing_keys)}."
            )
        kda_num_kv_heads = self.linear_attn_config.get("num_kv_heads")
        if (
            kda_num_kv_heads is not None
            and kda_num_kv_heads != self.linear_attn_config["num_heads"]
        ):
            raise ValueError(
                "vLLM's SolarOpen2 KDA implementation does not support "
                "`linear_attn_config['num_kv_heads']` != "
                f"`linear_attn_config['num_heads']`; got {kda_num_kv_heads}."
            )
        self.kda_use_full_proj = kda_use_full_proj
        self.kda_allow_neg_eigval = kda_allow_neg_eigval
        self.layer_types = layer_types


__all__ = ["SolarOpen2Config"]
