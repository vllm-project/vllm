# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
from dataclasses import asdict, dataclass
from typing import Any

from transformers.configuration_utils import PreTrainedConfig

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class KimiLinearDSAConfig:
    """Configuration class for DSA (DeepSeek Sparse Attention).

    Kimi-K3.1 puts DSA on every full-attention (MLA) layer; KDA layers
    are unaffected. The converted checkpoint exports this as a nested
    `dsa_config` sub-structure.
    """

    dsa_layers: list[int]
    index_topk: int = 2048
    index_head_dim: int = 64
    index_num_heads: int = 16
    # Leading dims of every indexer head that receive RoPE (neox/half-split
    # after checkpoint conversion; shares the main-attention rotary cache).
    # 0 = no indexer RoPE (Kimi-K3.1 is NoPE + index_rope_dim 0).
    index_rope_dim: int = 0
    # Norm on the indexer shared key: "rmsnorm" or "layernorm" (bias-full
    # fp32 LayerNorm; the converter detects it from k_index_norm.bias).
    index_k_norm_type: str = "rmsnorm"
    indexer_use_fp8: bool = False
    # Checkpoint schema parity fields; Kimi-K3.1 supports only
    # index_topk_freq == 1 (no index share), other values fail loudly.
    index_topk_freq: int = 1
    index_skip_topk_offset: int = 1

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "KimiLinearDSAConfig":
        config_dict = dict(config_dict)
        if "dsa_layers" not in config_dict and "layers" in config_dict:
            config_dict["dsa_layers"] = config_dict["layers"]
        config_dict.pop("layers", None)
        allowed_keys = inspect.signature(cls).parameters
        for key in config_dict:
            if key not in allowed_keys:
                logger.warning("%s is set but does not support. Ignore it.", key)
        cfg = cls(**{k: v for k, v in config_dict.items() if k in allowed_keys})
        if cfg.index_topk_freq < 1:
            raise ValueError(
                f"index_topk_freq must be >= 1, got {cfg.index_topk_freq}; "
                "0 would ZeroDivisionError in the index-share modulo check"
            )
        return cfg


class KimiLinearConfig(PreTrainedConfig):
    model_type = "kimi_linear"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        model_type="kimi_linear",
        vocab_size=163840,
        hidden_size=4096,
        head_dim=None,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        rope_parameters=None,
        tie_word_embeddings=False,
        moe_intermediate_size: int | None = None,
        moe_renormalize: bool = True,
        moe_router_activation_func: str = "sigmoid",
        num_experts: int | None = None,
        num_experts_per_token: int | None = None,
        num_shared_experts: int = 0,
        routed_scaling_factor: float = 1.0,
        first_k_dense_replace: int = 0,
        moe_layer_freq: int = 1,
        use_grouped_topk: bool = True,
        num_expert_group: int = 1,
        topk_group: int = 1,
        q_lora_rank: int | None = None,
        kv_lora_rank: int | None = None,
        qk_nope_head_dim: int | None = None,
        qk_rope_head_dim: int | None = None,
        v_head_dim: int | None = None,
        mla_use_nope: bool | None = False,
        mla_use_output_gate: bool | None = False,
        num_nextn_predict_layers: int = 0,
        linear_attn_config: dict | None = None,
        dsa_config: dict | KimiLinearDSAConfig | None = None,
        attn_res_block_size: int | None = None,
        latent_moe_use_norm: bool = False,
        activation_situ_beta: float | None = None,
        activation_situ_linear_beta: float | None = None,
        max_position_embeddings: int = 4096,
        routed_expert_hidden_size: int | None = None,
        topk_method: str = "noaux_tc",
        **kwargs,
    ):
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.head_dim = (
            head_dim if head_dim is not None else hidden_size // num_attention_heads
        )
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
        self.use_cache = use_cache
        # Try to set `rope_scaling` if available, otherwise use `rope_parameters`
        rope_scaling = kwargs.pop("rope_scaling", None)
        rope_parameters = rope_scaling or rope_parameters or {"rope_type": "default"}
        rope_theta = kwargs.pop("rope_theta", 10000.0)
        if "rope_theta" not in rope_parameters:
            rope_parameters["rope_theta"] = rope_theta
        self.rope_parameters = rope_parameters

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.mla_use_nope = mla_use_nope
        self.mla_use_output_gate = mla_use_output_gate
        # moe config
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.moe_renormalize = moe_renormalize
        self.num_shared_experts = num_shared_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.moe_router_activation_func = moe_router_activation_func
        assert self.moe_router_activation_func in ("softmax", "sigmoid")
        self.moe_intermediate_size = moe_intermediate_size
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_layer_freq = moe_layer_freq
        self.use_grouped_topk = use_grouped_topk
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.num_nextn_predict_layers = num_nextn_predict_layers

        self.attn_res_block_size = attn_res_block_size
        self.latent_moe_use_norm = latent_moe_use_norm
        self.activation_situ_beta = activation_situ_beta
        self.activation_situ_linear_beta = activation_situ_linear_beta
        self.max_position_embeddings = max_position_embeddings
        self.routed_expert_hidden_size = routed_expert_hidden_size
        self.topk_method = topk_method

        if linear_attn_config is not None:
            assert linear_attn_config["kda_layers"] is not None
            assert linear_attn_config["full_attn_layers"] is not None
        self.linear_attn_config = linear_attn_config

        # Kimi-K3.1 (KDA + DSA hybrid): DSA parameters arrive as a nested
        # sub-structure from the checkpoint conversion.
        if isinstance(dsa_config, dict):
            self.dsa_config: KimiLinearDSAConfig | None = KimiLinearDSAConfig.from_dict(
                dsa_config
            )
        else:
            self.dsa_config = dsa_config
        # SparseMLACommonMetadataBuilder reads top-level hf_config.index_topk
        # like upstream DeepSeek; expose it at the config level instead of
        # injecting it implicitly at model construction time.
        if self.dsa_config is not None:
            self.index_topk = self.dsa_config.index_topk

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def is_mla(self):
        return (
            self.q_lora_rank is not None
            or self.kv_lora_rank is not None
            or self.qk_nope_head_dim is not None
            or self.qk_rope_head_dim is not None
            or self.v_head_dim is not None
            or self.mla_use_nope is True
        )

    @property
    def is_moe(self):
        return self.num_experts is not None

    @property
    def is_linear_attn(self) -> bool:
        return not (
            self.linear_attn_config is None
            or (
                isinstance(self.linear_attn_config, dict)
                and self.linear_attn_config["kda_layers"] is not None
                and len(self.linear_attn_config["kda_layers"]) == 0
            )
        )

    def is_kda_layer(self, layer_idx: int):
        return (
            self.linear_attn_config is not None
            and (layer_idx + 1) in self.linear_attn_config["kda_layers"]
        )

    def is_dsa_layer(self, layer_idx: int):
        return (
            self.dsa_config is not None
            and (layer_idx + 1) in self.dsa_config.dsa_layers
        )

    def to_dict(self) -> dict[str, Any]:
        ret = super().to_dict()
        if isinstance(ret.get("dsa_config"), KimiLinearDSAConfig):
            ret["dsa_config"] = asdict(ret["dsa_config"])
        return ret
