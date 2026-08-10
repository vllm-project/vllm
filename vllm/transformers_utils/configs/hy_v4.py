# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from transformers.configuration_utils import PretrainedConfig


class HYV4Config(PretrainedConfig):
    r"""Configuration class for the HY V4 (`HYV4ForCausalLM`) model.

    This mirrors the on-disk `config.json` produced for the HY V4 internal
    checkpoints (`model_type = "hy_v4_internal"`). Registering it in vLLM's
    `_CONFIG_REGISTRY` lets the architecture load without `trust_remote_code`
    and without the `auto_map` remote-code module sitting next to the weights.

    Only the fields consumed by `vllm.models.hy_v4` need to be present; any
    extra checkpoint keys are still preserved by `PretrainedConfig`.

    Args:
        vocab_size: Vocabulary size of the model.
        hidden_size: Dimension of the hidden representations.
        intermediate_size: Dimension of the dense FFN intermediate size.
        num_hidden_layers: Number of decoder layers in the backbone.
        num_attention_heads: Number of attention heads per layer.
        num_key_value_heads: Number of key-value heads.
        head_dim: Dimension per attention head.
        hidden_act: FFN activation; only `"silu"` is supported.
        max_position_embeddings: Maximum sequence length supported.
        initializer_range: Std of the truncated-normal weight initializer.
        rms_norm_eps: Epsilon of the RMS normalization layers.
        attention_bias: Whether attention projections carry a bias.
        attention_dropout: Dropout ratio applied to attention weights.
        use_cache: Whether to return the KV cache.
        tie_word_embeddings: Whether to tie input and output embeddings.
        pad_token_id: Padding token id.
        bos_token_id: Beginning-of-sequence token id.
        eos_token_id: End-of-sequence token id (or list of ids).
        rope_parameters: RoPE settings; defaults to plain RoPE with
            `rope_theta` taken from the checkpoint.
        n_routed_experts: Number of routed MoE experts.
        n_shared_experts: Number of shared (always-on) experts.
        moe_intermediate_size: Intermediate size of a single expert.
        num_experts_per_tok: Number of experts selected per token.
        routed_scaling_factor: Scale applied to routed-expert outputs.
        norm_topk_prob: Whether to renormalize the top-k routing weights.
        mlp_layer_types: Per-layer MLP kind, `"dense"` or `"sparse"`.
        layer_types: Per-layer attention kind, e.g. `"full_attention"` or
            `"sparse"`.
        q_lora_rank: Rank of the query down-projection (`None` disables it).
        kv_lora_rank: Rank of the joint key-value down-projection.
        qk_nope_head_dim: Per-head dimension of the non-positional QK part.
        qk_rope_head_dim: Per-head dimension of the RoPE QK part.
        v_head_dim: Per-head dimension of the value projection.
        gated_mla: Whether attention output gating is enabled.
        gating_type: Gating granularity, `"headwise"` or `"elementwise"`.
        learnable_sink: Whether a per-head learnable attention sink exists.
        swiglu_limit: Clamp limit of the routed experts' SwiGLU; `0` disables.
        index_topk: Number of tokens kept by the lightning indexer.
        index_head_dim: Per-head dimension of the indexer.
        index_n_heads: Number of indexer heads.
        indexer_types: Per-layer indexer kind, `"full"` or `"shared"`.
        enable_ihc: Whether independent Hyper-Connections are enabled.
        hc_mult: Number of iHC residual channels.
        hc_magnitude: Scale of the iHC post gates.
        hc_eps: Epsilon added to the iHC gates.
        enable_lm_head_fp32: Whether the LM head runs in fp32.
        num_nextn_predict_layers: Number of MTP layers in the checkpoint.
        mtp_loss_factor: Training-time MTP loss weight (unused at inference).
    """

    model_type = "hy_v4_internal"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=120832,
        hidden_size=6144,
        intermediate_size=18432,
        num_hidden_layers=78,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=64,
        hidden_act="silu",
        max_position_embeddings=262144,
        initializer_range=0.006,
        rms_norm_eps=1e-5,
        attention_bias=False,
        attention_dropout=0.0,
        use_cache=True,
        tie_word_embeddings=False,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        rope_parameters: dict[str, Any] | None = None,
        # MoE specific
        n_routed_experts=256,
        n_shared_experts=1,
        moe_intermediate_size=2048,
        num_experts_per_tok=8,
        routed_scaling_factor=2.827,
        norm_topk_prob=True,
        mlp_layer_types=None,
        layer_types=None,
        # MLA / attention specific
        q_lora_rank=2048,
        kv_lora_rank=512,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        gated_mla=True,
        gating_type="elementwise",
        learnable_sink=True,
        swiglu_limit=10.0,
        # Sparse attention indexer (DSA)
        index_topk=2048,
        index_head_dim=128,
        index_n_heads=32,
        indexer_types=None,
        # iHC (independent Hyper-Connections)
        enable_ihc=True,
        hc_mult=4,
        hc_magnitude=2.0,
        hc_eps=1e-6,
        # misc
        enable_lm_head_fp32=True,
        # MTP
        num_nextn_predict_layers=1,
        mtp_loss_factor=0.1,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.use_cache = use_cache

        rope_theta = kwargs.pop("rope_theta", 10000000.0)
        self.rope_theta = rope_theta
        if rope_parameters is None:
            rope_parameters = {"rope_type": "default", "rope_theta": rope_theta}
        self.rope_parameters = rope_parameters

        # MoE specific
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.mlp_layer_types = mlp_layer_types
        # ``PretrainedConfig`` validates ``layer_types`` against a fixed
        # allow-list that does not include "sparse_attention". The model treats
        # "sparse_attention" and "sparse" identically, so normalize to the
        # accepted spelling.
        if layer_types is not None:
            layer_types = [
                "sparse" if lt == "sparse_attention" else lt for lt in layer_types
            ]
        self.layer_types = layer_types

        # MLA / attention specific
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.gated_mla = gated_mla
        self.gating_type = gating_type
        self.learnable_sink = learnable_sink
        self.swiglu_limit = swiglu_limit

        # Sparse attention indexer (DSA)
        self.index_topk = index_topk
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.indexer_types = indexer_types

        # iHC (independent Hyper-Connections)
        self.enable_ihc = enable_ihc
        self.hc_mult = hc_mult
        self.hc_magnitude = hc_magnitude
        self.hc_eps = hc_eps

        # misc
        self.enable_lm_head_fp32 = enable_lm_head_fp32

        # MTP
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.mtp_loss_factor = mtp_loss_factor

        if eos_token_id is not None and isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
