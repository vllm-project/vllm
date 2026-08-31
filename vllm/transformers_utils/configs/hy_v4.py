# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from huggingface_hub.dataclasses import strict
from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_rope_utils import RopeParameters


@strict
class HYV4Config(PreTrainedConfig):
    r"""Configuration class for the HY V4 (`HYV4ForCausalLM`) model.

    Kept field-for-field in sync with
    `transformers.models.hy_v4.configuration_hy_v4.HYV4Config` so that a
    checkpoint resolves to the same values through either class. Registering it
    in vLLM's `_CONFIG_REGISTRY` lets the architecture load without
    `trust_remote_code` and without an `auto_map` module next to the weights.

    HYV4 is a mixture-of-experts causal language model using Multi-head Latent
    Attention (MLA), DeepSeek-style sparse attention (DSA), gated MLA,
    learnable attention sinks, and independent Hyper-Connections (iHC).

    The `layer_types`, `num_nextn_predict_layers` and `mtp_loss_factor` fields
    below are vLLM-only additions: upstream leaves them to `**kwargs`, but
    vLLM reads them directly (sparse-attention selection and MTP speculative
    decoding), so they are declared to give checkpoints that omit them a
    working default instead of a `None`.

    Args:
        mlp_layer_types: Per-layer MLP kind, `"dense"` or `"sparse"`. Defaults
            to one dense layer followed by sparse MoE layers.
        layer_types: Per-layer attention kind, e.g. `"full_attention"` or
            `"deepseek_sparse_attention"`.
        indexer_types: Per-layer DSA indexer kind, `"full"` or `"shared"`. A
            shared layer reuses the most recent full indexer in the same
            forward request.
        index_topk: Maximum number of key positions selected by each DSA query.
        index_head_dim: Hidden dimension of each DSA indexer head.
        index_n_heads: Number of DSA indexer heads.
        enable_lm_head_fp32: Whether the language-model head emits float32
            logits. Surfaced as ``head_dtype`` so `LogitsProcessor` accumulates
            the projection into fp32 while the weight stays in the model dtype.
        enable_ihc: Whether independent Hyper-Connections are enabled.
        hc_mult: Number of hidden-state channels maintained by iHC.
        hc_magnitude: Scale applied to the iHC post-gating branch.
        hc_eps: Numerical epsilon added to iHC sigmoid gates.
        gated_mla: Whether to gate the MLA output.
        gating_type: MLA gate granularity, `"elementwise"` or `"headwise"`.
        learnable_sink: Whether to add a learned per-head attention sink.
        learnable_sink_init: Initial value of each learned attention-sink
            logit.
        swiglu_limit: Magnitude of the routed-expert SwiGLU clamp. Values at or
            below zero disable the clamp.
        num_nextn_predict_layers: Number of MTP layers in the checkpoint.
        mtp_loss_factor: Training-time MTP loss weight (unused at inference).
    """

    model_type = "hy_v4"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_local_experts": "n_routed_experts",
    }
    base_model_tp_plan = {
        "layers.*.self_attn.q_b_proj": "colwise",
        "layers.*.self_attn.kv_a_proj_with_mqa": "mla_kv_a_proj",
        "layers.*.self_attn.kv_b_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.self_attn.linear_gate": "colwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    vocab_size: int = 120832
    hidden_size: int = 2816
    intermediate_size: int = 6912
    moe_intermediate_size: int = 768
    num_hidden_layers: int = 34
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    head_dim: int = 256
    hidden_act: str = "silu"
    max_position_embeddings: int = 262144
    initializer_range: float = 0.006
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = 120002
    bos_token_id: int | None = 120000
    eos_token_id: int | list[int] | None = 120025
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    attention_dropout: float = 0.0
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    routed_scaling_factor: float = 2.827
    norm_topk_prob: bool = True
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    mlp_layer_types: list[str] | None = None
    layer_types: list[str] | None = None
    index_topk: int = 2048
    index_head_dim: int = 128
    index_n_heads: int = 16
    indexer_types: list[str] | None = None
    enable_lm_head_fp32: bool = True
    enable_ihc: bool = True
    hc_mult: int = 4
    hc_magnitude: float = 2.0
    hc_eps: float = 1e-6
    gated_mla: bool = True
    gating_type: str = "elementwise"
    learnable_sink: bool = True
    learnable_sink_init: float = 0.0
    swiglu_limit: float = 10.0
    rope_parameters: RopeParameters | dict | None = None
    num_nextn_predict_layers: int = 1
    mtp_loss_factor: float = 0.1

    def __post_init__(self, **kwargs):
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        # RoPE applies only to the rope slice, so point `head_dim` at it: the
        # inherited rotary embedding reads `config.head_dim` and then computes
        # the right frequencies with no override.
        self.head_dim = self.qk_rope_head_dim

        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] * min(1, self.num_hidden_layers) + [
                "sparse"
            ] * max(self.num_hidden_layers - 1, 0)
        if self.indexer_types is None:
            self.indexer_types = [
                "full" if layer_idx == 0 or (layer_idx - 1) % 4 == 0 else "shared"
                for layer_idx in range(self.num_hidden_layers)
            ]
        # `PreTrainedConfig` validates `layer_types` against a fixed allow-list
        # in which "deepseek_sparse_attention" is the only accepted spelling of
        # a DSA layer. The model treats "sparse_attention" and "sparse" as
        # equivalent, so fold them in before the base class validates.
        if self.layer_types is not None:
            self.layer_types = [
                "deepseek_sparse_attention"
                if lt in ("sparse_attention", "sparse")
                else lt
                for lt in self.layer_types
            ]

        # `ModelConfig.head_dtype` reads this off the HF config, so an fp32 head
        # needs no model-side dtype juggling: LogitsProcessor picks the
        # `torch.mm(out_dtype=float32)` path and leaves the weight in the model
        # dtype. Only fill it in when the checkpoint (or --hf-overrides) has not
        # already pinned a value.
        if self.enable_lm_head_fp32 and getattr(self, "head_dtype", None) is None:
            self.head_dtype = "float32"

        super().__post_init__(**kwargs)


__all__ = ["HYV4Config"]
