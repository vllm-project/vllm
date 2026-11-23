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
            logits. Surfaced as `head_dtype` so `LogitsProcessor` accumulates
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
    ignore_keys_at_rope_validation = {"mrope_section", "mrope_interleaved"}
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


class HYV4VLVisionConfig(PreTrainedConfig):
    """Vision tower and projector configuration for HYV4 VL."""

    model_type = "hy_v4_vl_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        num_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        spatial_patch_size=1,
        temporal_patch_size=1,
        out_hidden_size=None,
        hidden_act="gelu",
        rms_norm_eps=1e-5,
        max_image_size=2048,
        interpolate_mode="bilinear",
        cat_extra_token=0,
        perceive_pre_norm=True,
        perceive_post_norm=True,
        projector_type=None,
        perceive_intermediate_size=None,
        use_image_newline_token=True,
        use_2d_rope=False,
        rope_2d_theta=10000.0,
        rope_2d_percent=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.max_image_size = max_image_size
        self.interpolate_mode = interpolate_mode
        self.cat_extra_token = cat_extra_token
        self.perceive_pre_norm = perceive_pre_norm
        self.perceive_post_norm = perceive_post_norm
        self.projector_type = projector_type
        self.perceive_intermediate_size = perceive_intermediate_size
        self.use_image_newline_token = use_image_newline_token
        self.use_2d_rope = use_2d_rope
        self.rope_2d_theta = rope_2d_theta
        self.rope_2d_percent = rope_2d_percent


class HYV4VLConfig(PreTrainedConfig):
    """Local HYV4 vision/text composition, including flat checkpoint support."""

    model_type = "hy_v4_vl"
    architectures: list[str] | None = None
    sub_configs = {"text_config": HYV4Config, "vision_config": HYV4VLVisionConfig}
    keys_to_ignore_at_inference = ["past_key_values"]
    ignore_keys_at_rope_validation = {"mrope_section", "mrope_interleaved"}

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_start_token_id=120118,
        image_end_token_id=120119,
        image_token_id=120120,
        image_newline_token_id=120121,
        video_start_token_id=120122,
        video_end_token_id=120123,
        video_token_id=120124,
        **kwargs,
    ):
        metadata_keys = {"model_type", "architectures", "auto_map", "processor_class"}
        outer_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in metadata_keys or key.startswith("dfly_")
        }
        text_config_cls = self.sub_configs["text_config"]
        if isinstance(text_config, text_config_cls):
            text = text_config
        else:
            # Nested text fields take precedence; flat fields supply missing values.
            text_kwargs = {**kwargs, **(text_config or {})}
            for key in metadata_keys:
                text_kwargs.pop(key, None)
            self._prepare_text_config(text_kwargs)
            explicit_rope = dict(text_kwargs.get("rope_parameters") or {})
            text = text_config_cls(**text_kwargs)
            if not any(isinstance(value, dict) for value in explicit_rope.values()):
                rope = dict(text.rope_parameters or {})
                rope.update(explicit_rope)
                text.rope_parameters = rope
        super().__init__(**outer_kwargs)
        self.text_config = text
        self.vision_config = (
            vision_config
            if isinstance(vision_config, HYV4VLVisionConfig)
            else HYV4VLVisionConfig(**(vision_config or {}))
        )
        if self.vision_config.out_hidden_size is None:
            self.vision_config.out_hidden_size = text.hidden_size
        if self.vision_config.out_hidden_size != text.hidden_size:
            raise ValueError("HYV4 VL projector output must match text hidden_size")
        self.vision_config.text_hidden_size = text.hidden_size
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.image_token_id = image_token_id
        self.image_newline_token_id = image_newline_token_id
        self.video_start_token_id = video_start_token_id
        self.video_end_token_id = video_end_token_id
        self.video_token_id = video_token_id
        for name in (
            "dtype",
            "tie_word_embeddings",
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
        ):
            setattr(self, name, getattr(text, name))
        if getattr(text, "head_dtype", None) is not None:
            self.head_dtype = text.head_dtype
        if self.architectures is None:
            self.architectures = [self.default_architecture]

    default_architecture = "HYV4VLForConditionalGeneration"

    @staticmethod
    def _prepare_text_config(text_kwargs):
        rope = dict(text_kwargs.get("rope_parameters") or {})
        rope.setdefault("rope_type", "default")
        rope.setdefault("mrope_interleaved", True)
        rope.setdefault("mrope_section", [12, 10, 10])
        text_kwargs["rope_parameters"] = rope

    def __getattr__(self, name):
        text = self.__dict__.get("text_config")
        if text is not None:
            return getattr(text, name)
        raise AttributeError(name)


__all__ = ["HYV4Config", "HYV4VLConfig", "HYV4VLVisionConfig"]
