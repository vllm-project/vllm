# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers.configuration_utils import PretrainedConfig

from vllm.logger import init_logger

logger = init_logger(__name__)


class Glm5NextTextConfig(PretrainedConfig):
    model_type = "glm5_next_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        model_type="glm5_next_text",
        vocab_size: int = 154880,
        hidden_size: int = 4096,
        head_dim: int | None = None,
        intermediate_size: int = 12288,
        num_hidden_layers: int = 45,
        num_attention_heads: int = 64,
        num_key_value_heads: int | None = None,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-5,
        pad_token_id: int | None = 151329,
        bos_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        rope_parameters: dict | None = None,
        max_position_embeddings: int = 1013760,
        tie_word_embeddings: bool = False,
        moe_intermediate_size: int = 2048,
        moe_renormalize: bool = True,
        scoring_func: str = "sigmoid",
        n_routed_experts: int | None = 288,
        num_experts_per_token: int = 7,
        n_shared_experts: int = 1,
        routed_scaling_factor: float = 2.5,
        first_k_dense_replace: int = 0,
        moe_layer_freq: int = 1,
        use_grouped_topk: bool = True,
        n_group: int = 1,
        topk_group: int = 1,
        mla: bool = True,
        q_lora_rank: int | None = 1536,
        kv_lora_rank: int | None = 512,
        qk_nope_head_dim: int = 256,
        qk_rope_head_dim: int = 0,
        v_head_dim: int | None = 256,
        mla_nope: bool | None = True,
        num_nextn_predict_layers: int = 1,
        # Per-layer layout: "linear_attention" | "deepseek_sparse_attention"
        layer_types: list[str] | None = None,
        # Per-layer MLP: "dense" | "sparse"
        mlp_layer_types: list[str] | None = None,
        # Linear-attention (KDA) head config (flattened from the old
        # linear_attn_config dict).
        linear_head_dim: int = 128,
        linear_num_heads: int = 64,
        linear_conv_kernel_dim: int = 4,
        linear_lower_bound: float | None = -5.0,
        index_head_dim: int | None = None,
        index_topk: int | None = None,
        index_n_heads: int | None = None,
        index_dsa_use_layernorm: bool = True,
        index_kpool_compress: bool = True,
        index_kpool: int | None = 16,
        index_kpool_always_select_tail: bool = True,
        mhc: bool | None = True,
        mhc_num_residual_streams: int = 4,
        hc_eps: float | None = 1e-06,
        mhc_tau: float = 0.05,
        hres_vwnstyle: bool | None = True,
        mhc_no_norm_weight: bool | None = False,
        mhc_sinkhorn_iterations: int | None = 20,
        mhc_post_mult_value: float | None = 2.0,
        swiglu_limit: float | None = None,
        **kwargs,
    ):
        # The checkpoint ships sglang/standard field names that differ from our
        # __init__ params (num_experts_per_tok / hc_mult / hc_sinkhorn_iters).
        # Use .get (not .pop) so BOTH names stay exposed at the checkpoint
        # value: model.py reads the checkpoint name (num_experts_per_tok=8),
        # while older code may read our param name. Without this, our default
        # (num_experts_per_token=7) was used, routing 7 experts/token instead
        # of the trained 8.
        num_experts_per_token = kwargs.get("num_experts_per_tok", num_experts_per_token)
        mhc_num_residual_streams = kwargs.get("hc_mult", mhc_num_residual_streams)
        mhc_sinkhorn_iterations = kwargs.get(
            "hc_sinkhorn_iters", mhc_sinkhorn_iterations
        )

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
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.rope_parameters = rope_parameters

        # mla config
        self.mla = mla
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.mla_nope = mla_nope
        # moe config
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_token = num_experts_per_token
        self.moe_renormalize = moe_renormalize
        self.n_shared_experts = n_shared_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.scoring_func = scoring_func
        assert self.scoring_func in ("softmax", "sigmoid")
        self.moe_intermediate_size = moe_intermediate_size
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_layer_freq = moe_layer_freq
        self.use_grouped_topk = use_grouped_topk
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_nextn_predict_layers = num_nextn_predict_layers

        # Per-layer attention / MLP layout. Normalize mlp_layer_types from
        # first_k_dense_replace when the new-schema field is absent so layer
        # construction sees a consistent layout (mirrors cohere2_moe).
        self.layer_types = layer_types
        if mlp_layer_types is None:
            n = self.num_hidden_layers
            if first_k_dense_replace is not None:
                mlp_layer_types = ["dense"] * first_k_dense_replace + ["sparse"] * (
                    n - first_k_dense_replace
                )
            else:
                mlp_layer_types = ["sparse"] * n
        self.mlp_layer_types = mlp_layer_types

        # Linear-attention (KDA) head config.
        self.linear_head_dim = linear_head_dim
        self.linear_num_heads = linear_num_heads
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_lower_bound = linear_lower_bound

        # dsa index config
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.index_n_heads = index_n_heads
        self.index_dsa_use_layernorm = index_dsa_use_layernorm
        self.index_kpool_compress = index_kpool_compress
        self.index_kpool = index_kpool
        self.index_kpool_always_select_tail = index_kpool_always_select_tail

        # mhc config
        self.mhc = mhc
        self.mhc_num_residual_streams = mhc_num_residual_streams
        self.mhc_tau = mhc_tau
        self.hres_vwnstyle = hres_vwnstyle
        self.hc_eps = hc_eps
        self.mhc_no_norm_weight = mhc_no_norm_weight
        self.mhc_sinkhorn_iterations = mhc_sinkhorn_iterations
        self.mhc_post_mult_value = mhc_post_mult_value

        self.swiglu_limit = swiglu_limit

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
            or self.mla_nope is True
        )

    @property
    def is_moe(self):
        return self.n_routed_experts is not None

    @property
    def is_linear_attn(self) -> bool:
        return self.layer_types is not None and any(
            t == "linear_attention" for t in self.layer_types
        )

    def is_kda_layer(self, layer_idx: int):
        return (
            self.layer_types is not None
            and layer_idx < len(self.layer_types)
            and self.layer_types[layer_idx] == "linear_attention"
        )

    @property
    def layers_block_type(self):
        # Map the schema's per-layer types onto the block strings vLLM's hybrid
        # accounting (get_num_layers_by_block_type) recognizes: linear-attention
        # layers stay "linear_attention"; every other attention variant collapses
        # to "attention".
        if self.layer_types is None:
            return ["attention"] * self.num_hidden_layers
        return [
            "linear_attention" if t == "linear_attention" else "attention"
            for t in self.layer_types
        ]


class Glm5NextVisionConfig(PretrainedConfig):
    model_type = "glm5_next_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth: int = 24,
        hidden_size: int = 1024,
        hidden_act: str = "silu",
        image_size: int = 448,
        intermediate_size: int = 4096,
        num_heads: int = 16,
        out_hidden_size: int = 4096,
        projection_intermediate_size: int = 10240,
        in_channels: int = 3,
        initializer_range: float = 0.02,
        patch_size: int = 14,
        rms_norm_eps: float = 1e-5,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 2,
        attention_dropout: float = 0.0,
        attention_bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.image_size = image_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.out_hidden_size = out_hidden_size
        # GLM5-Next-specific merger bottleneck width (absent from the generic
        # GLM-OCR vision config); the tower uses it as the PatchMerger
        # context_dim instead of text_config.intermediate_size.
        self.projection_intermediate_size = projection_intermediate_size
        self.in_channels = in_channels
        self.initializer_range = initializer_range
        self.patch_size = patch_size
        # The open GLM5-Next checkpoints ship vision_config.rms_norm_eps = 1e-5,
        # but the vision tower was trained with 1e-6. Serving with 1e-5 drifts
        # the RMSNorm and produces repetitive/degraded image descriptions, so
        # force the trained value regardless of the checkpoint field.
        self.rms_norm_eps = 1e-6
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias


class Glm5NextConfig(PretrainedConfig):
    model_type = "glm5_next"
    sub_configs = {
        "vision_config": Glm5NextVisionConfig,
        "text_config": Glm5NextTextConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id: int = 154854,
        video_token_id: int = 154855,
        image_start_token_id: int = 154830,
        image_end_token_id: int = 154831,
        video_start_token_id: int = 154832,
        video_end_token_id: int = 154833,
        **kwargs,
    ):
        # Init super() first so base-class defaults don't clobber text-config
        # values set below (PretrainedConfig has many text-related defaults
        # that differ from Glm5NextTextConfig).
        super().__init__(**kwargs)

        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            # Backward compatibility: a flat top-level checkpoint (no nested
            # text_config) folds its text fields into Glm5NextTextConfig.
            self.text_config = self.sub_configs["text_config"](**kwargs)

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.video_start_token_id = video_start_token_id
        self.video_end_token_id = video_end_token_id

        # Mirror attention implementation recursively onto sub-configs.
        self._attn_implementation = kwargs.pop("attn_implementation", None)

    # Config-metadata fields that belong to the top-level (multimodal) config
    # and must NOT be mirrored onto text_config: ``architectures`` /
    # ``torch_dtype`` differ between the top-level config and the text
    # sub-config, and mirroring them makes the top-level ``architectures``
    # silently read back as None (PretrainedConfig initializes both to None),
    # which then fails model-class resolution ("No model architectures are
    # specified").
    _UNMIRRORED_KEYS = [
        "_name_or_path",
        "model_type",
        "dtype",
        "torch_dtype",
        "architectures",
        "_attn_implementation_internal",
    ]

    def __setattr__(self, key, value):
        unmirrored = type(self)._UNMIRRORED_KEYS
        if (
            (text_config := super().__getattribute__("__dict__").get("text_config"))
            is not None
            and key not in unmirrored
            and key in text_config.__dict__
        ):
            setattr(text_config, key, value)
        else:
            super().__setattr__(key, value)

    def __getattribute__(self, key):
        unmirrored = type(self)._UNMIRRORED_KEYS
        if (
            "text_config" in super().__getattribute__("__dict__")
            and key not in unmirrored
        ):
            text_config = super().__getattribute__("text_config")
            # Forward both instance attributes AND class-defined properties/
            # methods of the text config, so a flat text-only checkpoint
            # (model_type "glm5_next", no nested text_config) sees is_moe /
            # is_kda_layer / layers_block_type like a Glm5NextTextConfig.
            if key in text_config.__dict__ or key in type(text_config).__dict__:
                return getattr(text_config, key)

        return super().__getattribute__(key)
