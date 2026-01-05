# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, fields
from typing import Any, Literal

from transformers import AutoConfig, PretrainedConfig


@dataclass
class CLIPVisionConfig:
    attn_bias: bool = True

    hidden_size: int = 768
    head_width: int = 64
    image_size: int = 224
    in_channels: int = 3
    intp_freq: bool = False

    layers: int = 12
    layernorm_type: str | None = None
    layer_norm_eps: float = 1e-6
    ls_init_value: float | None = None  # layer scale initial value
    mlp_ratio: float = 4.0
    naiveswiglu: bool = False

    patch_size: int = 14
    postnorm: bool = False  # mlp postnorm
    pt_hw_seq_len: int = 16

    qk_scale: float | Literal["auto"] = "auto"

    # Whether the original weights were split into q,k,v before loading
    # Used during weight loading.
    split_qkv: bool = False
    use_rope: bool = False

    @classmethod
    def filter_kwargs(cls, kwargs: dict[str, Any]):
        clip_keys = set(cls.keys())
        vision_kwargs = dict()
        for key in clip_keys:
            if key in kwargs:
                vision_kwargs[key] = kwargs.pop(key)

        return vision_kwargs, kwargs

    def __post_init__(self):
        self.num_positions = (self.image_size // self.patch_size) ** 2 + 1
        self.num_heads: int = self.hidden_size // self.head_width

        if self.qk_scale == "auto":
            self.qk_scale = self.head_width**-0.5

    def __getitem__(self, idx: str):
        if not isinstance(idx, str):
            raise ValueError(f"{type(self)} does not support index type {type(idx)}")

        if not hasattr(self, idx):
            raise KeyError(f"Key {idx} not Found")

        return getattr(self, idx)

    @classmethod
    def keys(cls):
        return (f.name for f in fields(cls))


class EVACLIPVisionConfig(PretrainedConfig, CLIPVisionConfig):
    model_type = "eva_clip"
    base_config_key = "cross_vision_config"

    def __init__(
        self,
        attn_bias: bool = True,
        glu_intermediate_size: int = 11008,
        head_width: int = 112,
        hidden_act: str = "gelu",
        hidden_size: int = 1792,
        mlp_intermediate_size: int = 15360,
        num_hidden_layers: int = 63,
        outer_hidden_size: int = 4096,
        **kwargs,
    ):
        self.hidden_act = hidden_act
        self.head_width = head_width
        self.intermediate_size = glu_intermediate_size
        self.mlp_intermediate_size = mlp_intermediate_size
        self.outer_hidden_size = outer_hidden_size

        vision_kwargs, kwargs = CLIPVisionConfig.filter_kwargs(kwargs)
        CLIPVisionConfig.__init__(
            self,
            attn_bias=attn_bias,
            hidden_size=hidden_size,
            head_width=head_width,
            layers=num_hidden_layers,
            **vision_kwargs,
        )

        PretrainedConfig.__init__(self, **kwargs)


class EVALargeVisionConfig(PretrainedConfig, CLIPVisionConfig):
    model_type = "eva_large"
    base_config_key = "cross_vision_config"

    def __init__(
        self,
        attn_bias: bool = True,
        final_embed_dim: int = 768,
        hidden_size: int = 1024,
        hidden_act: str = "gelu",
        image_size: int = 1120,
        intp_freq: bool = True,
        layers: int = 24,
        layernorm_type: str = "BASE",
        mlp_ratio: float = 2.6667,
        naiveswiglu: bool = True,
        mlp_hidden_act: str = "silu",
        split_qkv: bool = True,
        use_abs_pos_emb: bool = True,
        use_mean_pooling: bool = False,
        use_rel_pos_bias: bool = False,
        use_rope: bool = True,
        use_shared_rel_pos_bias: bool = False,
        **kwargs,
    ):
        self.final_embed_dim = final_embed_dim
        self.hidden_act = hidden_act
        self.mlp_hidden_act = mlp_hidden_act

        self.use_abs_pos_emb = use_abs_pos_emb
        self.use_mean_pooling = use_mean_pooling
        self.use_rel_pos_bias = use_rel_pos_bias
        self.use_shared_rel_pos_bias = use_shared_rel_pos_bias

        vision_kwargs, kwargs = CLIPVisionConfig.filter_kwargs(kwargs)
        CLIPVisionConfig.__init__(
            self,
            attn_bias=attn_bias,
            hidden_size=hidden_size,
            image_size=image_size,
            intp_freq=intp_freq,
            layers=layers,
            layernorm_type=layernorm_type,
            mlp_ratio=mlp_ratio,
            naiveswiglu=naiveswiglu,
            split_qkv=split_qkv,
            use_rope=use_rope,
            **vision_kwargs,
        )

        PretrainedConfig.__init__(self, **kwargs)


class CogAgentConfig(PretrainedConfig):
    _auto_class = "AutoConfig"
    model_type = "cogagent"
    sub_configs = {
        "vision_config": EVACLIPVisionConfig,
        "cross_vision_config": EVALargeVisionConfig,
    }

    def __init__(
        self,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        vocab_size: int = 32000,
        image_token: str = "<EOI>",
        hidden_act: str = "silu",
        hidden_size: int = 4096,
        cross_image_size: int = 1120,
        cross_hidden_size: int = 1024,
        cross_compute_hidden_size: int = 1024,
        image_size: int = 224,
        initializer_range: float = 0.02,
        intermediate_size: int = 11008,
        max_position_embeddings: int = 2048,
        num_attention_heads: int = 32,
        num_hidden_layers: int = 32,
        rms_norm_eps: float = 1e-05,
        template_version: Literal["base", "chat", "chat_old"] = "chat_old",
        tie_word_embeddings: bool = False,
        torch_dtype: str = "bfloat16",
        cross_vision_config: EVALargeVisionConfig | dict[str, Any] | None = None,
        vision_config: EVACLIPVisionConfig | dict[str, Any] | None = None,
        **kwargs,
    ):
        self.cross_compute_hidden_size = cross_compute_hidden_size
        self.cross_hidden_size = cross_hidden_size
        self.hidden_size = hidden_size

        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.vocab_size = vocab_size

        # Replaces token_type_ids in model processing and forward.
        self.image_token = image_token

        self.num_hidden_layers = num_hidden_layers
        self.template_version = template_version
        self.hidden_act = hidden_act

        if vision_config is None:
            vision_config = EVACLIPVisionConfig(
                image_size=image_size,
                outer_hidden_size=hidden_size,
                glu_intermediate_size=intermediate_size,
                torch_dtype=torch_dtype,
            )
        elif isinstance(vision_config, dict):
            vision_hidden_size = vision_config.pop("hidden_size", 1792)
            mlp_intermediate_size = vision_config.pop("intermediate_size", 15360)

            vision_config["glu_intermediate_size"] = intermediate_size
            vision_config["torch_dtype"] = torch_dtype

            vision_config = EVACLIPVisionConfig(
                outer_hidden_size=hidden_size,
                hidden_size=vision_hidden_size,
                mlp_intermediate_size=mlp_intermediate_size,
                **vision_config,
            )

        if cross_vision_config is None:
            cross_vision_config = EVALargeVisionConfig(
                image_size=cross_image_size,
                hidden_size=cross_hidden_size,
                torch_dtype=torch_dtype,
            )

        elif isinstance(cross_vision_config, dict):
            cross_image_size = cross_vision_config.pop("image_size", cross_image_size)  # noqa: E501
            cross_hidden_size = cross_vision_config.pop(
                "hidden_size", cross_hidden_size
            )  # noqa: E501
            cross_vision_config["torch_dtype"] = torch_dtype

            cross_vision_config = EVALargeVisionConfig(
                image_size=cross_image_size,
                hidden_size=cross_hidden_size,
                **cross_vision_config,
            )

        self.cross_vision_config = cross_vision_config
        self.vision_config = vision_config

        self.image_size = vision_config.image_size
        self.cross_image_size = cross_vision_config.image_size

        # leaving this for documentation/future use. Tokenizer loads seperately.
        self.tokenizer = "lmsys/vicuna-7b-v1.5"

        super().__init__(
            num_attention_heads=num_attention_heads,
            cross_attention_hidden_size=cross_hidden_size,
            num_hidden_layers=num_hidden_layers,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            torch_dtype=torch_dtype,
            is_encoder_decoder=True,
            add_cross_attention=True,
            tie_encoder_decoder=True,
            **kwargs,
        )


AutoConfig.register(EVACLIPVisionConfig.model_type, EVACLIPVisionConfig)
AutoConfig.register(EVALargeVisionConfig.model_type, EVALargeVisionConfig)
AutoConfig.register(CogAgentConfig.model_type, CogAgentConfig)
