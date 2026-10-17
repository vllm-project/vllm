# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from transformers.configuration_utils import PreTrainedConfig

from vllm.transformers_utils.configs.step3p5 import Step3p5Config


class Step5SparseConfig:
    """The ``sparse_config`` block of a Step-5 text config."""

    def __init__(
        self,
        enabled: bool = True,
        proxy_dim: int = 256,
        sparse_indexer_rope_dim: int = 32,
        sparse_indexer_use_rope: bool = True,
        sparse_indexer_num_heads: int = 16,
        sparse_indexer_num_k_heads: int = 1,
        sparse_indexer_q_norm_type: str = "rmsnorm",
        sparse_indexer_k_norm_type: str = "layernorm",
        sparse_indexer_csa_z_norm_type: str = "none",
        sparse_indexer_softmax_variant: str = "ssmax",
        sparse_indexer_ssmax_s_granularity: str = "q_head",
        num_provider_groups: int = 4,
        topk: int = 512,
        region_block_size: int = 8,
        compression_method: str = "csa_block_compress",
        attention_impl: str = "sparse_gqa",
        apply_to_layer_types: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self.enabled = enabled
        self.proxy_dim = proxy_dim
        self.sparse_indexer_rope_dim = sparse_indexer_rope_dim
        self.sparse_indexer_use_rope = sparse_indexer_use_rope
        self.sparse_indexer_num_heads = sparse_indexer_num_heads
        self.sparse_indexer_num_k_heads = sparse_indexer_num_k_heads
        self.sparse_indexer_q_norm_type = sparse_indexer_q_norm_type
        self.sparse_indexer_k_norm_type = sparse_indexer_k_norm_type
        self.sparse_indexer_csa_z_norm_type = sparse_indexer_csa_z_norm_type
        self.sparse_indexer_softmax_variant = sparse_indexer_softmax_variant
        self.sparse_indexer_ssmax_s_granularity = sparse_indexer_ssmax_s_granularity
        self.num_provider_groups = num_provider_groups
        self.topk = topk
        self.region_block_size = region_block_size
        self.compression_method = compression_method
        self.attention_impl = attention_impl
        self.apply_to_layer_types = list(apply_to_layer_types or ["full_attention"])
        self._validate()

    _SUPPORTED = {
        "sparse_indexer_num_k_heads": 1,
        "sparse_indexer_use_rope": True,
        "sparse_indexer_q_norm_type": "rmsnorm",
        "sparse_indexer_k_norm_type": "layernorm",
        "sparse_indexer_csa_z_norm_type": "none",
        "sparse_indexer_softmax_variant": "ssmax",
        "sparse_indexer_ssmax_s_granularity": "q_head",
        "compression_method": "csa_block_compress",
        "attention_impl": "sparse_gqa",
    }

    def _validate(self) -> None:
        for name, value in self._SUPPORTED.items():
            if getattr(self, name) != value:
                raise NotImplementedError(
                    f"Step-5 sparse_config.{name}={getattr(self, name)!r} is "
                    f"unsupported; expected {value!r}."
                )
        if self.sparse_indexer_num_heads % self.num_provider_groups:
            raise ValueError(
                f"{self.sparse_indexer_num_heads} indexer heads do not divide "
                f"into {self.num_provider_groups} provider groups."
            )

    @property
    def indexer_heads_per_group(self) -> int:
        return self.sparse_indexer_num_heads // self.num_provider_groups

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class Step5Config(Step3p5Config):
    model_type = "step5"

    def __init__(
        self,
        sparse_config: dict[str, Any] | Step5SparseConfig | None = None,
        partial_rotary_factors: list[float] | None = None,
        sliding_window: int | None = None,
        moe_layer_list: list[int] | None = None,
        moe_layers_enum: str | None = None,
        moe_router_activation: str = "sigmoid",
        **kwargs: Any,
    ) -> None:
        if isinstance(sparse_config, dict):
            sparse_config = Step5SparseConfig(**sparse_config)
        self.sparse_config = sparse_config
        self.partial_rotary_factors = partial_rotary_factors
        self.sliding_window = sliding_window
        self.moe_layer_list = moe_layer_list
        if moe_layers_enum is None and moe_layer_list is not None:
            moe_layers_enum = ",".join(str(i) for i in moe_layer_list)
        self.moe_layers_enum = moe_layers_enum
        layer_types = kwargs.get("layer_types")
        super().__init__(moe_router_activation=moe_router_activation, **kwargs)
        # Step3p5Config drops the MTP layers' entries.
        if layer_types is not None:
            self.layer_types = list(layer_types)

    def is_sparse_layer(self, layer_idx: int) -> bool:
        sparse = self.sparse_config
        if sparse is None or not sparse.enabled or not self.layer_types:
            return False
        return self.layer_types[layer_idx] in sparse.apply_to_layer_types

    def to_dict(self) -> dict[str, Any]:
        output = super().to_dict()
        if isinstance(output.get("sparse_config"), Step5SparseConfig):
            output["sparse_config"] = output["sparse_config"].to_dict()
        return output


class Step5VisionConfig(PreTrainedConfig):
    model_type = "perception_encoder"

    def __init__(
        self,
        width: int = 1536,
        layers: int = 47,
        heads: int = 16,
        num_channels: int = 3,
        image_size: int = 728,
        mlp_ratio: float = 8960 / 1536,
        patch_size: int = 14,
        hidden_act: str = "quick_gelu",
        layer_norm_eps: float = 1e-5,
        use_cls_token: bool = False,
        use_ln_pre: bool = True,
        use_ln_post: bool = False,
        use_abs_posemb: bool = True,
        use_rope2d: bool = True,
        ls_init_value: float = 0.1,
        output_dim: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.width = width
        self.layers = layers
        self.heads = heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.use_cls_token = use_cls_token
        self.use_ln_pre = use_ln_pre
        self.use_ln_post = use_ln_post
        self.use_abs_posemb = use_abs_posemb
        self.use_rope2d = use_rope2d
        self.ls_init_value = ls_init_value
        self.output_dim = output_dim
        super().__init__(**kwargs)


class Step5VLConfig(PreTrainedConfig):
    model_type = "step5_vl"
    sub_configs = {"text_config": Step5Config, "vision_config": Step5VisionConfig}

    def __init__(
        self,
        text_config: dict[str, Any] | Step5Config | None = None,
        vision_config: dict[str, Any] | Step5VisionConfig | None = None,
        image_token_id: int = 128001,
        understand_projector_stride: int = 2,
        projector_bias: bool = False,
        **kwargs: Any,
    ) -> None:
        if isinstance(text_config, dict):
            text_config = Step5Config(**text_config)
        if isinstance(vision_config, dict):
            vision_config = Step5VisionConfig(**vision_config)
        self.text_config = text_config or Step5Config()
        self.vision_config = vision_config or Step5VisionConfig()
        self.image_token_id = image_token_id
        self.understand_projector_stride = understand_projector_stride
        self.projector_bias = projector_bias
        super().__init__(**kwargs)

    def get_text_config(self, decoder: bool = False) -> Step5Config:
        return self.text_config
