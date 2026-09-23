# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 multimodal configuration."""

from transformers.configuration_utils import PretrainedConfig

from vllm.logger import init_logger
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

logger = init_logger(__name__)


class KimiK3VisionConfig(PretrainedConfig):
    model_type = "kimi_k3_vision"

    def __init__(
        self,
        patch_size: int = 14,
        init_pos_emb_height: int = 64,
        init_pos_emb_width: int = 64,
        init_pos_emb_time: int = 4,
        pos_emb_type: str = "divided_fixed",
        vt_num_attention_heads: int = 12,
        vt_num_hidden_layers: int = 27,
        vt_hidden_size: int = 1024,
        vt_intermediate_size: int = 4096,
        merge_kernel_size: tuple[int, int] = (2, 2),
        video_attn_type: str = "spatial_temporal",
        merge_type: str = "sd2_tpool",
        _attn_implementation: str = "flash_attention_2",
        mm_projector_type: str = "patchmergerv2",
        mm_hidden_size: int | None = None,
        projector_hidden_act: str = "gelu",
        projector_ln_eps: float = 1e-5,
        qkv_hidden_size: int = 1536,
        norm_type: str = "rmsnorm",
        attn_bias: bool = False,
        patch_embed_proj_bias: bool = False,
        mlp_type: str = "mlp2",
        linear_bias: bool = False,
        activation_func: str = "gelu_pytorch_tanh",
        pos_emb_interpolation_mode: str = "bilinear",
        text_hidden_size: int = 2304,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.patch_size = patch_size
        self.init_pos_emb_height = init_pos_emb_height
        self.init_pos_emb_width = init_pos_emb_width
        self.init_pos_emb_time = init_pos_emb_time
        self.pos_emb_type = pos_emb_type
        self.vt_num_attention_heads = vt_num_attention_heads
        self.vt_num_hidden_layers = vt_num_hidden_layers
        self.vt_hidden_size = vt_hidden_size
        self.vt_intermediate_size = vt_intermediate_size
        self.merge_kernel_size = tuple(merge_kernel_size)
        self.video_attn_type = video_attn_type
        self.merge_type = merge_type
        self._attn_implementation = _attn_implementation

        self.mm_projector_type = mm_projector_type
        self.mm_hidden_size = (
            mm_hidden_size if mm_hidden_size is not None else vt_hidden_size
        )
        self.projector_hidden_act = projector_hidden_act
        self.projector_ln_eps = projector_ln_eps
        self.text_hidden_size = text_hidden_size

        self.qkv_hidden_size = qkv_hidden_size
        self.norm_type = norm_type
        self.attn_bias = attn_bias
        self.patch_embed_proj_bias = patch_embed_proj_bias
        self.mlp_type = mlp_type
        self.linear_bias = linear_bias
        self.activation_func = activation_func
        self.pos_emb_interpolation_mode = pos_emb_interpolation_mode

        # Aliases consumed by the Kimi-K2.5 vision implementation.
        self.num_attention_heads = vt_num_attention_heads
        self.num_hidden_layers = vt_num_hidden_layers
        self.hidden_size = vt_hidden_size
        self.intermediate_size = vt_intermediate_size


class KimiK3Config(PretrainedConfig):
    model_type = "kimi_k3"

    def __init__(
        self,
        text_config: dict | KimiLinearConfig | None = None,
        vision_config: dict | KimiK3VisionConfig | None = None,
        ignore_index: int = -100,
        media_placeholder_token_id: int = 163605,
        pad_token_id: int = 0,
        image_placeholder: str = "<|kimi_image_placeholder|>",
        **kwargs,
    ):
        if text_config is None:
            self.text_config = KimiLinearConfig()
        elif isinstance(text_config, dict):
            self.text_config = KimiLinearConfig(**text_config)
        else:
            self.text_config = text_config

        if vision_config is None:
            self.vision_config = KimiK3VisionConfig()
        elif isinstance(vision_config, dict):
            self.vision_config = KimiK3VisionConfig(**vision_config)
        else:
            self.vision_config = vision_config

        # K3's vision projector output must match the text model's hidden size.
        # Override any value provided in vision_config for safety.
        if self.vision_config.text_hidden_size != self.text_config.hidden_size:
            logger.info(
                "Overriding vision_config.text_hidden_size from %s to %s "
                "to match text_config.hidden_size",
                self.vision_config.text_hidden_size,
                self.text_config.hidden_size,
            )
            self.vision_config.text_hidden_size = self.text_config.hidden_size

        self.ignore_index = ignore_index
        self.media_placeholder_token_id = media_placeholder_token_id
        self.image_placeholder = image_placeholder

        if getattr(self.text_config, "quantization_config", None) is not None:
            self.quantization_config = self.text_config.quantization_config

        super().__init__(pad_token_id=pad_token_id, **kwargs)

    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size

    @property
    def vocab_size(self) -> int:
        return self.text_config.vocab_size
