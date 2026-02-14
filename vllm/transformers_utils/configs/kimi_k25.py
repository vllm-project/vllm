# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Kimi-K2.5 Model Configuration.

This configuration supports video-chunk as an internal modality type.
A video-chunk is the smallest independently processable unit of video.
"""

from transformers import DeepseekV3Config
from transformers.configuration_utils import PretrainedConfig


class KimiK25VisionConfig(PretrainedConfig):
    model_type = "kimi_k25_vision"

    def __init__(
        self,
        # Vision Tower
        patch_size: int = 14,
        init_pos_emb_height: int = 64,
        init_pos_emb_width: int = 64,
        init_pos_emb_time: int = 4,
        pos_emb_type: str = "divided_fixed",
        num_attention_heads: int = 16,
        num_hidden_layers: int = 27,
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
        merge_kernel_size: tuple[int, int] = (2, 2),
        video_attn_type: str = "spatial_temporal",
        merge_type: str = "sd2_tpool",
        # MM Projector
        mm_projector_type: str = "patchmerger",
        mm_hidden_size: int | None = None,
        projector_hidden_act: str = "gelu",
        projector_ln_eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Vision Tower
        self.patch_size = patch_size
        self.init_pos_emb_height = init_pos_emb_height
        self.init_pos_emb_width = init_pos_emb_width
        self.init_pos_emb_time = init_pos_emb_time
        self.pos_emb_type = pos_emb_type
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.merge_kernel_size = merge_kernel_size
        self.video_attn_type = video_attn_type
        self.merge_type = merge_type
        # MM Projector
        self.mm_projector_type = mm_projector_type
        if mm_hidden_size is not None:
            self.mm_hidden_size = mm_hidden_size
        else:
            self.mm_hidden_size = hidden_size
        self.projector_hidden_act = projector_hidden_act
        self.projector_ln_eps = projector_ln_eps


class KimiK25Config(PretrainedConfig):
    """Kimi-K2.5 model configuration.

    Kimi-K2.5 extends Kimi-K2 with vision support using video-chunks.
    A video-chunk consists of multiple consecutive frames
    that are processed together with temporal pooling.

    Args:
        vision_config: Configuration for the vision tower and projector.
        text_config: Configuration for the text model (DeepseekV3).
        ignore_index: The ignore index for the loss function.
        media_placeholder_token_id: The token ID for media placeholders.
        pad_token_id: The token ID for padding.
    """

    model_type = "kimi_k25"

    def __init__(
        self,
        vision_config: dict | KimiK25VisionConfig | None = None,
        text_config: dict | DeepseekV3Config | None = None,
        ignore_index: int = -100,
        media_placeholder_token_id: int = 163605,
        pad_token_id: int = 0,
        use_unified_vision_chunk: bool = False,
        video_placeholder: str = "<|kimi_k25_video_placeholder|>",
        **kwargs,
    ):
        # Vision config
        if vision_config is None:
            vision_config = KimiK25VisionConfig()
        elif isinstance(vision_config, dict):
            vision_config = KimiK25VisionConfig(**vision_config)
        self.vision_config: KimiK25VisionConfig = vision_config

        # Text config
        if text_config is None:
            text_config = DeepseekV3Config()
        elif isinstance(text_config, dict):
            text_config = DeepseekV3Config(**text_config)
        self.text_config: DeepseekV3Config = text_config

        # Set mm_hidden_size to text hidden size if not explicitly set
        if self.vision_config.mm_hidden_size == self.vision_config.hidden_size:
            self.vision_config.mm_hidden_size = self.text_config.hidden_size

        # Other config
        self.ignore_index = ignore_index
        self.media_placeholder_token_id = media_placeholder_token_id
        self.use_unified_vision_chunk = use_unified_vision_chunk
        self.video_placeholder = video_placeholder

        # Propagate quantization config from text model
        if getattr(self.text_config, "quantization_config", None) is not None:
            self.quantization_config = self.text_config.quantization_config

        super().__init__(pad_token_id=pad_token_id, **kwargs)

    @property
    def hidden_size(self) -> int:
        """Get hidden size from text config for compatibility."""
        return self.text_config.hidden_size

    @property
    def vocab_size(self) -> int:
        """Get vocab size from text config for compatibility."""
        return self.text_config.vocab_size
