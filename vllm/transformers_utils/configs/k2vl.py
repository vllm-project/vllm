# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
K2-VL Model Configuration.

This configuration supports video-chunk as an internal modality type.
A video-chunk is the smallest independently processable unit of video,
typically 4 frames per chunk (temporal_merge_kernel_size=4).
"""

from transformers import DeepseekV3Config
from transformers.configuration_utils import PretrainedConfig


class K2VLConfig(PretrainedConfig):
    """K2-VL model configuration.

    K2-VL extends Kimi-VL with video support using video-chunks.
    A video-chunk consists of multiple consecutive frames (default: 4)
    that are processed together with temporal pooling.

    Args:
        text_config: Configuration for the text model (DeepseekV3).

        Vision Tower Parameters:
            patch_size: Patch size for vision tower.
            init_pos_emb_height: Initial position embedding height.
            init_pos_emb_width: Initial position embedding width.
            init_pos_emb_time: Initial position embedding time dimension.
            pos_emb_type: Type of position embedding.
            vt_num_attention_heads: Number of attention heads in vision tower.
            vt_num_hidden_layers: Number of hidden layers in vision tower.
            vt_hidden_size: Hidden size of vision tower.
            vt_intermediate_size: Intermediate size in vision tower FFN.
            merge_kernel_size: Kernel size for spatial patch merging.
            video_attn_type: Type of video attention.
            merge_type: Type of merge operation.

        Video-Chunk Parameters:
            temporal_merge_kernel_size: Number of frames per video chunk.
                Default is 4, meaning 4 frames are merged into 1 chunk.
            sample_fps: Video sampling frame rate.
            timestamp_mode: Format for chunk timestamps.

        MM Projector Parameters:
            mm_projector_type: Type of multimodal projector.
            mm_hidden_size: Hidden size from vision tower.
            projector_hidden_act: Activation function for projector.
            projector_ln_eps: Layer norm epsilon for projector.

        Other Parameters:
            ignore_index: The ignore index for the loss function.
            media_placeholder_token_id: The token ID for media placeholders.
            pad_token_id: The token ID for padding.
    """

    model_type = "k2_vl"

    def __init__(
        self,
        text_config: dict | DeepseekV3Config | None = None,
        # Vision tower parameters
        patch_size: int = 14,
        init_pos_emb_height: int = 64,
        init_pos_emb_width: int = 64,
        init_pos_emb_time: int = 4,
        pos_emb_type: str = "divided_fixed",
        vt_num_attention_heads: int = 16,
        vt_num_hidden_layers: int = 27,
        vt_hidden_size: int = 1152,
        vt_intermediate_size: int = 4304,
        merge_kernel_size: tuple[int, int] = (2, 2),
        video_attn_type: str = "spatial_temporal",
        merge_type: str = "sd2_tpool",
        # Video-chunk parameters
        temporal_merge_kernel_size: int = 4,
        sample_fps: float = 2.0,
        timestamp_mode: str = "hh:mm:ss.fff",
        # MM Projector parameters
        mm_projector_type: str = "patchmerger",
        mm_hidden_size: int | None = None,
        projector_hidden_act: str = "gelu",
        projector_ln_eps: float = 1e-5,
        # Other parameters
        ignore_index: int = -100,
        media_placeholder_token_id: int = 163605,
        pad_token_id: int = 0,
        use_unified_vision_chunk: bool = False,
        video_placeholder: str = "<|k2vl_video_placeholder|>",
        **kwargs,
    ):
        if text_config is None:
            text_config = DeepseekV3Config()
        elif isinstance(text_config, dict):
            text_config = DeepseekV3Config(**text_config)

        self.text_config: DeepseekV3Config = text_config

        # Vision tower config
        self.patch_size = patch_size
        self.init_pos_emb_height = init_pos_emb_height
        self.init_pos_emb_width = init_pos_emb_width
        self.init_pos_emb_time = init_pos_emb_time
        self.pos_emb_type = pos_emb_type
        self.vt_num_attention_heads = vt_num_attention_heads
        self.vt_num_hidden_layers = vt_num_hidden_layers
        self.vt_hidden_size = vt_hidden_size
        self.vt_intermediate_size = vt_intermediate_size
        self.merge_kernel_size = merge_kernel_size
        self.video_attn_type = video_attn_type
        self.merge_type = merge_type

        # Video-chunk config
        self.temporal_merge_kernel_size = temporal_merge_kernel_size
        self.sample_fps = sample_fps
        self.timestamp_mode = timestamp_mode

        # MM Projector config
        self.mm_projector_type = mm_projector_type
        self.mm_hidden_size = (
            mm_hidden_size if mm_hidden_size is not None else vt_hidden_size
        )
        self.projector_hidden_act = projector_hidden_act
        self.projector_ln_eps = projector_ln_eps

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
