# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for the MiniCPM-RobotTrack trajectory policy.

This config lets vLLM load ``MiniCPMRobotTrackModel`` without
``trust_remote_code``. The bundled ``backbone_config`` dict is wrapped into a
real ``PretrainedConfig`` so the reused ``MiniCPMModel`` backbone and vLLM's
config machinery (KV cache sizing, RoPE patching) can consume it directly.
"""

from typing import Any

import transformers
from transformers import PretrainedConfig


class MiniCPMRobotTrackConfig(PretrainedConfig):
    model_type = "minicpm_robottrack"

    def __init__(
        self,
        backbone_config: dict[str, Any] | PretrainedConfig | None = None,
        vision_feature_dim: int = 1536,
        history_frames: int = 31,
        coarse_tokens_per_frame: int = 4,
        fine_tokens_current_frame: int = 64,
        num_waypoints: int = 8,
        action_dim: int = 3,
        max_text_tokens: int = 128,
        max_time_steps: int = 4096,
        trajectory_dropout: float = 0.4,
        xy_scale: float = 2.0,
        use_tanh_actions: bool = True,
        backbone_dtype: str = "bfloat16",
        **kwargs: Any,
    ) -> None:
        self.backbone_config = self._wrap_backbone_config(backbone_config)
        self.vision_feature_dim = int(vision_feature_dim)
        self.history_frames = int(history_frames)
        self.coarse_tokens_per_frame = int(coarse_tokens_per_frame)
        self.fine_tokens_current_frame = int(fine_tokens_current_frame)
        self.num_waypoints = int(num_waypoints)
        self.action_dim = int(action_dim)
        self.max_text_tokens = int(max_text_tokens)
        self.max_time_steps = int(max_time_steps)
        self.trajectory_dropout = float(trajectory_dropout)
        self.xy_scale = float(xy_scale)
        self.use_tanh_actions = bool(use_tanh_actions)
        self.backbone_dtype = str(backbone_dtype)
        super().__init__(**kwargs)

    @staticmethod
    def _wrap_backbone_config(
        backbone_config: dict[str, Any] | PretrainedConfig | None,
    ) -> PretrainedConfig:
        if isinstance(backbone_config, PretrainedConfig):
            return backbone_config
        raw = dict(backbone_config or {})
        # MiniCPM4 is not in transformers' CONFIG_MAPPING; fall back to a bare
        # PretrainedConfig, which still migrates rope_scaling -> rope_parameters.
        model_type = raw.pop("model_type", "minicpm")
        if model_type in transformers.CONFIG_MAPPING:
            return transformers.CONFIG_MAPPING[model_type](**raw)
        backbone = PretrainedConfig(**raw)
        backbone.model_type = model_type
        # The bundled MiniCPMConfig defaults rope_theta to 10000.0, which the raw
        # backbone dict omits; inject it so RoPE's base is not left as None.
        backbone.rope_theta = 10000.0
        rope_parameters = getattr(backbone, "rope_parameters", None)
        if isinstance(rope_parameters, dict):
            rope_parameters.setdefault("rope_theta", 10000.0)
        return backbone

    def get_text_config(
        self, decoder: bool | None = None, encoder: bool | None = None
    ) -> PretrainedConfig:
        # The MiniCPM backbone is the language model: expose it so vLLM sizes the
        # KV cache and patches RoPE from the backbone rather than this wrapper.
        return self.backbone_config
