# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Alpamayo 2 Super configuration support.

The released checkpoint (``model_type: alpamayo2_super``) wraps a Qwen3-VL
model in ``vlm_config`` and stores a trajectory expert beside it. This shim
exposes the Qwen3-VL fields at the top level, so vLLM's Qwen3-VL code can size
the model, and keeps the wrapper's fields for the trajectory tokens.
"""

from collections.abc import Mapping
from typing import Any

from transformers import Qwen3VLConfig


class Alpamayo2SuperConfig(Qwen3VLConfig):
    model_type = "alpamayo2_super"

    def __init__(
        self,
        *,
        vlm_config: Mapping[str, Any] | None = None,
        expert_config: Mapping[str, Any] | None = None,
        traj_ids: Mapping[str, int] | None = None,
        history_vocab_size: int = 1000,
        future_vocab_size: int = 3000,
        tokens_per_history_traj: int = 45,
        include_camera_ids: bool = True,
        frame_label: str = "frame_num",
        min_pixels: int | None = 163840,
        max_pixels: int | None = 196608,
        **kwargs: Any,
    ) -> None:
        nested = dict(vlm_config or {})
        text_config = nested.pop("text_config", kwargs.pop("text_config", None))
        vision_config = nested.pop("vision_config", kwargs.pop("vision_config", None))
        for key in (
            "image_token_id",
            "video_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
        ):
            if key in nested:
                kwargs.setdefault(key, nested[key])
        super().__init__(
            text_config=text_config,
            vision_config=vision_config,
            **kwargs,
        )
        self.vlm_config = dict(vlm_config or {})
        self.expert_config = dict(expert_config or {})
        self.traj_ids = dict(traj_ids or {})
        self.history_vocab_size = history_vocab_size
        self.future_vocab_size = future_vocab_size
        # <i0>..<i{traj_vocab_size - 1}>: history bins, then future bins
        self.traj_vocab_size = history_vocab_size + future_vocab_size
        self.tokens_per_history_traj = tokens_per_history_traj
        self.include_camera_ids = include_camera_ids
        self.frame_label = frame_label
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
