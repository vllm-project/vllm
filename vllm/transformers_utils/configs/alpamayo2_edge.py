# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Alpamayo 2 Edge configuration support.

The released checkpoint (`model_type: alpamayo2_edge`) wraps a Cosmos3-Edge
model in `vlm_config` and stores a trajectory expert beside it. This shim
exposes the Cosmos3-Edge fields at the top level, so vLLM's Cosmos3-Edge code
can size the model, and keeps the wrapper's fields for the trajectory tokens.
"""

from collections.abc import Mapping
from typing import Any

from transformers import Cosmos3EdgeConfig


class Alpamayo2EdgeConfig(Cosmos3EdgeConfig):
    model_type = "alpamayo2_edge"

    def __init__(
        self,
        *,
        vlm_config: Mapping[str, Any] | None = None,
        expert_config: Mapping[str, Any] | None = None,
        traj_ids: Mapping[str, int] | None = None,
        history_vocab_size: int = 1000,
        future_vocab_size: int = 3000,
        **kwargs: Any,
    ) -> None:
        nested = dict(vlm_config or {})
        text_config = nested.pop("text_config", kwargs.pop("text_config", None))
        vision_config = nested.pop("vision_config", kwargs.pop("vision_config", None))
        for key in (
            "projector_hidden_size",
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
