# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from typing import Any

from transformers import PreTrainedConfig, Qwen3_5Config


class SenseNovaLoopConfig(Qwen3_5Config):
    model_type = "sensenova_loop"

    def __init__(
        self,
        loop_settings: dict[str, Any] | None = None,
        mhc: dict[str, Any] | None = None,
        loop_physical_num_layers: int | None = None,
        loop_physical_layer_types: list[str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.loop_settings = loop_settings
        self.mhc = mhc
        self.loop_physical_num_layers = (
            loop_physical_num_layers or self.text_config.num_hidden_layers
        )
        self.loop_physical_layer_types = list(
            loop_physical_layer_types or self.text_config.layer_types
        )
        self.get_text_config()

    def get_text_config(self, **kwargs) -> PreTrainedConfig:
        text = super().get_text_config(**kwargs)
        # vLLM applies HF overrides after constructing this config.
        if not hasattr(self, "loop_physical_num_layers") or text is self:
            return text
        mhc = self.mhc
        num_layers = self.loop_physical_num_layers
        loop_settings = self.loop_settings
        if loop_settings is None:
            loop_settings = {"start": 0, "end": num_layers, "eval_count": 1}
        start, end, count = (loop_settings[k] for k in ("start", "end", "eval_count"))
        if (
            any(type(v) is not int for v in (start, end, count))
            or not 0 <= start < end <= num_layers
            or count < 1
            or loop_settings.get("use_gate", False)
        ):
            raise ValueError(
                "Loop requires valid fixed bounds/count and use_gate=false"
            )
        indices = list(range(start)) + list(range(start, end)) * count
        indices += list(range(end, num_layers))
        layer_types = self.loop_physical_layer_types
        if len(layer_types) != num_layers:
            raise ValueError("layer_types must describe the physical layers")
        if mhc is not None:
            pre_scale = mhc.get("pre_scale", 1.0)
            if (
                not isinstance(pre_scale, (int, float))
                or not math.isfinite(pre_scale)
                or pre_scale <= 0
            ):
                raise ValueError(
                    "Invalid mHC pre_scale: expected a positive finite number"
                )
            mhc_start, mhc_end, streams, iterations = (
                mhc[k] for k in ("start", "end", "streams", "iterations")
            )
            bounds = (0, start, end, num_layers)
            if (
                any(
                    type(v) is not int
                    for v in (mhc_start, mhc_end, streams, iterations)
                )
                or streams not in (2, 4)
                or iterations < 1
                or type(mhc["fixed_identity"]) is not bool
                or not any(
                    left <= mhc_start < mhc_end <= right
                    for left, right in zip(bounds, bounds[1:])
                )
            ):
                raise ValueError(
                    "Invalid mHC settings or range crossing a loop boundary"
                )
        text.loop_physical_indices = indices
        text.mhc = mhc
        text.num_hidden_layers = len(indices)
        text.layer_types = [layer_types[i] for i in indices]
        return text
