# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compatibility accessors for ERNIE 4.5 VL configuration layouts."""

from typing import Any

from transformers import PretrainedConfig


def _as_modality_pair(value: int | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(value, int):
        return [value, value]
    return list(value)


class _Ernie4_5VLConfigView:
    """Read-only flat view of the native Transformers composite config."""

    def __init__(self, config: PretrainedConfig) -> None:
        self._config = config
        self._text_config = config.text_config

    @property
    def im_patch_id(self) -> int:
        return self._config.image_token_id

    @property
    def spatial_conv_size(self) -> int:
        return self._config.vision_config.spatial_merge_size

    @property
    def temporal_conv_size(self) -> int:
        return self._config.vision_config.temporal_merge_size

    @property
    def moe_num_experts(self) -> list[int]:
        return _as_modality_pair(self._text_config.moe_num_experts)

    def __getattr__(self, name: str) -> Any:
        try:
            return getattr(self._config, name)
        except AttributeError:
            return getattr(self._text_config, name)


def get_ernie4_5_vl_config(config: PretrainedConfig) -> PretrainedConfig:
    """Return a model-facing view for native composite ERNIE configs."""
    if getattr(config, "text_config", None) is None:
        return config
    return _Ernie4_5VLConfigView(config)  # type: ignore[return-value]


def get_ernie4_5_vl_vision_norm_eps(config: PretrainedConfig) -> float:
    """Select the vision norm epsilon, with legacy flat-config fallback."""
    vision_config = config.vision_config
    return getattr(
        vision_config,
        "rms_norm_eps",
        getattr(config, "rms_norm_eps", 1e-6),
    )
