# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from vllm.config.multimodal import MultiModalConfig
from vllm.model_executor.models.qwen2_vl import Qwen2VLProcessingInfo
from vllm.model_executor.models.qwen3_vl import Qwen3VLProcessingInfo

pytestmark = pytest.mark.cpu_test

_UNIT = 16 * 2
_MODEL_MAX_PIXELS = 16_777_216
_MODEL_MIN_PIXELS = 65_536


@dataclass
class DummyVisionConfig:
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2


@dataclass
class DummyHFConfig:
    vision_config: DummyVisionConfig = field(default_factory=DummyVisionConfig)


class DummyImageProcessor:
    size = {
        "shortest_edge": _MODEL_MIN_PIXELS,
        "longest_edge": _MODEL_MAX_PIXELS,
    }


class DummyVideoProcessor:
    size = {
        "shortest_edge": _MODEL_MIN_PIXELS * 2,
        "longest_edge": _MODEL_MAX_PIXELS * 2,
    }
    temporal_patch_size = 2


class DummyQwenContext:
    def __init__(
        self,
        *,
        max_num_batched_tokens_hint: int | None,
        mm_processor_kwargs: dict[str, object] | None = None,
    ) -> None:
        mm_config = MultiModalConfig(mm_processor_kwargs=mm_processor_kwargs)
        self.max_num_batched_tokens_hint = max_num_batched_tokens_hint
        self.model_config = SimpleNamespace(
            get_multimodal_config=lambda: mm_config,
        )

    def get_merged_mm_kwargs(
        self,
        mm_kwargs: dict[str, object] | None,
    ) -> dict[str, object]:
        return self.model_config.get_multimodal_config().merge_mm_processor_kwargs(
            mm_kwargs or {}
        )


class DummyQwen2VLProcessingInfo(Qwen2VLProcessingInfo):
    def get_hf_config(self) -> DummyHFConfig:
        return DummyHFConfig()

    def get_image_processor(self, **kwargs: object) -> DummyImageProcessor:
        return DummyImageProcessor()


class DummyQwen3VLProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self) -> DummyHFConfig:
        return DummyHFConfig()

    def get_image_processor(self, **kwargs: object) -> DummyImageProcessor:
        return DummyImageProcessor()

    def get_video_processor(self, **kwargs: object) -> DummyVideoProcessor:
        return DummyVideoProcessor()


def _make_qwen2_info(
    *,
    max_num_batched_tokens_hint: int | None = 2048,
    mm_processor_kwargs: dict[str, object] | None = None,
) -> DummyQwen2VLProcessingInfo:
    return DummyQwen2VLProcessingInfo(
        DummyQwenContext(
            max_num_batched_tokens_hint=max_num_batched_tokens_hint,
            mm_processor_kwargs=mm_processor_kwargs,
        )
    )


def _make_qwen3_info(
    *,
    max_num_batched_tokens_hint: int | None = 2048,
    mm_processor_kwargs: dict[str, object] | None = None,
) -> DummyQwen3VLProcessingInfo:
    return DummyQwen3VLProcessingInfo(
        DummyQwenContext(
            max_num_batched_tokens_hint=max_num_batched_tokens_hint,
            mm_processor_kwargs=mm_processor_kwargs,
        )
    )


def _num_image_tokens_from_size(width: int, height: int) -> int:
    return (width // _UNIT) * (height // _UNIT)


def test_qwen_without_batched_token_hint_preserves_model_max_budget() -> None:
    info = _make_qwen2_info(max_num_batched_tokens_hint=None)

    image_size = info.get_image_size_with_most_features()

    assert _num_image_tokens_from_size(image_size.width, image_size.height) == 16_384


def test_qwen_batched_token_hint_caps_image_budget_path() -> None:
    info = _make_qwen2_info(max_num_batched_tokens_hint=2048)

    image_size = info.get_image_size_with_most_features()

    assert _num_image_tokens_from_size(image_size.width, image_size.height) == 2048


def test_qwen_batched_token_hint_caps_runtime_image_tokens() -> None:
    info = _make_qwen2_info(max_num_batched_tokens_hint=2048)

    num_tokens = info.get_num_image_tokens(
        image_width=4096,
        image_height=4096,
        image_processor=DummyImageProcessor(),
        mm_kwargs={},
    )

    assert num_tokens <= 2048


def test_qwen_explicit_max_pixels_preserves_model_max_budget() -> None:
    info = _make_qwen2_info(
        max_num_batched_tokens_hint=2048,
        mm_processor_kwargs={"max_pixels": _MODEL_MAX_PIXELS},
    )

    image_size = info.get_image_size_with_most_features()

    assert _num_image_tokens_from_size(image_size.width, image_size.height) == 16_384


def test_qwen_explicit_size_preserves_model_max_budget() -> None:
    info = _make_qwen2_info(
        max_num_batched_tokens_hint=2048,
        mm_processor_kwargs={
            "size": {
                "shortest_edge": _MODEL_MIN_PIXELS,
                "longest_edge": _MODEL_MAX_PIXELS,
            }
        },
    )

    image_size = info.get_image_size_with_most_features()

    assert _num_image_tokens_from_size(image_size.width, image_size.height) == 16_384


def test_qwen_min_pixels_is_floor_not_high_res_opt_in() -> None:
    info = _make_qwen2_info(
        max_num_batched_tokens_hint=1,
        mm_processor_kwargs={"min_pixels": _MODEL_MIN_PIXELS},
    )

    image_size = info.get_image_size_with_most_features()

    assert _num_image_tokens_from_size(image_size.width, image_size.height) == 64


def test_qwen2_video_budget_uses_batched_token_hint() -> None:
    info = _make_qwen2_info(max_num_batched_tokens_hint=2048)

    num_tokens = info.get_max_video_tokens(
        seq_len=32_768,
        mm_counts={"video": 1},
    )

    assert num_tokens <= 2048


def test_qwen3_video_budget_uses_batched_token_hint() -> None:
    info = _make_qwen3_info(max_num_batched_tokens_hint=2048)

    video_size = info.get_video_size_with_most_features()
    num_tokens = info.get_max_video_tokens(
        seq_len=32_768,
        mm_counts={"video": 1},
    )

    assert _num_image_tokens_from_size(video_size.width, video_size.height) <= 2048
    assert num_tokens <= 2048
