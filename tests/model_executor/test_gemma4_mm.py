# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import MethodType

import pytest
import torch

from vllm.model_executor.models.gemma4_mm import (
    _DEFAULT_VIDEO_VISION_BATCH_SIZE,
    Gemma4ForConditionalGeneration,
    Gemma4MultiModalProcessor,
    _Gemma4HFProcessorWrapper,
    _get_hf_processor_mm_kwargs,
    _validate_video_vision_batch_size,
)
from vllm.multimodal.inputs import MultiModalSharedField
from vllm.utils.func_utils import get_allowed_kwarg_only_overrides

pytestmark = pytest.mark.skip_global_cleanup


def test_validate_video_vision_batch_size_defaults() -> None:
    assert _validate_video_vision_batch_size(None) == _DEFAULT_VIDEO_VISION_BATCH_SIZE


def test_get_hf_processor_mm_kwargs_drops_local_only_fields() -> None:
    assert _get_hf_processor_mm_kwargs(
        {
            "max_soft_tokens": 70,
            "video_vision_batch_size": 4,
        }
    ) == {"max_soft_tokens": 70}


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "2"])
def test_validate_video_vision_batch_size_rejects_invalid_values(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="video_vision_batch_size"):
        _validate_video_vision_batch_size(value)


def test_get_mm_fields_config_keeps_video_batch_size_on_cpu() -> None:
    processor = object.__new__(Gemma4MultiModalProcessor)

    fields = Gemma4MultiModalProcessor._get_mm_fields_config(
        processor,
        {
            "video_frame_counts": torch.tensor([2, 1]),
            "video_vision_batch_size": 4,
        },
        {},
    )

    field = fields["video_vision_batch_size"].field
    assert isinstance(field, MultiModalSharedField)
    assert field.batch_size == 2
    assert field.keep_on_cpu is True


class _FakeProcessor:
    def __init__(self) -> None:
        self.captured_kwargs: dict[str, object] | None = None

    def __call__(self, *args: object, **kwargs: object):
        del args
        self.captured_kwargs = kwargs
        return kwargs


def test_wrapped_hf_processor_accepts_local_only_kwargs_without_forwarding() -> None:
    fake_hf_processor = _FakeProcessor()
    wrapped_processor = _Gemma4HFProcessorWrapper(fake_hf_processor)

    allowed_kwargs = get_allowed_kwarg_only_overrides(
        wrapped_processor,
        {
            "padding": True,
            "max_soft_tokens": 70,
            "video_vision_batch_size": 4,
            "not_valid": "drop-me",
        },
        requires_kw_only=False,
        allow_var_kwargs=True,
    )
    wrapped_processor(text="hello", **allowed_kwargs)

    assert allowed_kwargs == {
        "padding": True,
        "max_soft_tokens": 70,
        "video_vision_batch_size": 4,
    }
    assert fake_hf_processor.captured_kwargs == {
        "text": "hello",
        "max_soft_tokens": 70,
        "padding": True,
    }


def test_embed_multimodal_uses_request_video_batch_size() -> None:
    model = object.__new__(Gemma4ForConditionalGeneration)
    captured: dict[str, object] = {}

    def fake_process_video_input(self, video_input: dict[str, object]):
        captured["video_vision_batch_size"] = video_input["video_vision_batch_size"]
        return [torch.zeros(1)]

    model._process_video_input = MethodType(fake_process_video_input, model)

    outputs = Gemma4ForConditionalGeneration.embed_multimodal(
        model,
        pixel_values_videos=torch.zeros((1, 1, 1)),
        pixel_position_ids_videos=torch.zeros((1, 1, 2), dtype=torch.long),
        video_frame_counts=[1],
        video_num_soft_tokens=[[1]],
        video_vision_batch_size=4,
    )

    assert captured["video_vision_batch_size"] == 4
    assert len(outputs) == 1
