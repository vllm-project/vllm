# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import pytest
import torch

import vllm.platforms as vllm_platforms
from vllm.platforms.interface import UnspecifiedPlatform


@pytest.fixture()
def direct_input_contract(monkeypatch: pytest.MonkeyPatch):
    original_platform = vllm_platforms._current_platform
    test_platform = UnspecifiedPlatform()
    vllm_platforms._current_platform = test_platform

    try:
        from vllm.inputs import mm_input
        from vllm.multimodal.inputs import (
            MultiModalFieldElem,
            MultiModalKwargsItem,
            MultiModalKwargsItems,
            MultiModalSharedField,
            PlaceholderRange,
        )
        from vllm.sampling_params import SamplingParams
        from vllm.v1.engine import input_processor as input_processor_module
        from vllm.v1.engine.input_processor import InputProcessor

        monkeypatch.setattr(input_processor_module, "current_platform", test_platform)
        monkeypatch.setattr(
            test_platform, "validate_request", lambda *args, **kwargs: None
        )

        yield SimpleNamespace(
            InputProcessor=InputProcessor,
            MultiModalFieldElem=MultiModalFieldElem,
            MultiModalKwargsItem=MultiModalKwargsItem,
            MultiModalKwargsItems=MultiModalKwargsItems,
            MultiModalSharedField=MultiModalSharedField,
            PlaceholderRange=PlaceholderRange,
            SamplingParams=SamplingParams,
            mm_input=mm_input,
        )
    finally:
        vllm_platforms._current_platform = original_platform


class _Renderer:
    tokenizer = None

    @staticmethod
    def get_eos_token_id():
        return None


class _FailingPreprocessor:
    def preprocess(self, *args, **kwargs):
        raise AssertionError("Direct EngineInput must not be preprocessed")


def _build_input_processor(input_processor_cls: type[Any]):
    processor = input_processor_cls.__new__(input_processor_cls)
    processor.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_size_local=1,
            local_engines_only=False,
        ),
    )
    processor.model_config = SimpleNamespace(
        max_model_len=128,
        runner_type="generate",
    )
    processor.lora_config = None
    processor.generation_config_fields = {}
    processor.renderer = _Renderer()
    processor.supports_mm_inputs = True
    processor.mm_encoder_cache_size = 16
    processor.skip_prompt_length_check = False
    processor.input_preprocessor = _FailingPreprocessor()
    processor._validate_params = lambda *args, **kwargs: None
    processor._validate_lora = lambda *args, **kwargs: None
    return processor


def _mm_item(mm: Any, name: str, value: torch.Tensor):
    return mm.MultiModalKwargsItem(
        {
            name: mm.MultiModalFieldElem(
                data=value,
                field=mm.MultiModalSharedField(batch_size=1),
            )
        }
    )


def test_direct_multimodal_input_builds_mm_features(direct_input_contract: Any):
    mm = direct_input_contract
    processor = _build_input_processor(mm.InputProcessor)
    image_item = _mm_item(mm, "pixel_values", torch.arange(4))
    video_item = _mm_item(mm, "pixel_values_videos", torch.arange(8))
    is_embed = torch.tensor([True, False, True])
    image_position = mm.PlaceholderRange(offset=4, length=3, is_embed=is_embed)
    video_position = mm.PlaceholderRange(offset=1, length=2)

    engine_input = mm.mm_input(
        prompt_token_ids=[10, 11, 12, 13, 14, 15, 16],
        mm_kwargs=mm.MultiModalKwargsItems(
            {
                "image": [image_item],
                "video": [video_item],
            }
        ),
        mm_hashes={
            "image": ["image-hash"],
            "video": ["video-hash"],
        },
        mm_placeholders={
            "image": [image_position],
            "video": [video_position],
        },
        cache_salt="salt",
    )

    request = processor.process_inputs(
        request_id="request-id",
        prompt=engine_input,
        params=mm.SamplingParams(max_tokens=1),
        supported_tasks=("generate",),
        arrival_time=123.0,
    )

    assert request.prompt_token_ids == engine_input["prompt_token_ids"]
    assert request.prompt_embeds is None
    assert request.cache_salt == "salt"
    assert request.arrival_time == 123.0
    assert request.mm_features is not None

    # Features are flattened and sorted by their placeholder position.
    assert [feature.modality for feature in request.mm_features] == ["video", "image"]
    video_feature, image_feature = request.mm_features

    assert video_feature.data is video_item
    assert video_feature.identifier == "video-hash"
    assert video_feature.mm_hash == "video-hash"
    assert video_feature.mm_position == video_position

    assert image_feature.data is image_item
    assert image_feature.identifier == "image-hash"
    assert image_feature.mm_hash == "image-hash"
    assert image_feature.mm_position.offset == image_position.offset
    assert image_feature.mm_position.length == image_position.length
    assert image_feature.mm_position.is_embed is not None
    torch.testing.assert_close(image_feature.mm_position.is_embed, is_embed)


def test_direct_multimodal_input_rejects_non_string_hashes(direct_input_contract: Any):
    mm = direct_input_contract
    processor = _build_input_processor(mm.InputProcessor)
    engine_input = mm.mm_input(
        prompt_token_ids=[10, 11, 12],
        mm_kwargs=mm.MultiModalKwargsItems(
            {"image": [_mm_item(mm, "pixel_values", torch.arange(4))]}
        ),
        mm_hashes={"image": [123]},  # type: ignore[list-item]
        mm_placeholders={"image": [mm.PlaceholderRange(offset=1, length=1)]},
    )

    with pytest.raises(ValueError, match="mm_hashes must contain only strings"):
        processor.process_inputs(
            request_id="request-id",
            prompt=engine_input,
            params=mm.SamplingParams(max_tokens=1),
            supported_tasks=("generate",),
        )
