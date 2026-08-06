# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace

import pytest
from PIL import Image
from transformers import BatchFeature

from vllm.model_executor.models.phi4mm import (
    Phi4MMMultiModalProcessor,
    Phi4MMProcessingInfo,
)
from vllm.multimodal.parse import ImageSize
from vllm.multimodal.processing.processor import BaseMultiModalProcessor


class _FakeImageProcessor:
    def __init__(self, dynamic_hd: object):
        self.dynamic_hd = dynamic_hd

    def find_closest_aspect_ratio(
        self,
        aspect_ratio: float,
        target_ratios: list[tuple[int, int]],
        width: int,
        height: int,
        image_size: int,
    ) -> tuple[int, int]:
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            ratio_diff = abs(aspect_ratio - ratio[0] / ratio[1])
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio

        return best_ratio


class _Phi4MMProcessingContext:
    def __init__(self, dynamic_hd: object = 36):
        self.dynamic_hd = dynamic_hd

    def get_hf_config(self) -> SimpleNamespace:
        return SimpleNamespace(img_processor=None)

    def get_hf_processor(self, **kwargs: object) -> SimpleNamespace:
        dynamic_hd = kwargs.get("dynamic_hd", self.dynamic_hd)
        return SimpleNamespace(
            image_processor=_FakeImageProcessor(dynamic_hd),
            audio_processor=SimpleNamespace(sampling_rate=16000),
        )


def _get_num_image_tokens(
    *,
    deployment_dynamic_hd: object = 36,
    request_dynamic_hd: object = 36,
    image_width: int = 448 * 128,
    image_height: int = 448,
) -> int:
    info = Phi4MMProcessingInfo(_Phi4MMProcessingContext(deployment_dynamic_hd))
    processor = info.get_hf_processor(dynamic_hd=request_dynamic_hd)
    return info.get_num_image_tokens(
        image_width=image_width,
        image_height=image_height,
        processor=processor,
    )


def test_rejects_crop_fanout_above_deployment_ceiling():
    with pytest.raises(ValueError, match="exceeds the deployed limit"):
        _get_num_image_tokens(request_dynamic_hd=128)


def test_accepts_operator_selected_higher_trusted_ceiling():
    assert (
        _get_num_image_tokens(
            deployment_dynamic_hd=128,
            request_dynamic_hd=128,
        )
        > 0
    )


def test_accepts_high_named_cap_when_geometry_stays_within_ceiling():
    assert _get_num_image_tokens(
        request_dynamic_hd=128,
        image_width=448,
        image_height=448,
    ) == _get_num_image_tokens(
        request_dynamic_hd=36,
        image_width=448,
        image_height=448,
    )


def test_accepts_high_named_cap_when_search_result_stays_within_ceiling():
    assert _get_num_image_tokens(
        request_dynamic_hd=37,
        image_width=448 * 2,
        image_height=448 * 19,
    ) == _get_num_image_tokens(
        request_dynamic_hd=36,
        image_width=448 * 2,
        image_height=448 * 19,
    )


def test_ratio_search_candidate_count_is_sublinear_in_dynamic_hd(
    monkeypatch: pytest.MonkeyPatch,
):
    info = Phi4MMProcessingInfo(_Phi4MMProcessingContext())
    candidate_counts: list[int] = []
    original = _FakeImageProcessor.find_closest_aspect_ratio

    def capture_candidate_count(
        self: _FakeImageProcessor,
        aspect_ratio: float,
        target_ratios: list[tuple[int, int]],
        width: int,
        height: int,
        image_size: int,
    ) -> tuple[int, int]:
        candidate_counts.append(len(target_ratios))
        return original(self, aspect_ratio, target_ratios, width, height, image_size)

    monkeypatch.setattr(
        _FakeImageProcessor,
        "find_closest_aspect_ratio",
        capture_candidate_count,
    )

    info._find_target_aspect_ratio(
        448 * 129,
        448,
        448,
        128,
        min_num=1,
    )

    assert candidate_counts
    assert candidate_counts[0] <= 4 * math.isqrt(128)


@pytest.mark.parametrize("dynamic_hd", [0, -1, True, False, 1.5])
def test_rejects_invalid_dynamic_hd(dynamic_hd: object):
    with pytest.raises(ValueError, match="must be a positive integer"):
        _get_num_image_tokens(
            request_dynamic_hd=dynamic_hd,
            image_width=448,
            image_height=448,
        )


class _FakeParsedImages:
    def __init__(self, image_size: ImageSize | None = None):
        self.image_size = image_size or ImageSize(width=448 * 128, height=448)

    def __len__(self) -> int:
        return 1

    def get_image_size(self, item_idx: int) -> ImageSize:
        assert item_idx == 0
        return self.image_size


class _FakeMMItems:
    def __init__(self, image_size: ImageSize | None = None):
        self.image_size = image_size

    def get_items(self, modality: str, item_type):
        assert modality == "image"
        return _FakeParsedImages(self.image_size)


class _RejectingInfo:
    def parse_mm_data(self, mm_data, validate=False):
        del mm_data, validate
        return _FakeMMItems()

    def get_hf_processor(self, **mm_kwargs: object):
        del mm_kwargs
        return object()

    def get_feature_extractor(self, **mm_kwargs: object):
        del mm_kwargs
        return SimpleNamespace(sampling_rate=16000)

    def get_num_image_tokens(self, **kwargs: object) -> int:
        del kwargs
        raise ValueError("Phi4MM local HD crop count exceeds the deployed limit")


def test_rejects_before_hf_preprocessing(monkeypatch: pytest.MonkeyPatch):
    hf_called = False

    def fake_call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs):
        nonlocal hf_called
        del self, prompt, mm_data, mm_kwargs, tok_kwargs
        hf_called = True
        return BatchFeature(
            {
                "image_sizes": [[448 * 128, 448]],
                "input_audio_embeds": [],
            }
        )

    monkeypatch.setattr(
        BaseMultiModalProcessor,
        "_call_hf_processor",
        fake_call_hf_processor,
    )

    processor = object.__new__(Phi4MMMultiModalProcessor)
    processor.info = _RejectingInfo()

    with pytest.raises(ValueError, match="exceeds the deployed limit"):
        processor._call_hf_processor(
            "prompt",
            {"images": [object()]},
            {},
            {},
        )

    assert not hf_called


def test_high_named_cap_is_canonicalized_before_hf_preprocessing(
    monkeypatch: pytest.MonkeyPatch,
):
    captured_mm_kwargs: dict[str, object] = {}

    def fake_call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs):
        del self, prompt, mm_data, tok_kwargs
        captured_mm_kwargs.update(mm_kwargs)
        return BatchFeature(
            {
                "image_sizes": [[448, 448]],
                "input_audio_embeds": [],
            }
        )

    monkeypatch.setattr(
        BaseMultiModalProcessor,
        "_call_hf_processor",
        fake_call_hf_processor,
    )

    processor = object.__new__(Phi4MMMultiModalProcessor)
    processor.info = Phi4MMProcessingInfo(_Phi4MMProcessingContext())
    monkeypatch.setattr(
        processor.info,
        "parse_mm_data",
        lambda mm_data, validate=False: _FakeMMItems(ImageSize(width=448, height=448)),
    )
    processor._call_hf_processor(
        "prompt",
        {"images": [Image.new("RGB", (448, 448))]},
        {"dynamic_hd": 128},
        {},
    )

    assert captured_mm_kwargs == {"dynamic_hd": 36}
