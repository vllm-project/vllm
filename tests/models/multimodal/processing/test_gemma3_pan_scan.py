# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
from transformers import BatchFeature, Gemma3Processor
from transformers.models.gemma3.image_processing_gemma3 import Gemma3ImageProcessor
from transformers.models.gemma3.processing_gemma3 import Gemma3ProcessorKwargs

from vllm.model_executor.models.gemma3_mm import (
    Gemma3MultiModalProcessor,
    Gemma3ProcessingInfo,
)
from vllm.multimodal.parse import ImageSize
from vllm.multimodal.processing.processor import BaseMultiModalProcessor


class _Gemma3MMConfig:
    def __init__(self, mm_processor_kwargs: dict[str, object] | None = None):
        self.mm_processor_kwargs = mm_processor_kwargs or {}

    def merge_mm_processor_kwargs(
        self, inference_kwargs: dict[str, object]
    ) -> dict[str, object]:
        return self.mm_processor_kwargs | dict(inference_kwargs)


class _Gemma3ProcessingContext:
    def __init__(self, mm_processor_kwargs: dict[str, object] | None = None):
        self._mm_config = _Gemma3MMConfig(mm_processor_kwargs)

    def get_merged_mm_kwargs(self, kwargs: dict[str, object]) -> dict[str, object]:
        return self._mm_config.merge_mm_processor_kwargs(kwargs)

    def get_hf_processor(self, typ, **kwargs: object):
        assert typ is Gemma3Processor
        return _build_hf_processor(self.get_merged_mm_kwargs(dict(kwargs)))


def _build_hf_processor(mm_kwargs: dict[str, object]) -> Gemma3Processor:
    processor = object.__new__(Gemma3Processor)
    processor.tokenizer = SimpleNamespace(init_kwargs={})
    processor.image_processor = Gemma3ImageProcessor()
    images_kwargs = processor._merge_kwargs(
        Gemma3ProcessorKwargs,
        tokenizer_init_kwargs=processor.tokenizer.init_kwargs,
        **mm_kwargs,
    )["images_kwargs"]
    processor.image_processor = Gemma3ImageProcessor(**images_kwargs)
    return processor


def _get_num_crops(
    *,
    deployment_kwargs: dict[str, object] | None = None,
    request_kwargs: dict[str, object] | None = None,
    image_width: int = 65536,
    image_height: int = 512,
) -> int:
    info = Gemma3ProcessingInfo(_Gemma3ProcessingContext(deployment_kwargs))
    request_kwargs = request_kwargs or {}
    processor = info.get_hf_processor(**request_kwargs)
    return info.get_num_crops(
        image_width=image_width,
        image_height=image_height,
        processor=processor,
        mm_kwargs=request_kwargs,
    )


@pytest.mark.parametrize(
    "request_kwargs",
    [
        {"pan_and_scan_max_num_crops": 128},
        {"images_kwargs": {"pan_and_scan_max_num_crops": 128}},
    ],
)
def test_rejects_crop_fanout_above_deployment_ceiling(
    request_kwargs: dict[str, object],
):
    with pytest.raises(ValueError, match="exceeds the deployed limit"):
        _get_num_crops(
            deployment_kwargs={"do_pan_and_scan": True},
            request_kwargs=request_kwargs,
        )


def test_accepts_crop_fanout_at_operator_selected_ceiling():
    assert (
        _get_num_crops(
            deployment_kwargs={
                "do_pan_and_scan": True,
                "pan_and_scan_max_num_crops": 8,
            },
            request_kwargs={"pan_and_scan_max_num_crops": 8},
        )
        == 8
    )


def test_accepts_high_named_cap_when_geometry_stays_within_ceiling():
    assert (
        _get_num_crops(
            deployment_kwargs={"do_pan_and_scan": True},
            request_kwargs={"pan_and_scan_max_num_crops": 128},
            image_width=1024,
            image_height=512,
        )
        == 2
    )


@pytest.mark.parametrize(
    "request_kwargs",
    [
        {"pan_and_scan_min_crop_size": 0},
        {"pan_and_scan_max_num_crops": 0},
    ],
)
def test_rejects_non_positive_crop_arithmetic_inputs(
    request_kwargs: dict[str, object],
):
    with pytest.raises(ValueError, match="must be a positive integer"):
        _get_num_crops(
            deployment_kwargs={"do_pan_and_scan": True},
            request_kwargs=request_kwargs,
        )


class _FakeParsedImages:
    def __len__(self) -> int:
        return 1

    def get_image_size(self, item_idx: int) -> ImageSize:
        assert item_idx == 0
        return ImageSize(width=65536, height=512)


class _FakeMMItems:
    def get_items(self, modality: str, item_type):
        assert modality == "image"
        return _FakeParsedImages()


class _RejectingInfo:
    def parse_mm_data(self, mm_data, validate=False):
        del mm_data, validate
        return _FakeMMItems()

    def get_hf_processor(self, **mm_kwargs: object):
        del mm_kwargs
        return object()

    def get_num_crops(self, **kwargs: object) -> int:
        del kwargs
        raise ValueError("Gemma3 pan-and-scan crop count exceeds the deployed limit")


def test_rejects_before_hf_preprocessing(monkeypatch: pytest.MonkeyPatch):
    hf_called = False

    def fake_call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs):
        nonlocal hf_called
        del self, prompt, mm_data, mm_kwargs, tok_kwargs
        hf_called = True
        return BatchFeature({})

    monkeypatch.setattr(
        BaseMultiModalProcessor,
        "_call_hf_processor",
        fake_call_hf_processor,
    )

    processor = object.__new__(Gemma3MultiModalProcessor)
    processor.info = _RejectingInfo()

    with pytest.raises(ValueError, match="exceeds the deployed limit"):
        processor._call_hf_processor(
            "prompt",
            {"images": [object()]},
            {},
            {},
        )

    assert not hf_called
