# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for MiniCPMV's multimodal preprocessing."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.model_executor.models.minicpmo import MiniCPMOMultiModalProcessor
from vllm.model_executor.models.minicpmv import MiniCPMVMultiModalProcessor
from vllm.model_executor.models.minicpmv4_6 import MiniCPMV4_6MultiModalProcessor
from vllm.multimodal import MULTIMODAL_REGISTRY

from ...utils import build_model_context


@pytest.mark.parametrize("model_id", ["openbmb/MiniCPM-V-4"])
def test_get_hf_processor_for_same_model_different_kwargs(model_id: str):
    """Calls with different kwargs must not reuse stale processor instances."""
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    info = processor.info
    processor_1 = info.get_hf_processor(max_slice_nums=1)
    processor_2 = info.get_hf_processor(max_slice_nums=2)
    assert processor_1.image_processor.max_slice_nums == 1
    assert processor_2.image_processor.max_slice_nums == 2


@pytest.mark.parametrize(
    "model_ids", [("openbmb/MiniCPM-Llama3-V-2_5", "openbmb/MiniCPM-V-4")]
)
def test_image_processor_for_dif_model(model_ids):
    model_id_25, model_id_4 = model_ids

    ctx_25 = build_model_context(model_id_25, limit_mm_per_prompt={"image": 1})
    processor_25 = MULTIMODAL_REGISTRY.create_processor(ctx_25.model_config)
    image_processor_25 = processor_25.info.get_image_processor()

    ctx_4 = build_model_context(model_id_4, limit_mm_per_prompt={"image": 1})
    processor_4 = MULTIMODAL_REGISTRY.create_processor(ctx_4.model_config)
    image_processor_4 = processor_4.info.get_image_processor()

    assert type(image_processor_25) is not type(image_processor_4)
    assert type(image_processor_25).__module__ != type(image_processor_4).__module__


@pytest.mark.parametrize("model_id", ["openbmb/MiniCPM-V-4"])
def test_prompt_has_dif_BPE_boundaries_in_context(model_id: str):
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    tokenizer = ctx.get_tokenizer()

    messages = [
        {"role": "user", "content": "(<image>./</image>)\nWhat is in this image?"}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image = np.zeros((768, 1024, 3), dtype=np.uint8)

    mm_items = processor.info.parse_mm_data({"image": [image]})
    processed = processor(
        prompt,
        mm_items=mm_items,
        hf_processor_mm_kwargs={},
    )
    image_placeholders = processed["mm_placeholders"].get("image", [])
    assert len(image_placeholders) == 1
    assert image_placeholders[0].length > 0


class _ParsedMMItems:
    def get_items(self, modality, item_types):
        del modality, item_types
        return ["image"]


class _ImageProcessor:
    patch_size = 1

    def __init__(self, max_slice_nums: int):
        self.max_slice_nums = max_slice_nums
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def __call__(self, images, **kwargs):
        self.calls.append((images, kwargs))
        return {
            "pixel_values": torch.ones((1, 3, 1, 1)),
            "target_sizes": torch.tensor([[1, 1]]),
        }


class _ProcessingInfo:
    image_pattern = "<image>"

    def __init__(self, max_slice_nums: int):
        self.max_slice_nums = max_slice_nums
        self.image_processor = _ImageProcessor(max_slice_nums)
        self.hf_config = SimpleNamespace(insert_layer_id=-1)

    def parse_mm_data(self, data, validate=False):
        assert data == {"image": ["image"]}
        assert validate is False
        return _ParsedMMItems()

    def get_image_processor(self):
        return self.image_processor

    def get_image_processor_max_slice_num(self) -> int:
        return self.max_slice_nums

    def _get_downsample_mode(self) -> str:
        return "16x"

    def get_hf_config(self):
        return self.hf_config


def _build_shared_processor(processor_cls, max_slice_nums: int):
    processor = object.__new__(processor_cls)
    processor.info = _ProcessingInfo(max_slice_nums)
    processor.hf_calls = []

    def _base_call_hf_processor(*, mm_kwargs, **kwargs):
        del kwargs
        processor.hf_calls.append(dict(mm_kwargs))
        return {}

    processor._base_call_hf_processor = _base_call_hf_processor
    return processor


def _build_v46_processor(max_slice_nums: int):
    processor = object.__new__(MiniCPMV4_6MultiModalProcessor)
    processor.info = _ProcessingInfo(max_slice_nums)
    return processor


@pytest.mark.parametrize(
    "processor_cls",
    [
        pytest.param(MiniCPMVMultiModalProcessor, id="minicpmv"),
        pytest.param(MiniCPMOMultiModalProcessor, id="minicpmo"),
    ],
)
@pytest.mark.parametrize("request_max_slice_nums", [8, 10, 9.0])
def test_shared_minicpm_image_paths_reject_request_max_slice_nums_mismatch(
    processor_cls,
    request_max_slice_nums: object,
):
    processor = _build_shared_processor(processor_cls, max_slice_nums=9)

    with pytest.raises(
        ValueError,
        match="must match the deployed MiniCPM image processor configuration",
    ):
        processor.process_images(
            {"images": ["image"]},
            {"max_slice_nums": request_max_slice_nums},
            {},
        )

    assert processor.hf_calls == []


@pytest.mark.parametrize(
    "processor_cls",
    [
        pytest.param(MiniCPMVMultiModalProcessor, id="minicpmv"),
        pytest.param(MiniCPMOMultiModalProcessor, id="minicpmo"),
    ],
)
def test_shared_minicpm_image_paths_allow_deployed_max_slice_nums(processor_cls):
    processor = _build_shared_processor(processor_cls, max_slice_nums=4)

    processor.process_images(
        {"images": ["image"]},
        {"max_slice_nums": 4, "do_pad": False},
        {},
    )

    assert processor.hf_calls == [{"max_slice_nums": 4, "do_pad": False}]


@pytest.mark.parametrize(
    "mm_kwargs",
    [
        pytest.param({}, id="omitted"),
        pytest.param({"max_slice_nums": None}, id="explicit-none"),
    ],
)
def test_shared_minicpm_image_paths_allow_unset_max_slice_nums(mm_kwargs):
    processor = _build_shared_processor(MiniCPMVMultiModalProcessor, max_slice_nums=4)

    processor.process_images({"images": ["image"]}, mm_kwargs, {})

    assert len(processor.hf_calls) == 1


@pytest.mark.parametrize("request_max_slice_nums", [8, 10, 9.0])
def test_minicpmv46_image_path_rejects_request_max_slice_nums_mismatch(
    request_max_slice_nums: object,
):
    processor = _build_v46_processor(max_slice_nums=9)

    with pytest.raises(
        ValueError,
        match="must match the deployed MiniCPM image processor configuration",
    ):
        processor.process_images(
            {"images": ["image"]},
            {"max_slice_nums": request_max_slice_nums},
            {},
        )

    assert processor.info.image_processor.calls == []


def test_minicpmv46_image_path_allows_deployed_max_slice_nums():
    processor = _build_v46_processor(max_slice_nums=4)

    result = processor.process_images(
        {"images": ["image"]},
        {"max_slice_nums": 4},
        {},
    )

    assert processor.info.image_processor.calls == [(["image"], {"max_slice_nums": 4})]
    assert result["pixel_values"][0].shape == (1, 3, 1, 1)


@pytest.mark.parametrize(
    "mm_kwargs",
    [
        pytest.param({}, id="omitted"),
        pytest.param({"max_slice_nums": None}, id="explicit-none"),
    ],
)
def test_minicpmv46_image_path_allows_unset_max_slice_nums(mm_kwargs):
    processor = _build_v46_processor(max_slice_nums=4)

    processor.process_images({"images": ["image"]}, mm_kwargs, {})

    assert len(processor.info.image_processor.calls) == 1
