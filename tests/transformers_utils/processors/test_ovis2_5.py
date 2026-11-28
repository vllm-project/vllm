# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import SiglipImageProcessor, TokenizersBackend

from vllm.transformers_utils.processors.ovis2_5 import MIN_PIXELS, Ovis2_5Processor

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

IMAGE_BYTES = MIN_PIXELS * 3 * 4


@pytest.fixture
def processor():
    tokenizer = TokenizersBackend(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        unk_token="[UNK]",
    )
    return Ovis2_5Processor(SiglipImageProcessor(), tokenizer)


@pytest.mark.parametrize("modality", ["images", "video_list", "video_array"])
def test_expanded_media_rejected_before_preprocessing(processor, monkeypatch, modality):
    processor.temporal_patch_size = 2
    monkeypatch.setenv("VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES", str(2 * IMAGE_BYTES - 1))
    image = Image.new("RGB", (1, 1))
    inputs = {
        "images": {"images": image},
        "video_list": {"video": [image]},
        "video_array": {"video": np.zeros((1, 1, 1, 3), dtype=np.uint8)},
    }[modality]
    with (
        patch.object(processor.image_processor, "preprocess") as preprocess,
        pytest.raises(ValueError, match="VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES"),
    ):
        processor.preprocess_multidata(**inputs)
    preprocess.assert_not_called()


@pytest.mark.parametrize("modality", ["images", "videos", "mixed"])
def test_expanded_budget_shared_across_media(processor, monkeypatch, modality):
    monkeypatch.setenv("VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES", str(IMAGE_BYTES))
    images = [Image.new("RGB", (1, 1), (i % 256, i // 256, 0)) for i in range(999)]
    video = np.zeros((1, 1, 1, 3), dtype=np.uint8)
    inputs = {
        "images": {"images": images},
        "videos": {"videos": [video, video]},
        "mixed": {"images": images[:1], "videos": video},
    }[modality]
    with (
        patch.object(
            processor.image_processor,
            "preprocess",
            wraps=processor.image_processor.preprocess,
        ) as preprocess,
        pytest.raises(ValueError, match="VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES"),
    ):
        processor(**inputs)
    assert preprocess.call_count == 1


@pytest.mark.parametrize("temporal_patch_size", [1, 2])
@pytest.mark.parametrize("disabled", [False, True])
def test_expanded_budget_accepts_boundary_and_disabled(
    processor, monkeypatch, temporal_patch_size, disabled
):
    processor.temporal_patch_size = temporal_patch_size
    limit = 0 if disabled else temporal_patch_size * IMAGE_BYTES
    monkeypatch.setenv("VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES", str(limit))
    pixels, placeholders, grid = processor.preprocess_multidata(
        images=Image.new("RGB", (1, 1))
    )
    assert pixels.shape == (784, 768 * temporal_patch_size)
    assert pixels.numel() * pixels.element_size() == IMAGE_BYTES * temporal_patch_size
    assert len(placeholders) > 0
    assert grid.tolist() == [[1, 28, 28]]


def test_expanded_budget_is_per_call(processor, monkeypatch):
    monkeypatch.setenv("VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES", str(IMAGE_BYTES))
    _ = processor.extra_special_tokens

    def process(_):
        return processor(images=Image.new("RGB", (1, 1)), text="<image>")

    with ThreadPoolExecutor(max_workers=2) as executor:
        outputs = list(executor.map(process, range(2)))
    for output in outputs:
        assert output["pixel_values"][0].shape == (784, 768)
