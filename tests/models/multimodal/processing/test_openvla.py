# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for OpenVLA multimodal preprocessing."""

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import LlamaConfig

from vllm.model_executor.models.openvla import (
    OpenVLAForActionPrediction,
    OpenVLAMultiModalProcessor,
    OpenVLAProcessingInfo,
)
from vllm.multimodal.parse import ImageProcessorItems, MultiModalDataItems
from vllm.transformers_utils.configs.openvla import OpenVLAConfig
from vllm.transformers_utils.processors.openvla import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    SIGLIP_MEAN,
    SIGLIP_STD,
    OpenVLAImageProcessor,
    OpenVLAProcessor,
    preprocess_openvla_image,
    to_rgb_image,
)

pytestmark = pytest.mark.cpu_test


class _FakeTokenizer:
    bos_token_id = 1
    init_kwargs: dict[str, object] = {}

    def encode(self, prompt: str, **kwargs: object) -> list[int]:
        assert prompt == "In: test\nOut:"
        if kwargs == {"add_special_tokens": True}:
            return [self.bos_token_id, 10, 11]
        assert kwargs == {"add_special_tokens": False}
        return [10, 11]

    def __call__(self, text: str, **kwargs: object) -> dict[str, list[list[int]]]:
        return {"input_ids": [self.encode(text, **kwargs)]}


class _FakeProcessingInfo:
    def __init__(self) -> None:
        self.config = OpenVLAConfig()

    def get_hf_config(self) -> OpenVLAConfig:
        return self.config

    def get_tokenizer(self) -> _FakeTokenizer:
        return _FakeTokenizer()

    def get_num_image_tokens(self, *, image_width: int, image_height: int) -> int:
        assert image_width > 0
        assert image_height > 0
        return 256


class _FakeOpenVLAProcessingInfo(OpenVLAProcessingInfo):
    def get_hf_config(self) -> OpenVLAConfig:
        return OpenVLAConfig()


def _make_processor() -> OpenVLAMultiModalProcessor:
    processor = OpenVLAMultiModalProcessor.__new__(OpenVLAMultiModalProcessor)
    processor.info = _FakeProcessingInfo()
    return processor


def test_openvla_config_converts_text_config_dict() -> None:
    config = OpenVLAConfig(
        text_config={
            "vocab_size": 123,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
        },
    )

    assert isinstance(config.text_config, LlamaConfig)
    assert config.text_config.vocab_size == 123
    assert config.text_config.hidden_size == 64
    assert config.text_config.architectures == ["LlamaForCausalLM"]


@pytest.mark.parametrize(
    ("image", "expected_size", "expected_pixel"),
    [
        (
            torch.tensor(
                [
                    [[1.0, 1.0], [1.0, 1.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                ]
            ),
            (2, 2),
            (255, 0, 0),
        ),
        (
            np.full((4, 5, 1), 128, dtype=np.uint8),
            (5, 4),
            (128, 128, 128),
        ),
    ],
)
def test_openvla_to_rgb_image(
    image: torch.Tensor | np.ndarray,
    expected_size: tuple[int, int],
    expected_pixel: tuple[int, int, int],
) -> None:
    rgb_image = to_rgb_image(image)

    assert rgb_image.mode == "RGB"
    assert rgb_image.size == expected_size
    assert rgb_image.getpixel((0, 0)) == expected_pixel


def test_openvla_preprocess_image_matches_expected_normalization() -> None:
    image = Image.fromarray(
        np.arange(12 * 10 * 3, dtype=np.uint8).reshape(10, 12, 3),
        mode="RGB",
    )

    pixel_values = preprocess_openvla_image(image, image_size=224)

    resized = image.resize((224, 224), Image.Resampling.BICUBIC)
    raw = np.asarray(resized, dtype=np.float32) / 255.0
    expected_dinov2 = ((raw - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
    expected_siglip = ((raw - SIGLIP_MEAN) / SIGLIP_STD).transpose(2, 0, 1)
    expected = np.concatenate([expected_dinov2, expected_siglip], axis=0)

    assert pixel_values.shape == (6, 224, 224)
    assert pixel_values.dtype == torch.float32
    torch.testing.assert_close(pixel_values, torch.from_numpy(expected))


def test_openvla_processor_outputs_pixel_values() -> None:
    processor = OpenVLAProcessor(
        image_processor=OpenVLAImageProcessor(image_size=224),
        tokenizer=_FakeTokenizer(),
    )
    image = Image.new("RGB", (8, 8), color=(255, 0, 0))

    batch = processor(
        text="In: test\nOut:",
        images=image,
        text_kwargs={"add_special_tokens": True},
    )

    assert batch["input_ids"] == [[1, 10, 11]]
    assert batch["pixel_values"].shape == (1, 6, 224, 224)
    assert batch["pixel_values"].dtype == torch.float32


def test_openvla_image_processor_outputs_pixel_values() -> None:
    processor = OpenVLAImageProcessor(image_size=224)
    image = Image.new("RGB", (8, 8), color=(255, 0, 0))

    output = processor(images=image)

    assert output["pixel_values"].shape == (1, 6, 224, 224)
    assert output["pixel_values"].dtype == torch.float32


def test_openvla_processing_info_token_counts() -> None:
    info = _FakeOpenVLAProcessingInfo.__new__(_FakeOpenVLAProcessingInfo)

    assert info.get_supported_mm_limits() == {"image": 1}
    assert info.get_num_image_tokens(image_width=640, image_height=480) == 256
    assert info.get_image_size_with_most_features().width == 224
    assert info.get_image_size_with_most_features().height == 224
    assert info.get_mm_max_tokens_per_item(seq_len=2048, mm_counts={"image": 1}) == {
        "image": 256
    }


def test_openvla_prompt_update_inserts_image_tokens_after_bos() -> None:
    processor = _make_processor()
    image = Image.new("RGB", (640, 480), color=(255, 255, 255))
    mm_items = MultiModalDataItems({"image": ImageProcessorItems([image])})

    assert (
        processor._hf_processor_applies_updates("In: test\nOut:", mm_items, {}, {})
        is False
    )

    prompt_update = processor._get_prompt_updates(mm_items, {}, {})[0]
    resolved = prompt_update.resolve(0)
    content = resolved.content

    assert resolved.modality == "image"
    assert [
        (match.start_idx, match.end_idx)
        for match in resolved.iter_matches([1, 10, 11], _FakeTokenizer())
    ] == [(1, 1)]
    assert content.full == [32000] * 256

    is_embed = content.is_embed(None, content.full)
    assert is_embed.dtype == torch.bool
    assert is_embed.tolist() == [True] * 256


def test_openvla_placeholder_string() -> None:
    assert OpenVLAForActionPrediction.get_placeholder_str("image", 0) is None
