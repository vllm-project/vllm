# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for DeepSeek-OCR TensorSchema validation with empty images_crop.

When using the Gundam preset (BASE_SIZE=1024, IMAGE_SIZE=640, CROP_MODE=True),
images that are small enough to not require cropping produce an empty
images_crop tensor with shape (0, 3, 640, 640). The _parse_and_validate_image_input
method must correctly read image_size from this tensor's shape rather than
falling back to base_size, which would cause a TensorSchema mismatch.

Run with:
  pytest tests/models/multimodal/processing/test_deepseek_ocr.py -v
"""

from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from transformers import AutoTokenizer

from vllm.model_executor.models.deepseek_ocr import (
    DeepseekOCRForCausalLM,
    DeepseekOCRImagePixelInputs,
)
from vllm.model_executor.models.deepseek_ocr2 import DeepseekOCR2ForCausalLM
from vllm.transformers_utils.processors.deepseek_ocr import (
    DeepseekOCRProcessor,
    ImageTransform,
)

MODEL_ID = "deepseek-ai/DeepSeek-OCR"


@pytest.fixture(scope="module")
def processor():
    """Load the DeepseekOCRProcessor with tokenizer from HuggingFace."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    return DeepseekOCRProcessor(tokenizer=tokenizer)


class TestDeepseekOCREmptyImagesCrop:
    """Verify TensorSchema validation handles empty images_crop correctly."""

    def test_empty_images_crop_small_image(self, processor):
        """A small image (<=640px) produces empty images_crop and should
        not crash the TensorSchema validation.

        Previously, the code used ``numel() > 0`` to decide whether to read
        image_size from the tensor shape. When numel()==0, it fell back to
        base_size=1024, mismatching the actual tensor dim of 640.
        """
        # Small image: both dims <= IMAGE_SIZE (640) → no crops
        small_image = Image.new("RGB", (100, 100), color="red")

        result = processor(
            prompt="<image>\nDescribe this image.",
            images=[small_image],
        )

        pixel_values = result["pixel_values"]
        images_crop = result["images_crop"]
        images_spatial_crop = result["images_spatial_crop"]

        # Processor must produce an empty crop tensor for a small image
        assert images_crop.shape[0] == 0

        base_size = pixel_values.shape[-1]
        image_size = images_crop.shape[-1] if images_crop is not None else base_size

        # This should NOT raise ValueError
        schema = DeepseekOCRImagePixelInputs(
            type="pixel_values",
            data=pixel_values,
            images_crop=images_crop,
            images_spatial_crop=images_spatial_crop,
            resolve_bindings={
                "base_size": base_size,
                "image_size": image_size,
            },
        )

        assert schema.data.shape == (1, 3, 1024, 1024)
        assert schema.images_crop.shape == (0, 3, 640, 640)

    def test_populated_images_crop_large_image(self, processor):
        """A large image (>640px) produces populated images_crop."""
        # Large image: exceeds IMAGE_SIZE (640) → dynamic crop tiles
        large_image = Image.new("RGB", (1200, 800), color="blue")

        result = processor(
            prompt="<image>\nDescribe this image.",
            images=[large_image],
        )

        pixel_values = result["pixel_values"]
        images_crop = result["images_crop"]
        images_spatial_crop = result["images_spatial_crop"]

        assert images_crop.shape[0] > 0

        base_size = pixel_values.shape[-1]
        image_size = images_crop.shape[-1]

        schema = DeepseekOCRImagePixelInputs(
            type="pixel_values",
            data=pixel_values,
            images_crop=images_crop,
            images_spatial_crop=images_spatial_crop,
            resolve_bindings={
                "base_size": base_size,
                "image_size": image_size,
            },
        )

        assert schema.data.shape == (1, 3, 1024, 1024)
        assert schema.images_crop.shape[-1] == 640

    def test_mismatched_image_size_raises(self, processor):
        """Deliberately wrong image_size binding should still be caught
        by TensorSchema validation."""
        small_image = Image.new("RGB", (100, 100), color="green")

        result = processor(
            prompt="<image>\nDescribe this image.",
            images=[small_image],
        )

        pixel_values = result["pixel_values"]
        images_crop = result["images_crop"]
        images_spatial_crop = result["images_spatial_crop"]

        with pytest.raises(ValueError, match="images_crop"):
            DeepseekOCRImagePixelInputs(
                type="pixel_values",
                data=pixel_values,
                images_crop=images_crop,
                images_spatial_crop=images_spatial_crop,
                resolve_bindings={
                    "base_size": 1024,
                    "image_size": 1024,  # Wrong! Tensor has 640
                },
            )


class TestDeepseekOCRInputValidation:
    """Bare ``assert`` used to validate user input here returned HTTP 500
    (AssertionError) instead of HTTP 400 (ValueError) and disappeared under
    ``python -O``, silently corrupting the tokenized sequence. See #53850.
    """

    def test_mismatched_image_count_raises_value_error(self, processor):
        """Two ``<image>`` tokens with one image must raise ValueError, not
        AssertionError (which would yield HTTP 500 and, under ``-O``, silently
        drop the middle text segment)."""
        image = Image.new("RGB", (100, 100), color="red")
        with pytest.raises(ValueError, match="does not match number of images"):
            processor(prompt="<image> and <image>", images=[image])

    def test_missing_prompt_raises_value_error(self, processor):
        image = Image.new("RGB", (100, 100), color="red")
        with pytest.raises(ValueError, match="prompt and images"):
            processor(prompt=None, images=[image])

    def test_missing_images_raises_value_error(self, processor):
        with pytest.raises(ValueError, match="prompt and images"):
            processor(prompt="<image>\nDescribe this image.", images=None)


def _balanced_normalized_pixels(
    batch: int = 1, channels: int = 3, height: int = 64, width: int = 64
) -> torch.Tensor:
    """Build a pixel tensor whose values cancel to an exact sum of 0.

    Matches the normalized half-black / half-white case from the DeepSeek-OCR
    processor (Normalize(0.5, 0.5) maps 0→-1 and 255→+1).
    """
    assert width % 2 == 0
    pixel_values = torch.ones(batch, channels, height, width)
    pixel_values[..., : width // 2] = -1.0
    assert pixel_values.sum().item() == 0.0
    return pixel_values


class TestDeepseekOCRZeroSumPixelsAccepted:
    """Zero-sum normalized pixels must still produce validated image inputs.

    Returning None from ``_parse_and_validate_image_input`` makes
    ``embed_multimodal`` return None and trips the worker encoder-output
    sanity check, killing EngineCore. The zero-sum shortcut is therefore
    removed; balanced images must parse like any other valid tensor.
    """

    def test_ocr_accepts_balanced_normalized_pixels(self):
        pixel_values = _balanced_normalized_pixels()
        images_crop = torch.zeros(0, 3, 40, 40)
        images_spatial_crop = torch.tensor([[1, 1]])

        image_input = DeepseekOCRForCausalLM._parse_and_validate_image_input(
            None,
            pixel_values=pixel_values,
            images_crop=images_crop,
            images_spatial_crop=images_spatial_crop,
        )

        assert image_input is not None
        assert image_input.data is pixel_values

    def test_ocr2_accepts_balanced_normalized_pixels(self):
        pixel_values = _balanced_normalized_pixels()
        images_crop = torch.zeros(0, 3, 40, 40)
        images_spatial_crop = torch.tensor([[1, 1]])
        model = SimpleNamespace(vision_config=SimpleNamespace(image_size=64))

        image_input = DeepseekOCR2ForCausalLM._parse_and_validate_image_input(
            model,
            pixel_values=pixel_values,
            images_crop=images_crop,
            images_spatial_crop=images_spatial_crop,
        )

        assert image_input is not None
        assert image_input.data is pixel_values

    def test_image_transform_half_black_half_white_sums_to_zero(self):
        """A 1024×1024 half-black/half-white PNG through ImageTransform
        (ToTensor + Normalize(0.5, 0.5)) yields an exact zero sum, and parse
        must still accept it.
        """
        image = Image.new("RGB", (1024, 1024), (0, 0, 0))
        image.paste(Image.new("RGB", (512, 1024), (255, 255, 255)), (512, 0))
        pixel_values = ImageTransform()(image).unsqueeze(0)
        assert pixel_values.sum().item() == 0.0

        images_crop = torch.zeros(0, 3, 640, 640)
        images_spatial_crop = torch.tensor([[1, 1]])
        image_input = DeepseekOCRForCausalLM._parse_and_validate_image_input(
            None,
            pixel_values=pixel_values,
            images_crop=images_crop,
            images_spatial_crop=images_spatial_crop,
        )
        assert image_input is not None
        assert torch.equal(image_input.data, pixel_values)
