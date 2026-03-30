# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Processing tests for DeepSeek-OCR-2.

Validates that the processor handles images correctly with the v2 strategy
(IMAGE_SIZE=768, BASE_SIZE=1024, CROP_MODE=True).  The tests mirror the
structure of test_deepseek_ocr.py but use the OCR-2 specific configuration.

Run with:
  pytest tests/models/multimodal/processing/test_deepseek_ocr2.py -v
"""

import pytest
from PIL import Image
from transformers import AutoTokenizer

from vllm.model_executor.models.deepseek_ocr import DeepseekOCRImagePixelInputs
from vllm.transformers_utils.processors.deepseek_ocr import DeepseekOCRProcessor

MODEL_ID = "deepseek-ai/DeepSeek-OCR-2"

# OCR-2 uses different image_size (768) from OCR (640), same base_size (1024)
IMAGE_SIZE = 768
BASE_SIZE = 1024


@pytest.fixture(scope="module")
def processor():
    """Load DeepseekOCRProcessor with v2 strategy for OCR-2."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    return DeepseekOCRProcessor(
        tokenizer=tokenizer,
        image_size=IMAGE_SIZE,
        base_size=BASE_SIZE,
        crop_mode=True,
        strategy="v2",
    )


class TestDeepseekOCR2Processing:
    """Verify processing behaviour for DeepSeek-OCR-2."""

    def test_small_image_produces_empty_crop(self, processor):
        """An image smaller than IMAGE_SIZE should produce an empty
        images_crop tensor with the correct spatial dimension (768)."""
        small_image = Image.new("RGB", (100, 100), color="red")

        result = processor(
            prompt="<image>\nDescribe this image.",
            images=[small_image],
        )

        pixel_values = result["pixel_values"]
        images_crop = result["images_crop"]

        # Small image: no crops needed
        assert images_crop.shape[0] == 0
        # The base image should use BASE_SIZE
        assert pixel_values.shape[-1] == BASE_SIZE
        # Crop tensor spatial dim should be IMAGE_SIZE
        assert images_crop.shape[-1] == IMAGE_SIZE

    def test_large_image_produces_crops(self, processor):
        """An image larger than IMAGE_SIZE should produce non-empty
        images_crop tiles."""
        large_image = Image.new("RGB", (1200, 800), color="blue")

        result = processor(
            prompt="<image>\nDescribe this image.",
            images=[large_image],
        )

        images_crop = result["images_crop"]

        assert images_crop.shape[0] > 0
        assert images_crop.shape[-1] == IMAGE_SIZE

    def test_tensor_schema_with_empty_crop(self, processor):
        """TensorSchema validation must succeed when images_crop is empty,
        reading image_size from the tensor shape rather than falling back
        to base_size."""
        small_image = Image.new("RGB", (100, 100), color="green")

        result = processor(
            prompt="<image>\nDescribe this image.",
            images=[small_image],
        )

        pixel_values = result["pixel_values"]
        images_crop = result["images_crop"]
        images_spatial_crop = result["images_spatial_crop"]

        base_size = pixel_values.shape[-1]
        image_size = images_crop.shape[-1] if images_crop is not None else base_size

        # Should not raise
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

        assert schema.data.shape == (1, 3, BASE_SIZE, BASE_SIZE)
        assert schema.images_crop.shape == (0, 3, IMAGE_SIZE, IMAGE_SIZE)

    def test_tensor_schema_with_populated_crop(self, processor):
        """TensorSchema validation must succeed when images_crop is
        populated."""
        large_image = Image.new("RGB", (1200, 800), color="blue")

        result = processor(
            prompt="<image>\nDescribe this image.",
            images=[large_image],
        )

        pixel_values = result["pixel_values"]
        images_crop = result["images_crop"]
        images_spatial_crop = result["images_spatial_crop"]

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

        assert schema.data.shape == (1, 3, BASE_SIZE, BASE_SIZE)
        assert schema.images_crop.shape[-1] == IMAGE_SIZE

    def test_mismatched_image_size_raises(self, processor):
        """Deliberately wrong image_size binding should be caught by
        TensorSchema validation."""
        small_image = Image.new("RGB", (100, 100), color="yellow")

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
                    "base_size": BASE_SIZE,
                    "image_size": BASE_SIZE,  # Wrong! Tensor has IMAGE_SIZE
                },
            )
