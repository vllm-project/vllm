# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Regression test for DeepSeek-OCR TensorSchema validation with empty images_crop.

When using the Gundam preset (BASE_SIZE=1024, IMAGE_SIZE=640, CROP_MODE=True),
images that are small enough to not require cropping produce an empty
images_crop tensor with shape (0, 3, 640, 640). The _parse_and_validate_image_input
method must correctly read image_size from this tensor's shape rather than
falling back to base_size, which would cause a TensorSchema mismatch.

Run with:
  pytest tests/models/multimodal/processing/test_deepseek_ocr.py -v
"""

import pytest
from PIL import Image
from transformers import AutoTokenizer

from vllm.model_executor.models.deepseek_ocr import DeepseekOCRImagePixelInputs
from vllm.transformers_utils.processors.deepseek_ocr import DeepseekOCRProcessor

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
