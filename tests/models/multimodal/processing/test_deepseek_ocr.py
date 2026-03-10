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
import torch

from vllm.model_executor.models.deepseek_ocr import DeepseekOCRImagePixelInputs


class TestDeepseekOCREmptyImagesCrop:
    """Verify TensorSchema validation handles empty images_crop correctly."""

    def test_empty_images_crop_gundam_preset(self):
        """Empty images_crop (0, 3, 640, 640) should not crash when
        base_size=1024 and image_size=640 (Gundam preset).

        Previously, the code used `numel() > 0` to decide whether to read
        image_size from the tensor shape. When numel()==0, it fell back to
        base_size=1024, mismatching the actual tensor dim of 640.
        """
        base_size = 1024
        image_size = 640

        pixel_values = torch.randn(1, 3, base_size, base_size)
        images_crop = torch.zeros(0, 3, image_size, image_size)
        images_spatial_crop = torch.tensor([[1, 1]])

        # This should NOT raise ValueError
        result = DeepseekOCRImagePixelInputs(
            type="pixel_values",
            data=pixel_values,
            images_crop=images_crop,
            images_spatial_crop=images_spatial_crop,
            resolve_bindings={
                "base_size": base_size,
                "image_size": images_crop.shape[-1],  # 640, not base_size
            },
        )

        assert result.data.shape == (1, 3, 1024, 1024)
        assert result.images_crop.shape == (0, 3, 640, 640)

    def test_populated_images_crop_gundam_preset(self):
        """Populated images_crop should continue to work normally."""
        base_size = 1024
        image_size = 640

        pixel_values = torch.randn(1, 3, base_size, base_size)
        images_crop = torch.randn(2, 3, image_size, image_size)
        images_spatial_crop = torch.tensor([[2, 1]])

        result = DeepseekOCRImagePixelInputs(
            type="pixel_values",
            data=pixel_values,
            images_crop=images_crop,
            images_spatial_crop=images_spatial_crop,
            resolve_bindings={
                "base_size": base_size,
                "image_size": image_size,
            },
        )

        assert result.data.shape == (1, 3, 1024, 1024)
        assert result.images_crop.shape == (2, 3, 640, 640)

    def test_empty_images_crop_base_preset(self):
        """Base preset (image_size == base_size == 1024) should also work
        with empty crops."""
        size = 1024

        pixel_values = torch.randn(1, 3, size, size)
        images_crop = torch.zeros(0, 3, size, size)
        images_spatial_crop = torch.tensor([[1, 1]])

        result = DeepseekOCRImagePixelInputs(
            type="pixel_values",
            data=pixel_values,
            images_crop=images_crop,
            images_spatial_crop=images_spatial_crop,
            resolve_bindings={
                "base_size": size,
                "image_size": size,
            },
        )

        assert result.data.shape == (1, 3, 1024, 1024)
        assert result.images_crop.shape == (0, 3, 1024, 1024)

    def test_mismatched_image_size_raises(self):
        """Deliberately wrong image_size binding should still be caught
        by TensorSchema validation."""
        pixel_values = torch.randn(1, 3, 1024, 1024)
        images_crop = torch.zeros(0, 3, 640, 640)
        images_spatial_crop = torch.tensor([[1, 1]])

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
