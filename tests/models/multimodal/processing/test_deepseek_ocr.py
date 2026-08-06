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

from dataclasses import dataclass

import pytest
from PIL import Image
from transformers import AutoTokenizer

from vllm.model_executor.models.deepseek_ocr import (
    IMAGE_SIZE as DEEPSEEK_OCR_IMAGE_SIZE,
)
from vllm.model_executor.models.deepseek_ocr import (
    DeepseekOCRImagePixelInputs,
    DeepseekOCRProcessingInfo,
)
from vllm.model_executor.models.deepseek_ocr2 import (
    IMAGE_SIZE as DEEPSEEK_OCR2_IMAGE_SIZE,
)
from vllm.model_executor.models.deepseek_ocr2 import (
    DeepseekOCR2ProcessingInfo,
)
from vllm.model_executor.models.unlimited_ocr import UnlimitedOCRProcessingInfo
from vllm.transformers_utils.processors.deepseek_ocr import (
    BASE_SIZE,
    CROP_MODE,
    MAX_CROPS,
    DeepseekOCRProcessor,
)
from vllm.transformers_utils.processors.unlimited_ocr import UnlimitedOCRProcessor

MODEL_ID = "deepseek-ai/DeepSeek-OCR"
UNLIMITED_OCR_MAX_CROPS = 32


@dataclass
class _MMConfig:
    mm_processor_kwargs: dict[str, object] | None = None


class _ProcessorContext:
    def __init__(self, mm_processor_kwargs: dict[str, object] | None = None):
        self.mm_processor_kwargs = mm_processor_kwargs
        self.calls: list[tuple[type[object], dict[str, object]]] = []

    def get_mm_config(self) -> _MMConfig:
        return _MMConfig(self.mm_processor_kwargs)

    def get_hf_processor(
        self,
        processor_type: type[object],
        **kwargs: object,
    ) -> dict[str, object]:
        merged_kwargs = {**(self.mm_processor_kwargs or {}), **kwargs}
        self.calls.append((processor_type, merged_kwargs))
        return merged_kwargs


@pytest.mark.parametrize(
    ("info_cls", "unsafe_kwargs"),
    [
        pytest.param(
            DeepseekOCRProcessingInfo,
            {"image_size": DEEPSEEK_OCR_IMAGE_SIZE + 1},
            id="v1-image-size",
        ),
        pytest.param(
            DeepseekOCRProcessingInfo,
            {"base_size": BASE_SIZE + 1},
            id="v1-base-size",
        ),
        pytest.param(
            DeepseekOCRProcessingInfo,
            {"crop_mode": not CROP_MODE},
            id="v1-crop-mode",
        ),
        pytest.param(
            DeepseekOCRProcessingInfo,
            {"strategy": "v2"},
            id="v1-strategy",
        ),
        pytest.param(
            DeepseekOCRProcessingInfo,
            {"max_crops": MAX_CROPS + 1},
            id="v1-max-crops",
        ),
        pytest.param(
            DeepseekOCR2ProcessingInfo,
            {"image_size": DEEPSEEK_OCR2_IMAGE_SIZE + 1},
            id="v2-image-size",
        ),
        pytest.param(
            DeepseekOCR2ProcessingInfo,
            {"base_size": BASE_SIZE + 1},
            id="v2-base-size",
        ),
        pytest.param(
            DeepseekOCR2ProcessingInfo,
            {"crop_mode": not CROP_MODE},
            id="v2-crop-mode",
        ),
        pytest.param(
            DeepseekOCR2ProcessingInfo,
            {"strategy": "v1"},
            id="v2-strategy",
        ),
        pytest.param(
            DeepseekOCR2ProcessingInfo,
            {"max_crops": MAX_CROPS + 1},
            id="v2-max-crops",
        ),
        pytest.param(
            UnlimitedOCRProcessingInfo,
            {"image_size": DEEPSEEK_OCR_IMAGE_SIZE + 1},
            id="unlimited-image-size",
        ),
        pytest.param(
            UnlimitedOCRProcessingInfo,
            {"base_size": BASE_SIZE + 1},
            id="unlimited-base-size",
        ),
        pytest.param(
            UnlimitedOCRProcessingInfo,
            {"crop_mode": not CROP_MODE},
            id="unlimited-crop-mode",
        ),
        pytest.param(
            UnlimitedOCRProcessingInfo,
            {"strategy": "v2"},
            id="unlimited-strategy",
        ),
        pytest.param(
            UnlimitedOCRProcessingInfo,
            {"max_crops": UNLIMITED_OCR_MAX_CROPS + 1},
            id="unlimited-max-crops",
        ),
    ],
)
def test_processing_info_rejects_request_processor_overrides(
    info_cls: type[
        DeepseekOCRProcessingInfo
        | DeepseekOCR2ProcessingInfo
        | UnlimitedOCRProcessingInfo
    ],
    unsafe_kwargs: dict[str, object],
):
    ctx = _ProcessorContext()
    info = info_cls(ctx)  # type: ignore[arg-type]

    with pytest.raises(
        ValueError,
        match="must match the deployed DeepSeek OCR configuration",
    ):
        info.get_hf_processor(**unsafe_kwargs)

    assert ctx.calls == []


@pytest.mark.parametrize(
    ("info_cls", "trusted_config"),
    [
        pytest.param(
            DeepseekOCRProcessingInfo,
            {
                "image_size": DEEPSEEK_OCR_IMAGE_SIZE,
                "base_size": BASE_SIZE,
                "crop_mode": CROP_MODE,
                "strategy": "v1",
            },
            id="v1",
        ),
        pytest.param(
            DeepseekOCR2ProcessingInfo,
            {
                "image_size": DEEPSEEK_OCR2_IMAGE_SIZE,
                "base_size": BASE_SIZE,
                "crop_mode": CROP_MODE,
                "strategy": "v2",
            },
            id="v2",
        ),
    ],
)
@pytest.mark.parametrize(
    ("server_kwargs", "request_kwargs", "expected_max_crops"),
    [
        pytest.param(None, {"normalize": False}, MAX_CROPS, id="default"),
        pytest.param(
            {"max_crops": 20},
            {"normalize": False},
            MAX_CROPS,
            id="ignores-server-max-crops",
        ),
        pytest.param(
            None,
            {"max_crops": MAX_CROPS, "normalize": False},
            MAX_CROPS,
            id="request-repeats-fixed",
        ),
    ],
)
def test_processing_info_binds_fixed_processor_config(
    info_cls: type[DeepseekOCRProcessingInfo | DeepseekOCR2ProcessingInfo],
    trusted_config: dict[str, object],
    server_kwargs: dict[str, object] | None,
    request_kwargs: dict[str, object],
    expected_max_crops: int,
):
    ctx = _ProcessorContext(server_kwargs)
    info = info_cls(ctx)  # type: ignore[arg-type]

    result = info.get_hf_processor(**request_kwargs)
    expected_kwargs = {
        **trusted_config,
        "max_crops": expected_max_crops,
        "normalize": False,
    }

    assert result == expected_kwargs
    assert ctx.calls == [(DeepseekOCRProcessor, expected_kwargs)]


@pytest.mark.parametrize(
    "info_cls",
    [
        pytest.param(DeepseekOCRProcessingInfo, id="v1"),
        pytest.param(DeepseekOCR2ProcessingInfo, id="v2"),
    ],
)
def test_processing_info_rejects_request_max_crops_even_when_server_matches(
    info_cls: type[DeepseekOCRProcessingInfo | DeepseekOCR2ProcessingInfo],
):
    ctx = _ProcessorContext({"max_crops": 20})
    info = info_cls(ctx)  # type: ignore[arg-type]

    with pytest.raises(
        ValueError,
        match="must match the deployed DeepSeek OCR configuration",
    ):
        info.get_hf_processor(max_crops=20)

    assert ctx.calls == []


@pytest.mark.parametrize(
    "request_kwargs",
    [
        pytest.param({"normalize": False}, id="omitted"),
        pytest.param(
            {"max_crops": UNLIMITED_OCR_MAX_CROPS, "normalize": False},
            id="request-repeats-fixed",
        ),
    ],
)
def test_unlimited_ocr_processing_info_binds_fixed_processor_config(
    request_kwargs: dict[str, object],
):
    ctx = _ProcessorContext({"max_crops": 20})
    info = UnlimitedOCRProcessingInfo(ctx)  # type: ignore[arg-type]

    result = info.get_hf_processor(**request_kwargs)
    expected_kwargs = {
        "image_size": DEEPSEEK_OCR_IMAGE_SIZE,
        "base_size": BASE_SIZE,
        "crop_mode": CROP_MODE,
        "strategy": "v1",
        "max_crops": UNLIMITED_OCR_MAX_CROPS,
        "normalize": False,
    }

    assert result == expected_kwargs
    assert ctx.calls == [(UnlimitedOCRProcessor, expected_kwargs)]


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
