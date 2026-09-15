# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Cohere2 Vision request max_patches bounds."""

import sys
from unittest.mock import MagicMock

import pytest

from vllm.exceptions import VLLMValidationError
from vllm.model_executor.models.cohere2_max_patches import (
    validate_cohere2_max_patches,
)

_LIMIT = 12


@pytest.mark.parametrize(
    "mm_kwargs",
    [
        {"max_patches": 20000},
        {"images_kwargs": {"max_patches": 20000}},
        {"max_patches": 1, "images_kwargs": {"max_patches": 20000}},
        {"max_patches": 0},
        {"max_patches": -1},
        {"max_patches": 13},
        {"max_patches": True},
        {"max_patches": "12"},
        {"max_patches": 12.0},
    ],
)
def test_invalid_max_patches_rejected(mm_kwargs: dict[str, object]):
    with pytest.raises(VLLMValidationError, match="max_patches"):
        validate_cohere2_max_patches(mm_kwargs, _LIMIT)


@pytest.mark.parametrize(
    "mm_kwargs",
    [
        {},
        {"max_patches": 1},
        {"max_patches": _LIMIT},
        {"images_kwargs": {"max_patches": 4}},
        {"images_kwargs": "not-a-mapping"},
    ],
)
def test_valid_max_patches_accepted(mm_kwargs: dict[str, object]):
    validate_cohere2_max_patches(mm_kwargs, _LIMIT)


def test_non_mapping_mm_kwargs_rejected():
    with pytest.raises(VLLMValidationError, match="must be a mapping"):
        validate_cohere2_max_patches(["max_patches", 20000], _LIMIT)  # type: ignore[arg-type]


def _processing_info(
    *,
    processor_limit: int = _LIMIT,
    configured: dict[str, object] | None = None,
):
    if "torchvision" not in sys.modules:
        try:
            import torchvision  # noqa: F401
        except ModuleNotFoundError:
            sys.modules["torchvision"] = MagicMock()
            sys.modules["torchvision.transforms"] = MagicMock()
            sys.modules["torchvision.transforms.v2"] = MagicMock()
            sys.modules["torchvision.transforms.v2.functional"] = MagicMock()

    from vllm.model_executor.models.cohere2_vision import (
        Cohere2VisionProcessingInfo,
    )

    ctx = MagicMock()
    ctx.get_hf_processor.return_value.image_processor.max_patches = processor_limit
    mm_config = MagicMock()
    mm_config.mm_processor_kwargs = configured
    ctx.model_config.get_multimodal_config.return_value = mm_config
    ctx.get_merged_mm_kwargs.side_effect = lambda kwargs: dict(kwargs)
    return Cohere2VisionProcessingInfo(ctx), ctx


def test_get_hf_processor_rejects_oversized_max_patches():
    info, ctx = _processing_info()

    with pytest.raises(VLLMValidationError, match="max_patches"):
        info.get_hf_processor(max_patches=20000)

    for call in ctx.get_hf_processor.call_args_list:
        assert call.kwargs.get("max_patches") != 20000


def test_get_num_patches_rejects_oversized_max_patches():
    info, _ctx = _processing_info()
    processor = MagicMock()

    with pytest.raises(VLLMValidationError, match="max_patches"):
        info.get_num_patches(
            image_width=64,
            image_height=64,
            processor=processor,
            mm_kwargs={"max_patches": 20000},
        )

    processor.image_processor.get_number_of_image_patches.assert_not_called()


def test_get_num_patches_rejects_nested_oversized_max_patches():
    info, _ctx = _processing_info()
    processor = MagicMock()

    with pytest.raises(VLLMValidationError, match="max_patches"):
        info.get_num_patches(
            image_width=64,
            image_height=64,
            processor=processor,
            mm_kwargs={"images_kwargs": {"max_patches": 20000}},
        )

    processor.image_processor.get_number_of_image_patches.assert_not_called()


def test_get_num_patches_forwards_in_range_max_patches():
    info, _ctx = _processing_info()
    processor = MagicMock()
    processor.image_processor.get_number_of_image_patches.return_value = 2

    result = info.get_num_patches(
        image_width=64,
        image_height=64,
        processor=processor,
        mm_kwargs={"max_patches": 4},
    )

    assert result == 2
    processor.image_processor.get_number_of_image_patches.assert_called_once()


def test_operator_max_patches_is_the_request_cap():
    info, ctx = _processing_info(configured={"max_patches": 24})

    info.get_hf_processor(max_patches=24)
    ctx.get_hf_processor.assert_called()

    with pytest.raises(VLLMValidationError, match="max_patches"):
        info.get_hf_processor(max_patches=25)
