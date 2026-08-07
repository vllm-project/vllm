# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Isaac multimodal processor geometry ownership."""

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, cast

import pytest
import regex as re
import torch
from PIL import Image

from vllm.model_executor.models.isaac import IsaacProcessingInfo
from vllm.transformers_utils.processors import isaac as isaac_processors
from vllm.transformers_utils.processors.isaac import (
    IsaacImageProcessor,
    IsaacProcessor,
)


class _FakeIsaacProcessingContext:
    def __init__(
        self,
        *,
        video_patch_size: int = 16,
        vision_max_num_patches: int = 6144,
        vision_min_num_patches: int | None = 256,
        pixel_shuffle_scale: int = 2,
        mm_processor_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        self._config = SimpleNamespace(
            vision_config=None,
            video_patch_size=video_patch_size,
            vision_max_num_patches=vision_max_num_patches,
            vision_min_num_patches=vision_min_num_patches,
            pixel_shuffle_scale=pixel_shuffle_scale,
            max_sequence_length=16384,
            vision_token="<image>",
            vision_attn_implementation=None,
        )
        self._mm_processor_kwargs = dict(mm_processor_kwargs or {})

    def get_hf_config(self) -> SimpleNamespace:
        return self._config

    def get_merged_mm_kwargs(self, kwargs: Mapping[str, object]) -> dict[str, object]:
        return self._mm_processor_kwargs | dict(kwargs)


class _FakeTokenizer:
    init_kwargs: dict[str, object] = {}


@pytest.mark.cpu_test
def test_get_image_processor_uses_model_patch_size_by_default() -> None:
    info = IsaacProcessingInfo(_FakeIsaacProcessingContext(video_patch_size=32))

    image_processor = info.get_image_processor()

    assert image_processor.patch_size == 32


@pytest.mark.cpu_test
def test_get_image_processor_uses_model_geometry_by_default() -> None:
    info = IsaacProcessingInfo(
        _FakeIsaacProcessingContext(
            vision_max_num_patches=128,
            vision_min_num_patches=64,
            pixel_shuffle_scale=4,
        )
    )

    image_processor = info.get_image_processor()

    assert image_processor.vision_max_num_patches == 128
    assert image_processor.vision_min_num_patches == 64
    assert image_processor.pixel_shuffle_scale == 4


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("patch_size", "expected_error"),
    [
        (0, "patch_size=0"),
        (-1, "patch_size=-1"),
        (16.0, "patch_size=16.0"),
        (4096, "patch_size=4096"),
        (2**63, "patch_size=9223372036854775808"),
    ],
)
def test_get_image_processor_rejects_mismatched_request_patch_size(
    patch_size: object,
    expected_error: str,
) -> None:
    info = IsaacProcessingInfo(_FakeIsaacProcessingContext(video_patch_size=16))

    with pytest.raises(
        ValueError,
        match=rf"{re.escape(expected_error)} must match the configured Isaac "
        r"patch size of 16",
    ):
        info.get_image_processor(patch_size=patch_size)


@pytest.mark.cpu_test
def test_get_image_processor_allows_request_patch_size_that_matches_model() -> None:
    info = IsaacProcessingInfo(_FakeIsaacProcessingContext(video_patch_size=16))

    image_processor = info.get_image_processor(patch_size=16)

    assert image_processor.patch_size == 16


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("kwargs", "expected_error"),
    [
        (
            {"vision_max_num_patches": 999999},
            "vision_max_num_patches=999999",
        ),
        (
            {"vision_min_num_patches": 999999},
            "vision_min_num_patches=999999",
        ),
        (
            {"pixel_shuffle_scale": 4096},
            "pixel_shuffle_scale=4096",
        ),
    ],
)
def test_get_image_processor_rejects_request_geometry_that_differs_from_model(
    kwargs: dict[str, object],
    expected_error: str,
) -> None:
    info = IsaacProcessingInfo(_FakeIsaacProcessingContext())

    with pytest.raises(ValueError, match=expected_error):
        info.get_image_processor(**kwargs)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("images_kwargs", "expected_error"),
    [
        ({"patch_size": 4096}, "patch_size=4096"),
        ({"max_num_patches": 999999}, "max_num_patches=999999"),
        ({"min_num_patches": 999999}, "min_num_patches=999999"),
        ({"pixel_shuffle_scale": 4096}, "pixel_shuffle_scale=4096"),
    ],
)
def test_nested_geometry_is_rejected_before_patchification(
    monkeypatch: pytest.MonkeyPatch,
    images_kwargs: dict[str, object],
    expected_error: str,
) -> None:
    patchify_called = False

    def fake_process_vision_for_patches(
        _image: torch.Tensor,
        *,
        patch_size: int,
        max_num_patches: int,
        min_num_patches: int | None,
        pixel_shuffle_scale: int,
    ) -> tuple[torch.Tensor, list[int]]:
        nonlocal patchify_called
        patchify_called = True
        return torch.zeros((1, 1, 1, 3)), [1, 1, 1]

    monkeypatch.setattr(
        isaac_processors,
        "process_vision_for_patches",
        fake_process_vision_for_patches,
    )

    processor = IsaacProcessor(
        IsaacImageProcessor(patch_size=16),
        cast(Any, _FakeTokenizer()),
    )

    with pytest.raises(
        ValueError,
        match=expected_error,
    ):
        processor(
            images=Image.new("RGB", (1, 1), color="white"),
            images_kwargs=images_kwargs,
        )

    assert not patchify_called


@pytest.mark.cpu_test
def test_nested_geometry_that_matches_model_reaches_patchification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patchify_kwargs: dict[str, object] = {}

    def fake_process_vision_for_patches(
        _image: torch.Tensor,
        *,
        patch_size: int,
        max_num_patches: int,
        min_num_patches: int | None,
        pixel_shuffle_scale: int,
    ) -> tuple[torch.Tensor, list[int]]:
        patchify_kwargs.update(
            {
                "patch_size": patch_size,
                "max_num_patches": max_num_patches,
                "min_num_patches": min_num_patches,
                "pixel_shuffle_scale": pixel_shuffle_scale,
            }
        )
        return torch.zeros((1, 1, 1, 3)), [1, 1, 1]

    monkeypatch.setattr(
        isaac_processors,
        "process_vision_for_patches",
        fake_process_vision_for_patches,
    )

    processor = IsaacProcessor(
        IsaacImageProcessor(
            patch_size=16,
            vision_max_num_patches=6144,
            vision_min_num_patches=256,
            pixel_shuffle_scale=2,
        ),
        cast(Any, _FakeTokenizer()),
    )

    output = processor(
        images=Image.new("RGB", (1, 1), color="white"),
        images_kwargs={
            "patch_size": 16,
            "max_num_patches": 6144,
            "min_num_patches": 256,
            "pixel_shuffle_scale": 2,
        },
    )

    assert patchify_kwargs == {
        "patch_size": 16,
        "max_num_patches": 6144,
        "min_num_patches": 256,
        "pixel_shuffle_scale": 2,
    }
    assert output["pixel_values"].shape == torch.Size([1, 3])
