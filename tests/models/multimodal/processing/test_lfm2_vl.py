# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for LFM2-VL multimodal processor request bounds."""

from types import SimpleNamespace
from typing import Any, cast

import pytest

from vllm.model_executor.models.lfm2_vl import Lfm2VLMultiModalProcessor
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    ProcessorInputs,
    TimingContext,
)


class _FakeLfm2Info:
    def __init__(self, max_tiles: int) -> None:
        self._max_tiles = max_tiles

    def get_image_processor(self) -> SimpleNamespace:
        return SimpleNamespace(max_tiles=self._max_tiles)


def _make_processor(max_tiles: int) -> Lfm2VLMultiModalProcessor:
    processor = object.__new__(Lfm2VLMultiModalProcessor)
    processor.info = cast(Any, _FakeLfm2Info(max_tiles))
    return processor


@pytest.mark.parametrize(
    "hf_processor_mm_kwargs",
    [
        {"max_tiles": 700},
        {"images_kwargs": {"max_tiles": 700}},
        {"min_tiles": -1},
        {"images_kwargs": {"min_tiles": -1}},
    ],
)
def test_lfm2_rejects_unsafe_request_tile_bounds_before_base_processing(
    monkeypatch: pytest.MonkeyPatch,
    hf_processor_mm_kwargs: dict[str, object],
) -> None:
    processor = _make_processor(max_tiles=10)
    base_apply_called = False

    def fake_base_apply(
        _self: BaseMultiModalProcessor,
        _inputs: ProcessorInputs,
        _timing_ctx: TimingContext,
    ) -> object:
        nonlocal base_apply_called
        base_apply_called = True
        return object()

    monkeypatch.setattr(BaseMultiModalProcessor, "apply", fake_base_apply)

    inputs = ProcessorInputs(
        prompt="",
        mm_data_items=cast(Any, {}),
        hf_processor_mm_kwargs=hf_processor_mm_kwargs,
    )

    with pytest.raises(ValueError, match="LFM2-VL request"):
        processor.apply(inputs, TimingContext(enabled=False))

    assert not base_apply_called


def test_lfm2_allows_request_tile_bounds_within_operator_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = _make_processor(max_tiles=12)
    expected = object()
    base_apply_called = False

    def fake_base_apply(
        _self: BaseMultiModalProcessor,
        _inputs: ProcessorInputs,
        _timing_ctx: TimingContext,
    ) -> object:
        nonlocal base_apply_called
        base_apply_called = True
        return expected

    monkeypatch.setattr(BaseMultiModalProcessor, "apply", fake_base_apply)

    inputs = ProcessorInputs(
        prompt="",
        mm_data_items=cast(Any, {}),
        hf_processor_mm_kwargs={"min_tiles": 1, "max_tiles": 12},
    )

    assert processor.apply(inputs, TimingContext(enabled=False)) is expected
    assert base_apply_called
