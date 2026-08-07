# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest
import torch
from transformers import PretrainedConfig

from vllm.model_executor.models import qwen2_5_omni_thinker
from vllm.model_executor.models.qwen2_5_omni_thinker import (
    Qwen2_5OmniThinkerMultiModalProcessor,
)
from vllm.multimodal.parse import MultiModalDataItems, VideoProcessorItems
from vllm.multimodal.processing import ProcessorInputs, TimingContext
from vllm.multimodal.processing.processor import BaseMultiModalProcessor


@pytest.fixture
def mock_qwen2_5_omni_config() -> PretrainedConfig:
    config = Mock(spec=PretrainedConfig)
    config.audio_token_index = 151646
    config.video_token_index = 151656
    config.audio_start_token_id = 151647
    config.audio_end_token_id = 151648
    config.seconds_per_chunk = 1.0

    vision_config = Mock()
    vision_config.spatial_merge_size = 2
    vision_config.tokens_per_second = 25
    config.vision_config = vision_config
    return config


def _historical_dense_updates(
    thinker_config: PretrainedConfig,
    audio_len: int,
    video_grid_thw: list[int],
    video_second_per_grid_t: float,
) -> list[int]:
    audio_token_id = thinker_config.audio_token_index
    video_token_id = thinker_config.video_token_index
    audio_start_token_id = thinker_config.audio_start_token_id
    audio_end_token_id = thinker_config.audio_end_token_id
    seconds_per_chunk = thinker_config.seconds_per_chunk
    spatial_merge_size = thinker_config.vision_config.spatial_merge_size
    tokens_per_second = thinker_config.vision_config.tokens_per_second

    grid_t, grid_h, grid_w = video_grid_thw
    t_ntoken_per_chunk = int(tokens_per_second * seconds_per_chunk)
    t_index = (
        torch.arange(grid_t) * video_second_per_grid_t * tokens_per_second
    ).long()
    t_index_split_chunk: list[list[torch.Tensor]] = [
        [] for _ in range((max(t_index) // t_ntoken_per_chunk) + 1)
    ]
    for num in t_index:
        t_index_split_chunk[num // t_ntoken_per_chunk].append(num)

    updates = [audio_start_token_id]
    added_audio_len = 0
    for t_chunk in t_index_split_chunk:
        vision_ntoken_per_chunk = (
            len(t_chunk) * grid_h * grid_w // (spatial_merge_size**2)
        )
        updates.extend([video_token_id] * vision_ntoken_per_chunk)

        audio_chunk_size = min(t_ntoken_per_chunk, audio_len - added_audio_len)
        updates.extend(audio_chunk_size * [audio_token_id])
        added_audio_len += audio_chunk_size
    if added_audio_len < audio_len:
        updates.extend((audio_len - added_audio_len) * [audio_token_id])
    updates.extend([audio_end_token_id])
    return updates


def test_qwen2_5_omni_sparse_gap_does_not_call_dense_split(
    monkeypatch: pytest.MonkeyPatch,
    mock_qwen2_5_omni_config: PretrainedConfig,
) -> None:
    def fail_dense_split(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("dense empty-bucket allocation was called")

    monkeypatch.setattr(
        qwen2_5_omni_thinker,
        "split_list_into_ranges",
        fail_dense_split,
        raising=False,
    )

    updates = Qwen2_5OmniThinkerMultiModalProcessor.omni_get_updates_use_audio_in_video(
        thinker_config=mock_qwen2_5_omni_config,
        audio_len=50,
        video_grid_thw=[2, 4, 4],
        video_second_per_grid_t=100000.0,
    )

    assert updates == (
        [mock_qwen2_5_omni_config.audio_start_token_id]
        + [mock_qwen2_5_omni_config.video_token_index] * 4
        + [mock_qwen2_5_omni_config.audio_token_index] * 50
        + [mock_qwen2_5_omni_config.video_token_index] * 4
        + [mock_qwen2_5_omni_config.audio_end_token_id]
    )


def test_qwen2_5_omni_sparse_gap_matches_historical_dense_behavior(
    mock_qwen2_5_omni_config: PretrainedConfig,
) -> None:
    kwargs = {
        "thinker_config": mock_qwen2_5_omni_config,
        "audio_len": 85,
        "video_grid_thw": [6, 4, 4],
        "video_second_per_grid_t": 2.0,
    }

    assert Qwen2_5OmniThinkerMultiModalProcessor.omni_get_updates_use_audio_in_video(
        **kwargs
    ) == _historical_dense_updates(**kwargs)


@pytest.mark.parametrize(
    "second_per_grid_ts",
    [
        None,
        1.0,
        "1.0",
        [float("inf")],
        [float("nan")],
        [10**400],
        [0.0],
        [-1.0],
        [],
        [1.0, 2.0],
    ],
)
def test_qwen2_5_omni_rejects_invalid_second_per_grid_ts_before_processor_lookup(
    second_per_grid_ts: object,
) -> None:
    processor = Qwen2_5OmniThinkerMultiModalProcessor.__new__(
        Qwen2_5OmniThinkerMultiModalProcessor
    )
    processor.info = Mock()
    processor.info.get_hf_processor.side_effect = AssertionError(
        "processor lookup must not run for invalid second_per_grid_ts"
    )
    mm_items = MultiModalDataItems({"video": VideoProcessorItems([None])})

    with pytest.raises(ValueError, match="second_per_grid_ts"):
        processor._get_prompt_updates(
            mm_items=mm_items,
            hf_processor_mm_kwargs={
                "use_audio_in_video": True,
                "second_per_grid_ts": second_per_grid_ts,
            },
            out_mm_kwargs=Mock(),
        )


def test_qwen2_5_omni_cached_path_rejects_invalid_explicit_timing_before_processing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_cached_apply(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("base cached processor path must not run")

    monkeypatch.setattr(
        BaseMultiModalProcessor,
        "_cached_apply_hf_processor",
        fail_cached_apply,
    )
    processor = Qwen2_5OmniThinkerMultiModalProcessor.__new__(
        Qwen2_5OmniThinkerMultiModalProcessor
    )
    processor._apply_hf_processor = Mock(
        side_effect=AssertionError("HF processor work must not run")
    )
    inputs = ProcessorInputs(
        prompt=[],
        mm_data_items=MultiModalDataItems({"video": VideoProcessorItems([None])}),
        hf_processor_mm_kwargs={
            "use_audio_in_video": True,
            "second_per_grid_ts": [float("inf")],
        },
    )

    with pytest.raises(ValueError, match="second_per_grid_ts"):
        processor._cached_apply_hf_processor(inputs, TimingContext(enabled=False))

    processor._apply_hf_processor.assert_not_called()


def test_qwen2_5_omni_explicit_second_per_grid_ts_bypasses_processor_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_cached_apply(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("base cached processor path must not run")

    monkeypatch.setattr(
        BaseMultiModalProcessor,
        "_cached_apply_hf_processor",
        fail_cached_apply,
    )
    processor = Qwen2_5OmniThinkerMultiModalProcessor.__new__(
        Qwen2_5OmniThinkerMultiModalProcessor
    )
    expected = ([1], Mock(), False)
    processor._apply_hf_processor = Mock(return_value=expected)
    inputs = ProcessorInputs(
        prompt=[],
        mm_data_items=MultiModalDataItems({"video": VideoProcessorItems([None])}),
        hf_processor_mm_kwargs={
            "use_audio_in_video": True,
            "second_per_grid_ts": [1.0],
        },
    )
    timing_ctx = TimingContext(enabled=False)

    assert processor._cached_apply_hf_processor(inputs, timing_ctx) == expected
    processor._apply_hf_processor.assert_called_once_with(inputs, timing_ctx)


def test_qwen2_5_omni_prompt_only_timing_is_not_processor_init_kwarg() -> None:
    info = qwen2_5_omni_thinker.Qwen2_5OmniThinkerProcessingInfo.__new__(
        qwen2_5_omni_thinker.Qwen2_5OmniThinkerProcessingInfo
    )
    info.ctx = Mock()
    expected = Mock()
    info.ctx.get_hf_processor.return_value = expected

    assert (
        info.get_hf_processor(
            use_fast=False,
            use_audio_in_video=True,
            second_per_grid_ts=[2.0],
        )
        is expected
    )

    info.ctx.get_hf_processor.assert_called_once_with(
        qwen2_5_omni_thinker.Qwen2_5OmniProcessor,
        use_fast=False,
        use_audio_in_video=True,
    )


def test_qwen2_5_omni_omitted_second_per_grid_ts_uses_base_cached_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = ([1], Mock(), False)
    calls = []

    def base_cached_apply(
        self: BaseMultiModalProcessor,
        inputs: ProcessorInputs,
        timing_ctx: TimingContext,
    ) -> tuple[list[int], object, bool]:
        calls.append((self, inputs, timing_ctx))
        return expected

    monkeypatch.setattr(
        BaseMultiModalProcessor,
        "_cached_apply_hf_processor",
        base_cached_apply,
    )
    processor = Qwen2_5OmniThinkerMultiModalProcessor.__new__(
        Qwen2_5OmniThinkerMultiModalProcessor
    )
    processor._apply_hf_processor = Mock(
        side_effect=AssertionError("uncached processor path must not run")
    )
    inputs = ProcessorInputs(
        prompt=[],
        mm_data_items=MultiModalDataItems({"video": VideoProcessorItems([None])}),
        hf_processor_mm_kwargs={"use_audio_in_video": True},
    )
    timing_ctx = TimingContext(enabled=False)

    assert processor._cached_apply_hf_processor(inputs, timing_ctx) == expected
    assert calls == [(processor, inputs, timing_ctx)]
    processor._apply_hf_processor.assert_not_called()


def test_qwen2_5_omni_omitted_second_per_grid_ts_defaults_to_one() -> None:
    assert Qwen2_5OmniThinkerMultiModalProcessor._normalize_second_per_grid_ts(
        num_videos=2,
    ) == [1.0, 1.0]
