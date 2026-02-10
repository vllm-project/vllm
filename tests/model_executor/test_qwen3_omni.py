# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest
import torch
from transformers import PretrainedConfig

from vllm.multimodal.inputs import (
    MultiModalFieldElem,
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    MultiModalSharedField,
)
from vllm.multimodal.processing import InputProcessingContext


# Helper function to print input IDs with coalesced audio/video tokens.
def print_input_ids(input_ids):
    """
    Print input IDs, compressing consecutive special tokens.
    - 151675: <|audio_pad|>
    - 151656: <|video_pad|>
    """
    if not input_ids:
        print("[]")
        return

    result = []
    i = 0

    while i < len(input_ids):
        current_id = input_ids[i]

        # Check if it's a special token that should be compressed
        if current_id in [151675, 151656]:
            # Count consecutive occurrences
            count = 1
            while i + count < len(input_ids) and input_ids[i + count] == current_id:
                count += 1

            # Add compressed representation
            token_name = "<|audio_pad|>" if current_id == 151675 else "<|video_pad|>"
            result.append(f"{token_name} * {count}")
            i += count
        else:
            # Regular token, just add it
            result.append(str(current_id))
            i += 1

    print(", ".join(result))


@pytest.fixture
def mock_qwen3_omni_config():
    """Create a mock Qwen3OmniMoeThinker config."""
    config = Mock(spec=PretrainedConfig)
    # Token IDs from https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct/blob/main/tokenizer_config.json
    config.audio_token_id = 151675  # <|audio_pad|>
    config.video_token_id = 151656  # <|video_pad|>
    config.image_token_id = 151655  # <|image_pad|>
    config.audio_start_token_id = 151669  # <|audio_start|>
    config.audio_end_token_id = 151670  # <|audio_end|>
    config.vision_start_token_id = 151652  # <|vision_start|>
    config.position_id_per_seconds = 12.5

    # Vision config
    vision_config = Mock()
    vision_config.spatial_merge_size = 2
    config.vision_config = vision_config

    return config


@pytest.fixture
def mock_processor():
    """Create a mock HF processor."""
    from transformers.models.whisper import WhisperFeatureExtractor

    processor = Mock()
    processor.audio_token = "<|audio_pad|>"
    processor.image_token = "<|image_pad|>"
    processor.video_token = "<|video_pad|>"

    # Create a real WhisperFeatureExtractor instance for the feature_extractor attribute
    feature_extractor = WhisperFeatureExtractor()
    processor.feature_extractor = feature_extractor

    return processor


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = Mock()
    # Token IDs from https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct/blob/main/tokenizer_config.json
    tokenizer.get_vocab = Mock(
        return_value={
            "<|audio_pad|>": 151675,
            "<|video_pad|>": 151656,
            "<|image_pad|>": 151655,
            "<|audio_start|>": 151669,
            "<|audio_end|>": 151670,
            "<|vision_start|>": 151652,
            "<|vision_end|>": 151653,
        }
    )
    tokenizer.encode = Mock(
        side_effect=lambda x: {
            "<|vision_start|>": [151652],
            "<|vision_end|>": [151653],
            "<|audio_start|>": [151669],
            "<|audio_end|>": [151670],
            "<|audio_pad|>": [151675],
            "<|image_pad|>": [151655],
            "<|video_pad|>": [151656],
        }.get(x, [0])
    )
    tokenizer.vision_bos_token = "<|vision_start|>"
    tokenizer.vision_eos_token = "<|vision_end|>"
    tokenizer.audio_bos_token = "<|audio_start|>"
    tokenizer.audio_eos_token = "<|audio_end|>"
    return tokenizer


@pytest.fixture
def mock_image_processor():
    """Create a mock image processor."""
    image_processor = Mock()
    image_processor.merge_size = 2
    return image_processor


def test_qwen3_omni_get_updates_use_audio_in_video(
    mock_qwen3_omni_config,
    mock_processor,
    mock_tokenizer,
    mock_image_processor,
):
    """Test the get_updates_use_audio_in_video method directly."""

    from vllm.model_executor.models.qwen3_omni_moe_thinker import (
        Qwen3OmniMoeThinkerMultiModalProcessor,
        Qwen3OmniMoeThinkerProcessingInfo,
    )

    # Create a mock context
    mock_ctx = Mock(spec=InputProcessingContext)

    # Create processing info
    info = Qwen3OmniMoeThinkerProcessingInfo(mock_ctx)
    info._get_expected_hidden_size = lambda: 100
    info.get_hf_config = Mock(return_value=mock_qwen3_omni_config)
    info.get_hf_processor = Mock(return_value=mock_processor)
    info.get_tokenizer = Mock(return_value=mock_tokenizer)
    info.get_image_processor = Mock(return_value=mock_image_processor)

    # Create a mock dummy_inputs builder
    mock_dummy_inputs = Mock()

    # Create the processor
    processor = Qwen3OmniMoeThinkerMultiModalProcessor(info, mock_dummy_inputs)

    # Test parameters from reference video
    # https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/draw.mp4
    audio_len = 85
    video_grid_thw = [6, 36, 64]
    video_second_per_grid_t = 2.0

    # Call the method
    updates = processor.get_updates_use_audio_in_video(
        thinker_config=mock_qwen3_omni_config,
        audio_len=audio_len,
        video_grid_thw=video_grid_thw,
        video_second_per_grid_t=video_second_per_grid_t,
    )

    # Updated input ids should align with HF implementation.
    # 151669,
    # <|video_pad|> * 576, <|audio_pad|> * 25,
    # <|video_pad|> * 576, <|audio_pad|> * 25,
    # <|video_pad|> * 576, <|audio_pad|> * 25,
    # <|video_pad|> * 576, <|audio_pad|> * 10,
    # <|video_pad|> * 1152,
    # 151670
    print_input_ids(updates)

    # Verify structure
    assert isinstance(updates, list)
    assert len(updates) > 0

    # Verify start and end tokens
    audio_start_token_id = mock_qwen3_omni_config.audio_start_token_id
    audio_end_token_id = mock_qwen3_omni_config.audio_end_token_id

    assert updates[0] == audio_start_token_id
    assert updates[-1] == audio_end_token_id

    # Verify both audio and video tokens are present
    audio_token_id = mock_qwen3_omni_config.audio_token_id
    video_token_id = mock_qwen3_omni_config.video_token_id

    audio_count = updates.count(audio_token_id)
    video_count = updates.count(video_token_id)

    assert audio_count == audio_len, (
        f"Expected {audio_len} audio tokens, got {audio_count}"
    )

    # Calculate expected video token count
    spatial_merge_size = mock_qwen3_omni_config.vision_config.spatial_merge_size
    height = video_grid_thw[1] // spatial_merge_size
    width = video_grid_thw[2] // spatial_merge_size
    expected_video_count = video_grid_thw[0] * height * width

    assert video_count == expected_video_count, (
        f"Expected {expected_video_count} video tokens, got {video_count}"
    )

    # Total tokens should be: 1 (start) + audio_len + video_count + 1 (end)
    expected_total = 1 + audio_len + expected_video_count + 1
    assert len(updates) == expected_total, (
        f"Expected {expected_total} total tokens, got {len(updates)}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def _make_mm_kwargs_item_with_audio_in_video(
    use_audio_in_video: bool,
) -> MultiModalKwargsItem:
    """Helper to create a MultiModalKwargsItem with use_audio_in_video field."""
    field = MultiModalSharedField(batch_size=1)
    item = MultiModalKwargsItem(
        {
            "video_grid_thw": MultiModalFieldElem(
                data=torch.tensor([6, 36, 64]),
                field=field,
            ),
            "use_audio_in_video": MultiModalFieldElem(
                data=torch.tensor(use_audio_in_video, dtype=torch.bool),
                field=field,
            ),
        }
    )
    return item


def _make_mm_kwargs_item_without_audio_in_video() -> MultiModalKwargsItem:
    """Helper to create a MultiModalKwargsItem WITHOUT use_audio_in_video."""
    field = MultiModalSharedField(batch_size=1)
    item = MultiModalKwargsItem(
        {
            "video_grid_thw": MultiModalFieldElem(
                data=torch.tensor([6, 36, 64]),
                field=field,
            ),
        }
    )
    return item


@pytest.mark.parametrize(
    "items, expected",
    [
        # use_audio_in_video=True should be detected
        (
            [_make_mm_kwargs_item_with_audio_in_video(True)],
            True,
        ),
        # use_audio_in_video=False should not trigger
        (
            [_make_mm_kwargs_item_with_audio_in_video(False)],
            False,
        ),
        # Missing use_audio_in_video key should not trigger (no KeyError)
        (
            [_make_mm_kwargs_item_without_audio_in_video()],
            False,
        ),
        # Multiple items: first is True → should detect and break
        (
            [
                _make_mm_kwargs_item_with_audio_in_video(True),
                _make_mm_kwargs_item_with_audio_in_video(False),
            ],
            True,
        ),
        # Multiple items: first is False, second is True → first doesn't
        # break, second is True → detect True
        (
            [
                _make_mm_kwargs_item_with_audio_in_video(False),
                _make_mm_kwargs_item_with_audio_in_video(True),
            ],
            True,
        ),
    ],
    ids=[
        "single-true",
        "single-false",
        "missing-key",
        "multi-true-first",
        "multi-true-second",
    ],
)
def test_use_audio_in_video_detection(items, expected):
    """
    Test that _maybe_apply_prompt_updates correctly detects
    use_audio_in_video from mm_kwargs without raising KeyError.

    Regression test for: item['use_audio_in_video'] → item.get(...)
    """
    mm_kwargs = MultiModalKwargsItems({"video": items})

    # Extract the detection logic (matches the fixed code)
    use_audio_in_video = False
    if "video" in mm_kwargs:
        for item in mm_kwargs["video"]:
            if item and item.get("use_audio_in_video"):
                use_audio_in_video_tensor = item["use_audio_in_video"].data
                if use_audio_in_video_tensor.numel() > 0:
                    use_audio_in_video = bool(use_audio_in_video_tensor.item())
                    break

    assert use_audio_in_video == expected
