# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest
from transformers import Qwen3VLVideoProcessor

from vllm.model_executor.models.transformers.multimodal import (
    OffsetsMultiModalProcessor,
)

from .transformers_backend import (
    create_cached_processor,
    create_processor,
    offsets_only,
)

pytestmark = offsets_only

VIDEO_MODEL_SETTINGS = {
    "Qwen/Qwen3-VL-2B-Instruct": {
        "prompt": (
            "<|im_start|>user\n"
            "<|vision_start|><|video_pad|><|vision_end|>"
            "Describe this video.<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
    },
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf": {
        "prompt": (
            "<|im_start|>user <video>\nDescribe this video.<|im_end|>"
            "<|im_start|>assistant\n"
        ),
    },
    "CohereLabs/North-Micro-Vision-Instruct": {
        "prompt": (
            "<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>"
            "<|VISION_START|><|VIDEO_PAD|><|VISION_END|>Describe this video."
            "<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
        ),
    },
    "Qwen/Qwen2.5-VL-3B-Instruct": {
        "prompt": (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "<|vision_start|><|video_pad|><|vision_end|>"
            "Describe this video.<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
    },
}


def _video(num_frames: int = 8, seed: int = 0) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    frames = rng.integers(256, size=(num_frames, 64, 64, 3), dtype=np.uint8)
    metadata = {
        "fps": 1.0,
        "total_num_frames": num_frames,
        "frames_indices": list(range(num_frames)),
        "do_sample_frames": False,
    }
    return frames, metadata


def _hf_video_kwargs(*videos):
    return {
        "videos": [frames for frames, _ in videos],
        "video_metadata": [
            {k: v for k, v in metadata.items() if k != "do_sample_frames"}
            for _, metadata in videos
        ],
        "do_sample_frames": False,
    }


@pytest.mark.parametrize("model_id", list(VIDEO_MODEL_SETTINGS))
def test_video_multimodal_processor(model_id):
    mm_processor = create_processor(model_id, OffsetsMultiModalProcessor)
    hf_processor = mm_processor.info.get_hf_processor()

    video = _video()
    frames, _ = video
    result = mm_processor(
        prompt=VIDEO_MODEL_SETTINGS[model_id]["prompt"],
        mm_items=mm_processor.info.parse_mm_data({"video": video}),
        hf_processor_mm_kwargs={},
    )

    hf_inputs = hf_processor(
        text=VIDEO_MODEL_SETTINGS[model_id]["prompt"],
        **_hf_video_kwargs(video),
        return_mm_token_type_ids=True,
        return_tensors="pt",
    )
    assert result["prompt_token_ids"] == hf_inputs["input_ids"][0].tolist()

    (placeholder,) = result["mm_placeholders"]["video"]
    (num_video_tokens,) = hf_processor._get_num_multimodal_tokens(
        video_sizes=[frames.shape[:3]]
    )["num_video_tokens"]
    assert placeholder.get_num_embeds() == num_video_tokens
    assert placeholder.get_num_embeds() == int(
        (hf_inputs["mm_token_type_ids"] == 2).sum()
    )

    (item,) = result["mm_kwargs"]["video"]
    rows = item["pixel_values_videos"].data.shape[0]
    if "video_grid_thw" in item:
        assert rows == int(item["num_video_patches"].data)
    else:
        assert rows == len(frames)


@pytest.mark.parametrize("model_id", list(VIDEO_MODEL_SETTINGS))
def test_video_multiple_inputs(model_id):
    """Multiple videos per prompt are each detected as a separate placeholder
    and multi-modal item by the Transformers modelling backend."""
    mm_processor = create_processor(model_id, OffsetsMultiModalProcessor)
    hf_processor = mm_processor.info.get_hf_processor()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "video"},
                {"type": "text", "text": "Compare these videos."},
            ],
        }
    ]
    prompt = hf_processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    videos = [_video(8, seed=0), _video(8, seed=1)]
    result = mm_processor(
        prompt=prompt,
        mm_items=mm_processor.info.parse_mm_data({"video": videos}),
        hf_processor_mm_kwargs={},
    )

    assert len(result["mm_placeholders"]["video"]) == 2
    assert len(result["mm_kwargs"]["video"]) == 2
    hf_inputs = hf_processor(
        text=prompt, **_hf_video_kwargs(*videos), return_tensors="pt"
    )
    assert result["prompt_token_ids"] == hf_inputs["input_ids"][0].tolist()


def test_video_fields_not_claimed_by_image():
    """A prompt holding an image and a video keeps the fields of each on its own
    modality, so the image branch does not claim the video processor's outputs."""
    model_id = "Qwen/Qwen3-VL-2B-Instruct"
    mm_processor = create_processor(model_id, OffsetsMultiModalProcessor)

    result = mm_processor(
        prompt=(
            "<|vision_start|><|image_pad|><|vision_end|> and "
            "<|vision_start|><|video_pad|><|vision_end|> What do these show?"
        ),
        mm_items=mm_processor.info.parse_mm_data(
            {"image": _video()[0][0], "video": _video()}
        ),
        hf_processor_mm_kwargs={},
    )

    (image_item,) = result["mm_kwargs"]["image"]
    (video_item,) = result["mm_kwargs"]["video"]
    assert {"pixel_values", "image_grid_thw"} <= set(image_item.keys())
    assert {"pixel_values_videos", "video_grid_thw"} <= set(video_item.keys())
    assert len(result["mm_placeholders"]["image"]) == 1
    assert len(result["mm_placeholders"]["video"]) == 1


def test_repeated_video_hits_the_processor_cache():
    """Check that mm caching is actually working."""
    mm_processor, cache = create_cached_processor(
        "Qwen/Qwen3-VL-2B-Instruct", OffsetsMultiModalProcessor
    )
    video = _video()

    def process():
        return mm_processor(
            prompt=VIDEO_MODEL_SETTINGS["Qwen/Qwen3-VL-2B-Instruct"]["prompt"],
            mm_items=mm_processor.info.parse_mm_data({"video": video}),
            hf_processor_mm_kwargs={},
        )

    first, second = process(), process()

    assert cache.make_stats().hits > 0
    assert first["prompt_token_ids"] == second["prompt_token_ids"]
    assert first["mm_hashes"] == second["mm_hashes"]


def test_video_unsupported_when_processor_cannot_count_tokens(monkeypatch):
    """A processor that cannot count video tokens is served as image-only."""
    monkeypatch.delattr(Qwen3VLVideoProcessor, "get_num_of_video_patches")
    mm_processor = create_processor(
        "Qwen/Qwen3-VL-2B-Instruct", OffsetsMultiModalProcessor
    )

    assert "video" not in mm_processor.info.get_supported_mm_limits()


def test_merged_token_fields_split_per_video():
    """VideoLLaMA3's compression mask has one row per merged token, not per patch."""
    mm_processor = create_processor(
        "lkhl/VideoLLaMA3-2B-Image-HF", OffsetsMultiModalProcessor
    )

    result = mm_processor(
        prompt="<|im_start|>user\n<|video_pad|>\n<|video_pad|>\nDescribe.<|im_end|>\n",
        mm_items=mm_processor.info.parse_mm_data({"video": [_video(8), _video(4)]}),
        hf_processor_mm_kwargs={},
    )

    items = result["mm_kwargs"]["video"]
    assert [len(item["video_compression_mask"].data) for item in items] == [128, 64]
