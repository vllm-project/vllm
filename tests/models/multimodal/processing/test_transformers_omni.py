# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import numpy as np
import pytest

from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import (
    LruKeyReplicatedSenderCache,
    MultiModalProcessorOnlyCache,
)

OMNI_MODEL_SETTINGS: dict[str, dict[str, Any]] = {
    "Qwen/Qwen2.5-Omni-3B": {
        "prompt": (
            "<|im_start|>user\n"
            "<|audio_bos|><|AUDIO|><|audio_eos|>"
            "<|vision_bos|><|IMAGE|><|vision_eos|>"
            "<|vision_bos|><|VIDEO|><|vision_eos|>"
            "Describe what you see and hear.<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
    },
    "google/gemma-3n-E2B-it": {
        "prompt": (
            "<bos><start_of_turn>user\n"
            "<audio_soft_token><image_soft_token>"
            "Describe what you see and hear.<end_of_turn>\n<start_of_turn>model\n"
        ),
        "modalities": ("audio", "image"),
    },
    "google/gemma-4-E2B-it": {
        "prompt": (
            "<bos><|turn>user\n"
            "<|audio|><|image|><|video|>Describe what you see and hear.<turn|>\n"
            "<|turn>model\n<|channel>thought\n<channel|>"
        ),
    },
    "Qwen/Qwen3-Omni-30B-A3B-Instruct": {
        "prompt": (
            "<|im_start|>user\n"
            "<|audio_start|><|audio_pad|><|audio_end|>"
            "<|vision_start|><|image_pad|><|vision_end|>"
            "<|vision_start|><|video_pad|><|vision_end|>"
            "Describe what you see and hear.<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
    },
}

SAMPLING_RATE = 16000


def _audio(seconds: float = 1.0):
    return np.zeros(int(SAMPLING_RATE * seconds), dtype=np.float32)


def _image(height: int = 64, width: int = 64):
    return np.random.RandomState(0).randint(0, 256, (height, width, 3), dtype=np.uint8)


def _video(num_frames: int = 4, height: int = 64, width: int = 64):
    frames = np.random.RandomState(1).randint(
        0, 256, (num_frames, height, width, 3), dtype=np.uint8
    )
    metadata = {
        "fps": 1.0,
        "total_num_frames": num_frames,
        "frames_indices": list(range(num_frames)),
        "do_sample_frames": False,
    }
    return frames, metadata


def _modalities(model_id):
    return OMNI_MODEL_SETTINGS[model_id].get("modalities", ("audio", "image", "video"))


def _mm_data(model_id):
    data = {"audio": [_audio()], "image": [_image()], "video": [_video()]}
    return {m: data[m] for m in _modalities(model_id)}


def _prompt(hf_processor, *modalities):
    text = ""
    for modality in modalities:
        if modality == "audio":
            bos, token, eos = "audio_bos_token", "audio_token", "audio_eos_token"
        else:
            bos, eos = "vision_bos_token", "vision_eos_token"
            token = "image_token" if modality == "image" else "video_token"
        text += "".join(getattr(hf_processor, name) for name in (bos, token, eos))
    return text + " Describe this."


@pytest.mark.parametrize("model_id", list(OMNI_MODEL_SETTINGS))
def test_omni_all_modalities_are_supported(model_id):
    """An Omni processor reports every modality it takes, not one or two of them."""
    mm_processor = MULTIMODAL_REGISTRY.create_processor(
        ModelConfig(model=model_id, model_impl="transformers")
    )
    assert set(_modalities(model_id)) <= set(
        mm_processor.info.get_supported_mm_limits()
    )
    assert mm_processor.info.get_max_audio_tokens() > 0


@pytest.mark.parametrize("model_id", list(OMNI_MODEL_SETTINGS))
def test_omni_multimodal_processor(model_id):
    """One prompt holding audio, an image and a video yields one placeholder each."""
    mm_processor = MULTIMODAL_REGISTRY.create_processor(
        ModelConfig(model=model_id, model_impl="transformers")
    )
    mm_data = _mm_data(model_id)

    result = mm_processor(
        prompt=OMNI_MODEL_SETTINGS[model_id]["prompt"],
        mm_items=mm_processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs={},
    )

    for modality in _modalities(model_id):
        assert len(result["mm_placeholders"][modality]) == 1
        assert len(result["mm_kwargs"][modality]) == 1


def test_omni_fields_stay_on_their_own_modality():
    """The audio branch does not claim the image or video processor's outputs."""
    model_id = "Qwen/Qwen2.5-Omni-3B"
    mm_processor = MULTIMODAL_REGISTRY.create_processor(
        ModelConfig(model=model_id, model_impl="transformers")
    )
    mm_data = {"audio": [_audio()], "image": [_image()], "video": [_video()]}

    result = mm_processor(
        prompt=OMNI_MODEL_SETTINGS[model_id]["prompt"],
        mm_items=mm_processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs={},
    )

    (audio_item,) = result["mm_kwargs"]["audio"]
    (image_item,) = result["mm_kwargs"]["image"]
    (video_item,) = result["mm_kwargs"]["video"]
    assert {"input_features", "feature_attention_mask"} <= set(audio_item.keys())
    assert {"pixel_values", "image_grid_thw"} <= set(image_item.keys())
    assert {"pixel_values_videos", "video_grid_thw", "second_per_grid_ts"} <= set(
        video_item.keys()
    )
    assert not set(audio_item.keys()) & set(video_item.keys())


def test_omni_audio_placeholder_matches_the_processor_count():
    """The placeholder the backend locates is as long as the processor's own count."""
    model_id = "Qwen/Qwen2.5-Omni-3B"
    mm_processor = MULTIMODAL_REGISTRY.create_processor(
        ModelConfig(model=model_id, model_impl="transformers")
    )
    audio = _audio(1.7)

    result = mm_processor(
        prompt=_prompt(mm_processor.info.get_hf_processor(), "audio"),
        mm_items=mm_processor.info.parse_mm_data({"audio": [audio]}),
        hf_processor_mm_kwargs={},
    )

    hf_processor = mm_processor.info.get_hf_processor()
    expected = hf_processor._get_num_multimodal_tokens(
        audio_lengths=[len(audio)]
    ).num_audio_tokens[0]
    (placeholder,) = result["mm_placeholders"]["audio"]
    assert placeholder.get_num_embeds() == expected


@pytest.mark.parametrize("seconds", [1.3, 20.0])
def test_omni_audio_in_video_shares_one_span(seconds):
    """Audio folded into a video shares its span, so each gets a view of it."""
    model_id = "Qwen/Qwen2.5-Omni-3B"
    mm_processor = MULTIMODAL_REGISTRY.create_processor(
        ModelConfig(
            model=model_id,
            model_impl="transformers",
            mm_processor_kwargs={"use_audio_in_video": True},
        )
    )
    mm_data = {"audio": [_audio(seconds)], "video": [_video(num_frames=2)]}

    result = mm_processor(
        prompt=_prompt(mm_processor.info.get_hf_processor(), "video"),
        mm_items=mm_processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs={},
    )

    (audio,) = result["mm_placeholders"]["audio"]
    (video,) = result["mm_placeholders"]["video"]
    (video_item,) = result["mm_kwargs"]["video"]
    assert (audio.offset, audio.length) == (video.offset, video.length)
    assert not bool((audio.is_embed & video.is_embed).any())
    assert audio.get_num_embeds() > 0 and video.get_num_embeds() > 0
    assert bool(video_item["use_audio_in_video"].data.item())


def test_omni_audio_in_video_needs_one_audio_per_video():
    """A video carries one audio, so there cannot be more videos than audios."""
    model_id = "Qwen/Qwen2.5-Omni-3B"
    mm_processor = MULTIMODAL_REGISTRY.create_processor(
        ModelConfig(
            model=model_id,
            model_impl="transformers",
            mm_processor_kwargs={"use_audio_in_video": True},
        )
    )
    hf_processor = mm_processor.info.get_hf_processor()
    cache = MultiModalProcessorOnlyCache(mm_processor.info.ctx.model_config)

    def process(mm_data, *modalities):
        return mm_processor(
            prompt=_prompt(hf_processor, *modalities),
            mm_items=mm_processor.info.parse_mm_data(mm_data),
            hf_processor_mm_kwargs={},
            cache=cache,
        )

    too_many_videos = {"audio": [_audio()], "video": [_video(), _video(num_frames=2)]}
    with pytest.raises(ValueError, match="num_audios=1 and num_videos=2"):
        process(too_many_videos, "video", "video")

    # The counts are the request's own, so a cache hit cannot bypass the check
    process({"audio": [_audio()], "video": [_video()]}, "video")
    process({"audio": [_audio(1.3)], "video": [_video(num_frames=2)]}, "video")
    with pytest.raises(ValueError, match="num_audios=1 and num_videos=2"):
        process(too_many_videos, "video", "video")
    assert cache.make_stats().hits > 0

    result = mm_processor(
        prompt=_prompt(hf_processor, "audio", "video"),
        mm_items=mm_processor.info.parse_mm_data(
            {"audio": [_audio(), _audio(1.3)], "video": [_video()]}
        ),
        hf_processor_mm_kwargs={},
    )
    audios = result["mm_placeholders"]["audio"]
    assert len(audios) == 2
    assert len(result["mm_placeholders"]["video"]) == 1
    # Standalone audios come first, so the last one is the video's own track
    expected = hf_processor._get_num_multimodal_tokens(
        audio_lengths=[len(_audio()), len(_audio(1.3))]
    ).num_audio_tokens
    assert [audio.get_num_embeds() for audio in audios] == expected


@pytest.mark.parametrize("model_id", list(OMNI_MODEL_SETTINGS))
def test_repeated_omni_request_hits_the_processor_cache(model_id):
    """A repeated request returns the same ids and hashes, with its fields intact."""
    model_config = ModelConfig(model=model_id, model_impl="transformers")
    model_config.multimodal_config.mm_processor_cache_gb = 4
    mm_processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    cache = MultiModalProcessorOnlyCache(model_config)
    mm_data = _mm_data(model_id)
    prompt = OMNI_MODEL_SETTINGS[model_id]["prompt"]

    def process():
        return mm_processor(
            prompt=prompt,
            mm_items=mm_processor.info.parse_mm_data(mm_data),
            hf_processor_mm_kwargs={},
            cache=cache,
        )

    first, second = process(), process()

    assert cache.make_stats().hits > 0
    assert first["prompt_token_ids"] == second["prompt_token_ids"]
    assert first["mm_hashes"] == second["mm_hashes"]
    (audio_item,) = second["mm_kwargs"]["audio"]
    assert ("audio_feature_lengths" in audio_item) == model_config.uses_mrope


def test_repeated_audio_in_video_request_hits_the_processor_cache():
    """The fused path survives a cache hit, where the item comes back as None."""
    model_config = ModelConfig(
        model="Qwen/Qwen2.5-Omni-3B",
        model_impl="transformers",
        mm_processor_kwargs={"use_audio_in_video": True},
    )
    model_config.multimodal_config.mm_processor_cache_gb = 4
    mm_processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    cache = LruKeyReplicatedSenderCache(model_config)
    mm_data = {"audio": [_audio(1.3)], "video": [_video(num_frames=2)]}
    prompt = _prompt(mm_processor.info.get_hf_processor(), "video")

    results = [
        mm_processor(
            prompt=prompt,
            mm_items=mm_processor.info.parse_mm_data(mm_data),
            hf_processor_mm_kwargs={},
            cache=cache,
        )
        for _ in range(3)
    ]

    assert cache.make_stats().hits > 0
    for result in results[1:]:
        assert result["prompt_token_ids"] == results[0]["prompt_token_ids"]
