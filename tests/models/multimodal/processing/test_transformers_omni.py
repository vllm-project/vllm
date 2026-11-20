# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest
import torch
import transformers
from packaging.version import Version

from vllm.config import ModelConfig
from vllm.model_executor.models.transformers.multimodal import MultiModalProcessingInfo
from vllm.multimodal import MULTIMODAL_REGISTRY

pytestmark = pytest.mark.skipif(
    Version(transformers.__version__) < Version("5.23.0"),
    reason="Requires Omni token counting and text replacement offsets",
)

OMNI_MODEL_IDS = [
    "Qwen/Qwen2.5-Omni-3B",
    "google/gemma-4-E2B-it",
    "Qwen/Qwen3-Omni-30B-A3B-Instruct",
]

SAMPLING_RATE = 16000


def _audio(seconds: float = 1.0):
    return np.zeros(int(SAMPLING_RATE * seconds), dtype=np.float32)


def _image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(256, size=(64, 64, 3), dtype=np.uint8)


def _video(num_frames: int = 4, seed: int = 0) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    frames = rng.integers(256, size=(num_frames, 64, 64, 3), dtype=np.uint8)
    metadata = {
        "fps": 1.0,
        "total_num_frames": num_frames,
        "frames_indices": list(range(num_frames)),
        "do_sample_frames": False,
    }
    return frames, metadata


def _mm_data():
    return {"audio": [_audio()], "image": [_image()], "video": [_video()]}


def _hf_mm_kwargs(mm_data):
    kwargs = {"audio": mm_data.get("audio"), "images": mm_data.get("image")}
    if videos := mm_data.get("video"):
        kwargs |= {
            "videos": [frames for frames, _ in videos],
            "video_metadata": [
                {k: v for k, v in metadata.items() if k != "do_sample_frames"}
                for _, metadata in videos
            ],
            "do_sample_frames": videos[0][1]["do_sample_frames"],
        }
    return kwargs


def _prompt(hf_processor, *modalities):
    content: list[dict[str, str]] = [{"type": modality} for modality in modalities]
    content.append({"type": "text", "text": "Describe what you see and hear."})
    return hf_processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


@pytest.mark.parametrize("model_id", OMNI_MODEL_IDS)
def test_omni_multimodal_processor(model_id):
    """One prompt holding every modality yields one placeholder each,
    and the token ids the HF processor produces itself."""
    mm_processor = MULTIMODAL_REGISTRY.create_processor(
        ModelConfig(model=model_id, model_impl="transformers")
    )
    info = mm_processor.info
    assert isinstance(info, MultiModalProcessingInfo)
    hf_processor = info.get_hf_processor()
    mm_data = _mm_data()
    prompt = _prompt(hf_processor, *mm_data)

    result = mm_processor(
        prompt=prompt,
        mm_items=info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs={},
    )

    for modality in mm_data:
        assert len(result["mm_placeholders"][modality]) == 1
        assert len(result["mm_kwargs"][modality]) == 1

    hf_inputs = hf_processor(text=prompt, **_hf_mm_kwargs(mm_data), return_tensors="pt")
    assert result["prompt_token_ids"] == hf_inputs["input_ids"][0].tolist()


def test_omni_fields_stay_on_their_own_modality():
    """The audio branch does not claim the image or video processor's outputs."""
    model_id = "Qwen/Qwen2.5-Omni-3B"
    mm_processor = MULTIMODAL_REGISTRY.create_processor(
        ModelConfig(model=model_id, model_impl="transformers")
    )
    mm_data = _mm_data()

    result = mm_processor(
        prompt=_prompt(mm_processor.info.get_hf_processor(), *mm_data),
        mm_items=mm_processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs={},
    )

    (audio_item,) = result["mm_kwargs"]["audio"]
    (image_item,) = result["mm_kwargs"]["image"]
    (video_item,) = result["mm_kwargs"]["video"]
    assert audio_item is not None
    assert image_item is not None
    assert video_item is not None
    assert {"input_features", "feature_attention_mask"} <= set(audio_item.keys())
    assert {"pixel_values", "image_grid_thw"} <= set(image_item.keys())
    assert {"pixel_values_videos", "video_grid_thw", "second_per_grid_ts"} <= set(
        video_item.keys()
    )
    assert not set(audio_item.keys()) & set(video_item.keys())


def test_omni_audio_in_video_shares_one_span():
    """Audio folded into a video shares its span, so each gets a view of it."""
    model_id = "Qwen/Qwen2.5-Omni-3B"
    mm_processor = MULTIMODAL_REGISTRY.create_processor(
        ModelConfig(
            model=model_id,
            model_impl="transformers",
            mm_processor_kwargs={"load_audio_from_video": True},
        )
    )
    mm_data = {"audio": [_audio(20.0)], "video": [_video(num_frames=2)]}

    result = mm_processor(
        prompt=_prompt(mm_processor.info.get_hf_processor(), "video"),
        mm_items=mm_processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs={},
    )

    (audio,) = result["mm_placeholders"]["audio"]
    (video,) = result["mm_placeholders"]["video"]
    (video_item,) = result["mm_kwargs"]["video"]
    assert audio.is_embed is not None
    assert video.is_embed is not None
    assert video_item is not None
    assert (audio.offset, audio.length) == (video.offset, video.length)
    assert not bool((audio.is_embed & video.is_embed).any())
    assert audio.get_num_embeds() > 0 and video.get_num_embeds() > 0
    use_audio_in_video = video_item["use_audio_in_video"].data
    assert isinstance(use_audio_in_video, torch.Tensor)
    assert bool(use_audio_in_video.item())


def test_omni_audio_in_video_needs_one_audio_per_video():
    """A video carries one audio, so a request with videos needs as many audios."""
    model_id = "Qwen/Qwen2.5-Omni-3B"
    mm_processor = MULTIMODAL_REGISTRY.create_processor(
        ModelConfig(
            model=model_id,
            model_impl="transformers",
            mm_processor_kwargs={"use_audio_in_video": True},
        )
    )
    hf_processor = mm_processor.info.get_hf_processor()

    def process(mm_data, *modalities):
        return mm_processor(
            prompt=_prompt(hf_processor, *modalities),
            mm_items=mm_processor.info.parse_mm_data(mm_data),
            hf_processor_mm_kwargs={},
        )

    too_many_videos = {"audio": [_audio()], "video": [_video(), _video(num_frames=2)]}
    with pytest.raises(ValueError, match="num_audios=1, num_videos=2"):
        process(too_many_videos, "video", "video")

    too_many_audios = {"audio": [_audio(), _audio(1.3)], "video": [_video()]}
    with pytest.raises(ValueError, match="num_audios=2, num_videos=1"):
        process(too_many_audios, "audio", "video")

    (audio,) = process({"audio": [_audio()]}, "audio")["mm_placeholders"]["audio"]
    assert audio.get_num_embeds() > 0
