# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import LruKeyReplicatedSenderCache
from vllm.transformers_utils.configs.lfm3_omni import Lfm3OmniConfig

from ....multimodal.utils import random_audio, random_video
from ...utils import build_model_context

LAYER_TYPES = ["conv", "conv", "full_attention"] * 5 + ["full_attention"] * 2
ENCODER = {"num_mel_bins": 128, "conv_channels": 256}
LFM2_ENCODER = {
    "model_type": "lfm2",
    "vocab_size": 2,
    "hidden_size": 768,
    "intermediate_size": 2560,
    "num_hidden_layers": 17,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "layer_types": LAYER_TYPES,
    "rope_parameters": {"rope_type": "default", "rope_theta": 1e6},
}
LFM3_ENCODER = {
    **LFM2_ENCODER,
    "model_type": "lfm3",
    "intermediate_size": 1792,
    "head_dim": 96,
    "sliding_window": 32,
    "use_attention_gate": False,
    "use_sandwich_norm": False,
    "layer_types": [
        "sliding_attention" if t == "full_attention" else t for t in LAYER_TYPES
    ],
    "rope_parameters": {
        "sliding_attention": {"rope_type": "default", "rope_theta": 1e6}
    },
}
# The hub checkpoints keep an LFM2 encoder; the transformers-native export an
# LFM3 one without the attention gate and sandwich norms
HUB = {
    **ENCODER,
    "model_type": "lfm3_omni_encoder",
    "sliding_window": 32,
    "text_config": LFM2_ENCODER,
}
TRANSFORMERS_NATIVE = {**ENCODER, "text_config": LFM3_ENCODER}


@pytest.mark.parametrize("encoder", [HUB, TRANSFORMERS_NATIVE])
def test_config_layouts(encoder):
    config = Lfm3OmniConfig(
        text_config={"model_type": "lfm3"},
        vision_config={"model_type": "lfm3_vl_vision"},
        audio_encoder_config=encoder,
        depth_config={"model_type": "lfm3"},
    )
    assert (config.image_token_id, config.video_token_id) == (125030, 125029)
    assert config.audio_token_id == 125031
    assert config.vision_config.spatial_merge_size == 2
    audio_encoder = config.audio_encoder_config
    assert audio_encoder.sliding_window == 32
    assert audio_encoder.text_config.model_type == "lfm2"
    assert audio_encoder.text_config.layer_types == LAYER_TYPES
    restored = Lfm3OmniConfig(**config.to_dict())
    assert restored.audio_encoder_config.to_dict() == audio_encoder.to_dict()
    assert restored.vision_config.to_dict() == config.vision_config.to_dict()


def soundtrack_data(num_videos: int) -> dict[str, list]:
    rng = np.random.RandomState(0)
    data: dict[str, list] = {"video": [], "audio": []}
    for _ in range(num_videos):
        video = random_video(rng, min_frames=8, max_frames=9, min_wh=64, max_wh=65)
        metadata = {
            "total_num_frames": len(video),
            "fps": 2.0,
            "duration": len(video) / 2.0,
            "video_backend": "opencv",
            "frames_indices": list(range(len(video))),
            "do_sample_frames": False,
        }
        data["video"].append((video, metadata))
        data["audio"].append(random_audio(rng, min_len=16000, max_len=16001, sr=16000))
    return data


@pytest.mark.parametrize("num_videos", [1, 2])
@pytest.mark.parametrize("standalone_clip", [False, True])
def test_soundtrack_without_placeholder(num_videos, standalone_clip):
    """With `use_audio_in_video` the chat parser adds each video's soundtrack as
    the audio item right after it, and the openai content format puts no
    placeholder for it in the prompt. The processor adds one after each video;
    a standalone clip keeps its own placeholder."""
    num_audios = num_videos + standalone_clip
    ctx = build_model_context(
        "LiquidAI/LFM3-Omni-2B",
        limit_mm_per_prompt={"image": 0, "video": num_videos, "audio": num_audios},
        mm_processor_cache_gb=1,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    config = processor.info.get_hf_config()
    video, audio = config.video_token_id, config.audio_token_id
    data = soundtrack_data(num_videos)
    if standalone_clip:
        data["audio"].append(random_audio(np.random.RandomState(1), 8000, 8001, 16000))
    mm_items = processor.info.parse_mm_data(data)

    def run(prompt, cache=None):
        return processor(
            prompt,
            mm_items=mm_items,
            hf_processor_mm_kwargs={"use_audio_in_video": True},
            cache=cache,
        )

    explicit = [video, audio] * num_videos + [audio] * standalone_clip
    out = run(explicit)
    expected = out["prompt_token_ids"]
    # each soundtrack is interleaved inside its video's span
    spans = [(p.offset, p.offset + p.length) for p in out["mm_placeholders"]["video"]]
    assert len(spans) == num_videos
    for start, end in spans:
        assert audio in expected[start:end]
    # a standalone clip keeps its own span after the videos
    assert (audio in expected[spans[-1][1] :]) == standalone_clip

    cache = LruKeyReplicatedSenderCache(ctx.model_config)
    chat_prompt = [video] * num_videos + [audio] * standalone_clip
    for _ in range(2):  # cache miss, then cache hit
        assert run(chat_prompt, cache)["prompt_token_ids"] == expected
