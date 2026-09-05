# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.assets.video import VideoAsset
from vllm.config import ModelConfig, VllmConfig
from vllm.multimodal.utils import encode_video_url
from vllm.renderers.hf import HfRenderer
from vllm.renderers.params import ChatParams
from vllm.tokenizers.registry import cached_tokenizer_from_config

pytestmark = pytest.mark.cpu_test

video_data_url = encode_video_url(VideoAsset("baby_reading").np_ndarrays)

DEFAULT_MEDIA_IO_KWARGS = {"video": {"num_frames": 32}}
DEFAULT_MM_PROCESSOR_KWARGS = {"videos_kwargs": {"do_resize": True}}


def _build_video_renderer() -> HfRenderer:
    model_config = ModelConfig(
        model="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        max_model_len=4096,
        mm_processor_cache_gb=4.0,
    )

    return HfRenderer(
        VllmConfig(model_config=model_config),
        cached_tokenizer_from_config(model_config),
    )


def _get_message_hash(
    renderer: HfRenderer,
    messages: list[dict],
    media_io_kwargs: dict | None = None,
    mm_processor_kwargs: dict | None = None,
) -> str:
    media_io_kwargs = media_io_kwargs or DEFAULT_MEDIA_IO_KWARGS
    mm_processor_kwargs = mm_processor_kwargs or DEFAULT_MM_PROCESSOR_KWARGS
    _, inputs = renderer.render_chat(
        [messages],
        ChatParams(
            media_io_kwargs=media_io_kwargs,
            mm_processor_kwargs=mm_processor_kwargs,
        ),
        prompt_extras={"mm_processor_kwargs": mm_processor_kwargs},
    )

    return inputs[0]["mm_hashes"]["video"][0]


def _video_chat_messages(client_uuid: str):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {"url": video_data_url},
                    "uuid": client_uuid,
                },
                {"type": "text", "text": "Describe the video."},
            ],
        }
    ]


def test_mm_hash_includes_uuid_and_processing_kwargs():
    renderer = _build_video_renderer()
    messages = _video_chat_messages(video_data_url)

    base_hash = _get_message_hash(renderer, messages)
    media_io_hash = _get_message_hash(
        renderer,
        messages,
        media_io_kwargs={"video": {"num_frames": 4}},
    )
    uuid_hash = _get_message_hash(
        renderer,
        _video_chat_messages("different-video-uuid"),
    )
    mm_processor_hash = _get_message_hash(
        renderer,
        messages,
        mm_processor_kwargs={"videos_kwargs": {"do_resize": False}},
    )

    assert base_hash != media_io_hash
    assert base_hash != uuid_hash
    assert base_hash != mm_processor_hash
