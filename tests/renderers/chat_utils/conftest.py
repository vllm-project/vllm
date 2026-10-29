# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.assets.audio import AudioAsset
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.multimodal.utils import encode_audio_url, encode_image_url, encode_video_url


@pytest.fixture(scope="module")
def image_url():
    image = ImageAsset("cherry_blossom")
    return encode_image_url(image.pil_image)


@pytest.fixture(scope="module")
def video_url():
    video = VideoAsset("baby_reading", 1)
    return encode_video_url(video.np_ndarrays)


@pytest.fixture(scope="module")
def audio_url():
    audio = AudioAsset("mary_had_lamb")
    return encode_audio_url(*audio.audio_and_sample_rate)
