# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test: an explicit video ``num_frames`` must take effect for
models whose video loader samples by fps (Qwen2-VL here), not be ignored.
"""

import json
import shutil
from pathlib import Path

import pytest
import requests
from transformers import AutoTokenizer

from tests.utils import RemoteOpenAIServer
from vllm.assets.video import VideoAsset

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"


@pytest.fixture(scope="module")
def video_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    # 243 frames at 25 fps, 640x360. Copied out of the HF cache, whose
    # symlinks resolve outside --allowed-local-media-path.
    path = tmp_path_factory.mktemp("video") / "baby_reading.mp4"
    shutil.copy(VideoAsset(name="baby_reading").video_path, path)
    return path


@pytest.fixture(scope="module")
def video_pad_id() -> int:
    return AutoTokenizer.from_pretrained(MODEL_NAME).convert_tokens_to_ids(
        "<|video_pad|>"
    )


@pytest.fixture(scope="module")
def server(video_path: Path):
    args = [
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "1",
        "--enforce-eager",
        "--limit-mm-per-prompt",
        json.dumps({"video": 1}),
        "--mm-processor-kwargs",
        json.dumps({"min_pixels": 784, "max_pixels": 151200}),
        "--media-io-kwargs",
        json.dumps({"video": {"num_frames": 4}}),
        "--allowed-local-media-path",
        str(video_path.parent),
        "--trust-request-mm-kwargs",
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


def _count_video_tokens(
    server: RemoteOpenAIServer,
    video_path: Path,
    video_pad_id: int,
    num_frames: int | None = None,
) -> int:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": video_path.as_uri()}},
                {"type": "text", "text": "Describe this video."},
            ],
        }
    ]
    body = {"model": MODEL_NAME, "messages": messages}
    if num_frames is not None:
        body["media_io_kwargs"] = {"video": {"num_frames": num_frames}}
    response = requests.post(server.url_for("tokenize"), json=body)
    response.raise_for_status()
    return response.json()["tokens"].count(video_pad_id)


def test_tokenize_video_honours_explicit_num_frames(
    server: RemoteOpenAIServer, video_path: Path, video_pad_id: int
):
    # Each pair of frames is one temporal patch of 20x36 / 4 = 180 tokens.
    # The server-level num_frames=4 applies, and a request can override it.
    assert _count_video_tokens(server, video_path, video_pad_id) == 360
    assert _count_video_tokens(server, video_path, video_pad_id, num_frames=8) == 720
